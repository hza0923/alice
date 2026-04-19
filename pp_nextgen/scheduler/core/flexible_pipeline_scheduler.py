"""DP-based fine-grained pipeline scheduler (legacy-compatible core + tail_n)."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pp_nextgen.utils import setup_logger


class DeviceStateManager:
    def __init__(self, device_counts: Dict[str, int], designated_device: str):
        self.designated_device = designated_device
        self.other_device_counts: Dict[str, int] = {}
        for dt, cnt in device_counts.items():
            if dt == designated_device:
                if cnt > 1:
                    self.other_device_counts[dt] = cnt - 1
            elif cnt > 0:
                self.other_device_counts[dt] = cnt

        self.device_types = sorted(self.other_device_counts.keys())
        self.total_states = 1
        self.multipliers: Dict[str, int] = {}
        for dt in self.device_types:
            self.multipliers[dt] = self.total_states
            self.total_states *= self.other_device_counts[dt] + 1

    def state_to_counts(self, state: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        temp = state
        for dt in self.device_types:
            max_c = self.other_device_counts[dt] + 1
            counts[dt] = temp % max_c
            temp //= max_c
        return counts

    def counts_to_state(self, counts: Dict[str, int]) -> int:
        state = 0
        for dt in self.device_types:
            state += counts.get(dt, 0) * self.multipliers[dt]
        return state


class FlexiblePipelineScheduler:
    """Compute time: decode ms(seq_len) = base + increase * seq_len (per bs bucket in registry)."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_perf: Dict[str, Any],
        network_bw: Dict[str, Any],
        device_group: Dict[str, int],
        designated_device: str,
        *,
        use_fine_grained: bool = True,
        designated_tail_n: int = 1,
        strategy_output_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.logger = setup_logger("Scheduler")
        self.model_cfg = model_config
        self.device_perf = device_perf
        self.network_bw = network_bw
        self.device_group = device_group
        self.device_memories = {dev: float(perf["memory_gb"]) for dev, perf in device_perf.items()}
        self.designated_device = designated_device
        self.use_fine_grained = use_fine_grained
        self.designated_tail_n = int(designated_tail_n)
        if self.designated_tail_n < 1:
            raise ValueError("designated_tail_n must be >= 1")
        self.strategy_output_path = strategy_output_path
        # KV / attention buffer scheduling cap: when set, memory checks use this seq length
        # instead of target_seq_len (compute time still uses target_seq_len).
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else None
        if self.max_seq_len is not None and self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1 when set")

        self._decoder_modules: List[str] = list(self.model_cfg["component_config"]["decoder_layer"])
        self.layer_names = self._build_layer_names()
        self.num_layers = len(self.layer_names)
        self.dtype_bytes = int(self.model_cfg.get("dtype_bytes", 2))

        self.logger.info(
            "调度器初始化: layers=%s devices=%s fine=%s tail_n=%s",
            self.num_layers,
            device_group,
            use_fine_grained,
            self.designated_tail_n,
        )

    def _bandwidth_mbps(self, dev1: str, dev2: str) -> float:
        key1 = f"{dev1}-{dev2}"
        key2 = f"{dev2}-{dev1}"
        if key1 in self.network_bw:
            return float(self.network_bw[key1])
        if key2 in self.network_bw:
            return float(self.network_bw[key2])
        return float(self.network_bw.get("default", 10000.0))

    def _build_layer_names(self) -> List[str]:
        layers: List[str] = []
        layers.append(f"input_embed_{self.model_cfg['component_config']['input_embed']}")
        if self.use_fine_grained:
            for i in range(int(self.model_cfg["num_layers"])):
                layers.extend([f"layer{i}_{m}" for m in self.model_cfg["component_config"]["decoder_layer"]])
        else:
            for i in range(int(self.model_cfg["num_layers"])):
                layers.append(f"layer{i}_decoder_layer")
        layers.append(f"output_embed_{self.model_cfg['component_config']['output_embed']}")
        return layers

    def _get_module_name(self, layer_name: str) -> str:
        if layer_name.startswith("input_embed_"):
            return layer_name.replace("input_embed_", "")
        if layer_name.startswith("output_embed_"):
            return layer_name.replace("output_embed_", "")
        return layer_name.split("_", 1)[1]

    def _eval_poly(self, model: Dict[str, Any], x: float) -> float:
        form = str(model.get("form", "linear")).lower()
        c0 = float(model.get("c0", 0.0))
        c1 = float(model.get("c1", 0.0))
        c2 = float(model.get("c2", 0.0))
        if form == "constant":
            return c0
        if form == "quadratic":
            return c0 + c1 * x + c2 * x * x
        return c0 + c1 * x

    def _module_time_model(self, device_type: str, module_name: str, phase: str) -> Dict[str, Any]:
        params = self.device_perf[device_type]["modules"][module_name]
        m = params.get(phase) or params.get("decode") or {}
        return {
            "form": str(m.get("form", "linear")).lower(),
            "c0": float(m.get("c0", 0.0)),
            "c1": float(m.get("c1", 0.0)),
            "c2": float(m.get("c2", 0.0)),
            "x": str(m.get("x", "seq_len")),
        }

    def _get_layer_latency(
        self, device_type: str, layer_name: str, seq_len: int, bs: int, phase: str = "decode"
    ) -> float:
        module_name = self._get_module_name(layer_name)
        if module_name != "decoder_layer":
            model = self._module_time_model(device_type, module_name, phase)
            return self._eval_poly(model, float(seq_len))
        total = 0.0
        for m in self._decoder_modules:
            model = self._module_time_model(device_type, m, phase)
            total += self._eval_poly(model, float(seq_len))
        return total

    def _get_comm_model_bytes(self, module_name: str, bs: int, phase: str) -> Tuple[int, int, int]:
        if module_name == "decoder_layer":
            module_name = "down_projection"

        hidden_size = int(self.model_cfg.get("hidden_size", 4096))
        n_head = int(self.model_cfg.get("n_head", 32))
        n_kv_head = int(self.model_cfg.get("n_kv_head", 8))
        head_dim = hidden_size // n_head
        ffn_dim = int(self.model_cfg.get("ffn_dim", 11008))
        vocab_size = int(self.model_cfg.get("vocab_size", 32000))

        base_elems = 0
        inc_elems = 0
        if module_name == "input_embed":
            base_elems = hidden_size
        elif module_name == "qkv_projection":
            base_elems = (n_head + 2 * n_kv_head) * head_dim
        elif module_name == "attn_qk":
            inc_elems = n_head
        elif module_name == "attn_av":
            base_elems = n_head * head_dim
        elif module_name == "o_projection":
            base_elems = hidden_size
        elif module_name == "up_projection":
            base_elems = ffn_dim
        elif module_name == "down_projection":
            base_elems = hidden_size
        elif module_name == "lm_head":
            base_elems = vocab_size
        else:
            base_elems = hidden_size

        bs_val = int(bs)
        if phase == "prefill":
            if module_name == "lm_head":
                c0 = int(base_elems) * self.dtype_bytes * bs_val
                return c0, 0, 0
            # Prefill transfers full-sequence outputs: c0 + c1*x + c2*x^2 where x=seq_len.
            c0 = 0
            c1 = int(base_elems) * self.dtype_bytes * bs_val
            c2 = int(inc_elems) * self.dtype_bytes * bs_val
            return c0, c1, c2
        # Decode transfers single-token outputs: c0 + c1*x where x=context_len.
        c0 = int(base_elems) * self.dtype_bytes * bs_val
        c1 = int(inc_elems) * self.dtype_bytes * bs_val
        return c0, c1, 0

    def _eval_comm_bytes(self, c0: int, c1: int, c2: int, x: int) -> int:
        return int(c0 + c1 * int(x) + c2 * int(x) * int(x))

    def _get_comm_time_ms(
        self,
        prev_dev_type: str,
        curr_dev_type: str,
        prev_layer_idx: int,
        bs: int,
        target_len: int,
        phase: str = "decode",
    ) -> float:
        prev_layer_name = self.layer_names[prev_layer_idx]
        module_name = self._get_module_name(prev_layer_name)
        c0, c1, c2 = self._get_comm_model_bytes(module_name, bs, phase)
        comm_bytes = self._eval_comm_bytes(c0, c1, c2, target_len)
        bandwidth_mbps = self._bandwidth_mbps(prev_dev_type, curr_dev_type)
        bandwidth_bytes_per_sec = bandwidth_mbps * 1e6 / 8
        if bandwidth_bytes_per_sec == 0:
            return float("inf")
        return (comm_bytes / bandwidth_bytes_per_sec) * 1000

    def _get_layer_memory(self, layer_name: str, bs: int, seq_len: Optional[int] = None) -> float:
        if seq_len is None:
            seq_len = 0

        def _get_module_mem(module_name: str) -> float:
            mm = self.model_cfg["module_memory_gb"].get(module_name, 0)
            if isinstance(mm, dict):
                base = float(mm.get("base", 0.0))
                kv_per_token = float(mm.get("kv_per_token_gb", 0.0))
                if kv_per_token > 0.0:
                    value = base + kv_per_token * float(seq_len) * float(bs)
                else:
                    value = base + float(mm.get("inc", 0.0)) * float(seq_len)
                return value
            return float(mm)

        module_name = self._get_module_name(layer_name)
        if module_name == "decoder_layer":
            return float(sum(_get_module_mem(m) for m in self._decoder_modules))
        return float(_get_module_mem(module_name))

    def _check_memory(self, device_type: str, layer_indices: List[int], bs: int, seq_len: int) -> bool:
        total_memory = sum(
            self._get_layer_memory(self.layer_names[i], bs, seq_len=seq_len) for i in layer_indices
        )
        return total_memory <= self.device_memories[device_type]

    def schedule(self, bs: int, target_seq_len: int, *, quiet: bool = False) -> Optional[Dict[str, Any]]:
        k = self.designated_tail_n
        n_layers = self.num_layers
        if k > n_layers - 1:
            self.logger.error("designated_tail_n=%s too large for n_layers=%s", k, n_layers)
            return None

        if not quiet:
            self.logger.info("开始调度: bs=%s seq_len=%s tail_n=%s", bs, target_seq_len, k)
        mem_seq_len = self.max_seq_len if self.max_seq_len is not None else target_seq_len
        if not quiet and mem_seq_len != target_seq_len:
            self.logger.info("KV 显存估算 seq_len=%s（与计算用的 target_seq_len=%s 分离）", mem_seq_len, target_seq_len)
        latencies = {
            dev: [self._get_layer_latency(dev, layer, target_seq_len, bs) for layer in self.layer_names]
            for dev in self.device_group
        }

        state_mgr = DeviceStateManager(self.device_group, self.designated_device)
        n_states = state_mgr.total_states
        INF = float("inf")
        all_device_types = list(self.device_group.keys())

        best_global_tbt = INF
        best_global_alloc = None

        dp = [[[None for _ in all_device_types] for _ in range(n_states)] for _ in range(n_layers)]

        suffix_layers = list(range(n_layers - k, n_layers))

        for i in range(0, n_layers - k - 1):
            prefix_layers = list(range(i + 1))
            dd_layers = prefix_layers + suffix_layers
            if not self._check_memory(self.designated_device, dd_layers, bs, seq_len=mem_seq_len):
                continue
            dd_comp_time = sum(latencies[self.designated_device][idx] for idx in dd_layers)
            dd_idx = all_device_types.index(self.designated_device)
            alloc_init = {self.designated_device: {0: prefix_layers.copy()}}
            dp[i][0][dd_idx] = (dd_comp_time, alloc_init)

        last_dp_row = n_layers - k - 1
        for m_curr in range(1, n_layers - k):
            if not quiet and m_curr % 20 == 0:
                print(f"当前层: {m_curr}/{n_layers - k - 1}")
            for state in range(n_states):
                counts = state_mgr.state_to_counts(state)
                for d_curr in state_mgr.device_types:
                    if counts.get(d_curr, 0) == 0:
                        continue
                    curr_idx = all_device_types.index(d_curr)
                    prev_counts = counts.copy()
                    prev_counts[d_curr] -= 1
                    prev_state = state_mgr.counts_to_state(prev_counts)
                    for m_prev in range(0, m_curr):
                        start_layer = m_prev + 1
                        layer_indices = list(range(start_layer, m_curr + 1))
                        if not self._check_memory(d_curr, layer_indices, bs, seq_len=mem_seq_len):
                            continue
                        curr_comp_time = sum(latencies[d_curr][idx] for idx in layer_indices)
                        for d_prev in all_device_types:
                            prev_idx = all_device_types.index(d_prev)
                            if dp[m_prev][prev_state][prev_idx] is None:
                                continue
                            prev_tbt, prev_alloc = dp[m_prev][prev_state][prev_idx]
                            comm_time = self._get_comm_time_ms(d_prev, d_curr, m_prev, bs, target_seq_len)
                            stage_time = max(curr_comp_time, comm_time)
                            new_tbt = max(prev_tbt, stage_time)
                            if dp[m_curr][state][curr_idx] is None or new_tbt < dp[m_curr][state][curr_idx][0]:
                                new_alloc = {
                                    dt: {kk: v.copy() for kk, v in inst.items()} for dt, inst in prev_alloc.items()
                                }
                                if d_curr not in new_alloc:
                                    new_alloc[d_curr] = {}
                                new_inst_id = len(new_alloc[d_curr])
                                new_alloc[d_curr][new_inst_id] = layer_indices
                                dp[m_curr][state][curr_idx] = (new_tbt, new_alloc)

        for state in range(n_states):
            for d_last in all_device_types:
                d_last_idx = all_device_types.index(d_last)
                if dp[last_dp_row][state][d_last_idx] is None:
                    continue
                prev_tbt, prev_alloc = dp[last_dp_row][state][d_last_idx]
                comm_time_to_dd = self._get_comm_time_ms(
                    d_last, self.designated_device, last_dp_row, bs, target_seq_len
                )
                final_tbt = max(prev_tbt, comm_time_to_dd)
                if final_tbt < best_global_tbt:
                    best_global_tbt = final_tbt
                    best_global_alloc = copy.deepcopy(prev_alloc)
                    tail = list(range(n_layers - k, n_layers))
                    best_global_alloc[self.designated_device][0].extend(tail)

        if best_global_alloc is None:
            if not quiet:
                self.logger.error("调度失败：无可行解。")
            return None

        if not quiet:
            self.logger.info("最优调度完成，全局 TBT=%.2f ms", best_global_tbt)
        strategy = self._build_strategy_file(
            best_global_alloc, best_global_tbt, bs, target_seq_len, mem_seq_len=mem_seq_len
        )
        if self.strategy_output_path:
            Path(self.strategy_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.strategy_output_path, "w", encoding="utf-8") as f:
                json.dump(strategy, f, indent=2, ensure_ascii=False)
            if not quiet:
                self.logger.info("策略已写入 %s", self.strategy_output_path)
        if not quiet:
            self._print_schedule_summary(strategy)
        return strategy

    def _iter_effective_modules_for_layer(self, layer_name: str) -> List[str]:
        module_name = self._get_module_name(layer_name)
        if module_name == "decoder_layer":
            return list(self._decoder_modules)
        return [module_name]

    def _get_stage_time_params(
        self, device_type: str, layer_indices: List[int], phase: str
    ) -> Tuple[float, float, float]:
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        for layer_idx in layer_indices:
            layer_name = self.layer_names[layer_idx]
            for m in self._iter_effective_modules_for_layer(layer_name):
                model = self._module_time_model(device_type, m, phase)
                c0 += float(model.get("c0", 0.0))
                c1 += float(model.get("c1", 0.0))
                c2 += float(model.get("c2", 0.0))
        return c0, c1, c2

    def _make_model_params(
        self,
        *,
        time_c0: float,
        time_c1: float,
        time_c2: float,
        size_c0: int,
        size_c1: int,
        size_c2: int,
        target_seq_len: int,
    ) -> Dict[str, Any]:
        return {
            "base_time": float(time_c0),
            "increase_time": float(time_c1),
            "quadratic_time": float(time_c2),
            "a": float(time_c0),
            "b": float(time_c1),
            "c": float(time_c2),
            "comp_time_ms": float(time_c0 + time_c1 * target_seq_len + time_c2 * target_seq_len * target_seq_len),
            "base_size": int(size_c0),
            "inc_size": int(size_c1),
            "quadratic_size": int(size_c2),
            "size_a": int(size_c0),
            "size_b": int(size_c1),
            "size_c": int(size_c2),
        }

    def _poly_doc(self, c0: float, c1: float, c2: float, *, x: str, unit: str) -> Dict[str, Any]:
        if abs(float(c2)) > 1e-18:
            return {
                "form": "quadratic",
                "c0": float(c0),
                "c1": float(c1),
                "c2": float(c2),
                "x": x,
                "unit": unit,
                "expr": "c0 + c1 * x + c2 * x^2",
            }
        if abs(float(c1)) > 1e-18:
            return {
                "form": "linear",
                "c0": float(c0),
                "c1": float(c1),
                "c2": 0.0,
                "x": x,
                "unit": unit,
                "expr": "c0 + c1 * x",
            }
        return {
            "form": "constant",
            "c0": float(c0),
            "c1": 0.0,
            "c2": 0.0,
            "x": x,
            "unit": unit,
            "expr": "c0",
        }

    def _build_strategy_file(
        self,
        allocation: Dict[str, Any],
        tbt: float,
        bs: int,
        target_seq_len: int,
        *,
        mem_seq_len: int,
    ) -> Dict[str, Any]:
        flat_stages: List[Dict[str, Any]] = []
        for dev_type, instances in allocation.items():
            for inst_id, layers in instances.items():
                if not layers:
                    continue
                worker_name = f"{dev_type}_{inst_id}"
                flat_stages.append(
                    {"worker_name": worker_name, "layers": sorted(layers), "min_layer": min(layers)}
                )
        sorted_stages = sorted(flat_stages, key=lambda x: x["min_layer"])
        pipeline_stages: List[Dict[str, Any]] = []

        for i, stage_info in enumerate(sorted_stages):
            worker_name = stage_info["worker_name"]
            layer_indices = stage_info["layers"]
            device_type = worker_name.split("_")[0]
            module_names = [self.layer_names[idx] for idx in layer_indices]
            d_c0, d_c1, d_c2 = self._get_stage_time_params(device_type, layer_indices, phase="decode")
            p_c0, p_c1, p_c2 = self._get_stage_time_params(device_type, layer_indices, phase="prefill")
            d_c0 = max(0.0, float(d_c0))
            p_c0 = max(0.0, float(p_c0))
            comp_time_ms = d_c0 + d_c1 * target_seq_len + d_c2 * target_seq_len * target_seq_len

            if i < len(sorted_stages) - 1:
                next_worker_name = sorted_stages[i + 1]["worker_name"]
                next_device_type = next_worker_name.split("_")[0]
                send_layer_idx = sorted_stages[i + 1]["min_layer"] - 1
                is_last_worker = False
            else:
                next_worker_name = sorted_stages[0]["worker_name"]
                next_device_type = next_worker_name.split("_")[0]
                send_layer_idx = max(layer_indices)
                is_last_worker = True

            last_module_name = self._get_module_name(self.layer_names[send_layer_idx])
            d_s0, d_s1, d_s2 = self._get_comm_model_bytes(last_module_name, bs, phase="decode")
            p_s0, p_s1, p_s2 = self._get_comm_model_bytes(last_module_name, bs, phase="prefill")
            comm_bytes_to_next = self._eval_comm_bytes(d_s0, d_s1, d_s2, target_seq_len)
            comm_time_ms = self._get_comm_time_ms(device_type, next_device_type, send_layer_idx, bs, target_seq_len)
            bw_mbps = self._bandwidth_mbps(device_type, next_device_type)
            bw_bytes_per_sec = max(1e-12, bw_mbps * 1e6 / 8)
            comm_scale = 1000.0 / bw_bytes_per_sec
            d_t0, d_t1, d_t2 = d_s0 * comm_scale, d_s1 * comm_scale, d_s2 * comm_scale
            p_t0, p_t1, p_t2 = p_s0 * comm_scale, p_s1 * comm_scale, p_s2 * comm_scale
            d_mem_c0 = 0.0
            d_mem_kv_per_token = 0.0
            for layer_idx in layer_indices:
                layer_name = self.layer_names[layer_idx]
                for m in self._iter_effective_modules_for_layer(layer_name):
                    mm = self.model_cfg["module_memory_gb"].get(m, 0.0)
                    if isinstance(mm, dict):
                        d_mem_c0 += float(mm.get("base", 0.0))
                        d_mem_kv_per_token += float(mm.get("kv_per_token_gb", 0.0))
                    else:
                        d_mem_c0 += float(mm)

            decode_default_model = self._make_model_params(
                time_c0=d_c0,
                time_c1=d_c1,
                time_c2=d_c2,
                size_c0=d_s0,
                size_c1=d_s1,
                size_c2=d_s2,
                target_seq_len=target_seq_len,
            )
            prefill_default_model = self._make_model_params(
                time_c0=p_c0,
                time_c1=p_c1,
                time_c2=p_c2,
                size_c0=p_s0,
                size_c1=p_s1,
                size_c2=p_s2,
                target_seq_len=target_seq_len,
            )
            decode_phase: Dict[str, Any] = {"single_model": dict(decode_default_model)}
            prefill_phase: Dict[str, Any] = {"single_model": dict(prefill_default_model)}

            # The designated worker0 runs both first-pass head and ring-return tail.
            if worker_name == f"{self.designated_device}_0":
                tail_layer_start = max(0, self.num_layers - self.designated_tail_n)
                head_layers = [idx for idx in layer_indices if idx < tail_layer_start]
                tail_layers = [idx for idx in layer_indices if idx >= tail_layer_start]
                if head_layers and tail_layers:
                    h_d0, h_d1, h_d2 = self._get_stage_time_params(device_type, head_layers, phase="decode")
                    t_d0, t_d1, t_d2 = self._get_stage_time_params(device_type, tail_layers, phase="decode")
                    h_p0, h_p1, h_p2 = self._get_stage_time_params(device_type, head_layers, phase="prefill")
                    t_p0, t_p1, t_p2 = self._get_stage_time_params(device_type, tail_layers, phase="prefill")
                    h_d0 = max(0.0, float(h_d0))
                    t_d0 = max(0.0, float(t_d0))
                    h_p0 = max(0.0, float(h_p0))
                    t_p0 = max(0.0, float(t_p0))
                    # Head sends to next worker; tail loops locally on worker0.
                    head_send_layer = max(head_layers)
                    head_mod_name = self._get_module_name(self.layer_names[head_send_layer])
                    h_d_s0, h_d_s1, h_d_s2 = self._get_comm_model_bytes(head_mod_name, bs, phase="decode")
                    h_p_s0, h_p_s1, h_p_s2 = self._get_comm_model_bytes(head_mod_name, bs, phase="prefill")
                    decode_phase = {
                        "head_model": self._make_model_params(
                            time_c0=h_d0,
                            time_c1=h_d1,
                            time_c2=h_d2,
                            size_c0=h_d_s0,
                            size_c1=h_d_s1,
                            size_c2=h_d_s2,
                            target_seq_len=target_seq_len,
                        ),
                        "tail_model": self._make_model_params(
                            time_c0=t_d0,
                            time_c1=t_d1,
                            time_c2=t_d2,
                            size_c0=0,
                            size_c1=0,
                            size_c2=0,
                            target_seq_len=target_seq_len,
                        ),
                    }
                    prefill_phase = {
                        "head_model": self._make_model_params(
                            time_c0=h_p0,
                            time_c1=h_p1,
                            time_c2=h_p2,
                            size_c0=h_p_s0,
                            size_c1=h_p_s1,
                            size_c2=h_p_s2,
                            target_seq_len=target_seq_len,
                        ),
                        "tail_model": self._make_model_params(
                            time_c0=t_p0,
                            time_c1=t_p1,
                            time_c2=t_p2,
                            size_c0=0,
                            size_c1=0,
                            size_c2=0,
                            target_seq_len=target_seq_len,
                        ),
                    }

            pipeline_stages.append(
                {
                    "worker_name": worker_name,
                    "modules_to_execute": module_names,
                    "stage_params": {
                        "decode": decode_phase,
                        "prefill": prefill_phase,
                        # Legacy compatibility fallback.
                        "base_time": d_c0,
                        "increase_time": d_c1,
                        "comp_time_ms": comp_time_ms,
                        "base_size": d_s0,
                        "inc_size": d_s1,
                    },
                    "comm_bytes_to_next": comm_bytes_to_next,
                    "comm_time_ms": comm_time_ms,
                    "next_worker": next_worker_name,
                    "is_last_worker": is_last_worker,
                    "stage_models": {
                        "decode": {
                            "time_ms": {
                                "single": self._poly_doc(d_c0, d_c1, d_c2, x="context_len", unit="ms")
                            },
                            "comm_bytes": {
                                "single": self._poly_doc(d_s0, d_s1, d_s2, x="context_len", unit="bytes")
                            },
                            "comm_time_ms": {
                                "single": self._poly_doc(d_t0, d_t1, d_t2, x="context_len", unit="ms")
                            },
                            "memory_gb": {
                                "single": {
                                    "form": "linear",
                                    "c0": float(d_mem_c0),
                                    "c1": float(d_mem_kv_per_token),
                                    "c2": 0.0,
                                    "x": "seq_len",
                                    "batch_term": "batch_size",
                                    "unit": "GB",
                                    "expr": "c0 + c1 * seq_len * batch_size",
                                }
                            },
                        },
                        "prefill": {
                            "time_ms": {
                                "single": self._poly_doc(p_c0, p_c1, p_c2, x="seq_len", unit="ms")
                            },
                            "comm_bytes": {
                                "single": self._poly_doc(p_s0, p_s1, p_s2, x="seq_len", unit="bytes")
                            },
                            "comm_time_ms": {
                                "single": self._poly_doc(p_t0, p_t1, p_t2, x="seq_len", unit="ms")
                            },
                        },
                    },
                }
            )

            if worker_name == f"{self.designated_device}_0":
                d = pipeline_stages[-1]["stage_models"]["decode"]
                p = pipeline_stages[-1]["stage_models"]["prefill"]
                if "head_model" in decode_phase and "tail_model" in decode_phase:
                    d["time_ms"] = {
                        "head": self._poly_doc(
                            decode_phase["head_model"]["a"],
                            decode_phase["head_model"]["b"],
                            decode_phase["head_model"]["c"],
                            x="context_len",
                            unit="ms",
                        ),
                        "tail": self._poly_doc(
                            decode_phase["tail_model"]["a"],
                            decode_phase["tail_model"]["b"],
                            decode_phase["tail_model"]["c"],
                            x="context_len",
                            unit="ms",
                        ),
                    }
                    d["comm_bytes"] = {
                        "head": self._poly_doc(
                            decode_phase["head_model"]["size_a"],
                            decode_phase["head_model"]["size_b"],
                            decode_phase["head_model"]["size_c"],
                            x="context_len",
                            unit="bytes",
                        ),
                        "tail": self._poly_doc(0, 0, 0, x="context_len", unit="bytes"),
                    }
                    d["comm_time_ms"] = {
                        "head": self._poly_doc(
                            decode_phase["head_model"]["size_a"] * comm_scale,
                            decode_phase["head_model"]["size_b"] * comm_scale,
                            decode_phase["head_model"]["size_c"] * comm_scale,
                            x="context_len",
                            unit="ms",
                        ),
                        "tail": self._poly_doc(0, 0, 0, x="context_len", unit="ms"),
                    }
                    d["memory_gb"] = {
                        "head": dict(d["memory_gb"]["single"]),
                        "tail": {
                            "form": "linear",
                            "c0": 0.0,
                            "c1": 0.0,
                            "c2": 0.0,
                            "x": "seq_len",
                            "batch_term": "batch_size",
                            "unit": "GB",
                            "expr": "c0 + c1 * seq_len * batch_size",
                        },
                    }
                    p["time_ms"] = {
                        "head": self._poly_doc(
                            prefill_phase["head_model"]["a"],
                            prefill_phase["head_model"]["b"],
                            prefill_phase["head_model"]["c"],
                            x="seq_len",
                            unit="ms",
                        ),
                        "tail": self._poly_doc(
                            prefill_phase["tail_model"]["a"],
                            prefill_phase["tail_model"]["b"],
                            prefill_phase["tail_model"]["c"],
                            x="seq_len",
                            unit="ms",
                        ),
                    }
                    p["comm_bytes"] = {
                        "head": self._poly_doc(
                            prefill_phase["head_model"]["size_a"],
                            prefill_phase["head_model"]["size_b"],
                            prefill_phase["head_model"]["size_c"],
                            x="seq_len",
                            unit="bytes",
                        ),
                        "tail": self._poly_doc(0, 0, 0, x="seq_len", unit="bytes"),
                    }
                    p["comm_time_ms"] = {
                        "head": self._poly_doc(
                            prefill_phase["head_model"]["size_a"] * comm_scale,
                            prefill_phase["head_model"]["size_b"] * comm_scale,
                            prefill_phase["head_model"]["size_c"] * comm_scale,
                            x="seq_len",
                            unit="ms",
                        ),
                        "tail": self._poly_doc(0, 0, 0, x="seq_len", unit="ms"),
                    }

        model_name = str(self.model_cfg.get("name", "llama2-7b"))
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "schema_version": "pipeline_strategy.v2",
            "generated_at": generated_at,
            "model": {
                "name": model_name,
                "num_layers": self.model_cfg.get("num_layers"),
                "hidden_size": self.model_cfg.get("hidden_size"),
                "n_head": self.model_cfg.get("n_head"),
                "n_kv_head": self.model_cfg.get("n_kv_head"),
                "ffn_dim": self.model_cfg.get("ffn_dim"),
                "vocab_size": self.model_cfg.get("vocab_size"),
                "dtype_bytes": self.dtype_bytes,
            },
            "schedule_input": {
                "bs": bs,
                "target_seq_len": target_seq_len,
                "max_seq_len": self.max_seq_len,
                "kv_memory_seq_len": mem_seq_len,
                "designated_device": self.designated_device,
                "designated_tail_n": self.designated_tail_n,
                "use_fine_grained": self.use_fine_grained,
            },
            "objective": {"name": "min_tbt_ms", "value": tbt},
            "tbt_ms": tbt,
            "pipeline_stages": pipeline_stages,
        }

    def _print_schedule_summary(self, strategy: Dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print("         流水线 DP 调度结果摘要")
        print("=" * 50)
        print(f" TBT (含通信瓶颈): {strategy['tbt_ms']:.2f} ms\n")
        for i, stage in enumerate(strategy["pipeline_stages"]):
            worker = stage["worker_name"]
            modules = stage["modules_to_execute"]
            comp_time = stage["stage_params"]["comp_time_ms"]
            comm_time = stage["comm_time_ms"]
            bytes_out = stage["comm_bytes_to_next"]
            next_worker = stage["next_worker"]
            stage_bottleneck = max(comp_time, comm_time)
            print(f" [Stage {i:02d}]  {worker}")
            if len(modules) > 3:
                print(f"    modules: ['{modules[0]}', '{modules[1]}' ... '{modules[-1]}'] ({len(modules)})")
            else:
                print(f"    modules: {modules}")
            print(f"    comp_ms: {comp_time:.2f}  comm_ms: {comm_time:.2f}  bytes: {bytes_out} -> {next_worker}")
            print(f"    stage_bottleneck_ms: {stage_bottleneck:.2f}")
        print("=" * 50 + "\n")
