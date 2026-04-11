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

    def _get_layer_latency(self, device_type: str, layer_name: str, seq_len: int, bs: int) -> float:
        module_name = self._get_module_name(layer_name)
        if module_name != "decoder_layer":
            params = self.device_perf[device_type]["modules"][module_name]
            return float(params["base"]) + float(params["increase"]) * float(seq_len)
        total = 0.0
        for m in self._decoder_modules:
            params = self.device_perf[device_type]["modules"][m]
            total += float(params["base"]) + float(params["increase"]) * float(seq_len)
        return total

    def _get_comm_model_bytes(self, module_name: str, bs: int) -> Tuple[int, int]:
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
        base_bytes = int(base_elems) * self.dtype_bytes * bs_val
        inc_bytes = int(inc_elems) * self.dtype_bytes * bs_val
        return base_bytes, inc_bytes

    def _get_comm_bytes(self, module_name: str, bs: int, target_len: int) -> int:
        base_b, inc_b = self._get_comm_model_bytes(module_name, bs)
        return int(base_b + inc_b * int(target_len))

    def _get_comm_time_ms(
        self,
        prev_dev_type: str,
        curr_dev_type: str,
        prev_layer_idx: int,
        bs: int,
        target_len: int,
    ) -> float:
        prev_layer_name = self.layer_names[prev_layer_idx]
        module_name = self._get_module_name(prev_layer_name)
        comm_bytes = self._get_comm_bytes(module_name, bs, target_len)
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
                value = float(mm.get("base", 0.0)) + float(mm.get("inc", 0.0)) * float(seq_len)
                # print(f"module_name: {module_name} mm: {value}")
                return value
            # print(f"module_name: {module_name} mm: {mm}")
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
            if not self._check_memory(self.designated_device, dd_layers, bs, seq_len=target_seq_len):
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
                        if not self._check_memory(d_curr, layer_indices, bs, seq_len=target_seq_len):
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
        strategy = self._build_strategy_file(best_global_alloc, best_global_tbt, bs, target_seq_len)
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

    def _get_stage_time_params(self, device_type: str, layer_indices: List[int]) -> Tuple[float, float]:
        stage_base_time = 0.0
        stage_inc_time = 0.0
        for layer_idx in layer_indices:
            layer_name = self.layer_names[layer_idx]
            for m in self._iter_effective_modules_for_layer(layer_name):
                params = self.device_perf[device_type]["modules"][m]
                stage_base_time += float(params["base"])
                stage_inc_time += float(params["increase"])
        return stage_base_time, stage_inc_time

    def _build_strategy_file(self, allocation: Dict[str, Any], tbt: float, bs: int, target_seq_len: int) -> Dict[str, Any]:
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
            stage_base_time, stage_inc_time = self._get_stage_time_params(device_type, layer_indices)
            comp_time_ms = stage_base_time + stage_inc_time * target_seq_len

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
            comm_base_bytes, comm_inc_bytes = self._get_comm_model_bytes(last_module_name, bs)
            comm_bytes_to_next = int(comm_base_bytes + comm_inc_bytes * int(target_seq_len))
            comm_time_ms = self._get_comm_time_ms(device_type, next_device_type, send_layer_idx, bs, target_seq_len)

            pipeline_stages.append(
                {
                    "worker_name": worker_name,
                    "modules_to_execute": module_names,
                    "stage_params": {
                        "base_time": stage_base_time,
                        "increase_time": stage_inc_time,
                        "comp_time_ms": comp_time_ms,
                        "base_size": comm_base_bytes,
                        "inc_size": comm_inc_bytes,
                    },
                    "comm_bytes_to_next": comm_bytes_to_next,
                    "comm_time_ms": comm_time_ms,
                    "next_worker": next_worker_name,
                    "is_last_worker": is_last_worker,
                    "stage_models": {
                        "decode": {
                            "time_ms": {
                                "form": "linear",
                                "base": stage_base_time,
                                "inc": stage_inc_time,
                                "x": "seq_len",
                                "unit": "ms",
                                "expr": "base + inc * x",
                            },
                            "comm_bytes": {
                                "form": "linear",
                                "base": comm_base_bytes,
                                "inc": comm_inc_bytes,
                                "x": "seq_len",
                                "unit": "bytes",
                                "expr": "base + inc * x",
                            },
                        },
                        "prefill": {"time_ms": None, "comm_bytes": None},
                    },
                }
            )

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
