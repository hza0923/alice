"""Torch shape pipeline: microbench-style ops + shared max-shaped K/V pools (shape_executor only)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from pp_nextgen.runtime.config import RuntimeConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_attn_slots(names: List[str]) -> Tuple[int, int]:
    nk = nv = 0
    for m in names:
        s = str(m).lower()
        if s.endswith("attn_qk"):
            nk += 1
        elif s.endswith("attn_av"):
            nv += 1
    return nk, nv


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


@dataclass
class _ShapeHyper:
    hidden_size: int
    n_head: int
    n_kv_head: int
    head_dim: int
    n_kv_groups: int
    ffn_dim: int
    vocab_size: int
    dtype: torch.dtype
    device: torch.device
    max_batch: int
    max_seq: int
    rope_theta: float
    rms_eps: float


OpFn = Callable[[str, int, int, str], None]


class ShapeTorchPipelineModel:
    """Torch microbench-style decode steps; shared K/V tensors (no per-request KV alloc)."""

    def __init__(
        self,
        merged_model: Dict[str, Any],
        head_ordered_modules: List[str],
        tail_ordered_modules: Optional[List[str]],
        rt: RuntimeConfig,
    ) -> None:
        self._rt = rt
        self.head_ordered_modules = list(head_ordered_modules or [])
        self.tail_ordered_modules = list(tail_ordered_modules or [])
        self._merged = merged_model

        hs = int(merged_model.get("hidden_size", 4096))
        nh = int(merged_model.get("n_head", merged_model.get("num_attention_heads", 32)))
        nkv = int(merged_model.get("n_kv_head", merged_model.get("num_key_value_heads", 32)))
        head_dim = max(1, hs // max(1, nh))
        ffn = int(merged_model.get("ffn_dim", 11008))
        vocab = int(merged_model.get("vocab_size", 32000))
        dt_b = int(merged_model.get("dtype_bytes", 2))
        dtype = torch.float16 if dt_b <= 2 else torch.float32

        device = torch.device(str(rt.shape_device).strip() if rt.shape_device else "cpu")

        max_batch = max(1, int(rt.shape_max_batch_size))
        max_seq = max(1, int(rt.shape_max_seq_len))

        rope_theta = float(
            merged_model.get("rope_theta")
            or (merged_model.get("scheduler") or {}).get("rope_theta")
            or 10000.0
        )
        rms_eps = float(merged_model.get("rms_norm_eps") or 1e-6)

        assert nh % nkv == 0
        self._hyp = _ShapeHyper(
            hidden_size=hs,
            n_head=nh,
            n_kv_head=nkv,
            head_dim=head_dim,
            n_kv_groups=nh // nkv,
            ffn_dim=ffn,
            vocab_size=vocab,
            dtype=dtype,
            device=device,
            max_batch=max_batch,
            max_seq=max_seq,
            rope_theta=rope_theta,
            rms_eps=rms_eps,
        )

        combined = self.head_ordered_modules + self.tail_ordered_modules
        nk, nv = _count_attn_slots(combined)
        self._nk = nk
        self._nv = nv

        self._k_pools: List[torch.Tensor] = []
        self._v_pools: List[torch.Tensor] = []
        self._head_ops: List[OpFn] = []
        self._tail_ops: List[OpFn] = []
        self._layers_built = False

        self._base_seed = int(rt.shape_random_seed) & ((1 << 63) - 1)

    @property
    def has_tail(self) -> bool:
        return bool(self.tail_ordered_modules)

    def _generator(self, req_id: str, ctx: int) -> torch.Generator:
        msg = f"{self._base_seed}:{req_id}:{ctx}".encode()
        digest = hashlib.blake2b(msg, digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**63)
        g = torch.Generator(device=self._hyp.device)
        g.manual_seed(seed)
        return g

    def init_layers(self) -> None:
        """Allocate shared KV pools and build nn.Module ops (once)."""
        if self._layers_built:
            return

        h = self._hyp
        shape_kv = (h.max_batch, h.n_kv_head, h.max_seq, h.head_dim)
        self._k_pools = [torch.zeros(shape_kv, device=h.device, dtype=h.dtype) for _ in range(self._nk)]
        self._v_pools = [torch.zeros(shape_kv, device=h.device, dtype=h.dtype) for _ in range(self._nv)]

        self._head_ops = self._build_ops(self.head_ordered_modules)
        self._tail_ops = self._build_ops(self.tail_ordered_modules)
        self._layers_built = True

    def ensure_kv_session(self, req_id: str, batch_size: int, target_len: int) -> None:
        """No per-request KV; validate bounds against shared pools."""
        if int(target_len) > self._hyp.max_seq:
            raise ValueError(
                f"target_len={target_len} exceeds shape_executor.max_seq_len={self._hyp.max_seq}"
            )
        if int(batch_size) > self._hyp.max_batch:
            raise ValueError(
                f"batch_size={batch_size} exceeds shape_executor.max_batch_size={self._hyp.max_batch}"
            )

    def close_kv_session(self, req_id: str) -> None:
        return

    def close_all_kv_sessions(self) -> None:
        return

    def forward_decode_step(self, req_id: str, context_len: int, batch_size: int, phase: str = "decode") -> None:
        self._run_chain(self._head_ops, req_id, context_len, batch_size, phase)

    def forward_decode_step_head(self, req_id: str, context_len: int, batch_size: int, phase: str = "decode") -> None:
        self.forward_decode_step(req_id, context_len, batch_size, phase)

    def forward_decode_step_tail(self, req_id: str, context_len: int, batch_size: int, phase: str = "decode") -> None:
        self._run_chain(self._tail_ops, req_id, context_len, batch_size, phase)

    def _run_chain(self, ops: List[OpFn], req_id: str, context_len: int, batch_size: int, phase: str) -> None:
        self.init_layers()
        with torch.no_grad():
            for op in ops:
                op(req_id, int(context_len), int(batch_size), phase)

    def _build_ops(self, names: List[str]) -> List[OpFn]:
        h = self._hyp
        ops: List[OpFn] = []
        ki = vi = 0
        for raw in names:
            name = str(raw).lower()
            if name.endswith("attn_qk"):
                if ki >= len(self._k_pools):
                    raise RuntimeError("attn_qk slot index out of range")
                k_tensor = self._k_pools[ki]
                ops.append(self._make_attn_qk(raw, k_tensor, ki))
                ki += 1
            elif name.endswith("attn_av"):
                if vi >= len(self._v_pools):
                    raise RuntimeError("attn_av slot index out of range")
                v_tensor = self._v_pools[vi]
                ops.append(self._make_attn_av(raw, v_tensor, vi))
                vi += 1
            elif "qkv" in name or name.endswith("qkv_projection"):
                ops.append(self._make_qkv(raw))
            elif name.endswith("o_projection") or "attn_wo" in name:
                ops.append(self._make_o_proj(raw))
            elif "up_projection" in name or name.endswith("up_projection"):
                ops.append(self._make_up_proj(raw))
            elif "down_projection" in name or name.endswith("down_projection"):
                ops.append(self._make_down_proj(raw))
            elif name == "input_embed" or "embed" in name:
                ops.append(self._make_embed(raw))
            elif name == "lm_head" or "output_embed" in name:
                ops.append(self._make_lm_head(raw))
            elif "norm" in name or "rms" in name or "layernorm" in name:
                ops.append(self._make_rmsnorm(raw))
            else:
                ops.append(self._make_generic_linear(raw))

        return ops

    def _make_qkv(self, label: str) -> OpFn:
        h = self._hyp
        rotary = RotaryEmbedding(h.head_dim, max_position_embeddings=h.max_seq, base=h.rope_theta).to(h.device).to(h.dtype)
        q_proj = nn.Linear(h.hidden_size, h.n_head * h.head_dim, bias=False, device=h.device).to(h.dtype)
        k_proj = nn.Linear(h.hidden_size, h.n_kv_head * h.head_dim, bias=False, device=h.device).to(h.dtype)
        v_proj = nn.Linear(h.hidden_size, h.n_kv_head * h.head_dim, bias=False, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            if phase == "prefill":
                sl = min(max(1, context_len + 1), h.max_seq)
                x = torch.randn(bs, sl, h.hidden_size, generator=g, device=h.device, dtype=h.dtype)
                pos = torch.arange(0, sl, device=h.device).unsqueeze(0).expand(bs, -1)
            else:
                x = torch.randn(bs, 1, h.hidden_size, generator=g, device=h.device, dtype=h.dtype)
                pos = torch.full((bs, 1), min(context_len, h.max_seq - 1), device=h.device, dtype=torch.long)
            cos, sin = rotary(x, pos.float())
            q = q_proj(x).view(bs, x.shape[1], h.n_head, h.head_dim).transpose(1, 2)
            k = k_proj(x).view(bs, x.shape[1], h.n_kv_head, h.head_dim).transpose(1, 2)
            v = v_proj(x).view(bs, x.shape[1], h.n_kv_head, h.head_dim).transpose(1, 2)
            apply_rotary_pos_emb(q, k, cos, sin)
            _ = v

        return run

    def _make_attn_qk(self, label: str, k_cache: torch.Tensor, _slot: int) -> OpFn:
        h = self._hyp

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            pos = min(max(0, context_len), h.max_seq - 1)
            active = pos + 1
            if phase == "prefill":
                sl = min(max(1, context_len + 1), h.max_seq)
                q = torch.randn(bs, h.n_head, sl, h.head_dim, generator=g, device=h.device, dtype=h.dtype)
                kt = torch.randn(bs, h.n_kv_head, sl, h.head_dim, generator=g, device=h.device, dtype=h.dtype)
                k_cache[:bs, :, :sl, :] = kt
                k_rep = repeat_kv(k_cache[:bs, :, :sl, :], h.n_kv_groups)
                _ = torch.matmul(q, k_rep.transpose(2, 3)) / (h.head_dim ** 0.5)
            else:
                q = torch.randn(bs, h.n_head, 1, h.head_dim, generator=g, device=h.device, dtype=h.dtype)
                k_new = torch.randn(bs, h.n_kv_head, 1, h.head_dim, generator=g, device=h.device, dtype=h.dtype)
                k_cache[:bs, :, pos : pos + 1, :] = k_new
                k_rep = repeat_kv(k_cache[:bs, :, :active, :], h.n_kv_groups)
                _ = torch.matmul(q, k_rep.transpose(2, 3)) / (h.head_dim ** 0.5)

        return run

    def _make_attn_av(self, label: str, v_cache: torch.Tensor, _slot: int) -> OpFn:
        h = self._hyp

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            pos = min(max(0, context_len), h.max_seq - 1)
            active = pos + 1
            if phase == "prefill":
                sl = min(max(1, context_len + 1), h.max_seq)
                attn_w = torch.randn(bs, h.n_head, sl, sl, generator=g, device=h.device, dtype=h.dtype)
                attn_w = F.softmax(attn_w, dim=-1)
                vt = torch.randn(bs, h.n_kv_head, sl, h.head_dim, generator=g, device=h.device, dtype=h.dtype)
                v_cache[:bs, :, :sl, :] = vt
                v_rep = repeat_kv(v_cache[:bs, :, :sl, :], h.n_kv_groups)
                _ = torch.matmul(attn_w, v_rep)
            else:
                attn_w = torch.randn(bs, h.n_head, 1, active, generator=g, device=h.device, dtype=h.dtype)
                attn_w = F.softmax(attn_w, dim=-1)
                v_new = torch.randn(bs, h.n_kv_head, 1, h.head_dim, generator=g, device=h.device, dtype=h.dtype)
                v_cache[:bs, :, pos : pos + 1, :] = v_new
                v_rep = repeat_kv(v_cache[:bs, :, :active, :], h.n_kv_groups)
                _ = torch.matmul(attn_w, v_rep)

        return run

    def _make_o_proj(self, label: str) -> OpFn:
        h = self._hyp
        lin = nn.Linear(h.n_head * h.head_dim, h.hidden_size, bias=False, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            x = torch.randn(bs, seq, h.n_head * h.head_dim, generator=g, device=h.device, dtype=h.dtype)
            _ = lin(x)

        return run

    def _make_up_proj(self, label: str) -> OpFn:
        h = self._hyp
        up = nn.Linear(h.hidden_size, h.ffn_dim, bias=False, device=h.device).to(h.dtype)
        gate = nn.Linear(h.hidden_size, h.ffn_dim, bias=False, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            x = torch.randn(bs, seq, h.hidden_size, generator=g, device=h.device, dtype=h.dtype)
            _ = F.silu(gate(x)) * up(x)

        return run

    def _make_down_proj(self, label: str) -> OpFn:
        h = self._hyp
        lin = nn.Linear(h.ffn_dim, h.hidden_size, bias=False, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            x = torch.randn(bs, seq, h.ffn_dim, generator=g, device=h.device, dtype=h.dtype)
            _ = lin(x)

        return run

    def _make_embed(self, label: str) -> OpFn:
        h = self._hyp
        emb = nn.Embedding(h.vocab_size, h.hidden_size, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            idx = torch.randint(0, h.vocab_size, (bs, seq), generator=g, device=h.device)
            _ = emb(idx)

        return run

    def _make_lm_head(self, label: str) -> OpFn:
        h = self._hyp
        head = nn.Linear(h.hidden_size, h.vocab_size, bias=False, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            x = torch.randn(bs, seq, h.hidden_size, generator=g, device=h.device, dtype=h.dtype)
            _ = head(x)

        return run

    def _make_rmsnorm(self, label: str) -> OpFn:
        h = self._hyp
        norm = RMSNorm(h.hidden_size, eps=h.rms_eps).to(h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            x = torch.randn(bs, seq, h.hidden_size, generator=g, device=h.device, dtype=h.dtype)
            _ = norm(x)

        return run

    def _make_generic_linear(self, label: str) -> OpFn:
        h = self._hyp
        lin = nn.Linear(h.hidden_size, h.hidden_size, bias=False, device=h.device).to(h.dtype)

        def run(req_id: str, context_len: int, batch_size: int, phase: str) -> None:
            g = self._generator(req_id, context_len)
            bs = min(batch_size, h.max_batch)
            seq = 1 if phase != "prefill" else min(max(1, context_len + 1), h.max_seq)
            x = torch.randn(bs, seq, h.hidden_size, generator=g, device=h.device, dtype=h.dtype)
            _ = lin(x)

        return run

    @classmethod
    def from_configs(
        cls,
        merged_model: Dict[str, Any],
        head_ordered_modules: Optional[List[str]] = None,
        tail_ordered_modules: Optional[List[str]] = None,
        ordered_modules: Optional[List[str]] = None,
        rt: Optional[RuntimeConfig] = None,
    ) -> ShapeTorchPipelineModel:
        if rt is None:
            raise ValueError("RuntimeConfig required")
        head = list(head_ordered_modules or ordered_modules or [])
        tail = list(tail_ordered_modules or [])
        return cls(merged_model, head, tail if tail else None, rt)
