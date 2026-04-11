"""Model placeholder + per-request KV reservation (no weights) + numpy module chain.

Worker0 uses head_ordered_modules (ingress) and tail_ordered_modules (ring return);
other workers use head + empty tail from the worker strategy export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _count_module_slots(ordered_modules: List[str]) -> Tuple[int, int]:
    """Count attn_qk (K only) and attn_av (V only) occurrences on this worker."""
    nk = 0
    nv = 0
    for m in ordered_modules:
        if str(m).lower().endswith("attn_qk"):
            nk += 1
        elif str(m).lower().endswith("attn_av"):
            nv += 1
    return nk, nv


@dataclass
class KvSession:
    """Per-request KV storage: attn_qk stages use k_slots; attn_av stages use v_slots."""

    req_id: str
    batch_size: int
    target_len: int
    head_dim: int
    k_slots: List[np.ndarray] = field(default_factory=list)
    v_slots: List[np.ndarray] = field(default_factory=list)
    hidden_state: Optional[np.ndarray] = None

    def total_bytes(self) -> int:
        b = 0
        for a in self.k_slots:
            b += a.nbytes
        for a in self.v_slots:
            b += a.nbytes
        return b


def _small_linear_weights(
    rng: np.random.Generator, in_dim: int, out_dim: int, scale: float = 0.02
) -> np.ndarray:
    return (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)


class _LinearDecodeModule:
    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        dtype: type,
        rng: np.random.Generator,
        activation: Optional[str] = None,
    ) -> None:
        self.name = name
        self._dtype = dtype
        self._w = _small_linear_weights(rng, in_dim, out_dim).astype(dtype)
        self._activation = activation

    def forward(
        self,
        hidden: np.ndarray,
        session: KvSession,
        context_len: int,
        batch_size: int,
    ) -> np.ndarray:
        _ = (session, context_len, batch_size)
        h = np.matmul(hidden.astype(np.float32), self._w.astype(np.float32)).astype(self._dtype)
        if self._activation == "tanh":
            h = np.tanh(h)
        return h


class _AttnSlotWrite:
    """Writes projected hidden into K or V cache at decode position `context_len`."""

    def __init__(
        self,
        model: "PipelineModel",
        kind: str,
        slot_idx: int,
        rng: np.random.Generator,
    ) -> None:
        if kind not in ("k", "v"):
            raise ValueError("kind must be 'k' or 'v'")
        self._model = model
        self._kind = kind
        self._slot_idx = slot_idx
        h = model.hidden_size
        need = model.n_kv_head * model.head_dim
        self._proj = _small_linear_weights(rng, h, need, scale=0.02).astype(model._dtype)

    def forward(
        self,
        hidden: np.ndarray,
        session: KvSession,
        context_len: int,
        batch_size: int,
    ) -> np.ndarray:
        _ = batch_size
        pos = int(context_len)
        if session.target_len > 0:
            pos = min(pos, session.target_len - 1)
        bsz = hidden.shape[0]
        proj = np.matmul(hidden.astype(np.float32), self._proj.astype(np.float32)).astype(
            self._model._dtype
        )
        reshaped = proj.reshape(bsz, self._model.n_kv_head, self._model.head_dim)
        if self._kind == "k":
            session.k_slots[self._slot_idx][:, :, pos, :] = reshaped
        else:
            session.v_slots[self._slot_idx][:, :, pos, :] = reshaped
        return hidden


def _build_module_list(
    model: "PipelineModel", ordered: List[str], rng: np.random.Generator
) -> List[Any]:
    modules: List[Any] = []
    k_idx = v_idx = 0
    h = model.hidden_size
    f = model.ffn_dim
    for raw in ordered:
        name = str(raw).lower()
        if name.endswith("attn_qk"):
            modules.append(_AttnSlotWrite(model, "k", k_idx, rng))
            k_idx += 1
        elif name.endswith("attn_av"):
            modules.append(_AttnSlotWrite(model, "v", v_idx, rng))
            v_idx += 1
        elif "up_projection" in name or name.endswith("up_projection"):
            modules.append(_LinearDecodeModule(raw, h, f, model._dtype, rng, activation="tanh"))
        elif "down_projection" in name or name.endswith("down_projection"):
            modules.append(_LinearDecodeModule(raw, f, h, model._dtype, rng, activation=None))
        else:
            modules.append(_LinearDecodeModule(raw, h, h, model._dtype, rng, activation=None))
    return modules


class PipelineModel:
    """
    head_ordered_modules: compute for ingress frames (master or internal head queue).
    tail_ordered_modules: optional second chain for ring-return path on worker0 only.
    """

    def __init__(
        self,
        name: str,
        num_layers: int,
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        ffn_dim: int,
        vocab_size: int,
        dtype_bytes: int,
        head_ordered_modules: List[str],
        tail_ordered_modules: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size
        self.dtype_bytes = dtype_bytes
        self.head_ordered_modules = list(head_ordered_modules or [])
        self.tail_ordered_modules = list(tail_ordered_modules or [])
        self.head_dim = max(1, hidden_size // max(1, n_head))
        self._dtype = np.float16 if dtype_bytes <= 2 else np.float32
        self._kv_sessions: Dict[str, KvSession] = {}
        combined = self.head_ordered_modules + self.tail_ordered_modules
        self._num_k_slots, self._num_v_slots = _count_module_slots(combined)
        self._rng = np.random.default_rng(42)
        self._head_modules: List[Any] = []
        self._tail_modules: List[Any] = []
        self._layers_built = False

    @property
    def has_tail(self) -> bool:
        return bool(self.tail_ordered_modules)

    @classmethod
    def from_configs(
        cls,
        merged_model: Dict[str, Any],
        head_ordered_modules: Optional[List[str]] = None,
        tail_ordered_modules: Optional[List[str]] = None,
        ordered_modules: Optional[List[str]] = None,
    ) -> "PipelineModel":
        """Backward compatible: if only ordered_modules is set, head=head list, no tail."""
        head = list(head_ordered_modules or ordered_modules or [])
        tail = list(tail_ordered_modules or [])
        return cls(
            name=str(merged_model.get("name", "llama2-7b")),
            num_layers=int(merged_model.get("num_layers", 32)),
            hidden_size=int(merged_model.get("hidden_size", 4096)),
            n_head=int(merged_model.get("n_head", merged_model.get("num_attention_heads", 32))),
            n_kv_head=int(
                merged_model.get("n_kv_head", merged_model.get("num_key_value_heads", 32))
            ),
            ffn_dim=int(merged_model.get("ffn_dim", 11008)),
            vocab_size=int(merged_model.get("vocab_size", 32000)),
            dtype_bytes=int(merged_model.get("dtype_bytes", 2)),
            head_ordered_modules=head,
            tail_ordered_modules=tail if tail else None,
        )

    def init_layers(self) -> None:
        """Eager build (optional). Prefer lazy build on first forward to speed process startup."""
        self._ensure_layers_built()

    def _ensure_layers_built(self) -> None:
        if self._layers_built:
            return
        self._head_modules = _build_module_list(self, self.head_ordered_modules, self._rng)
        if self.tail_ordered_modules:
            self._tail_modules = _build_module_list(self, self.tail_ordered_modules, self._rng)
        else:
            self._tail_modules = []
        self._layers_built = True

    def ensure_kv_session(self, req_id: str, batch_size: int, target_len: int) -> None:
        """Create KV tensors for this request if missing; replace if shape parameters change."""
        self._ensure_layers_built()
        if target_len <= 0:
            return
        if req_id in self._kv_sessions:
            ex = self._kv_sessions[req_id]
            if ex.batch_size == batch_size and ex.target_len == target_len:
                return
            self.close_kv_session(req_id)

        shape = (batch_size, self.n_kv_head, target_len, self.head_dim)
        k_slots = [np.zeros(shape, dtype=self._dtype) for _ in range(self._num_k_slots)]
        v_slots = [np.zeros(shape, dtype=self._dtype) for _ in range(self._num_v_slots)]
        hidden = np.zeros((batch_size, self.hidden_size), dtype=self._dtype)
        self._kv_sessions[req_id] = KvSession(
            req_id=req_id,
            batch_size=batch_size,
            target_len=target_len,
            head_dim=self.head_dim,
            k_slots=k_slots,
            v_slots=v_slots,
            hidden_state=hidden,
        )

    def get_kv_session(self, req_id: str) -> Optional[KvSession]:
        return self._kv_sessions.get(req_id)

    def close_kv_session(self, req_id: str) -> None:
        self._kv_sessions.pop(req_id, None)

    def close_all_kv_sessions(self) -> None:
        self._kv_sessions.clear()

    def forward_decode_step(self, req_id: str, context_len: int, batch_size: int) -> None:
        """Single-path decode: all head modules (middle workers, or worker0 head-only if no tail list)."""
        self._ensure_layers_built()
        sess = self.get_kv_session(req_id)
        if sess is None:
            return
        if sess.hidden_state is None or sess.hidden_state.shape[0] != batch_size:
            sess.hidden_state = np.zeros((batch_size, self.hidden_size), dtype=self._dtype)
        h = sess.hidden_state
        for mod in self._head_modules:
            h = mod.forward(h, sess, context_len, batch_size)
        sess.hidden_state = h

    def forward_decode_step_head(self, req_id: str, context_len: int, batch_size: int) -> None:
        """Worker0 head path only."""
        self.forward_decode_step(req_id, context_len, batch_size)

    def forward_decode_step_tail(self, req_id: str, context_len: int, batch_size: int) -> None:
        """Worker0 tail path only (requires tail_ordered_modules)."""
        self._ensure_layers_built()
        if not self._tail_modules:
            return
        sess = self.get_kv_session(req_id)
        if sess is None:
            return
        if sess.hidden_state is None or sess.hidden_state.shape[0] != batch_size:
            sess.hidden_state = np.zeros((batch_size, self.hidden_size), dtype=self._dtype)
        h = sess.hidden_state
        for mod in self._tail_modules:
            h = mod.forward(h, sess, context_len, batch_size)
        sess.hidden_state = h
