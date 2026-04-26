"""Split-module micro-benchmarks that emit legacy ``*_all_results.json`` for ``tools/build_registry.py``."""
# 优化兼容cp36
import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Populated by configure_runtime() before any benchmark runs.
device = torch.device("cpu")
device_name = "unset"
REPEAT_CONFIG: Dict[str, int] = {"n_repeats": 20, "warmup_repeats": 1}


def configure_runtime(
    *,
    force_cpu: bool,
    cuda_device: int,
    device_label: str,
    n_repeats: int,
    warmup_repeats: int,
) -> None:
    """Set module-level torch device, JSON ``device`` label, and repeat counts."""
    global device, device_name, REPEAT_CONFIG
    device_name = device_label
    REPEAT_CONFIG = {"n_repeats": n_repeats, "warmup_repeats": warmup_repeats}
    if not force_cpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")


def _empty_cache_if_cuda() -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ==================== 配置类 ====================
@dataclass
class ModelConfig:
    """模型配置参数"""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 1
    num_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 5632
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_key_value_heads == 0
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    """Llama-style RMSNorm (shared by layernorm microbench and pre-norm blocks)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# ==================== 辅助函数 ====================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def gqa_qk_matmul(q: torch.Tensor, k: torch.Tensor, n_groups: int, head_dim: int) -> torch.Tensor:
    """GQA-friendly QK matmul without materializing repeated KV heads."""
    if n_groups == 1:
        return torch.matmul(q, k.transpose(2, 3)) / (head_dim ** 0.5)
    bsz, q_heads, q_len, dim = q.shape
    kv_heads = k.shape[1]
    q_grouped = q.view(bsz, kv_heads, n_groups, q_len, dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped, k)
    return (scores / (head_dim ** 0.5)).reshape(bsz, q_heads, q_len, k.shape[2])

def gqa_av_matmul(attn_w: torch.Tensor, v: torch.Tensor, n_groups: int) -> torch.Tensor:
    """GQA-friendly AV matmul without materializing repeated KV heads."""
    if n_groups == 1:
        return torch.matmul(attn_w, v)
    bsz, q_heads, q_len, kv_len = attn_w.shape
    kv_heads = v.shape[1]
    head_dim = v.shape[3]
    attn_grouped = attn_w.view(bsz, kv_heads, n_groups, q_len, kv_len)
    out = torch.einsum("bhgqk,bhkd->bhgqd", attn_grouped, v)
    return out.reshape(bsz, q_heads, q_len, head_dim)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def measure_time(func, n_repeats=1, warmup=0):
    """测量函数执行时间（毫秒）"""
    for _ in range(warmup):
        func()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    times = []
    for _ in range(n_repeats):
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            func()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            func()
            times.append((time.perf_counter() - start) * 1000)
    
    return float(np.mean(times))

def save_results_to_file(model_name, results, component_name="all"):
    """保存结果到文件"""
    output_file = f"{model_name.replace('/', '_')}_{component_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] 结果已保存到: {output_file}")

def calculate_model_size(tensor_size_bytes):
    """将字节转换为GB"""
    return tensor_size_bytes / (1024 ** 3)

def _get_device_memory_gb() -> float:
    """获取当前设备显存容量（GB）。"""
    try:
        if torch.cuda.is_available() and device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            return float(props.total_memory) / (1024**3)
    except Exception:
        pass
    return 0.0

# ==================== 单组件测试函数 ====================

def test_embed_tokens(config, batch_size, p_max_len, d_max_len, step):
    """测试 Embedding 层 - 从1到max_len逐步测试"""
    print("\n>>> 测试组件: Embedding")
    
    embed = nn.Embedding(
        config.vocab_size, config.hidden_size, device=device, dtype=config.dtype
    )
    embed.eval()
    
    # 计算权重占用
    weight_bytes = embed.weight.numel() * embed.weight.element_size()
    weight_gb = calculate_model_size(weight_bytes)

    results = {
        "component": "embed_tokens",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},  # {seq_len: time_ms}
        "decode_times": {},   # 记录平均值
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill: 从1到p_max_len
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    input_ids = torch.randint(
                        0,
                        config.vocab_size,
                        (batch_size, seq_len),
                        device=device,
                        dtype=torch.int64,
                    )
                    time_ms = measure_time(
                        lambda: embed(input_ids),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_seq_len": seq_len
                    }
                    print(f"\n[err] Prefill seq_len={seq_len} 失败: {e}")
                    break
            
            # Decode: 从1到d_max_len (取平均值)
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    single_token = torch.randint(
                        0,
                        config.vocab_size,
                        (batch_size, 1),
                        device=device,
                        dtype=torch.int64,
                    )
                    time_ms = measure_time(
                        lambda: embed(single_token),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_decode": True
                    }
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        print(f"\n[err] 测试失败: {e}")
    
    finally:
        del embed
        _empty_cache_if_cuda()
    
    return results


def test_layernorm(config, batch_size, p_max_len, d_max_len, step):
    """测试 LayerNorm - 从1到p_max_len逐步测试"""
    print("\n>>> 测试组件: RMSNorm")
    
    norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=config.dtype)
    norm.eval()
    
    # 计算权重占用
    weight_bytes = norm.weight.numel() * norm.weight.element_size()
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "layernorm",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: norm(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_seq_len": seq_len
                    }
                    break
            
            # Decode
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: norm(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del norm
        _empty_cache_if_cuda()
    
    return results


def test_qkv_rope(config, batch_size, p_max_len, d_max_len, step):
    """测试 Pre-norm + fused QKV + RoPE（与 Llama 子层中 QKV 前接 input_layernorm 一致；残差在完整 attention 之后）。"""
    print("\n>>> 测试组件: QKV + RoPE (pre-norm RMSNorm + fused QKV)")
    
    class RotaryEmbedding(nn.Module):
        def __init__(
            self,
            dim: int,
            max_position_embeddings: int = 2048,
            base: float = 10000.0,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ):
            super().__init__()
            self.out_dtype = dtype
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=self.out_dtype), emb.sin().to(dtype=self.out_dtype)

    q_out = config.num_heads * config.head_dim
    k_out = config.num_key_value_heads * config.head_dim
    v_out = config.num_key_value_heads * config.head_dim
    fused_out = q_out + k_out + v_out
    input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=config.dtype)
    qkv_proj = nn.Linear(
        config.hidden_size, fused_out, bias=False, device=device, dtype=config.dtype
    )
    rotary = RotaryEmbedding(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta,
        device=device,
        dtype=config.dtype,
    )
    
    weight_bytes = (
        qkv_proj.weight.numel() * qkv_proj.weight.element_size()
        + input_norm.weight.numel() * input_norm.weight.element_size()
    )
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "qkv_rope",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=config.dtype)
                    pos_ids = torch.arange(
                        0, seq_len, device=device, dtype=torch.int64
                    ).unsqueeze(0).expand(batch_size, -1)
                    
                    def forward_fn():
                        h = input_norm(x)
                        cos, sin = rotary(h, pos_ids)
                        q_part, k_part, v_part = qkv_proj(h).split((q_out, k_out, v_out), dim=-1)
                        q = q_part.view(batch_size, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
                        k = k_part.view(batch_size, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
                        v = v_part.view(batch_size, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
                        return apply_rotary_pos_emb(q, k, cos, sin)
                    
                    time_ms = measure_time(forward_fn, n_repeats=REPEAT_CONFIG['n_repeats'], warmup=REPEAT_CONFIG['warmup_repeats'])
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_seq_len": seq_len
                    }
                    break
            
            # Decode
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    pos_ids = torch.full(
                        (batch_size, 1), seq_len - 1, device=device, dtype=torch.int64
                    )
                    
                    def decode_fn():
                        h = input_norm(x)
                        cos, sin = rotary(h, pos_ids)
                        q_part, k_part, v_part = qkv_proj(h).split((q_out, k_out, v_out), dim=-1)
                        q = q_part.view(batch_size, 1, config.num_heads, config.head_dim).transpose(1, 2)
                        k = k_part.view(batch_size, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
                        v = v_part.view(batch_size, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
                        return apply_rotary_pos_emb(q, k, cos, sin)
                    
                    time_ms = measure_time(decode_fn, n_repeats=REPEAT_CONFIG['n_repeats'], warmup=REPEAT_CONFIG['warmup_repeats'])
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del input_norm, qkv_proj, rotary
        _empty_cache_if_cuda()
    
    return results


def test_attn_qk(config, batch_size, p_max_len, d_max_len, step):
    """测试 Q @ K^T - prefill和decode都记录每一个context_len"""
    print("\n>>> 测试组件: Attention Q@K^T")
    
    # 计算KV Cache占用
    k_cache_bytes = batch_size * config.num_key_value_heads * d_max_len * config.head_dim * 2  # float16
    k_cache_gb = calculate_model_size(k_cache_bytes)
    
    results = {
        "component": "attn_qk",
        "weight_size_gb": 0.0,
        "kvcache_size_gb": round(k_cache_gb, 6),
        "prefill_times": {},  # {seq_len: time}
        "decode_times": {},   # {context_len: time}
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # 预分配 KV Cache (静态方式)，以d_max_len为最大长度，后续根据实际context_len更新
            k_cache = torch.zeros(batch_size, config.num_key_value_heads, d_max_len, 
                                  config.head_dim, device=device, dtype=config.dtype)

            # Prefill: 从1到p_max_len
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    q = torch.randn(batch_size, config.num_heads, seq_len, 
                                   config.head_dim, device=device, dtype=config.dtype)
                    k = torch.randn(batch_size, config.num_key_value_heads, seq_len, 
                                   config.head_dim, device=device, dtype=config.dtype)
                    
                    # 更新 cache
                    k_cache[:, :, :seq_len, :] = k
                    
                    def qk_fn():
                        return gqa_qk_matmul(
                            q,
                            k_cache[:, :, :seq_len, :],
                            config.num_key_value_groups,
                            config.head_dim,
                        )
                    
                    time_ms = measure_time(qk_fn, n_repeats=REPEAT_CONFIG['n_repeats'], warmup=REPEAT_CONFIG['warmup_repeats'])
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()
                
                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_seq_len": seq_len
                    }
                    print(f"\n[err] Prefill seq_len={seq_len} 失败: {e}")
                    break
            
            # Decode: 从1到max_len，每次增加1个token
            kv_len = 1  # 初始context_len
            for step_idx in range(0, (d_max_len - 1) // step + 1):
                context_len = 1 + step_idx * step
                if context_len > d_max_len:
                    break
                
                try:
                    q = torch.randn(batch_size, config.num_heads, 1, 
                                   config.head_dim, device=device, dtype=config.dtype)
                    k_new = torch.randn(batch_size, config.num_key_value_heads, 1, 
                                       config.head_dim, device=device, dtype=config.dtype)
                    
                    # 更新 cache
                    k_cache[:, :, context_len-1:context_len, :] = k_new
                    
                    def decode_qk_fn():
                        return gqa_qk_matmul(
                            q,
                            k_cache[:, :, :context_len, :],
                            config.num_key_value_groups,
                            config.head_dim,
                        )
                    
                    time_ms = measure_time(decode_qk_fn, n_repeats=REPEAT_CONFIG['n_repeats'], warmup=REPEAT_CONFIG['warmup_repeats'])
                    results["decode_times"][str(context_len)] = time_ms

                
                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_context_len": context_len
                    }
                    print(f"\n[err] Decode context_len={context_len} 失败: {e}")
                    break
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        print(f"\n[err] 测试失败: {e}")
    
    finally:
        _empty_cache_if_cuda()
    
    return results


def test_attn_av(config, batch_size, p_max_len, d_max_len, step):
    """测试 Attn_weights @ V - prefill和decode都记录每一个context_len"""
    print("\n>>> 测试组件: Attention Weights @ V")
    
    # 计算KV Cache占用 (V cache)
    v_cache_bytes = batch_size * config.num_key_value_heads * d_max_len * config.head_dim * 2  # float16
    v_cache_gb = calculate_model_size(v_cache_bytes)
    
    results = {
        "component": "attn_av",
        "weight_size_gb": 0.0,
        "kvcache_size_gb": round(v_cache_gb, 6),
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # 预分配 KV Cache, 以d_max_len为最大长度，后续根据实际context_len更新
            v_cache = torch.zeros(batch_size, config.num_key_value_heads, d_max_len, 
                                  config.head_dim, device=device, dtype=config.dtype)
            
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    attn_w = torch.randn(batch_size, config.num_heads, seq_len, seq_len, device=device, dtype=config.dtype)
                    attn_w = torch.nn.functional.softmax(attn_w, dim=-1)
                    v = torch.randn(batch_size, config.num_key_value_heads, seq_len, 
                                   config.head_dim, device=device, dtype=config.dtype)
                    
                    v_cache[:, :, :seq_len, :] = v
                    
                    def av_fn():
                        return gqa_av_matmul(
                            attn_w,
                            v_cache[:, :, :seq_len, :],
                            config.num_key_value_groups,
                        )
                    
                    time_ms = measure_time(av_fn, n_repeats=REPEAT_CONFIG['n_repeats'], warmup=REPEAT_CONFIG['warmup_repeats'])
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()
                    

                
                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_seq_len": seq_len
                    }
                    break
            
            # Decode
            for step_idx in range(0, (d_max_len - 1) // step + 1):
                context_len = 1 + step_idx * step
                if context_len > d_max_len:
                    break
                
                try:
                    attn_w = torch.randn(batch_size, config.num_heads, 1, context_len, device=device, dtype=config.dtype)
                    attn_w = torch.nn.functional.softmax(attn_w, dim=-1)
                    v_new = torch.randn(batch_size, config.num_key_value_heads, 1, 
                                       config.head_dim, device=device, dtype=config.dtype)
                    
                    v_cache[:, :, context_len-1:context_len, :] = v_new
                    
                    def decode_av_fn():
                        return gqa_av_matmul(
                            attn_w,
                            v_cache[:, :, :context_len, :],
                            config.num_key_value_groups,
                        )
                    
                    time_ms = measure_time(decode_av_fn, n_repeats=REPEAT_CONFIG['n_repeats'], warmup=REPEAT_CONFIG['warmup_repeats'])
                    results["decode_times"][str(context_len)] = time_ms
                    
                
                except Exception as e:
                    results["status"] = "partial"
                    results["error_info"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "failed_at_context_len": context_len
                    }
                    break
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        print(f"\n[err] 测试失败: {e}")
    
    finally:
        _empty_cache_if_cuda()
    
    return results


def test_attn_wo(config, batch_size, p_max_len, d_max_len, step):
    """测试 Attention 输出投影 W_o + 残差（hidden + o_proj(attn_out)）。"""
    print("\n>>> 测试组件: Attention Output Projection + residual")
    
    o_proj = nn.Linear(
        config.num_heads * config.head_dim,
        config.hidden_size,
        bias=False,
        device=device,
        dtype=config.dtype,
    )
    o_proj.eval()
    
    # 计算权重占用
    weight_bytes = o_proj.weight.numel() * o_proj.weight.element_size()
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "attn_wo",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    residual = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=config.dtype)
                    attn_merged = torch.randn(
                        batch_size,
                        seq_len,
                        config.num_heads * config.head_dim,
                        device=device,
                        dtype=config.dtype,
                    )
                    time_ms = measure_time(
                        lambda: residual + o_proj(attn_merged),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            # Decode (只记平均值)
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    residual = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    attn_merged = torch.randn(
                        batch_size,
                        1,
                        config.num_heads * config.head_dim,
                        device=device,
                        dtype=config.dtype,
                    )
                    time_ms = measure_time(
                        lambda: residual + o_proj(attn_merged),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del o_proj
        _empty_cache_if_cuda()
    
    return results


def test_up_proj(config, batch_size, p_max_len, d_max_len, step):
    """测试 Pre-norm + MLP gate+up（fused gate_up(RMSNorm(x))，与 Llama post_attention_layernorm 后接 SwiGLU 前半段一致）。"""
    print("\n>>> 测试组件: MLP gate+up (pre-norm RMSNorm + fused)")
    
    input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=config.dtype)
    gate_up = nn.Linear(
        config.hidden_size,
        2 * config.intermediate_size,
        bias=False,
        device=device,
        dtype=config.dtype,
    )
    gate_up.eval()

    def fused_swiglu_branch(x):
        gate, up = gate_up(input_norm(x)).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    
    weight_bytes = (
        gate_up.weight.numel() * gate_up.weight.element_size()
        + input_norm.weight.numel() * input_norm.weight.element_size()
    )
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "mlp_up_proj",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: fused_swiglu_branch(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    break
            
            # Decode (只记平均值)
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: fused_swiglu_branch(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del input_norm, gate_up
        _empty_cache_if_cuda()
    
    return results


def test_down_proj(config, batch_size, p_max_len, d_max_len, step):
    """测试 MLP 下投影 + 残差（hidden + down(intermediate_act)，对应子层末尾相加）。"""
    print("\n>>> 测试组件: MLP 下投影 + residual")
    
    down_proj = nn.Linear(
        config.intermediate_size,
        config.hidden_size,
        bias=False,
        device=device,
        dtype=config.dtype,
    )
    down_proj.eval()
    
    # 计算权重占用
    weight_bytes = down_proj.weight.numel() * down_proj.weight.element_size()
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "mlp_down_proj",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    residual = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=config.dtype)
                    hidden_act = torch.randn(batch_size, seq_len, config.intermediate_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: residual + down_proj(hidden_act),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    break
            
            # Decode (只记平均值)
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    residual = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    hidden_act = torch.randn(batch_size, 1, config.intermediate_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: residual + down_proj(hidden_act),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del down_proj
        _empty_cache_if_cuda()
    
    return results


def test_mlp(config, batch_size, p_max_len, d_max_len, step):
    """测试 Pre-norm + SwiGLU MLP + 残差（x + down(silu(gate)*up(norm(x)))）。"""
    print("\n>>> 测试组件: MLP (pre-norm RMSNorm + SwiGLU + residual)")
    
    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.input_norm = RMSNorm(
                config.hidden_size, config.rms_norm_eps, device=device, dtype=config.dtype
            )
            self.gate_up = nn.Linear(
                config.hidden_size,
                2 * config.intermediate_size,
                bias=False,
                device=device,
                dtype=config.dtype,
            )
            self.down_proj = nn.Linear(
                config.intermediate_size,
                config.hidden_size,
                bias=False,
                device=device,
                dtype=config.dtype,
            )

        def forward(self, x):
            gate, up = self.gate_up(self.input_norm(x)).chunk(2, dim=-1)
            return x + self.down_proj(torch.nn.functional.silu(gate) * up)
    
    mlp = MLP(config)
    mlp.eval()
    
    weight_bytes = (
        mlp.gate_up.weight.numel()
        + mlp.down_proj.weight.numel()
        + mlp.input_norm.weight.numel()
    ) * mlp.gate_up.weight.element_size()
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "mlp",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: mlp(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    break
            
            # Decode (只记平均值)
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: mlp(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del mlp
        _empty_cache_if_cuda()
    
    return results


def test_lm_head(config, batch_size, p_max_len, d_max_len, step):
    """测试 LM Head"""
    print("\n>>> 测试组件: LM Head")
    # Allocate lm_head with ``device=`` + ``dtype=`` so large vocab weights never
    # materialize as a temporary fp32 copy on device.
    gc.collect()
    _empty_cache_if_cuda()
    lm_head = nn.Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=config.dtype,
    )
    lm_head.eval()
    
    # 计算权重占用
    weight_bytes = lm_head.weight.numel() * lm_head.weight.element_size()
    weight_gb = calculate_model_size(weight_bytes)
    
    results = {
        "component": "lm_head",
        "weight_size_gb": round(weight_gb, 6),
        "kvcache_size_gb": 0.0,
        "prefill_times": {},
        "decode_times": {},
        "status": "running",
        "error_info": None
    }
    
    try:
        with torch.no_grad():
            # Prefill JSON keys follow seq_len; timing is last-token logits only, so one
            # [B,1,H] buffer avoids growing B×S×H activations alongside the huge vocab head.
            x_one = torch.randn(
                batch_size, 1, config.hidden_size, device=device, dtype=config.dtype
            )
            for seq_len in range(1, p_max_len + 1, step):
                try:
                    time_ms = measure_time(
                        lambda: lm_head(x_one),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats'],
                    )
                    results["prefill_times"][str(seq_len)] = time_ms
                    _empty_cache_if_cuda()

                except Exception as e:
                    results["status"] = "partial"
                    break
            
            # Decode (只记平均值)
            decode_times = []
            for seq_len in range(1, d_max_len + 1, step):
                try:
                    x = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=config.dtype)
                    time_ms = measure_time(
                        lambda: lm_head(x),
                        n_repeats=REPEAT_CONFIG['n_repeats'],
                        warmup=REPEAT_CONFIG['warmup_repeats']
                    )
                    decode_times.append(time_ms)
                except Exception as e:
                    results["status"] = "partial"
                    break
            
            if decode_times:
                results["decode_times"]["average"] = float(np.mean(decode_times))
            
            if results["status"] == "running":
                results["status"] = "completed"
    
    except Exception as e:
        results["status"] = "failed"
        results["error_info"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    
    finally:
        del lm_head
        _empty_cache_if_cuda()
    
    return results


DEFAULT_MODEL_CONFIGS: List[Tuple[Any, ...]] = [
    # ("llama2-7b", 32000, 4096, 1, 32, 32, 11008),
    # ("llama2-13b", 32000, 5120, 1, 40, 40, 13824),
    ("llama3-3b", 128256, 3072, 1, 24, 8, 8192),
    ("llama3-8b", 128256, 4096, 1, 32, 8, 14336),
    ("mistral-7b", 32768, 4096, 1, 32, 8, 14336),
    # ("aya-8b", 256000, 4096, 1, 32, 8, 14336),
]

DEFAULT_TEST_CONFIGS: List[Tuple[int, int, int, int]] = [
    # (1, 700, 4000, 100),
    # (2, 700, 4000, 100),
    # (4, 700, 4000, 100),
    (8, 600, 1000, 100),
    (16, 600, 1000, 100),
    # (32, 500, 800, 100),
]

QUICK_TEST_CONFIGS: List[Tuple[int, int, int, int]] = [
    (1, 25, 50, 25),
]

DEFAULT_SELECTED_COMPONENTS: Tuple[str, ...] = (
    "embed_tokens",
    "qkv_rope",
    "attn_qk",
    "attn_av",
    "attn_wo",
    "up_proj",
    "down_proj",
    "lm_head",
)


def component_test_registry() -> Dict[str, Any]:
    return {
        "embed_tokens": test_embed_tokens,
        "layernorm": test_layernorm,
        "qkv_rope": test_qkv_rope,
        "attn_qk": test_attn_qk,
        "attn_av": test_attn_av,
        "attn_wo": test_attn_wo,
        "up_proj": test_up_proj,
        "down_proj": test_down_proj,
        "mlp": test_mlp,
        "lm_head": test_lm_head,
    }


def run_all_benchmarks(
    *,
    model_configs: Optional[List[Tuple[Any, ...]]] = None,
    test_configs: Optional[List[Tuple[int, int, int, int]]] = None,
    selected_components: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Run configured sweeps and write one legacy JSON per model. Returns written paths."""
    model_cfgs = model_configs or list(DEFAULT_MODEL_CONFIGS)
    test_cfgs = test_configs or list(DEFAULT_TEST_CONFIGS)
    comps = tuple(selected_components) if selected_components is not None else DEFAULT_SELECTED_COMPONENTS
    out_dir = Path(output_dir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    component_tests = component_test_registry()
    written: List[Path] = []

    print("开始单组件性能测试\n")
    print(f"使用设备: {device}  device_id={device_name!r}\n")

    for name, vocab, hidden, layers, heads, kv_heads, inter_dim in model_cfgs:
        print("\n" + "=" * 80)
        print(f"测试模型: {name}")
        print("=" * 80)

        config = ModelConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads,
            num_key_value_heads=kv_heads,
            intermediate_size=inter_dim,
            dtype=torch.float16,
        )

        model_all_results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": name,
            "device": device_name,
            "device_memory_gb": round(_get_device_memory_gb(), 6),
            "test_configurations": [],
        }

        for batch_size, p_max_len, d_max_len, step_size in test_cfgs:
            print(f"\n{'='*80}")
            print(f"测试配置: batch={batch_size}, p_max_len={p_max_len}, d_max_len={d_max_len}, step={step_size}")
            print("=" * 80)

            config_results: Dict[str, Any] = {
                "batch_size": batch_size,
                "p_max_len": p_max_len,
                "d_max_len": d_max_len,
                "step": step_size,
                "components": {},
            }

            for comp_name in comps:
                fn = component_tests.get(comp_name)
                if not fn:
                    continue
                try:
                    print(f"\n>>> 开始测试: {comp_name}")
                    config_results["components"][comp_name] = fn(
                        config, batch_size, p_max_len, d_max_len, step_size
                    )
                    _empty_cache_if_cuda()
                    gc.collect()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[err] 组件 {comp_name} 测试失败: {e}")
                    import traceback

                    traceback.print_exc()

            model_all_results["test_configurations"].append(config_results)
            gc.collect()
            _empty_cache_if_cuda()
            time.sleep(1)

        out_path = out_dir / f"{name.replace('/', '_')}_all_results.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(model_all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[ok] 模型 {name} 的所有结果已保存到: {out_path}")
        written.append(out_path)

    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device-id",
        default="3060",
        help="Written to JSON ``device``; must match cluster ``device_group`` keys (e.g. 3060, tx2).",
    )
    p.add_argument("--cuda-device", type=int, default=0, help="CUDA device index when GPU profiling.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for <model>_all_results.json",
    )
    p.add_argument(
        "--components",
        nargs="*",
        default=None,
        help="Subset of component keys; default matches scheduler ingest (see profiling README).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use a small batch/len grid for smoke tests.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    configure_runtime(
        force_cpu=args.cpu,
        cuda_device=args.cuda_device,
        device_label=args.device_id,
        n_repeats=args.n_repeats,
        warmup_repeats=args.warmup,
    )
    test_cfgs = QUICK_TEST_CONFIGS if args.quick else None
    sel = args.components if args.components else None
    run_all_benchmarks(
        test_configs=test_cfgs,
        selected_components=sel,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())