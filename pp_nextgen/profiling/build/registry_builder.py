"""Assemble device_registry.v3 from legacy profiles + model memory contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from pp_nextgen.profiling.constants import KV_MODULES, LM_HEAD_MODULE
from pp_nextgen.profiling.fit.fitter import fit_decode_time, fit_prefill_time
from pp_nextgen.profiling.legacy_ingest import iter_legacy_module_samples, load_legacy_all_results


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gb_per_token_for_kv(kv_per_token_bytes: int, batch_size: int) -> float:
    return float(kv_per_token_bytes) * float(batch_size) / (1024.0**3)


def _sanitize_filename_component(name: str) -> str:
    out = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in str(name))
    return out.strip() or "unknown"


def _predict_y(fit: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    form = str(fit.get("form", "linear"))
    c0 = float(fit.get("c0", 0.0))
    c1 = float(fit.get("c1", 0.0))
    c2 = float(fit.get("c2", 0.0))
    xv = np.asarray(x, dtype=np.float64)
    if form == "constant":
        return np.full_like(xv, c0, dtype=np.float64)
    if form == "linear":
        return c0 + c1 * xv
    if form == "quadratic":
        return c0 + c1 * xv + c2 * xv * xv
    return c0 + c1 * xv


def _sorted_xy_from_str_float_map(d: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    items = sorted(((int(k), float(v)) for k, v in d.items()), key=lambda t: t[0])
    if not items:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = np.array([t[0] for t in items], dtype=np.float64)
    y = np.array([t[1] for t in items], dtype=np.float64)
    return x, y


def _decode_xy_for_plot(
    module_name: str, decode_times: Dict[str, Any], dec_fit: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scatter (xs, ys) and smooth line (xf, yf) for decode phase."""
    if module_name in KV_MODULES:
        d = {k: float(v) for k, v in decode_times.items() if k != "average"}
        xs, ys = _sorted_xy_from_str_float_map(d)
        if xs.size == 0:
            return xs, ys, np.array([]), np.array([])
        xf = np.linspace(float(xs.min()), float(xs.max()), max(50, min(200, xs.size * 10)))
        yf = _predict_y(dec_fit, xf)
        return xs, ys, xf, yf
    avg = float(decode_times.get("average", dec_fit.get("c0", 0.0)))
    xf = np.linspace(0.0, 1.0, 50)
    yf = np.full_like(xf, avg, dtype=np.float64)
    xs = np.array([0.5], dtype=np.float64)
    ys = np.array([avg], dtype=np.float64)
    return xs, ys, xf, yf


def _max_decode_context_among_kv(sessions: Sequence[Tuple[Any, ...]], batch_size: int) -> int:
    m = 1
    for row in sessions:
        bs, mod_name, _, dec, _, _ = row
        if bs != batch_size or mod_name not in KV_MODULES:
            continue
        if not isinstance(dec, dict):
            continue
        for k in dec:
            if k == "average":
                continue
            try:
                m = max(m, int(k))
            except (TypeError, ValueError):
                pass
    return m


def _write_device_fit_plots(
    plot_dir: Path,
    device_id: str,
    model: str,
    source_tag: str,
    sessions: List[Tuple[int, str, Dict[str, float], Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
) -> List[Path]:
    """Write prefill and decode multi-panel fit figures for one legacy profile."""
    import matplotlib.pyplot as plt

    if not sessions:
        return []

    batch_sizes = sorted({s[0] for s in sessions})
    modules: List[str] = []
    seen: set[str] = set()
    for s in sessions:
        mod = s[1]
        if mod not in seen:
            seen.add(mod)
            modules.append(mod)

    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = _sanitize_filename_component(device_id) + "_" + _sanitize_filename_component(source_tag)
    written: List[Path] = []

    def _style_axis(ax: Any, xlabel: str) -> None:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ms")

    nrows, ncols = len(batch_sizes), len(modules)
    fig_w = max(12.0, 2.2 * ncols)
    fig_h = max(8.0, 2.2 * nrows)

    # --- prefill ---
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig1.suptitle(f"{model} / {device_id} — prefill time (samples vs fit)", fontsize=12)
    for i, bs in enumerate(batch_sizes):
        for j, mod in enumerate(modules):
            ax = axes1[i][j]
            row = next((s for s in sessions if s[0] == bs and s[1] == mod), None)
            if row is None:
                ax.axis("off")
                continue
            _, _, prefill_times, _, pre_fit, _ = row
            xp, yp = _sorted_xy_from_str_float_map(prefill_times)
            if xp.size == 0 and mod != LM_HEAD_MODULE:
                ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{mod}\nbs={bs}")
                continue
            if xp.size:
                ax.scatter(xp, yp, s=22, alpha=0.85, label="prefill samples", color="C0", zorder=3)
            if str(pre_fit.get("form")) == "constant" and xp.size:
                xf = np.linspace(float(xp.min()), float(xp.max()), 50)
            elif xp.size:
                xf = np.linspace(float(xp.min()), float(xp.max()), max(50, min(200, xp.size * 12)))
            else:
                xf = np.linspace(0.0, 1.0, 50)
            yf = _predict_y(pre_fit, xf)
            fit_lbl = f'{pre_fit.get("form", "?")} fit'
            if mod == LM_HEAD_MODULE:
                fit_lbl = "constant (decode avg)"
            ax.plot(xf, yf, color="C1", lw=2.0, label=fit_lbl)
            ax.legend(loc="upper left", fontsize=7)
            ax.set_title(f"{mod}\nbs={bs}")
            _style_axis(ax, str(pre_fit.get("x", "seq_len")))
    fig1.tight_layout()
    p_pre = plot_dir / f"{tag}_prefill_fits.png"
    fig1.savefig(p_pre, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    written.append(p_pre)

    # --- decode ---
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig2.suptitle(f"{model} / {device_id} — decode time (samples vs fit)", fontsize=12)
    for i, bs in enumerate(batch_sizes):
        max_ctx = _max_decode_context_among_kv(sessions, bs)
        for j, mod in enumerate(modules):
            ax = axes2[i][j]
            row = next((s for s in sessions if s[0] == bs and s[1] == mod), None)
            if row is None:
                ax.axis("off")
                continue
            _, _, _, decode_times, _, dec_fit = row
            xs, ys, xf, yf = _decode_xy_for_plot(mod, decode_times, dec_fit)
            if mod in KV_MODULES:
                if xs.size:
                    ax.scatter(xs, ys, s=22, alpha=0.85, label="samples", color="C0", zorder=3)
                if xf.size:
                    ax.plot(xf, yf, color="C1", lw=2.0, label=f'{dec_fit.get("form", "?")} fit')
                ax.legend(loc="upper left", fontsize=7)
                _style_axis(ax, str(dec_fit.get("x", "context_len")))
            else:
                avg = float(ys[0]) if ys.size else float(dec_fit.get("c0", 0.0))
                xband = np.linspace(0.0, float(max_ctx), 50)
                ax.plot(xband, np.full_like(xband, avg), color="C1", lw=2.0, label="constant fit")
                ax.scatter(np.array([max_ctx / 2.0]), np.array([avg]), s=36, color="C0", zorder=3, label="avg sample")
                ax.legend(loc="upper left", fontsize=7)
                _style_axis(ax, "context_len (band for constant decode)")
            ax.set_title(f"{mod}\nbs={bs}")
    fig2.tight_layout()
    p_dec = plot_dir / f"{tag}_decode_fits.png"
    fig2.savefig(p_dec, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    written.append(p_dec)

    return written


def _memory_decode_model(
    module_name: str,
    weight_size_gb: float,
    kv_per_token_bytes: int,
    batch_size: int,
) -> Dict[str, Any]:
    if module_name in KV_MODULES:
        c1 = _gb_per_token_for_kv(kv_per_token_bytes, batch_size)
        return {
            "form": "linear",
            "c0": 0.0,
            "c1": c1,
            "c2": 0.0,
            "x": "seq_len",
            "unit": "GB",
            "note": "KV decode memory from model kv_per_token_bytes * batch_size per token (no profile kvcache).",
        }
    return {
        "form": "linear",
        "c0": float(weight_size_gb),
        "c1": 0.0,
        "c2": 0.0,
        "x": "seq_len",
        "unit": "GB",
    }


def device_entry_from_legacy_path(
    path: str | Path,
    kv_per_token_bytes: int,
    plot_dir: Path | None = None,
) -> Dict[str, Any]:
    bundle = load_legacy_all_results(path)
    device_entry: Dict[str, Any] = {
        "device_type": bundle.device_id,
        "model": bundle.model,
        "memory_gb": float(bundle.device_memory_gb),
        "source": str(Path(path).resolve()),
        "modules": {},
    }
    plot_sessions: List[
        Tuple[int, str, Dict[str, float], Dict[str, Any], Dict[str, Any], Dict[str, Any]]
    ] = []
    for bs, mod_name, comp in iter_legacy_module_samples(bundle):
        weight_size_gb = float(comp.get("weight_size_gb", 0.0))
        prefill_times = {str(k): float(v) for k, v in comp.get("prefill_times", {}).items()}
        decode_times = comp.get("decode_times", {})
        if not isinstance(decode_times, dict):
            decode_times = {}

        dev_mod = device_entry["modules"].setdefault(
            mod_name,
            {
                "weight_size_gb": weight_size_gb,
                "time_models": {"prefill": {"by_bs": {}}, "decode": {"by_bs": {}}},
                "memory_models": {"decode": {"by_bs": {}}},
            },
        )
        if float(dev_mod.get("weight_size_gb", 0.0)) == 0.0 and weight_size_gb > 0:
            dev_mod["weight_size_gb"] = weight_size_gb

        pre_fit = fit_prefill_time(mod_name, prefill_times, decode_times=decode_times)
        dec_fit = fit_decode_time(mod_name, decode_times)
        dev_mod["time_models"]["prefill"]["by_bs"][str(bs)] = pre_fit
        dev_mod["time_models"]["decode"]["by_bs"][str(bs)] = dec_fit
        dev_mod["memory_models"]["decode"]["by_bs"][str(bs)] = _memory_decode_model(
            mod_name, weight_size_gb, kv_per_token_bytes, bs
        )
        if plot_dir is not None:
            plot_sessions.append((bs, mod_name, prefill_times, decode_times, pre_fit, dec_fit))
    if plot_dir is not None and plot_sessions:
        _write_device_fit_plots(
            plot_dir.resolve(),
            device_id=bundle.device_id,
            model=str(bundle.model),
            source_tag=Path(path).stem,
            sessions=plot_sessions,
        )
    return device_entry


def merge_device_entries(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    if not dst:
        out = dict(src)
        out["sources"] = [out.pop("source")] if "source" in out else []
        return out
    if dst.get("model") != src.get("model"):
        raise ValueError(f"model mismatch merging device: {dst.get('model')} vs {src.get('model')}")
    out = dst
    out.setdefault("sources", [])
    if "source" in src:
        out["sources"].append(src["source"])
    if float(out.get("memory_gb", 0.0)) == 0.0 and float(src.get("memory_gb", 0.0)) != 0.0:
        out["memory_gb"] = float(src["memory_gb"])
    for mod_name, mod_entry in src.get("modules", {}).items():
        dst_mod = out.setdefault("modules", {}).setdefault(mod_name, mod_entry)
        if dst_mod is mod_entry:
            continue
        if float(dst_mod.get("weight_size_gb", 0.0)) == 0.0:
            dst_mod["weight_size_gb"] = float(mod_entry.get("weight_size_gb", 0.0))
        for section in ("time_models", "memory_models"):
            if section not in mod_entry:
                continue
            dst_mod.setdefault(section, {})
            for phase, phase_entry in mod_entry[section].items():
                dst_mod[section].setdefault(phase, {})
                src_by_bs = phase_entry.get("by_bs", {})
                dst_mod[section][phase].setdefault("by_bs", {})
                dst_mod[section][phase]["by_bs"].update(src_by_bs)
    return out


def build_device_registry_v3(
    input_paths: List[str | Path],
    model_name: str,
    kv_per_token_bytes: int,
    plot_output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    devices: Dict[str, Any] = {}
    resolved_model = model_name
    plot_base = Path(plot_output_dir).resolve() if plot_output_dir else None
    for p in input_paths:
        bundle = load_legacy_all_results(p)
        resolved_model = bundle.model or model_name
        dev_id = bundle.device_id
        entry = device_entry_from_legacy_path(
            p, kv_per_token_bytes=kv_per_token_bytes, plot_dir=plot_base
        )
        entry["model"] = resolved_model
        devices[dev_id] = merge_device_entries(devices.get(dev_id, {}), entry)
    return {
        "schema_version": "device_registry.v3",
        "generated_at": _iso_now(),
        "model": resolved_model,
        "devices": devices,
    }


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
