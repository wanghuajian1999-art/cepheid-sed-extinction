#!/usr/bin/env python3
"""Filtered latest MW pure-photometry SED fitter with dynamic phase floors.

This wrapper reuses
``jupyter_mw_sed_latest_pc_source_dependent_floor_filtered_distanceprior_emcee_direct``
for the SED model, quality filters, missing-Wang25 handling, and residual
diagnostics, but replaces the fixed JHKW1 source floors by a Skowron-style
phase-error model.

JHK single-epoch amplitudes are estimated from Gaia RP amplitudes:

    A_X = (A_X / A_I) * (A_I / A_RP) * A_RP

with ``A_I/A_RP = 0.95`` by default.  Single-mode F Bras2025+RP
phase-corrected JHK magnitudes use fixed residual floors, and literature
intensity-mean JHK magnitudes receive no extra floor.  W1 amplitudes use the
LMC-calibrated conversion from the historical ``A_W1 = 0.32 A_BP`` relation to
``A_W1 = 0.535 A_RP`` by default.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import jupyter_mw_sed_latest_pc_source_dependent_floor_filtered_distanceprior_emcee_direct as ffloor
from compute_band_extinction_from_saved_posteriors import (
    compute_band_extinction_from_saved_posteriors,
)


sfloor = ffloor.sfloor

DEFAULT_OUT_ROOT = SCRIPT_DIR / "outputs" / "sed_latest_dynamic_phase_floor_filtered"
NEW_QUALITY_MODE = ffloor.NEW_QUALITY_MODE
SED_BANDS = list(ffloor.SED_BANDS)

ORIGINAL_JHK_BAND_PASSES = None
ORIGINAL_W1_BAND_PASSES = None
ACTIVE_DYNAMIC_CONFIG: dict = {}

DEFAULT_ARP_CLIP_P95_BY_FAMILY = {"F": 0.7335, "1O": 0.3210}
DEFAULT_LMC_ARP_OVER_ABP_MEDIAN = 0.5978
DEFAULT_LMC_ARP_OVER_ABP_BY_FAMILY = {"F": 0.5988, "1O": 0.5964}
DEFAULT_A_W1_OVER_ARP = 0.32 / DEFAULT_LMC_ARP_OVER_ABP_MEDIAN
DEFAULT_AX_OVER_AI_F_BY_LOGP = {
    "logP_break": 1.3,
    "short": {"J": 0.63, "H": 0.50, "K": 0.49},
    "long": {"J": 0.70, "H": 0.63, "K": 0.62},
}

MULTIMODE_LABELS = ("F1O", "F1O2O", "1O2O", "1O2O3O", "1OX")
MODE_LABEL_ORDER = ("F", "1O", *MULTIMODE_LABELS, "2O", "unknown")
DEFAULT_PHOT_FILE = SCRIPT_DIR / "MW_3640_10band_GBPRPVIJHKW1S36_phase_corrected_Bras2025_RPonly.csv"


def _finite(x):
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def _clean(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _slug_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if np.isfinite(value) and abs(value - round(value)) < 1e-9:
            value = int(round(value))
        else:
            return str(value).replace(".", "p").replace("-", "m")
    return str(value).strip().replace(" ", "").replace(".", "p").replace("-", "m").replace("/", "")


def _copy_band_dict(values, default):
    out = dict(default)
    if values is None:
        return out
    for key, value in dict(values).items():
        out[str(key)] = float(value)
    return out


def _copy_source_coeff(values, default):
    out = dict(default)
    if values is None:
        return out
    for key, value in dict(values).items():
        out[str(key)] = float(value)
    return out


def _copy_F_coeff_by_logp(values=None, flat_values=None):
    cfg = {
        "logP_break": float(DEFAULT_AX_OVER_AI_F_BY_LOGP["logP_break"]),
        "short": dict(DEFAULT_AX_OVER_AI_F_BY_LOGP["short"]),
        "long": dict(DEFAULT_AX_OVER_AI_F_BY_LOGP["long"]),
    }
    if flat_values is not None:
        flat = _copy_band_dict(flat_values, cfg["short"])
        cfg["short"] = dict(flat)
        cfg["long"] = dict(flat)
    if values is None:
        return cfg
    values = dict(values)
    if any(k in values for k in ("short", "long", "logP_break")):
        if "logP_break" in values:
            cfg["logP_break"] = float(values["logP_break"])
        if "short" in values:
            cfg["short"] = _copy_band_dict(values["short"], cfg["short"])
        if "long" in values:
            cfg["long"] = _copy_band_dict(values["long"], cfg["long"])
    else:
        flat = _copy_band_dict(values, cfg["short"])
        cfg["short"] = dict(flat)
        cfg["long"] = dict(flat)
    return cfg


def _mode_token(value) -> str:
    token = _clean(value).upper()
    for old, new in ((" ", ""), ("/", ""), ("+", ""), ("-", ""), ("_", "")):
        token = token.replace(old, new)
    return token


def _normalize_allowed_multimode_modes(values):
    """Return canonical multimode labels, preserving historical ``None=all``."""
    if values is None:
        return None
    if isinstance(values, str):
        values = (values,)
    token_to_label = {_mode_token(label): label for label in MULTIMODE_LABELS}
    normalized = []
    invalid = []
    for value in values:
        token = _mode_token(value)
        label = token_to_label.get(token)
        if label is None:
            invalid.append(str(value))
        elif label not in normalized:
            normalized.append(label)
    if invalid:
        raise ValueError(
            "allowed_multimode_modes contains unsupported modes: "
            f"{invalid}; valid choices are {list(MULTIMODE_LABELS)}"
        )
    return tuple(label for label in MULTIMODE_LABELS if label in normalized)


def _mode_label_for_floor(row) -> str:
    for key in ("Mode", "mode_for_PC"):
        token = _mode_token(row.get(key, ""))
        if not token or token == "NAN":
            continue
        if token in {"F", "FUNDAMENTAL", "FUNDAMENTALMODE", "DCEPFU"}:
            return "F"
        if token in {"1O", "FIRSTOVERTONE", "FIRSTOVERTONEMODE", "DCEPFO"}:
            return "1O"
        if token in {"F1O", "F1O2O"}:
            return token
        if token in {"1O2O", "1O2O3O", "1OX"}:
            return token
        if token in {"2O", "SECONDOVERTONE", "SECONDOVERTONEMODE"}:
            return "2O"
        if token.startswith("F1O"):
            return "F1O2O" if "2O" in token else "F1O"
        if token.startswith("1O"):
            if "3O" in token:
                return "1O2O3O"
            if "2O" in token:
                return "1O2O"
            return "1OX" if "X" in token else "1O"
        if token.startswith("F"):
            return "F"
    return "unknown"


def _mode_for_floor(row) -> str:
    label = _mode_label_for_floor(row)
    if label in {"F", "F1O", "F1O2O"}:
        return "F"
    if label in {"1O", "1O2O", "1O2O3O", "1OX"}:
        return "1O"
    if label == "2O":
        return "2O"
    return "unknown"


def _is_multimode_for_floor(row) -> bool:
    return _mode_label_for_floor(row) in {"F1O", "F1O2O", "1O2O", "1O2O3O", "1OX"}


def _is_f_mode(row) -> bool:
    return _mode_for_floor(row) == "F"


def _is_1o_mode(row) -> bool:
    return _mode_for_floor(row) == "1O"


def _is_single_f_mode(row) -> bool:
    return _mode_label_for_floor(row) == "F"


def _mode_excluded(row) -> bool:
    if _mode_for_floor(row) not in {"F", "1O"}:
        return True
    if _is_multimode_for_floor(row):
        if not ACTIVE_DYNAMIC_CONFIG.get("include_multimode", True):
            return True
        allowed = ACTIVE_DYNAMIC_CONFIG.get("allowed_multimode_modes")
        if allowed is not None and _mode_label_for_floor(row) not in set(allowed):
            return True
    return False


def _logp_for_floor(row):
    for key in ("logP_used", "logP_select", "logP_for_PC", "logP"):
        val = _finite(row.get(key, np.nan))
        if np.isfinite(val):
            return float(val)
    period = _finite(row.get("period_used", np.nan))
    if np.isfinite(period) and period > 0:
        return float(np.log10(period))
    return np.nan


def _clip_positive(value, max_value):
    raw = _finite(value)
    if not (np.isfinite(raw) and raw > 0):
        return np.nan, np.nan, False
    if max_value is None:
        return raw, raw, False
    used = min(raw, float(max_value))
    return raw, used, bool(raw > used)


def _family_arp_clip_max(row):
    family = _mode_for_floor(row)
    by_family = ACTIVE_DYNAMIC_CONFIG.get("arp_clip_max_by_family", {}) or {}
    value = by_family.get(family, ACTIVE_DYNAMIC_CONFIG.get("arp_clip_max", None))
    if value is None:
        return None
    value = _finite(value)
    return float(value) if np.isfinite(value) and value > 0 else None


def _rp_amplitude(row):
    raw = row.get("peak_to_peak_rp", np.nan)
    if not np.isfinite(_finite(raw)):
        raw = row.get("peak_to_peak_rp_gaia_rp", np.nan)
    clip_max = _family_arp_clip_max(row)
    raw_val = _finite(raw)
    if np.isfinite(raw_val) and raw_val > 0:
        return _clip_positive(raw_val, clip_max)
    if clip_max is not None:
        return np.nan, float(clip_max), False
    return np.nan, np.nan, False


def _bp_amplitude(row):
    raw = row.get("peak_to_peak_bp", np.nan)
    if not np.isfinite(_finite(raw)):
        raw = row.get("peak_to_peak_bp_gaia_bp", np.nan)
    return _clip_positive(raw, ACTIVE_DYNAMIC_CONFIG.get("abp_clip_max", None))


def _estimate_jhk_amplitude(row, band: str):
    if band not in {"J", "H", "K"}:
        return np.nan
    if _mode_for_floor(row) not in {"F", "1O"}:
        return np.nan
    _raw, arp, _clipped = _rp_amplitude(row)
    if not np.isfinite(arp):
        return np.nan
    if _is_f_mode(row):
        f_cfg = ACTIVE_DYNAMIC_CONFIG["ax_over_ai_F_by_logp"]
        logp = _logp_for_floor(row)
        branch = "long" if np.isfinite(logp) and logp >= float(f_cfg["logP_break"]) else "short"
        coeffs = f_cfg[branch]
    else:
        coeffs = ACTIVE_DYNAMIC_CONFIG["ax_over_ai_1o"]
    coeff = coeffs.get(band, np.nan)
    ai_over_arp = float(ACTIVE_DYNAMIC_CONFIG["ai_over_arp"])
    if not (np.isfinite(coeff) and coeff > 0 and np.isfinite(ai_over_arp) and ai_over_arp > 0):
        return np.nan
    return float(coeff * ai_over_arp * arp)


def _estimate_w1_amplitude(row):
    _raw, arp, _clipped = _rp_amplitude(row)
    coeff = float(ACTIVE_DYNAMIC_CONFIG["a_w1_over_arp"])
    if not (np.isfinite(arp) and np.isfinite(coeff) and coeff > 0):
        return np.nan
    return float(coeff * arp)


def _gaia_dynamic_floor(row, band: str):
    group, source_label = sfloor._gaia_floor_group(row, band)
    if group == "gaia_vari_cepheid":
        floor = ACTIVE_DYNAMIC_CONFIG["gaia_vari_floor"].get(band, 0.0)
        rule = "gaia_vari_cepheid_floor"
    else:
        floor = ACTIVE_DYNAMIC_CONFIG["gaia_source_mean_floor"].get(band, 0.0)
        rule = "gaia_source_mean_floor"
    return float(floor), rule, source_label


def _jhk_dynamic_floor(row, band: str):
    group, source_label = sfloor._jhk_floor_group(row, band)
    if group == "jhk_literature_intensity_mean":
        floor = ACTIVE_DYNAMIC_CONFIG["floor_literature_JHK"].get(band, 0.0)
        return float(floor), "literature_no_floor", source_label

    if group == "jhk_bras2025_rp_phase_corrected" and _is_single_f_mode(row):
        floor = ACTIVE_DYNAMIC_CONFIG["floor_F_phasecorr_JHK"].get(band, 0.0)
        return float(floor), "F_Bras2025_RP_band_floor", source_label

    if group in {"jhk_2mass_single_epoch", "jhk_bras2025_rp_phase_corrected"}:
        amp = _estimate_jhk_amplitude(row, band)
        if np.isfinite(amp):
            label = "F_RP_amplitude_single_epoch" if _is_f_mode(row) else "1O_RP_amplitude_single_epoch"
            if group == "jhk_bras2025_rp_phase_corrected":
                source_label = f"{source_label}|treated_as_unphasecorrected_multimode"
            return float(amp / (2.0 * np.sqrt(2.0))), label, source_label
        return 0.0, "RP_amplitude_missing", source_label

    return 0.0, "jhk_unknown_no_floor", source_label


def _w1_dynamic_floor(row):
    group, source_label = sfloor._w1_floor_group(row)
    amp = _estimate_w1_amplitude(row)
    if not np.isfinite(amp):
        return 0.0, "W1_missing_ARP", source_label

    if group == "w1_unwise":
        coeff_key = "unWISE"
        rule = "W1_skowron_unwise"
    elif group == "w1_wise_allsky":
        coeff_key = "WISE_AllSky"
        rule = "W1_skowron_allsky"
    else:
        coeff_key = "AllWISE"
        rule = "W1_skowron_allwise"

    coeff = ACTIVE_DYNAMIC_CONFIG["w1_epoch_coeff"].get(coeff_key, 0.0)
    return float(coeff * amp), rule, source_label


def dynamic_source_floor_for_band(row, band: str, floor_config=None):
    band = str(band)
    if band in {"BP", "G", "RP"}:
        floor, rule, label = _gaia_dynamic_floor(row, band)
    elif band in {"J", "H", "K"}:
        floor, rule, label = _jhk_dynamic_floor(row, band)
    elif band == "W1":
        floor, rule, label = _w1_dynamic_floor(row)
    else:
        floor, rule, label = 0.0, "unsupported_band_no_floor", ""
    return float(floor), str(rule), str(label)


def dynamic_source_floor_vector(row, sed_bands, floor_config=None):
    floors, groups, labels = [], [], []
    for band in list(sed_bands):
        floor, group, label = dynamic_source_floor_for_band(row, band, floor_config=floor_config)
        floors.append(floor)
        groups.append(group)
        labels.append(label)
    return np.asarray(floors, dtype=float), groups, labels


def _jhk_band_passes_dynamic(row, band: str) -> bool:
    if _mode_excluded(row):
        return False
    if not ORIGINAL_JHK_BAND_PASSES(row, band):
        return False
    return True


def _w1_band_passes_dynamic(row) -> bool:
    if _mode_excluded(row):
        return False
    return ORIGINAL_W1_BAND_PASSES(row)


def _patch_dynamic_floor_behavior():
    global ORIGINAL_JHK_BAND_PASSES, ORIGINAL_W1_BAND_PASSES
    if ORIGINAL_JHK_BAND_PASSES is None:
        ORIGINAL_JHK_BAND_PASSES = ffloor._jhk_band_passes
    if ORIGINAL_W1_BAND_PASSES is None:
        ORIGINAL_W1_BAND_PASSES = ffloor._w1_band_passes
    sfloor.source_floor_for_band = dynamic_source_floor_for_band
    sfloor.source_floor_vector = dynamic_source_floor_vector
    ffloor._jhk_band_passes = _jhk_band_passes_dynamic
    ffloor._w1_band_passes = _w1_band_passes_dynamic


def _set_dynamic_config(
    *,
    ai_over_arp,
    ax_over_ai_F,
    ax_over_ai_F_by_logp,
    ax_over_ai_1o,
    floor_F_phasecorr_JHK,
    floor_literature_JHK,
    arp_clip_max,
    arp_clip_max_by_family,
    a_w1_over_abp,
    a_w1_over_arp,
    lmc_arp_over_abp_median,
    lmc_arp_over_abp_by_family,
    w1_epoch_coeff,
    abp_clip_max,
    gaia_vari_floor,
    gaia_source_mean_floor,
    include_multimode,
    allowed_multimode_modes,
):
    global ACTIVE_DYNAMIC_CONFIG
    lmc_arp_over_abp_median = float(lmc_arp_over_abp_median)
    if not (np.isfinite(lmc_arp_over_abp_median) and lmc_arp_over_abp_median > 0):
        raise ValueError("lmc_arp_over_abp_median must be positive and finite")
    a_w1_over_abp = float(a_w1_over_abp)
    if a_w1_over_arp is None:
        a_w1_over_arp = a_w1_over_abp / lmc_arp_over_abp_median
    ACTIVE_DYNAMIC_CONFIG = {
        "ai_over_arp": float(ai_over_arp),
        "ax_over_ai_F": _copy_band_dict(ax_over_ai_F, DEFAULT_AX_OVER_AI_F_BY_LOGP["short"]),
        "ax_over_ai_F_by_logp": _copy_F_coeff_by_logp(ax_over_ai_F_by_logp, flat_values=ax_over_ai_F),
        "ax_over_ai_1o": _copy_band_dict(ax_over_ai_1o, {"J": 0.628, "H": 0.417, "K": 0.398}),
        "floor_F_phasecorr_JHK": _copy_band_dict(
            floor_F_phasecorr_JHK, {"J": 0.040, "H": 0.025, "K": 0.025}
        ),
        "floor_literature_JHK": _copy_band_dict(
            floor_literature_JHK, {"J": 0.0, "H": 0.0, "K": 0.0}
        ),
        "arp_clip_max": None if arp_clip_max is None else float(arp_clip_max),
        "arp_clip_max_by_family": _copy_source_coeff(
            arp_clip_max_by_family, DEFAULT_ARP_CLIP_P95_BY_FAMILY
        ),
        "a_w1_over_abp": a_w1_over_abp,
        "a_w1_over_arp": float(a_w1_over_arp),
        "lmc_arp_over_abp_median": lmc_arp_over_abp_median,
        "lmc_arp_over_abp_by_family": _copy_source_coeff(
            lmc_arp_over_abp_by_family, DEFAULT_LMC_ARP_OVER_ABP_BY_FAMILY
        ),
        "w1_epoch_coeff": _copy_source_coeff(
            w1_epoch_coeff,
            {"unWISE": 0.11, "AllWISE": 0.25, "WISE_AllSky": 1.0 / (2.0 * np.sqrt(2.0))},
        ),
        "abp_clip_max": None if abp_clip_max is None else float(abp_clip_max),
        "gaia_vari_floor": _copy_band_dict(gaia_vari_floor, {"BP": 0.005, "G": 0.005, "RP": 0.005}),
        "gaia_source_mean_floor": _copy_band_dict(
            gaia_source_mean_floor, {"BP": 0.060, "G": 0.010, "RP": 0.030}
        ),
        "include_multimode": bool(include_multimode),
        "allowed_multimode_modes": _normalize_allowed_multimode_modes(
            allowed_multimode_modes
        ),
    }


def _dynamic_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _base_slug(
    fix_logg_mh: bool,
    use_vi_pc_prior: bool,
    fixed_logg=2.0,
    fixed_mh=0.0,
) -> str:
    if fix_logg_mh:
        suffix = "withVI" if use_vi_pc_prior else "noVI"
        return f"fixedLogg{_slug_value(fixed_logg)}_MH{_slug_value(fixed_mh)}_{suffix}"
    return "variableLoggMH_withVI" if use_vi_pc_prior else "variableLoggMH_noVI"


def _run_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _multimode_slug(include_multimode, allowed_multimode_modes) -> str:
    if not include_multimode:
        return "singleModeOnly"
    if allowed_multimode_modes is None:
        return "withMultiAll"
    if len(allowed_multimode_modes) == 0:
        return "withMultiNone"
    return "withMulti" + "-".join(allowed_multimode_modes)


def _auto_out_dir(
    out_root,
    quality_mode,
    fix_logg_mh,
    use_vi_pc_prior,
    bp_max,
    require_gaia_vari_cepheid,
    include_multimode,
    run_config,
):
    parameterization = str(
        run_config.get("extinction_parameterization", "")
    ).strip().lower()
    if parameterization == "a550_r550":
        parameterization_slug = "nativeA550R550"
    elif parameterization == "av_rv":
        parameterization_slug = "nativeAvRv"
    else:
        parameterization_slug = "legacyAvRv"
    uses_native_cache = parameterization in {"a550_r550", "av_rv"}
    bits = [
        f"{_base_slug(bool(fix_logg_mh), bool(use_vi_pc_prior), run_config.get('fixed_logg', 2.0), run_config.get('fixed_mh', 0.0))}_dynamicPhaseFloor_filtered",
        str(quality_mode).replace("/", ""),
        f"bp{str(bp_max).replace('.', 'p')}" if bp_max is not None else "bpnone",
        "gaiaVar" if require_gaia_vari_cepheid else "gaiaAny",
        _multimode_slug(
            include_multimode,
            run_config.get("allowed_multimode_modes"),
        ),
        parameterization_slug,
        "scaleLogRoverDOnly" if str(run_config.get("scale_parameterization", "")) == "logRoverD_only" else str(run_config.get("scale_parameterization", "scaleUnknown")),
        (
            f"bandExt{int(run_config.get('band_extinction_draws', 0))}"
            if uses_native_cache
            else "bandExtNA"
        ),
        (
            "VegaCALSPEC012"
            if "alpha_lyr_stis_012" in Path(str(run_config.get("zp_file", ""))).name.lower()
            else "VegaDefault"
        ),
        (
            f"iterClip{str(run_config.get('residual_clip_sigma', 3.0)).replace('.', 'p')}"
            f"_minBand{int(run_config.get('min_bands_after_clip', 6))}"
            if bool(run_config.get("iterative_residual_clipping", False))
            else "noResidualClip"
        ),
        f"cfg{_dynamic_config_hash(ACTIVE_DYNAMIC_CONFIG)}",
        f"run{_run_config_hash(run_config)}",
    ]
    return Path(out_root) / "_".join(bits)


def _dynamic_metadata_for_row(row, summary_row=None):
    summary_row = {} if summary_row is None else summary_row
    arp_raw, arp_used, arp_clipped = _rp_amplitude(row)
    abp_raw, abp_used, abp_clipped = _bp_amplitude(row)
    logp_floor = _logp_for_floor(row)
    f_cfg = ACTIVE_DYNAMIC_CONFIG.get("ax_over_ai_F_by_logp", DEFAULT_AX_OVER_AI_F_BY_LOGP)
    f_branch = ""
    if _is_f_mode(row):
        f_branch = "long" if np.isfinite(logp_floor) and logp_floor >= float(f_cfg["logP_break"]) else "short"
    mode_label = _mode_label_for_floor(row)
    mode_family = _mode_for_floor(row)
    is_multimode = _is_multimode_for_floor(row)
    mode_for_pc = _clean(summary_row.get("pc_mode_used", "")) or mode_family
    logp_pc_used = _finite(summary_row.get("logP_pc_used", np.nan))
    if not np.isfinite(logp_pc_used):
        period_used = _finite(row.get("period_used", np.nan))
        logp_pc_used = np.log10(period_used) if np.isfinite(period_used) and period_used > 0 else np.nan
    logp0_used = _finite(summary_row.get("logP0_used", row.get("logP_F_plot_Pilecki2024_MW", np.nan)))
    out = {
        "mode_original": _clean(row.get("Mode", mode_label)),
        "mode_family_for_floor": mode_family,
        "mode_for_PC": mode_for_pc,
        "is_multimode": bool(is_multimode),
        "multimode_phase_correction_allowed": False if is_multimode else np.nan,
        "jhk_phase_correction_allowed_by_mode": bool(_is_single_f_mode(row)),
        "logP_pc_used": logp_pc_used,
        "logP0_used": logp0_used,
        "floor_mode_label": mode_label,
        "floor_mode_family": mode_family,
        "logP_for_floor": logp_floor,
        "F_ax_over_ai_branch": f_branch,
        "F_ax_over_ai_logP_break": float(f_cfg["logP_break"]),
        "A_RP_p95_mode_family": _family_arp_clip_max(row),
        "A_RP_raw": arp_raw,
        "A_RP_used": arp_used,
        "ARP_clipped": bool(arp_clipped),
        "ARP_missing_used_family_p95": bool((not np.isfinite(arp_raw)) and np.isfinite(arp_used)),
        "A_BP_raw": abp_raw,
        "A_BP_used": abp_used,
        "ABP_clipped": bool(abp_clipped),
        "A_J_est_mag": _estimate_jhk_amplitude(row, "J"),
        "A_H_est_mag": _estimate_jhk_amplitude(row, "H"),
        "A_K_est_mag": _estimate_jhk_amplitude(row, "K"),
        "A_W1_est_mag": _estimate_w1_amplitude(row),
    }
    for band in SED_BANDS:
        floor, rule, _label = dynamic_source_floor_for_band(row, band)
        out[f"floor_rule_{band}"] = rule
        out[f"dynamic_floor_{band}_used"] = floor
    return out


def _augment_summary_with_dynamic_metadata(selected, summary):
    if selected is None or summary is None or len(selected) == 0 or len(summary) == 0:
        return summary
    selected_by_tag = selected.set_index("selection_tag", drop=False)
    rows = []
    for _, srow in summary.iterrows():
        tag = srow.get("selection_tag", "")
        if tag in selected_by_tag.index:
            row = selected_by_tag.loc[tag]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            meta = _dynamic_metadata_for_row(row, summary_row=srow)
        else:
            meta = {}
        rows.append(meta)
    extra = pd.DataFrame(rows, index=summary.index)
    out = pd.concat([summary.drop(columns=[c for c in extra.columns if c in summary.columns], errors="ignore"), extra], axis=1)
    for band in SED_BANDS:
        old_cat = f"{band}_mag_err_catalog"
        old_eff = f"{band}_mag_err_effective"
        if old_cat in out.columns:
            out[f"magerr_catalog_{band}"] = out[old_cat]
        if old_eff in out.columns:
            out[f"magerr_effective_{band}"] = out[old_eff]
    for key, value in ACTIVE_DYNAMIC_CONFIG.items():
        out[f"dynamic_{key}"] = (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    return out


def _mode_counts(frame):
    if frame is None or len(frame) == 0:
        return {}
    counts = {}
    for _, row in frame.iterrows():
        label = _mode_label_for_floor(row)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _read_optional_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_mode_selection_statistics(out_dir, selected, summary, phot_file=None, n_stars=None):
    """Write an auditable mode-by-mode selection and fit accounting table."""
    out_dir = Path(out_dir)
    mother_path = Path(phot_file) if phot_file is not None else DEFAULT_PHOT_FILE
    mother = _read_optional_csv(mother_path)
    rejected = _read_optional_csv(out_dir / "residual_clipping_rejected_sources.csv")

    selected_counts = _mode_counts(selected)
    mother_counts = _mode_counts(mother)
    successful = summary
    if "fit_success" in summary.columns:
        successful = summary[summary["fit_success"].fillna(False).astype(bool)]
    success_counts = _mode_counts(successful)

    selected_mode_by_tag = {}
    if selected is not None and len(selected) > 0 and "selection_tag" in selected.columns:
        selected_mode_by_tag = {
            str(row["selection_tag"]): _mode_label_for_floor(row)
            for _, row in selected.iterrows()
        }

    rejected_counts = {}
    if len(rejected) > 0:
        for _, row in rejected.iterrows():
            label = selected_mode_by_tag.get(str(row.get("selection_tag", "")), "unknown")
            rejected_counts[label] = rejected_counts.get(label, 0) + 1

    refit_counts = {}
    for frame in (summary, rejected):
        if len(frame) == 0 or "n_clip_refits" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            if _finite(row.get("n_clip_refits", 0)) <= 0:
                continue
            label = _mode_label_for_floor(row)
            if label == "unknown":
                label = selected_mode_by_tag.get(str(row.get("selection_tag", "")), "unknown")
            refit_counts[label] = refit_counts.get(label, 0) + 1

    allowed = ACTIVE_DYNAMIC_CONFIG.get("allowed_multimode_modes")
    if not ACTIVE_DYNAMIC_CONFIG.get("include_multimode", True):
        allowed_set = set()
    else:
        allowed_set = set(MULTIMODE_LABELS if allowed is None else allowed)
    labels = list(MODE_LABEL_ORDER)
    rows = []
    for label in labels:
        selected_n = int(selected_counts.get(label, 0))
        success_n = int(success_counts.get(label, 0))
        rows.append(
            {
                "mode": label,
                "mother_sample_count": int(mother_counts.get(label, 0)) if len(mother) else np.nan,
                "quality_selected_for_mcmc_count": selected_n,
                "entered_mcmc_count": selected_n,
                "clip_refit_source_count": int(refit_counts.get(label, 0)),
                "residual_clipping_rejected_count": int(rejected_counts.get(label, 0)),
                "fit_failed_or_rejected_count": max(selected_n - success_n, 0),
                "final_success_count": success_n,
                "allowed_by_mode_policy": bool(
                    label in {"F", "1O"} or label in allowed_set
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "mode_selection_fit_statistics.csv", index=False)

    unexpected = {
        label: count
        for label, count in selected_counts.items()
        if label not in {"F", "1O"} | allowed_set
    }
    if unexpected:
        raise RuntimeError(
            "mode whitelist failure: disallowed modes entered selected table: "
            f"{unexpected}"
        )

    validation = {
        "mother_photometry_file": str(mother_path),
        "n_stars_requested": n_stars,
        "quality_count_scope": (
            "all quality-selected sources" if n_stars is None
            else "quality-selected sources after n_stars limit"
        ),
        "include_multimode": bool(ACTIVE_DYNAMIC_CONFIG.get("include_multimode", True)),
        "allowed_multimode_modes": None if allowed is None else list(allowed),
        "selected_mode_counts": selected_counts,
        "final_success_mode_counts": success_counts,
        "unexpected_selected_modes": unexpected,
    }
    with open(out_dir / "mode_selection_fit_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)
    return table


def _rewrite_outputs_with_dynamic_metadata(out_dir, selected, summary, residual_summary):
    out_dir = Path(out_dir)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summary.to_csv(out_dir / "MW_SED_latest_dynamic_phase_floor_summary.csv", index=False)
    if residual_summary is not None:
        residual_summary.to_csv(out_dir / "normalized_residual_robust_scatter_summary.csv", index=False)
    runinfo_path = out_dir / "runinfo.json"
    if runinfo_path.exists():
        with open(runinfo_path, "r", encoding="utf-8") as f:
            runinfo = json.load(f)
    else:
        runinfo = {}
    runinfo.update(
        {
            "output_dir": str(out_dir.parent),
            "method": "latest pure-photometry SED direct emcee with dynamic Skowron-style phase floors",
            "dynamic_phase_floor_config": copy.deepcopy(ACTIVE_DYNAMIC_CONFIG),
            "include_multimode": bool(ACTIVE_DYNAMIC_CONFIG.get("include_multimode", True)),
            "allowed_multimode_modes": (
                None
                if ACTIVE_DYNAMIC_CONFIG.get("allowed_multimode_modes") is None
                else list(ACTIVE_DYNAMIC_CONFIG["allowed_multimode_modes"])
            ),
            "use_distance_prior": bool(summary["uses_distance_prior_for_source"].any())
            if hasattr(summary, "__getitem__") and "uses_distance_prior_for_source" in summary.columns
            else None,
            "scale_parameterization": str(summary["scale_parameterization"].iloc[0])
            if hasattr(summary, "__getitem__") and "scale_parameterization" in summary.columns and len(summary) > 0
            else str(runinfo.get("scale_parameterization", "unknown")),
            "pc_prior_mode": str(summary["pc_prior_mode"].iloc[0])
            if hasattr(summary, "__getitem__") and "pc_prior_mode" in summary.columns and len(summary) > 0
            else None,
            "pc_prior_enabled": bool(summary["pc_prior_enabled"].iloc[0])
            if hasattr(summary, "__getitem__") and "pc_prior_enabled" in summary.columns and len(summary) > 0
            else None,
            "pc_prior_scale": float(summary["pc_prior_scale"].iloc[0])
            if hasattr(summary, "__getitem__") and "pc_prior_scale" in summary.columns and len(summary) > 0 and pd.notna(summary["pc_prior_scale"].iloc[0])
            else None,
            "extinction_law_used": str(summary["extinction_law_used"].iloc[0])
            if hasattr(summary, "__getitem__") and "extinction_law_used" in summary.columns and len(summary) > 0
            else runinfo.get("extinction_law_used"),
            "extinction_cache_file": str(summary["extinction_cache_file"].iloc[0])
            if hasattr(summary, "__getitem__") and "extinction_cache_file" in summary.columns and len(summary) > 0
            else runinfo.get("extinction_cache_file"),
            "multimode_policy": (
                "F1O/F1O2O use the F family and F PC branch; "
                "1O2O/1O2O3O/1OX use the 1O family and 1O PC branch; "
                "2O and unknown modes are excluded; recognized multimodes are included only "
                "when include_multimode=True and, when supplied, present in allowed_multimode_modes"
            ),
            "jhk_amp_anchor": "RP",
            "jhk_ai_over_arp": float(ACTIVE_DYNAMIC_CONFIG["ai_over_arp"]),
            "jhk_literature_floor_policy": "no_extra_floor",
            "F_phase_corrected_JHK_policy": "band_fixed_residual_floor",
            "W1_amp_anchor": "RP",
            "W1_amp_conversion": "a_w1_over_arp = a_w1_over_abp / median_LMC(A_RP/A_BP)",
        }
    )
    with open(runinfo_path, "w", encoding="utf-8") as f:
        json.dump(runinfo, f, ensure_ascii=False, indent=2)


def run_mw_sed_latest_pc_dynamic_phase_floor_filtered_distanceprior_emcee_direct(
    n_stars=30,
    quality_mode=NEW_QUALITY_MODE,
    bp_max=20.3,
    require_gaia_vari_cepheid=False,
    include_multimode=True,
    allowed_multimode_modes=None,
    jhk_phqual_allowed=("A", "B", "C", "D"),
    jhk_cc_flg_allowed=("0",),
    apply_jhk_quality_to_literature_mean=False,
    require_wise_profile_quality=True,
    wise_phqual_allowed=("A", "B", "C"),
    wise_cc_flags_allowed=("0",),
    fix_logg_mh=False,
    fixed_logg=2.0,
    fixed_mh=0.0,
    use_vi_pc_prior=False,
    use_distance_prior=True,
    scale_parameterization="logR_logD",
    pc_prior_mode="1sigma",
    pc_prior_scale=None,
    extinction_law=ffloor.DEFAULT_EXTINCTION_LAW,
    extinction_parameterization="A550_R550",
    extinction_law_cache_map=None,
    build_native_cache_if_missing=True,
    native_cache_build_kwargs=None,
    compute_band_extinction_posterior=True,
    band_extinction_draws=512,
    band_extinction_batch_size=512,
    save_band_extinction_draws=False,
    native_vertex_cache_size=128,
    ai_over_arp=0.95,
    ax_over_ai_F=None,
    ax_over_ai_F_by_logp=None,
    ax_over_ai_1o=None,
    floor_F_phasecorr_JHK=None,
    floor_literature_JHK=None,
    exclude_uncorrected_F_single_epoch_JHK=None,
    fallback_F_single_epoch_JHK=None,
    arp_clip_max=1.5,
    arp_clip_max_by_family=None,
    a_w1_over_abp=0.32,
    a_w1_over_arp=None,
    lmc_arp_over_abp_median=DEFAULT_LMC_ARP_OVER_ABP_MEDIAN,
    lmc_arp_over_abp_by_family=None,
    w1_epoch_coeff=None,
    abp_clip_max=2.2,
    gaia_vari_floor=None,
    gaia_source_mean_floor=None,
    non_gaia_vari_floor=None,
    pc_coeff_file=None,
    pc_cov_file=None,
    phot_file=None,
    wang25_file=None,
    old_phot_file=None,
    zp_file=None,
    cache_file=None,
    vi_cache_file=None,
    out_root=DEFAULT_OUT_ROOT,
    out_dir=None,
    nwalkers=32,
    nsteps=2500,
    burnin=800,
    thin=10,
    random_seed=42,
    save_samples="diagnostic",
    make_corner=True,
    make_plots=True,
    progress=False,
    n_workers=1,
    resume=False,
    return_samples_to_notebook=True,
    overwrite=False,
    cleanup_incomplete_run=True,
    iterative_residual_clipping=False,
    residual_clip_sigma=3.0,
    min_bands_after_clip=6,
    required_g23_dust_extinction_version="1.7",
):
    """Run the filtered SED fit with dynamic phase floors."""
    _set_dynamic_config(
        ai_over_arp=ai_over_arp,
        ax_over_ai_F=ax_over_ai_F,
        ax_over_ai_F_by_logp=ax_over_ai_F_by_logp,
        ax_over_ai_1o=ax_over_ai_1o,
        floor_F_phasecorr_JHK=floor_F_phasecorr_JHK,
        floor_literature_JHK=floor_literature_JHK,
        arp_clip_max=arp_clip_max,
        arp_clip_max_by_family=arp_clip_max_by_family,
        a_w1_over_abp=a_w1_over_abp,
        a_w1_over_arp=a_w1_over_arp,
        lmc_arp_over_abp_median=lmc_arp_over_abp_median,
        lmc_arp_over_abp_by_family=lmc_arp_over_abp_by_family,
        w1_epoch_coeff=w1_epoch_coeff,
        abp_clip_max=abp_clip_max,
        gaia_vari_floor=gaia_vari_floor,
        gaia_source_mean_floor=gaia_source_mean_floor
        if gaia_source_mean_floor is not None
        else non_gaia_vari_floor,
        include_multimode=include_multimode,
        allowed_multimode_modes=allowed_multimode_modes,
    )
    _patch_dynamic_floor_behavior()

    run_config = {
        "n_stars": n_stars,
        "quality_mode": quality_mode,
        "bp_max": bp_max,
        "require_gaia_vari_cepheid": bool(require_gaia_vari_cepheid),
        "include_multimode": bool(include_multimode),
        "allowed_multimode_modes": (
            None
            if ACTIVE_DYNAMIC_CONFIG.get("allowed_multimode_modes") is None
            else list(ACTIVE_DYNAMIC_CONFIG["allowed_multimode_modes"])
        ),
        "jhk_phqual_allowed": list(jhk_phqual_allowed) if jhk_phqual_allowed is not None else None,
        "jhk_cc_flg_allowed": list(jhk_cc_flg_allowed) if jhk_cc_flg_allowed is not None else None,
        "apply_jhk_quality_to_literature_mean": bool(apply_jhk_quality_to_literature_mean),
        "require_wise_profile_quality": bool(require_wise_profile_quality),
        "wise_phqual_allowed": list(wise_phqual_allowed) if wise_phqual_allowed is not None else None,
        "wise_cc_flags_allowed": list(wise_cc_flags_allowed) if wise_cc_flags_allowed is not None else None,
        "fix_logg_mh": bool(fix_logg_mh),
        "fixed_logg": fixed_logg,
        "fixed_mh": fixed_mh,
        "use_vi_pc_prior": bool(use_vi_pc_prior),
        "use_distance_prior": bool(use_distance_prior),
        "scale_parameterization": str(scale_parameterization),
        "pc_prior_mode": str(pc_prior_mode),
        "pc_prior_scale": None if pc_prior_scale is None else float(pc_prior_scale),
        "extinction_law": str(extinction_law),
        "extinction_parameterization": str(extinction_parameterization),
        "extinction_law_cache_map": {str(k): str(v) for k, v in dict(extinction_law_cache_map).items()} if extinction_law_cache_map is not None else None,
        "build_native_cache_if_missing": bool(build_native_cache_if_missing),
        "native_cache_build_kwargs": native_cache_build_kwargs,
        "compute_band_extinction_posterior": bool(compute_band_extinction_posterior),
        "band_extinction_draws": int(band_extinction_draws),
        "band_extinction_batch_size": int(band_extinction_batch_size),
        "save_band_extinction_draws": bool(save_band_extinction_draws),
        "native_vertex_cache_size": int(native_vertex_cache_size),
        "cache_file": None if cache_file is None else str(cache_file),
        "zp_file": None if zp_file is None else str(zp_file),
        "iterative_residual_clipping": bool(iterative_residual_clipping),
        "residual_clip_sigma": float(residual_clip_sigma),
        "min_bands_after_clip": int(min_bands_after_clip),
        "required_g23_dust_extinction_version": required_g23_dust_extinction_version,
        "dynamic_phase_floor_config": copy.deepcopy(ACTIVE_DYNAMIC_CONFIG),
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "burnin": burnin,
        "thin": thin,
        "random_seed": random_seed,
        "save_samples": save_samples,
        "return_samples_to_notebook": bool(return_samples_to_notebook),
        "make_corner": make_corner,
        "make_plots": make_plots,
    }

    auto_out_dir = _auto_out_dir(
        out_root=out_root,
        quality_mode=quality_mode,
        fix_logg_mh=fix_logg_mh,
        use_vi_pc_prior=use_vi_pc_prior,
        bp_max=bp_max,
        require_gaia_vari_cepheid=require_gaia_vari_cepheid,
        include_multimode=include_multimode,
        run_config=run_config,
    )
    final_out_dir = Path(out_dir) if out_dir is not None else auto_out_dir
    final_out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((final_out_dir / name).exists() for name in ("summary.csv", "runinfo.json", "selected.csv")):
        raise FileExistsError(
            f"completed output already exists in {final_out_dir}; "
            "choose a new out_dir or set overwrite=True"
        )
    out_dir = final_out_dir / f".inprogress_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=False)

    try:
        selected, summary, samples, residual_summary = (
            ffloor.run_mw_sed_latest_pc_source_dependent_floor_filtered_distanceprior_emcee_direct(
            n_stars=n_stars,
            quality_mode=quality_mode,
            bp_max=bp_max,
            require_gaia_vari_cepheid=require_gaia_vari_cepheid,
            jhk_phqual_allowed=jhk_phqual_allowed,
            jhk_cc_flg_allowed=jhk_cc_flg_allowed,
            apply_jhk_quality_to_literature_mean=apply_jhk_quality_to_literature_mean,
            require_wise_profile_quality=require_wise_profile_quality,
            wise_phqual_allowed=wise_phqual_allowed,
            wise_cc_flags_allowed=wise_cc_flags_allowed,
            fix_logg_mh=fix_logg_mh,
            fixed_logg=fixed_logg,
            fixed_mh=fixed_mh,
            use_vi_pc_prior=use_vi_pc_prior,
            use_distance_prior=use_distance_prior,
            scale_parameterization=scale_parameterization,
            pc_prior_mode=pc_prior_mode,
            pc_prior_scale=pc_prior_scale,
            extinction_law=extinction_law,
            extinction_parameterization=extinction_parameterization,
            extinction_law_cache_map=extinction_law_cache_map,
            build_native_cache_if_missing=build_native_cache_if_missing,
            native_cache_build_kwargs=native_cache_build_kwargs,
            compute_band_extinction_posterior=compute_band_extinction_posterior,
            band_extinction_draws=band_extinction_draws,
            band_extinction_batch_size=band_extinction_batch_size,
            save_band_extinction_draws=save_band_extinction_draws,
            native_vertex_cache_size=native_vertex_cache_size,
            use_source_dependent_floor=True,
            floor_config={},
            pc_coeff_file=pc_coeff_file,
            pc_cov_file=pc_cov_file,
            phot_file=phot_file,
            wang25_file=wang25_file,
            old_phot_file=old_phot_file,
            zp_file=zp_file,
            cache_file=cache_file,
            vi_cache_file=vi_cache_file,
            out_dir=out_dir,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            random_seed=random_seed,
            save_samples=save_samples,
            make_corner=make_corner,
            make_plots=make_plots,
            progress=progress,
            n_workers=n_workers,
            resume=resume,
            return_samples_to_notebook=return_samples_to_notebook,
            iterative_residual_clipping=iterative_residual_clipping,
            residual_clip_sigma=residual_clip_sigma,
            min_bands_after_clip=min_bands_after_clip,
            required_g23_dust_extinction_version=required_g23_dust_extinction_version,
            )
        )

        summary = _augment_summary_with_dynamic_metadata(selected, summary)
        summary["fit_output_dir"] = str(final_out_dir)
        _write_mode_selection_statistics(
            out_dir,
            selected,
            summary,
            phot_file=phot_file,
            n_stars=n_stars,
        )
        _rewrite_outputs_with_dynamic_metadata(out_dir, selected, summary, residual_summary)
        for child in out_dir.iterdir():
            target = final_out_dir / child.name
            if target.exists():
                if not overwrite:
                    raise FileExistsError(f"output collision: {target}")
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        out_dir.rmdir()
        return selected, summary, samples, residual_summary
    except BaseException:
        if cleanup_incomplete_run:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise


print("Loaded filtered latest MW pure-photometry SED fitter with dynamic Skowron-style phase floors.")
print(
    "Call: selected, summary, samples, residual_summary = "
    "run_mw_sed_latest_pc_dynamic_phase_floor_filtered_distanceprior_emcee_direct(...)"
)
