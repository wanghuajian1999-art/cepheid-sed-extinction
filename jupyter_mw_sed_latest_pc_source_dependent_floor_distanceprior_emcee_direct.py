#!/usr/bin/env python3
"""Latest MW pure-photometry SED fitter with fixed source-dependent floors.

This wrapper is based on
``jupyter_mw_sed_latest_pc_flexible_config_distanceprior_emcee_direct.py``.
It keeps the same SED model, PC prior, distance prior, optional logg/MH
configuration, and emcee driver, but replaces the catalog magnitude errors in
the seven-band SED likelihood by

    sigma_eff^2 = sigma_cat^2 + sigma_floor^2

where ``sigma_floor`` is fixed from the band and photometry source.  The floor
is not sampled by MCMC.
"""

from __future__ import annotations

import copy
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_OUT_ROOT = SCRIPT_DIR / "outputs" / "sed_latest_source_dependent_floor"

SED_BANDS = ["BP", "G", "RP", "J", "H", "K", "W1"]

FULL_PC_COLOR_DEFS = [
    ("BP-RP", "BP", "RP"),
    ("RP-J", "RP", "J"),
    ("RP-H", "RP", "H"),
    ("RP-K", "RP", "K"),
    ("V-I", "V", "I"),
]
NO_VI_PC_COLOR_DEFS = [item for item in FULL_PC_COLOR_DEFS if item[0] != "V-I"]

DEFAULT_FLOOR_CONFIG = {
    "gaia_vari_cepheid": {"BP": 0.01, "G": 0.01, "RP": 0.01},
    "gaia_source_mean": {"BP": 0.06, "G": 0.01, "RP": 0.03},
    "jhk_literature_intensity_mean": {"J": 0.02, "H": 0.02, "K": 0.02},
    "jhk_bras2025_rp_phase_corrected": {"J": 0.04, "H": 0.02, "K": 0.02},
    "jhk_2mass_single_epoch": {"J": 0.04, "H": 0.02, "K": 0.02},
    "w1_unwise": {"W1": 0.03},
    "w1_allwise": {"W1": 0.04},
    "w1_wise_allsky": {"W1": 0.04},
}

ORIGINAL_ROW_SED_VECTOR = None
ORIGINAL_SUMMARIZE_SAMPLES = None
ACTIVE_FLOOR_CONFIG = copy.deepcopy(DEFAULT_FLOOR_CONFIG)
USE_SOURCE_DEPENDENT_FLOOR = True


def _config_slug(fix_logg_mh: bool, use_vi_pc_prior: bool) -> str:
    if fix_logg_mh:
        base = "fixedLogg2_MH0_withVI" if use_vi_pc_prior else "fixedLogg2_MH0_noVI"
    else:
        base = "variableLoggMH_withVI" if use_vi_pc_prior else "variableLoggMH_noVI"
    return f"{base}_sourceFloor"


def _pc_color_defs(use_vi_pc_prior: bool):
    return list(FULL_PC_COLOR_DEFS if use_vi_pc_prior else NO_VI_PC_COLOR_DEFS)


def _load_modules():
    import jupyter_all_mw_sed_pcprior_method2_distanceprior as mapbase_module
    import jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct as directbase_module
    import jupyter_mw_sed_latest_pc_loggF_rgcFeH_distanceprior_emcee_direct as latest_module

    mapbase_module = importlib.reload(mapbase_module)
    directbase_module = importlib.reload(directbase_module)
    latest_module = importlib.reload(latest_module)
    return mapbase_module, directbase_module, latest_module


def _load_fixed_module():
    import jupyter_mw_sed_latest_pc_fixedMH0_logg2_distanceprior_emcee_direct as fixed_module

    fixed_module = importlib.reload(fixed_module)
    fixed_module._reset_base_modules()
    return fixed_module


def _set_pc_colors(latest_module, use_vi_pc_prior: bool):
    latest_module.PC_COLOR_DEFS = _pc_color_defs(use_vi_pc_prior)
    latest_module.VI_MODEL = None


def _deep_update_floor_config(user_config):
    cfg = copy.deepcopy(DEFAULT_FLOOR_CONFIG)
    if user_config is None:
        return cfg
    for group, band_values in dict(user_config).items():
        group = str(group)
        cfg.setdefault(group, {})
        for band, value in dict(band_values).items():
            band = str(band)
            value = float(value)
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"invalid floor for {group}/{band}: {value}")
            cfg[group][band] = value
    return cfg


def _clean_source_string(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _gaia_floor_group(row, band: str) -> tuple[str, str]:
    source = _clean_source_string(row.get(f"{band}_source", ""))
    if source == "Gaia_DR3_vari_cepheid":
        return "gaia_vari_cepheid", source
    if source == "Gaia_DR3_gaia_source":
        return "gaia_source_mean", source
    # Conservative fallback for any non-vari Gaia mean-like source.
    return "gaia_source_mean", source or "unknown_gaia_source"


def _jhk_floor_group(row, band: str) -> tuple[str, str]:
    recommended_col = f"{band}_source_recommended_phasecorr_rp"
    source = _clean_source_string(row.get(recommended_col, ""))
    fallback = _clean_source_string(row.get(f"{band}_source", ""))
    data_level = _clean_source_string(row.get(f"{band}_data_level", ""))
    source_for_label = source or fallback or data_level
    source_l = source_for_label.lower()
    fallback_l = fallback.lower()
    level_l = data_level.lower()

    if "bras2025" in source_l or "phase_corrected" in source_l or "phasecorr" in source_l:
        return "jhk_bras2025_rp_phase_corrected", source_for_label
    if source_for_label in {"Wang2018", "Bhardwaj2024"} or "literature" in level_l:
        return "jhk_literature_intensity_mean", source_for_label
    if "2mass" in source_l or "2mass" in fallback_l or "single_epoch" in level_l:
        return "jhk_2mass_single_epoch", source_for_label
    # Conservative fallback: unknown JHK values are treated like 2MASS single epoch.
    return "jhk_2mass_single_epoch", source_for_label or "unknown_jhk_source"


def _w1_floor_group(row) -> tuple[str, str]:
    source = _clean_source_string(row.get("W1_source", ""))
    source_l = source.lower()
    if source_l == "unwise":
        return "w1_unwise", source
    if source_l == "wise_allsky":
        return "w1_wise_allsky", source
    if source_l == "allwise":
        return "w1_allwise", source
    # Conservative fallback for profile-fit WISE-like measurements.
    return "w1_allwise", source or "unknown_w1_source"


def source_floor_for_band(row, band: str, floor_config=None) -> tuple[float, str, str]:
    """Return ``(floor_mag, floor_group, source_label)`` for one SED band."""
    cfg = ACTIVE_FLOOR_CONFIG if floor_config is None else floor_config
    band = str(band)
    if band in {"BP", "G", "RP"}:
        group, source_label = _gaia_floor_group(row, band)
    elif band in {"J", "H", "K"}:
        group, source_label = _jhk_floor_group(row, band)
    elif band == "W1":
        group, source_label = _w1_floor_group(row)
    else:
        group, source_label = "no_floor", "unsupported_band"

    floor = float(cfg.get(group, {}).get(band, 0.0))
    return floor, group, source_label


def source_floor_vector(row, sed_bands, floor_config=None):
    floors = []
    groups = []
    labels = []
    for band in list(sed_bands):
        floor, group, label = source_floor_for_band(row, band, floor_config=floor_config)
        floors.append(floor)
        groups.append(group)
        labels.append(label)
    return np.asarray(floors, dtype=float), groups, labels


def row_sed_vector_with_source_floor(row):
    sed_bands, sed_obs, sed_err = ORIGINAL_ROW_SED_VECTOR(row)
    if not USE_SOURCE_DEPENDENT_FLOOR:
        return sed_bands, sed_obs, sed_err
    floors, _groups, _labels = source_floor_vector(row, sed_bands)
    eff_err = np.sqrt(np.asarray(sed_err, dtype=float) ** 2 + floors**2)
    return sed_bands, sed_obs, eff_err


def _summary_model_and_zps(directbase, mapbase):
    model = getattr(directbase, "_WORKER_MODEL", None)
    zps = getattr(directbase, "_WORKER_ZPS", None)
    if model is not None and zps is not None:
        return model, zps
    zps = mapbase.load_zps()
    PreconvolvedBOSZModel = mapbase.sedbase._import_model()
    model = PreconvolvedBOSZModel(cache_file=mapbase.sedbase.CACHE_FILE)
    return model, zps


def _theta_from_summary(summary_row, param_names):
    return np.asarray([summary_row.get(f"{p}_p50", np.nan) for p in param_names], dtype=float)


def make_summarize_samples_with_source_floor(directbase, mapbase):
    def summarize_samples_source_floor(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
        out = ORIGINAL_SUMMARIZE_SAMPLES(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin)
        out["uses_fixed_band_floor"] = bool(USE_SOURCE_DEPENDENT_FLOOR)
        out["fixed_floor_config"] = json.dumps(ACTIVE_FLOOR_CONFIG, sort_keys=True)

        try:
            sed_bands_no, sed_obs, sed_err_no = ORIGINAL_ROW_SED_VECTOR(row)
            floors, groups, labels = source_floor_vector(row, sed_bands_no)
            sed_err_eff = np.sqrt(np.asarray(sed_err_no, dtype=float) ** 2 + floors**2)
            for band in SED_BANDS:
                if band in list(sed_bands_no):
                    idx = list(sed_bands_no).index(band)
                    out[f"floor_{band}_used"] = float(floors[idx])
                    out[f"floor_group_{band}"] = str(groups[idx])
                    out[f"floor_source_{band}"] = str(labels[idx])
                    out[f"{band}_mag_err_catalog"] = float(np.asarray(sed_err_no, dtype=float)[idx])
                    out[f"{band}_mag_err_effective"] = float(sed_err_eff[idx])
                else:
                    out[f"floor_{band}_used"] = np.nan
                    out[f"floor_group_{band}"] = ""
                    out[f"floor_source_{band}"] = ""
                    out[f"{band}_mag_err_catalog"] = np.nan
                    out[f"{band}_mag_err_effective"] = np.nan

            model, zps = _summary_model_and_zps(directbase, mapbase)
            theta = _theta_from_summary(out, directbase.PARAM_NAMES)
            pred = mapbase.model_mags(theta, model, zps, sed_bands_no)
            delta = np.asarray(sed_obs, dtype=float) - np.asarray(pred, dtype=float)
            var_no = np.asarray(sed_err_no, dtype=float) ** 2
            var_floor = sed_err_eff**2
            out["sed_chi2_no_floor"] = float(np.nansum(delta * delta / var_no))
            out["sed_chi2_with_floor"] = float(np.nansum(delta * delta / var_floor))
            out["sed_chi2_no_floor_per_band"] = float(out["sed_chi2_no_floor"] / max(len(sed_bands_no), 1))
            out["sed_chi2_with_floor_per_band"] = float(out["sed_chi2_with_floor"] / max(len(sed_bands_no), 1))
            out["sed_loglike_with_floor_p50"] = -0.5 * float(
                np.nansum(delta * delta / var_floor + np.log(2.0 * np.pi * var_floor))
            )
        except Exception as exc:
            for band in SED_BANDS:
                out[f"floor_{band}_used"] = np.nan
                out[f"floor_group_{band}"] = ""
                out[f"floor_source_{band}"] = ""
                out[f"{band}_mag_err_catalog"] = np.nan
                out[f"{band}_mag_err_effective"] = np.nan
            out["sed_chi2_no_floor"] = np.nan
            out["sed_chi2_with_floor"] = np.nan
            out["sed_chi2_no_floor_per_band"] = np.nan
            out["sed_chi2_with_floor_per_band"] = np.nan
            out["sed_loglike_with_floor_p50"] = np.nan
            out["source_floor_summary_warning"] = str(exc)
        return out

    return summarize_samples_source_floor


def _apply_source_floor_patches(directbase, mapbase, out_dir, use_source_dependent_floor, floor_config):
    global ORIGINAL_ROW_SED_VECTOR, ORIGINAL_SUMMARIZE_SAMPLES
    global ACTIVE_FLOOR_CONFIG, USE_SOURCE_DEPENDENT_FLOOR

    ACTIVE_FLOOR_CONFIG = _deep_update_floor_config(floor_config)
    USE_SOURCE_DEPENDENT_FLOOR = bool(use_source_dependent_floor)
    ORIGINAL_ROW_SED_VECTOR = mapbase.row_sed_vector
    ORIGINAL_SUMMARIZE_SAMPLES = directbase.summarize_samples

    mapbase.row_sed_vector = row_sed_vector_with_source_floor
    directbase.summarize_samples = make_summarize_samples_with_source_floor(directbase, mapbase)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    directbase.OUT_DIR = out_dir
    directbase.SELECTED_OUT = out_dir / "MW_SED_latest_source_dependent_floor_selected.csv"
    directbase.SUMMARY_OUT = out_dir / "MW_SED_latest_source_dependent_floor_summary.csv"
    directbase.FAIL_OUT = out_dir / "MW_SED_latest_source_dependent_floor_failures.csv"
    directbase.RUNINFO_OUT = out_dir / "MW_SED_latest_source_dependent_floor_runinfo.json"
    directbase._WORKER_MODEL = None
    directbase._WORKER_ZPS = None
    directbase._WORKER_PC_COEFF = None
    directbase._WORKER_PC_COV_LONG = None
    directbase._WORKER_BOUNDS = None


def _write_standard_outputs(out_dir: Path, selected: pd.DataFrame, summary: pd.DataFrame, runinfo: dict, fail_path: Path | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "selected.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if fail_path is not None and fail_path.exists():
        shutil.copy2(fail_path, out_dir / "failures.csv")
    else:
        pd.DataFrame().to_csv(out_dir / "failures.csv", index=False)
    with open(out_dir / "runinfo.json", "w", encoding="utf-8") as f:
        json.dump(runinfo, f, ensure_ascii=False, indent=2)


def run_mw_sed_latest_pc_source_dependent_floor_distanceprior_emcee_direct(
    n_stars=30,
    quality_mode="primary",
    fix_logg_mh=False,
    fixed_logg=2.0,
    fixed_mh=0.0,
    use_vi_pc_prior=False,
    use_source_dependent_floor=True,
    floor_config=None,
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
):
    """Run latest pure-photometry SED fitting with fixed source-dependent floors."""
    fix_logg_mh = bool(fix_logg_mh)
    use_vi_pc_prior = bool(use_vi_pc_prior)
    out_dir = Path(out_dir) if out_dir is not None else Path(out_root) / _config_slug(fix_logg_mh, use_vi_pc_prior)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fix_logg_mh:
        fixed_module = _load_fixed_module()
        latest_module = fixed_module.latest
        directbase = fixed_module.directbase
        mapbase = fixed_module.mapbase
        _set_pc_colors(latest_module, use_vi_pc_prior)
        latest_module._apply_latest_patches(
            out_dir=out_dir,
            phot_file=phot_file or latest_module.DEFAULT_PHOT_FILE,
            wang25_file=wang25_file or latest_module.DEFAULT_WANG25_FILE,
            old_phot_file=old_phot_file or latest_module.DEFAULT_OLD_PHOT_FILE,
            zp_file=zp_file or latest_module.DEFAULT_ZP_FILE,
            cache_file=cache_file or latest_module.DEFAULT_CACHE_FILE,
            vi_cache_file=vi_cache_file or latest_module.DEFAULT_VI_CACHE_FILE,
            pc_coeff_file=pc_coeff_file or latest_module.DEFAULT_PC_COEFF_FILE,
            pc_cov_file=pc_cov_file or latest_module.DEFAULT_PC_COV_FILE,
        )
        fixed_module._apply_fixed_patches(out_dir)
        directbase = fixed_module.directbase
        mapbase = fixed_module.mapbase
    else:
        _mapbase0, _directbase0, latest_module = _load_modules()
        directbase = latest_module.directbase
        mapbase = latest_module.mapbase
        _set_pc_colors(latest_module, use_vi_pc_prior)
        latest_module._apply_latest_patches(
            out_dir=out_dir,
            phot_file=phot_file or latest_module.DEFAULT_PHOT_FILE,
            wang25_file=wang25_file or latest_module.DEFAULT_WANG25_FILE,
            old_phot_file=old_phot_file or latest_module.DEFAULT_OLD_PHOT_FILE,
            zp_file=zp_file or latest_module.DEFAULT_ZP_FILE,
            cache_file=cache_file or latest_module.DEFAULT_CACHE_FILE,
            vi_cache_file=vi_cache_file or latest_module.DEFAULT_VI_CACHE_FILE,
            pc_coeff_file=pc_coeff_file or latest_module.DEFAULT_PC_COEFF_FILE,
            pc_cov_file=pc_cov_file or latest_module.DEFAULT_PC_COV_FILE,
        )

    _apply_source_floor_patches(
        directbase=directbase,
        mapbase=mapbase,
        out_dir=out_dir,
        use_source_dependent_floor=use_source_dependent_floor,
        floor_config=floor_config,
    )

    selected, summary, samples = directbase.run_mw_sed_pcprior_method2_distanceprior_emcee_direct(
        n_stars=n_stars,
        quality_mode=quality_mode,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        thin=thin,
        random_seed=random_seed,
        save_samples=save_samples,
        make_corner=make_corner,
        progress=progress,
        n_workers=n_workers,
    )

    if len(summary) > 0:
        summary = summary.copy()
        summary["fit_configuration"] = _config_slug(fix_logg_mh, use_vi_pc_prior)
        summary["fix_logg_mh"] = bool(fix_logg_mh)
        summary["fixed_logg_value"] = float(fixed_logg) if fix_logg_mh else np.nan
        summary["fixed_MH_value"] = float(fixed_mh) if fix_logg_mh else np.nan
        summary["use_vi_pc_prior"] = bool(use_vi_pc_prior)
        summary["pc_prior_color_set"] = "with_VI" if use_vi_pc_prior else "no_VI"
        summary["uses_fixed_band_floor"] = bool(use_source_dependent_floor)
        summary["floor_config_name"] = "default_source_dependent" if floor_config is None else "custom_source_dependent"
        summary["floor_config_json"] = json.dumps(ACTIVE_FLOOR_CONFIG, sort_keys=True)
        summary.to_csv(directbase.SUMMARY_OUT, index=False)

    fail_path = directbase.FAIL_OUT
    runinfo = {
        "method": "latest pure-photometry SED direct emcee with fixed source-dependent magnitude floors",
        "out_dir": str(out_dir),
        "fit_configuration": _config_slug(fix_logg_mh, use_vi_pc_prior),
        "n_selected": int(len(selected)),
        "n_success": int(len(summary)),
        "quality_mode": quality_mode,
        "fix_logg_mh": bool(fix_logg_mh),
        "fixed_logg": float(fixed_logg) if fix_logg_mh else None,
        "fixed_MH": float(fixed_mh) if fix_logg_mh else None,
        "use_vi_pc_prior": bool(use_vi_pc_prior),
        "pc_color_defs": _pc_color_defs(use_vi_pc_prior),
        "sed_bands": SED_BANDS,
        "uses_pc_prior": True,
        "uses_distance_prior": True,
        "uses_fixed_band_floor": bool(use_source_dependent_floor),
        "floor_config_name": "default_source_dependent" if floor_config is None else "custom_source_dependent",
        "floor_config": ACTIVE_FLOOR_CONFIG,
        "floor_formula": "sigma_eff^2 = sigma_cat^2 + sigma_floor^2",
        "nwalkers_requested": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "thin": int(thin),
        "random_seed": int(random_seed),
        "n_workers": int(n_workers),
        "save_samples": str(save_samples),
        "make_corner": bool(make_corner),
        "make_plots": bool(make_plots),
        "progress": bool(progress),
        "resume_argument": bool(resume),
    }
    _write_standard_outputs(out_dir, selected, summary, runinfo, fail_path)
    return selected, summary, samples


print("Loaded latest MW pure-photometry SED fitter with fixed source-dependent floors.")
print(
    "Call: selected, summary, samples = "
    "run_mw_sed_latest_pc_source_dependent_floor_distanceprior_emcee_direct(...)"
)
