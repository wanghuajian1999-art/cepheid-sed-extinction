#!/usr/bin/env python3
"""Latest MW pure-photometry SED fit with fixed logg=2 and [M/H]=0.

This is a thin wrapper around
``jupyter_mw_sed_latest_pc_loggF_rgcFeH_distanceprior_emcee_direct.py``.
It keeps the same photometry, PC prior, distance prior, sample selection, and
emcee driver, but samples only

    (Teff, Av, Rv, logD, logR)

while all synthetic photometry calls use logg=2.0 and [M/H]=0.0.
"""

from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import jupyter_mw_sed_latest_pc_loggF_rgcFeH_distanceprior_emcee_direct as latest


FIXED_LOGG = 2.0
FIXED_MH = 0.0
FIT_PARAM_NAMES = ["Teff", "Av", "Rv", "logD", "logR"]
FULL_PARAM_NAMES = ["Teff", "logg", "MH", "Av", "Rv", "logD", "logR"]
FULL_TO_FIT_INDEX = np.asarray([0, 3, 4, 5, 6], dtype=int)

DEFAULT_OUT_DIR = SCRIPT_DIR / "sed_latest_noqualityPC_fixedMH0_logg2_distanceprior_emcee_direct"


directbase = latest.directbase
mapbase = latest.mapbase


def _finite(x):
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def _reset_base_modules():
    """Reload patched modules so repeated notebook runs start from a clean base."""
    global latest, directbase, mapbase
    import jupyter_all_mw_sed_pcprior_method2_distanceprior as mapbase_module
    import jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct as directbase_module

    importlib.reload(mapbase_module)
    directbase = importlib.reload(directbase_module)
    latest = importlib.reload(latest)
    directbase = latest.directbase
    mapbase = latest.mapbase


def expand_theta_fixed(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.size == 7:
        full = theta.copy()
        full[1] = FIXED_LOGG
        full[2] = FIXED_MH
        return full
    if theta.size != 5:
        raise ValueError(f"expected 5 or 7 parameters, got {theta.size}")
    teff, av, rv, logd, logr = theta
    return np.asarray([teff, FIXED_LOGG, FIXED_MH, av, rv, logd, logr], dtype=float)


def _fit_bounds_from_full_bounds(full_bounds):
    return {
        "lower": np.asarray(full_bounds["lower"], dtype=float)[FULL_TO_FIT_INDEX],
        "upper": np.asarray(full_bounds["upper"], dtype=float)[FULL_TO_FIT_INDEX],
    }


def _full_bounds_for_model_from_grid(model):
    lower = np.asarray(
        [
            4000.0,
            0.5,
            -1.5,
            0.0,
            max(2.1, float(model.rv_grid[0])),
            np.log10(0.05),
            np.log10(3.0),
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            7500.0,
            3.5,
            0.5,
            20.0,
            min(5.6, float(model.rv_grid[-1])),
            np.log10(50.0),
            np.log10(1000.0),
        ],
        dtype=float,
    )
    return {"lower": lower, "upper": upper}


def bounds_for_model_fixed(model):
    full_bounds = _full_bounds_for_model_from_grid(model)
    return _fit_bounds_from_full_bounds(full_bounds)


def _direct_center_fixed(row, bounds):
    lower = bounds["lower"]
    upper = bounds["upper"]

    d = _finite(row.get("<d>", np.nan))
    if not np.isfinite(d) or d <= 0:
        d = 5.0

    center = np.asarray(
        [
            directbase._teff_center_from_period(row),
            directbase._av_center(row),
            3.1,
            np.log10(d),
            np.log10(directbase._radius_seed_from_period(row)),
        ],
        dtype=float,
    )
    return np.clip(center, lower + 1.0e-6, upper - 1.0e-6)


def _draw_one_prior_position_fixed(row, bounds, rng):
    lower = bounds["lower"]
    upper = bounds["upper"]
    center = _direct_center_fixed(row, bounds)

    teff = rng.normal(center[0], 350.0)
    teff = np.clip(teff, lower[0] + 1.0e-6, upper[0] - 1.0e-6)

    av = rng.normal(center[1], 0.6)
    rv = rng.normal(3.1, 0.35)

    d_mu = _finite(row.get("<d>", np.nan))
    d_sig = _finite(row.get("<d_error>", np.nan))
    if np.isfinite(d_mu) and np.isfinite(d_sig) and d_mu > 0 and d_sig > 0:
        d = rng.normal(d_mu, d_sig)
        if (not np.isfinite(d)) or d <= 0:
            d = d_mu
    else:
        d = 10.0 ** center[3]

    log_radius = rng.uniform(np.log10(3.0), np.log10(600.0))
    theta = np.asarray([teff, av, rv, np.log10(d), log_radius], dtype=float)
    return np.clip(theta, lower + 1.0e-6, upper - 1.0e-6)


def initialize_walkers_direct_fixed(row, bounds, nwalkers, random_seed):
    rng = np.random.default_rng(random_seed)
    ndim = len(FIT_PARAM_NAMES)
    nwalkers = int(max(nwalkers, 2 * ndim + 2))
    lower = bounds["lower"]
    upper = bounds["upper"]

    p0 = []
    n_try = 0
    while len(p0) < nwalkers and n_try < 10000:
        n_try += 1
        p = _draw_one_prior_position_fixed(row, bounds, rng)
        if np.all(np.isfinite(p)) and np.all(p > lower) and np.all(p < upper):
            p0.append(p)

    if len(p0) < nwalkers:
        center = _direct_center_fixed(row, bounds)
        scale = np.asarray([50.0, 0.05, 0.03, 0.005, 0.02], dtype=float)
        while len(p0) < nwalkers:
            p = center + rng.normal(0.0, scale)
            p = np.clip(p, lower + 1.0e-6, upper - 1.0e-6)
            p0.append(p)

    return np.asarray(p0, dtype=float)


def prior_residuals_fixed(theta, row):
    full = expand_theta_fixed(theta)
    logd = full[5]
    d_prior = _finite(row.get("<d>", np.nan))
    d_err = _finite(row.get("<d_error>", np.nan))
    d_model = 10.0 ** logd
    if np.isfinite(d_prior) and np.isfinite(d_err) and d_prior > 0 and d_err > 0:
        return np.asarray([(d_model - d_prior) / d_err], dtype=float)
    return np.asarray([], dtype=float)


def intrinsic_color_syn_fixed(theta, model, zps, pc_prior):
    return latest.intrinsic_color_syn_latest(expand_theta_fixed(theta), model, zps, pc_prior)


def pc_prior_residuals_fixed(theta, model, zps, pc_prior, pc_cov):
    return latest.pc_prior_residuals_latest(expand_theta_fixed(theta), model, zps, pc_prior, pc_cov)


def model_mags_fixed(theta, model, zps, sed_bands):
    return latest.model_mags_latest(expand_theta_fixed(theta), model, zps, sed_bands)


def residual_fixed(theta, row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds):
    theta = np.asarray(theta, dtype=float)
    lower, upper = bounds["lower"], bounds["upper"]
    n_bad = len(sed_obs) + len(latest.PC_COLOR_DEFS) + 1
    if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
        return np.full(n_bad, 1.0e6)
    try:
        full = expand_theta_fixed(theta)
        sed_pred = latest.model_mags_latest(full, model, zps, sed_bands)
        sed_res = (sed_obs - sed_pred) / sed_err
        pc_res, _method = latest.pc_prior_residuals_latest(full, model, zps, pc_prior, pc_cov)
        prior_res = prior_residuals_fixed(theta, row)
        return np.concatenate([sed_res, pc_res, prior_res])
    except Exception:
        return np.full(n_bad, 1.0e6)


def run_emcee_direct_fixed(*args, **kwargs):
    sampler, samples = directbase._fixed_original_run_emcee_direct(*args, **kwargs)
    if "logg" not in samples.columns:
        insert_at = list(samples.columns).index("Teff") + 1 if "Teff" in samples.columns else 1
        samples.insert(insert_at, "logg", FIXED_LOGG)
    if "MH" not in samples.columns:
        insert_at = list(samples.columns).index("logg") + 1 if "logg" in samples.columns else 2
        samples.insert(insert_at, "MH", FIXED_MH)
    return sampler, samples


def summarize_samples_fixed(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
    out = directbase.summarize_samples_original(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin)

    for p, fixed in [("logg", FIXED_LOGG), ("MH", FIXED_MH)]:
        out[f"{p}_p16"] = float(fixed)
        out[f"{p}_p50"] = float(fixed)
        out[f"{p}_p84"] = float(fixed)
        out[f"{p}_err_minus"] = 0.0
        out[f"{p}_err_plus"] = 0.0
        out[f"{p}_map_sample"] = float(fixed)

    logp0, logp0_source = latest.logp0_for_loggF_latest(row)
    teff = out.get("Teff_p50", np.nan)
    loggF = FIXED_LOGG - 4.0 * np.log10(teff / 1.0e4) if np.isfinite(teff) and teff > 0 else np.nan
    loggF_mu = latest.loggF_prior_mu_latest(row)
    loggF_delta = (loggF - loggF_mu) / latest.LOGGF_PRIOR_SIGMA if np.isfinite(loggF) and np.isfinite(loggF_mu) else np.nan

    out.update(
        {
            "pc_prior_version": "LMC_temperature_six_WC2023_Rv3p4_no_quality_3sigma",
            "pc_quality_filter": "finite_only_plus_3sigma_clip",
            "fixed_logg": True,
            "fixed_logg_value": FIXED_LOGG,
            "fixed_MH": True,
            "fixed_MH_value": FIXED_MH,
            "mh_prior_mode": "fixed_MH0_no_prior",
            "mh_prior_sigma": np.nan,
            "MH_prior": np.nan,
            "MH_prior_sigma": np.nan,
            "FeH_Wang25_original": _finite(row.get("FeH_Wang25_original", np.nan)),
            "RGC_Wang25_kpc": _finite(row.get("RGC_Wang25_kpc", np.nan)),
            "FeH_prior_mu_RGC_gradient": _finite(row.get("FeH_prior_mu_RGC_gradient", np.nan)),
            "FeH_gradient_alpha": latest.FE_GRAD_ALPHA,
            "FeH_gradient_beta": latest.FE_GRAD_BETA,
            "FeH_gradient_rms": latest.FE_GRAD_SIGMA,
            "logP_pc_used": latest.logp_for_pc_latest(row),
            "logP0_used": logp0,
            "logP0_source": logp0_source,
            "loggF_relation_mu_diagnostic": loggF_mu,
            "loggF_relation_sigma_diagnostic": latest.LOGGF_PRIOR_SIGMA,
            "loggF_p50": float(loggF) if np.isfinite(loggF) else np.nan,
            "loggF_relation_delta_sigma_diagnostic": float(loggF_delta) if np.isfinite(loggF_delta) else np.nan,
            "loggF_prior_mu": np.nan,
            "loggF_prior_sigma": np.nan,
            "logg_prior_sigma": np.nan,
            "uses_latest_PC_prior": True,
            "uses_pc_prior": True,
            "uses_loggF_prior": False,
            "uses_logg_prior": False,
            "uses_RGC_FeH_prior": False,
            "uses_mh_prior": False,
            "uses_distance_prior": True,
        }
    )
    return out


def make_corner_plot_fixed(samples, row, out_dir):
    import corner

    plot_cols = ["Teff", "Av", "Rv", "D_kpc", "R_Rsun"]
    labels = [r"$T_{\rm eff}$", r"$A_V$", r"$R_V$", r"$D$ kpc", r"$R/R_\odot$"]
    fig = corner.corner(
        samples[plot_cols],
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 9},
    )
    out = out_dir / f"{row['selection_tag']}_{directbase._safe_name(row.get('Name', 'source'))}_corner.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _apply_fixed_patches(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapbase.bounds_for_model = bounds_for_model_fixed
    mapbase.prior_residuals = prior_residuals_fixed
    mapbase.intrinsic_color_syn = intrinsic_color_syn_fixed
    mapbase.pc_prior_residuals = pc_prior_residuals_fixed
    mapbase.model_mags = model_mags_fixed
    mapbase.residual = residual_fixed

    directbase.PARAM_NAMES = list(FIT_PARAM_NAMES)
    directbase._direct_center = _direct_center_fixed
    directbase._draw_one_prior_position = _draw_one_prior_position_fixed
    directbase.initialize_walkers_direct = initialize_walkers_direct_fixed
    directbase._fixed_original_run_emcee_direct = directbase.run_emcee_direct
    directbase.run_emcee_direct = run_emcee_direct_fixed
    directbase.summarize_samples = summarize_samples_fixed
    directbase.make_corner_plot = make_corner_plot_fixed

    directbase.OUT_DIR = out_dir
    directbase.SELECTED_OUT = out_dir / "MW_SED_latest_noqualityPC_fixedMH0_logg2_selected.csv"
    directbase.SUMMARY_OUT = out_dir / "MW_SED_latest_noqualityPC_fixedMH0_logg2_summary.csv"
    directbase.FAIL_OUT = out_dir / "MW_SED_latest_noqualityPC_fixedMH0_logg2_failures.csv"
    directbase.RUNINFO_OUT = out_dir / "MW_SED_latest_noqualityPC_fixedMH0_logg2_runinfo.json"
    directbase._WORKER_MODEL = None
    directbase._WORKER_ZPS = None
    directbase._WORKER_PC_COEFF = None
    directbase._WORKER_PC_COV_LONG = None
    directbase._WORKER_BOUNDS = None


def run_mw_sed_latest_pc_fixedMH0_logg2_distanceprior_emcee_direct(
    n_stars=30,
    quality_mode="primary",
    pc_coeff_file=latest.DEFAULT_PC_COEFF_FILE,
    pc_cov_file=latest.DEFAULT_PC_COV_FILE,
    phot_file=latest.DEFAULT_PHOT_FILE,
    wang25_file=latest.DEFAULT_WANG25_FILE,
    old_phot_file=latest.DEFAULT_OLD_PHOT_FILE,
    zp_file=latest.DEFAULT_ZP_FILE,
    cache_file=latest.DEFAULT_CACHE_FILE,
    vi_cache_file=latest.DEFAULT_VI_CACHE_FILE,
    out_dir=DEFAULT_OUT_DIR,
    fixed_logg=FIXED_LOGG,
    fixed_mh=FIXED_MH,
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
    """Run latest pure-photometry SED fit with fixed logg=2 and [M/H]=0."""
    if abs(float(fixed_logg) - FIXED_LOGG) > 1.0e-12:
        raise ValueError("This fixed-parameter version requires fixed_logg=2.0")
    if abs(float(fixed_mh) - FIXED_MH) > 1.0e-12:
        raise ValueError("This fixed-parameter version requires fixed_mh=0.0")

    _reset_base_modules()
    latest._apply_latest_patches(
        out_dir=out_dir,
        phot_file=phot_file,
        wang25_file=wang25_file,
        old_phot_file=old_phot_file,
        zp_file=zp_file,
        cache_file=cache_file,
        vi_cache_file=vi_cache_file,
        pc_coeff_file=pc_coeff_file,
        pc_cov_file=pc_cov_file,
    )
    _apply_fixed_patches(out_dir)

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

    runinfo_path = Path(out_dir) / "MW_SED_latest_noqualityPC_fixedMH0_logg2_runinfo_extra.json"
    runinfo = {
        "pc_prior_version": "LMC_temperature_six_WC2023_Rv3p4_no_quality_3sigma",
        "pc_quality_filter": "finite_only_plus_3sigma_clip",
        "pc_coeff_file": str(pc_coeff_file),
        "pc_cov_file": str(pc_cov_file),
        "phot_file": str(phot_file),
        "zp_file": str(zp_file),
        "cache_file": str(cache_file),
        "vi_cache_file": str(vi_cache_file),
        "fixed_logg": FIXED_LOGG,
        "fixed_MH": FIXED_MH,
        "sampled_parameters": FIT_PARAM_NAMES,
        "full_model_parameters": FULL_PARAM_NAMES,
        "uses_pc_prior": True,
        "uses_distance_prior": True,
        "uses_loggF_prior": False,
        "uses_mh_prior": False,
        "sed_bands": latest.SED_BANDS,
        "pc_color_defs": latest.PC_COLOR_DEFS,
        "make_plots_argument": bool(make_plots),
        "resume_argument": bool(resume),
    }
    with open(runinfo_path, "w", encoding="utf-8") as f:
        json.dump(runinfo, f, ensure_ascii=False, indent=2)

    return selected, summary, samples


print("Loaded fixed-MH/logg latest MW pure-photometry SED fitter.")
print("Call: selected, fixed_summary, fixed_samples = run_mw_sed_latest_pc_fixedMH0_logg2_distanceprior_emcee_direct(...)")
