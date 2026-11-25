#!/usr/bin/env python3
"""Direct-emcee MW Cepheid SED fit with intrinsic-color PC prior.

Jupyter usage
-------------

    %run -i "jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct.py"

    selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_method2_distanceprior_emcee_direct(
        n_stars=30,
        quality_mode="primary",
        nwalkers=32,
        nsteps=2500,
        burnin=800,
        thin=10,
        random_seed=42,
        save_samples="diagnostic",
        make_corner=True,
        progress=False,
    )

This script uses the same physical model and inputs as
``jupyter_all_mw_sed_pcprior_method2_distanceprior.py`` but replaces the
least-squares/Hessian fit with direct emcee sampling. Walkers are initialized
from broad physical/prior centers, not from a MAP fit.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = BASE_DIR
OUT_DIR = SCRIPT_DIR / "sed_pcprior_method2_distanceprior_emcee_direct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
import jupyter_all_mw_sed_pcprior_method2_distanceprior as mapbase


PARAM_NAMES = ["Teff", "logg", "MH", "Av", "Rv", "logD", "logR"]
DERIVED_NAMES = ["D_kpc", "R_Rsun"]

SELECTED_OUT = OUT_DIR / "MW_SED_PCprior_method2_distanceprior_emcee_direct_selected.csv"
SUMMARY_OUT = OUT_DIR / "MW_SED_PCprior_method2_distanceprior_emcee_direct_summary.csv"
FAIL_OUT = OUT_DIR / "MW_SED_PCprior_method2_distanceprior_emcee_direct_failures.csv"
RUNINFO_OUT = OUT_DIR / "MW_SED_PCprior_method2_distanceprior_emcee_direct_runinfo.json"


def _finite(x):
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def _safe_name(x) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(x))[:80]


def _period_logp(row) -> float:
    logp = _finite(row.get("logP_F_plot_Pilecki2024_MW", np.nan))
    if np.isfinite(logp):
        return logp
    period = _finite(row.get("period_used", np.nan))
    return np.log10(period) if np.isfinite(period) and period > 0 else np.nan


def _teff_center_from_period(row) -> float:
    """A deliberately weak period-dependent Teff center for walker seeding."""
    logp = _period_logp(row)
    if not np.isfinite(logp):
        return 5800.0
    return float(np.clip(5800.0 - 350.0 * (logp - 1.0), 4000.0, 7500.0))


def _radius_seed_from_period(row) -> float:
    period = _finite(row.get("period_used", np.nan))
    if not np.isfinite(period) or period <= 0:
        logp = _period_logp(row)
        period = 10.0 ** logp if np.isfinite(logp) else 10.0
    return float(np.clip(35.0 * period**0.70, 3.0, 600.0))


def _av_center(row) -> float:
    for col in ["Av_p50", "Av_fit", "Av"]:
        val = _finite(row.get(col, np.nan))
        if np.isfinite(val):
            return float(np.clip(val, 0.0, 20.0))
    return 2.0


def _direct_center(row, bounds) -> np.ndarray:
    lower = bounds["lower"]
    upper = bounds["upper"]

    teff = _teff_center_from_period(row)
    logg = mapbase.logg_prior_mu(teff, row)
    if not np.isfinite(logg):
        logg = 2.0

    mh = _finite(row.get("[Fe/H]", np.nan))
    if not np.isfinite(mh):
        mh = 0.0

    d = _finite(row.get("<d>", np.nan))
    if not np.isfinite(d) or d <= 0:
        d = 5.0

    center = np.array(
        [
            teff,
            logg,
            mh,
            _av_center(row),
            3.1,
            np.log10(d),
            np.log10(_radius_seed_from_period(row)),
        ],
        dtype=float,
    )
    return np.clip(center, lower + 1.0e-6, upper - 1.0e-6)


def _draw_one_prior_position(row, bounds, rng) -> np.ndarray:
    """Draw one walker from broad priors around physical centers."""
    lower = bounds["lower"]
    upper = bounds["upper"]
    center = _direct_center(row, bounds)

    teff = rng.normal(center[0], 350.0)
    teff = np.clip(teff, lower[0] + 1.0e-6, upper[0] - 1.0e-6)

    logg_mu = mapbase.logg_prior_mu(teff, row)
    if not np.isfinite(logg_mu):
        logg_mu = center[1]
    logg = rng.normal(logg_mu, mapbase.LOGG_PRIOR_SIGMA)

    mh_mu = _finite(row.get("[Fe/H]", np.nan))
    if not np.isfinite(mh_mu):
        mh_mu = center[2]
    mh = rng.normal(mh_mu, mapbase.FEH_PRIOR_SIGMA)

    av = rng.normal(center[3], 0.6)
    rv = rng.normal(3.1, 0.35)

    d_mu = _finite(row.get("<d>", np.nan))
    d_sig = _finite(row.get("<d_error>", np.nan))
    if np.isfinite(d_mu) and np.isfinite(d_sig) and d_mu > 0 and d_sig > 0:
        d = rng.normal(d_mu, d_sig)
        if (not np.isfinite(d)) or d <= 0:
            d = d_mu
    else:
        d = 10.0 ** center[5]

    log_radius = rng.uniform(np.log10(3.0), np.log10(600.0))

    theta = np.array([teff, logg, mh, av, rv, np.log10(d), log_radius], dtype=float)
    return np.clip(theta, lower + 1.0e-6, upper - 1.0e-6)


def initialize_walkers_direct(row, bounds, nwalkers, random_seed):
    rng = np.random.default_rng(random_seed)
    ndim = len(PARAM_NAMES)
    nwalkers = int(max(nwalkers, 2 * ndim + 2))
    lower = bounds["lower"]
    upper = bounds["upper"]

    p0 = []
    n_try = 0
    while len(p0) < nwalkers and n_try < 10000:
        n_try += 1
        p = _draw_one_prior_position(row, bounds, rng)
        if np.all(np.isfinite(p)) and np.all(p > lower) and np.all(p < upper):
            p0.append(p)

    if len(p0) < nwalkers:
        center = _direct_center(row, bounds)
        scale = np.array([50.0, 0.02, 0.02, 0.05, 0.03, 0.005, 0.02])
        while len(p0) < nwalkers:
            p = center + rng.normal(0.0, scale)
            p = np.clip(p, lower + 1.0e-6, upper - 1.0e-6)
            p0.append(p)

    return np.asarray(p0, dtype=float)


def log_probability(theta, row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds):
    res = mapbase.residual(
        theta,
        row,
        model,
        zps,
        sed_bands,
        sed_obs,
        sed_err,
        pc_prior,
        pc_cov,
        bounds,
    )
    if np.any(~np.isfinite(res)) or np.any(np.abs(res) > 1.0e5):
        return -np.inf
    return -0.5 * float(np.sum(res * res))


def run_emcee_direct(
    row,
    model,
    zps,
    sed_bands,
    sed_obs,
    sed_err,
    pc_prior,
    pc_cov,
    bounds,
    nwalkers,
    nsteps,
    burnin,
    thin,
    random_seed,
    progress=False,
):
    import emcee

    p0 = initialize_walkers_direct(row, bounds, nwalkers, random_seed=random_seed)
    ndim = len(PARAM_NAMES)
    sampler = emcee.EnsembleSampler(
        p0.shape[0],
        ndim,
        log_probability,
        args=(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds),
    )
    sampler.run_mcmc(p0, int(nsteps), progress=bool(progress))

    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    logp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
    samples = pd.DataFrame(flat, columns=PARAM_NAMES)
    samples["D_kpc"] = 10.0 ** samples["logD"]
    samples["R_Rsun"] = 10.0 ** samples["logR"]
    samples["log_prob"] = logp
    return sampler, samples


def summarize_samples(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
    out = {
        "selection_tag": row.get("selection_tag", ""),
        "row_id": row.get("row_id", np.nan),
        "Name": str(row.get("Name", "")),
        "GaiaDR3": str(row.get("GaiaDR3", "")),
        "Mode": row.get("Mode", ""),
        "period_used": _finite(row.get("period_used", np.nan)),
        "logP_used": _period_logp(row),
        "quality_tier": row.get("quality_tier", ""),
        "sed_bands_used": ",".join(sed_bands),
        "n_sed_bands": int(len(sed_bands)),
        "pc_mode_used": mapbase.pc_mode_for_row(row),
        "pc_colors_used": ",".join(pc_prior["color"].astype(str)),
        "n_pc_colors": int(len(pc_prior)),
        "distance_prior_source": "<d>",
        "distance_prior_kpc": _finite(row.get("<d>", np.nan)),
        "distance_prior_err_kpc": _finite(row.get("<d_error>", np.nan)),
        "MH_prior": _finite(row.get("[Fe/H]", np.nan)),
        "MH_prior_sigma": mapbase.FEH_PRIOR_SIGMA,
        "logg_prior_sigma": mapbase.LOGG_PRIOR_SIGMA,
        "nwalkers": int(len(sampler.acceptance_fraction)),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "thin": int(thin),
        "n_posterior_samples": int(len(samples)),
        "acceptance_fraction": float(np.nanmedian(sampler.acceptance_fraction)),
        "acceptance_fraction_min": float(np.nanmin(sampler.acceptance_fraction)),
        "acceptance_fraction_max": float(np.nanmax(sampler.acceptance_fraction)),
        "fit_success": bool(len(samples) > 0 and np.isfinite(samples["log_prob"]).any()),
        "fail_reason": "",
    }

    best_idx = int(np.nanargmax(samples["log_prob"].to_numpy(float)))
    for p in PARAM_NAMES + DERIVED_NAMES:
        vals = samples[p].to_numpy(float)
        p16, p50, p84 = np.nanpercentile(vals, [16, 50, 84])
        out[f"{p}_p16"] = float(p16)
        out[f"{p}_p50"] = float(p50)
        out[f"{p}_p84"] = float(p84)
        out[f"{p}_err_minus"] = float(p50 - p16)
        out[f"{p}_err_plus"] = float(p84 - p50)
        out[f"{p}_map_sample"] = float(samples.iloc[best_idx][p])

    out["log_prob_max"] = float(samples.iloc[best_idx]["log_prob"])
    if out["distance_prior_err_kpc"] > 0:
        out["D_prior_pull"] = (out["D_kpc_p50"] - out["distance_prior_kpc"]) / out["distance_prior_err_kpc"]
    else:
        out["D_prior_pull"] = np.nan
    return out


def model_mags_from_summary(summary_row, model, zps, sed_bands):
    theta = np.array([summary_row[f"{p}_p50"] for p in PARAM_NAMES], dtype=float)
    return mapbase.model_mags(theta, model, zps, sed_bands)


def make_corner_plot(samples, row, out_dir):
    import corner

    plot_cols = ["Teff", "logg", "MH", "Av", "Rv", "D_kpc", "R_Rsun"]
    labels = [r"$T_{\rm eff}$", r"$\log g$", r"$[M/H]$", r"$A_V$", r"$R_V$", r"$D$ kpc", r"$R/R_\odot$"]
    fig = corner.corner(
        samples[plot_cols],
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 9},
    )
    out = out_dir / f"{row['selection_tag']}_{_safe_name(row.get('Name', 'source'))}_corner.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_sed_mag_plot(row, summary_row, model, zps, sed_bands, sed_obs, sed_err, out_dir):
    pred = model_mags_from_summary(summary_row, model, zps, sed_bands)
    x = np.arange(len(sed_bands), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(x, sed_obs, yerr=sed_err, fmt="o", color="k", label="observed")
    ax.plot(x, pred, "s-", color="tab:red", label="model p50")
    ax.set_xticks(x)
    ax.set_xticklabels(sed_bands)
    ax.invert_yaxis()
    ax.set_ylabel("Vega mag")
    ax.set_title(f"{row['selection_tag']} {row.get('Name', '')}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"{row['selection_tag']}_{_safe_name(row.get('Name', 'source'))}_sed_mag_fit.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_selected_table(df, quality_mode, n_stars, random_seed):
    work = mapbase.build_work_table(df, quality_mode=quality_mode, max_stars=None)
    usable = []
    for _, row in work.iterrows():
        fail = mapbase.base_fail_reason(row, quality_mode="all")
        if fail:
            continue
        sed_bands, sed_obs, sed_err = mapbase.row_sed_vector(row)
        if len(sed_bands) < 4:
            continue
        usable.append(row)
    if not usable:
        return pd.DataFrame()

    cand = pd.DataFrame(usable).copy()
    cand["logP_select"] = cand.apply(_period_logp, axis=1)
    cand = cand[np.isfinite(cand["logP_select"])].copy()
    if cand.empty:
        return cand

    if n_stars is None:
        selected = cand.copy()
    else:
        rng = np.random.default_rng(random_seed)
        n_stars = min(int(n_stars), len(cand))
        selected_parts = []
        edges = np.nanquantile(cand["logP_select"], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        quota_base = n_stars // 3
        quotas = [quota_base, quota_base, n_stars - 2 * quota_base]
        used_index = set()
        for i, quota in enumerate(quotas):
            lo, hi = edges[i], edges[i + 1]
            if i == 2:
                sub = cand[(cand["logP_select"] >= lo) & (cand["logP_select"] <= hi)]
            else:
                sub = cand[(cand["logP_select"] >= lo) & (cand["logP_select"] < hi)]
            if sub.empty or quota <= 0:
                continue
            take = min(quota, len(sub))
            chosen = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            selected_parts.append(cand.loc[chosen])
            used_index.update(map(int, chosen))
        selected = pd.concat(selected_parts, axis=0) if selected_parts else cand.head(0)
        if len(selected) < n_stars:
            remaining = cand.loc[[idx for idx in cand.index if int(idx) not in used_index]]
            selected = pd.concat([selected, remaining.head(n_stars - len(selected))], axis=0)

    selected = selected.sort_values("logP_select").reset_index(drop=True)
    selected.insert(0, "selection_tag", [f"EMCEE{i+1:03d}" for i in range(len(selected))])
    return selected


def _should_save_sample(save_samples, i):
    if save_samples is True or str(save_samples).lower() == "all":
        return True
    if str(save_samples).lower() == "diagnostic":
        return i < 5
    return False


def _atomic_to_csv_gz(samples: pd.DataFrame, target: Path) -> None:
    """Write a posterior table atomically so interrupted jobs leave no bad gzip."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        samples.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _terminate_executor(executor, futures) -> None:
    """Cancel queued work and forcibly stop currently running workers."""
    for future in futures:
        future.cancel()

    # ProcessPoolExecutor has no public terminate_workers() before Python 3.14.
    # Capture its process objects before shutdown clears the private registry.
    processes = list(getattr(executor, "_processes", {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # Python versions without cancel_futures
        executor.shutdown(wait=False)

    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=3.0)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
    for process in processes:
        process.join(timeout=1.0)


def _clip_failure(row, reason, **extra):
    failure = {
        "selection_tag": row.get("selection_tag", ""),
        "row_id": row.get("row_id", np.nan),
        "Name": row.get("Name", ""),
        "GaiaDR3": row.get("GaiaDR3", ""),
        "quality_tier": row.get("quality_tier", ""),
        "fail_reason": str(reason),
    }
    failure.update(extra)
    return failure


def _fit_source_with_optional_clipping(
    *,
    row,
    model,
    zps,
    pc_coeff,
    pc_cov_long,
    bounds,
    nwalkers,
    nsteps,
    burnin,
    thin,
    random_seed,
    progress,
    iterative_residual_clipping,
    residual_clip_sigma,
    min_bands_after_clip,
):
    """Fit one source, optionally removing model-inconsistent photometry."""
    initial_bands, initial_obs, initial_err = mapbase.row_sed_vector(row)
    initial_bands = list(initial_bands)
    initial_obs = np.asarray(initial_obs, dtype=float)
    initial_err = np.asarray(initial_err, dtype=float)
    if bool(iterative_residual_clipping) and len(initial_bands) < int(min_bands_after_clip):
        raise ValueError(
            f"initial usable bands ({len(initial_bands)}) are fewer than min_bands_after_clip="
            f"{int(min_bands_after_clip)}"
        )

    pc_prior, pc_cov = mapbase.build_pc_prior(row, pc_coeff, pc_cov_long)
    current_bands = list(initial_bands)
    current_obs = initial_obs.copy()
    current_err = initial_err.copy()
    clipped_bands = []
    history = []
    initial_summary = None
    final_summary = None
    final_sampler = None
    final_samples = None
    final_pred = None
    final_residual = None
    reject_reason = ""
    max_iterations = max(1, len(initial_bands) - int(min_bands_after_clip) + 1)

    for iteration in range(max_iterations):
        sampler, samples = run_emcee_direct(
            row=row,
            model=model,
            zps=zps,
            sed_bands=current_bands,
            sed_obs=current_obs,
            sed_err=current_err,
            pc_prior=pc_prior,
            pc_cov=pc_cov,
            bounds=bounds,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            random_seed=int(random_seed) + iteration * 1000003,
            progress=progress,
        )
        summary_row = summarize_samples(
            row, samples, sampler, current_bands, pc_prior, nsteps, burnin, thin
        )
        pred = model_mags_from_summary(summary_row, model, zps, current_bands)
        normalized = (current_obs - pred) / current_err
        if np.any(~np.isfinite(normalized)):
            raise ValueError("non-finite normalized residual at posterior median")

        bad = np.flatnonzero(np.abs(normalized) > float(residual_clip_sigma))
        can_refit = (
            bool(iterative_residual_clipping)
            and len(bad) > 0
            and len(current_bands) - len(bad) >= int(min_bands_after_clip)
        )
        action = "clip_and_refit" if can_refit else (
            "reject_source" if bool(iterative_residual_clipping) and len(bad) > 0 else "retain"
        )
        bad_set = set(map(int, bad))
        for j, band in enumerate(current_bands):
            history.append(
                {
                    "selection_tag": row.get("selection_tag", ""),
                    "row_id": row.get("row_id", np.nan),
                    "Name": row.get("Name", ""),
                    "GaiaDR3": row.get("GaiaDR3", ""),
                    "iteration": int(iteration),
                    "band": band,
                    "observed_mag": float(current_obs[j]),
                    "model_mag_p50": float(pred[j]),
                    "effective_mag_error": float(current_err[j]),
                    "normalized_residual": float(normalized[j]),
                    "exceeds_clip_threshold": bool(j in bad_set),
                    "action": action if j in bad_set else "retain",
                }
            )

        if initial_summary is None:
            initial_summary = dict(summary_row)
            initial_summary["max_abs_residual_initial"] = float(np.max(np.abs(normalized)))

        final_summary = summary_row
        final_sampler = sampler
        final_samples = samples
        final_pred = pred
        final_residual = normalized

        if not iterative_residual_clipping or len(bad) == 0:
            break
        if not can_refit:
            reject_reason = (
                "residual_clipping_would_leave_fewer_than_"
                f"{int(min_bands_after_clip)}_bands"
            )
            break

        clipped_bands.extend(current_bands[j] for j in bad)
        keep = np.ones(len(current_bands), dtype=bool)
        keep[bad] = False
        current_bands = [band for j, band in enumerate(current_bands) if keep[j]]
        current_obs = current_obs[keep]
        current_err = current_err[keep]

    passed = not reject_reason and (
        not iterative_residual_clipping
        or np.all(np.abs(final_residual) <= float(residual_clip_sigma))
    )
    if not passed and not reject_reason:
        reject_reason = "residual_clipping_did_not_converge"

    metadata = {
        "n_bands_initial": int(len(initial_bands)),
        "n_bands_final": int(len(current_bands)),
        "bands_initial": ",".join(initial_bands),
        "bands_final": ",".join(current_bands),
        "n_bands_clipped": int(len(clipped_bands)),
        "clipped_bands": ",".join(clipped_bands),
        "n_clip_refits": int(len(clipped_bands) > 0),
        "max_abs_residual_initial": float(initial_summary["max_abs_residual_initial"]),
        "max_abs_residual_final": float(np.max(np.abs(final_residual))),
        "passed_residual_clipping": bool(passed),
        "iterative_residual_clipping": bool(iterative_residual_clipping),
        "residual_clip_sigma": float(residual_clip_sigma),
        "min_bands_after_clip": int(min_bands_after_clip),
    }
    initial_summary.update(metadata)
    initial_summary["initial_fit_only"] = True
    final_summary.update(metadata)
    final_summary["initial_fit_only"] = False

    if not passed:
        failure = _clip_failure(row, reject_reason, **metadata)
        return {
            "ok": False,
            "summary": None,
            "initial_summary": initial_summary,
            "failure": failure,
            "history": history,
            "sample_item": None,
            "samples": None,
            "sampler": None,
            "sed_bands": current_bands,
            "sed_obs": current_obs,
            "sed_err": current_err,
        }

    return {
        "ok": True,
        "summary": final_summary,
        "initial_summary": initial_summary,
        "failure": None,
        "history": history,
        "sample_item": None,
        "samples": final_samples,
        "sampler": final_sampler,
        "sed_bands": current_bands,
        "sed_obs": current_obs,
        "sed_err": current_err,
        "model_pred": final_pred,
    }


_WORKER_MODEL = None
_WORKER_ZPS = None
_WORKER_PC_COEFF = None
_WORKER_PC_COV_LONG = None
_WORKER_BOUNDS = None


def _close_direct_worker_model():
    global _WORKER_MODEL
    if _WORKER_MODEL is not None:
        try:
            _WORKER_MODEL.close()
        finally:
            _WORKER_MODEL = None


def _init_direct_worker():
    global _WORKER_MODEL, _WORKER_ZPS, _WORKER_PC_COEFF, _WORKER_PC_COV_LONG, _WORKER_BOUNDS
    _WORKER_ZPS = mapbase.load_zps()
    _WORKER_PC_COEFF, _WORKER_PC_COV_LONG = mapbase.load_pc_tables()
    PreconvolvedBOSZModel = mapbase.sedbase._import_model()
    _WORKER_MODEL = PreconvolvedBOSZModel(cache_file=mapbase.sedbase.CACHE_FILE)
    _WORKER_BOUNDS = mapbase.bounds_for_model(_WORKER_MODEL)
    atexit.register(_close_direct_worker_model)


def _fit_direct_worker(task):
    (
        row_dict,
        i,
        nwalkers,
        nsteps,
        burnin,
        thin,
        random_seed,
        save_samples,
        make_corner,
        progress,
        return_samples_to_notebook,
        iterative_residual_clipping,
        residual_clip_sigma,
        min_bands_after_clip,
    ) = task
    if _WORKER_MODEL is None:
        _init_direct_worker()

    row = pd.Series(row_dict)
    try:
        result = _fit_source_with_optional_clipping(
            row=row,
            model=_WORKER_MODEL,
            zps=_WORKER_ZPS,
            pc_coeff=_WORKER_PC_COEFF,
            pc_cov_long=_WORKER_PC_COV_LONG,
            bounds=_WORKER_BOUNDS,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            random_seed=random_seed + i,
            progress=progress,
            iterative_residual_clipping=iterative_residual_clipping,
            residual_clip_sigma=residual_clip_sigma,
            min_bands_after_clip=min_bands_after_clip,
        )
        if not result["ok"]:
            return result

        summary_row = result["summary"]
        samples = result.pop("samples")
        result.pop("sampler", None)
        sed_bands = result["sed_bands"]
        sed_obs = result["sed_obs"]
        sed_err = result["sed_err"]
        sample_item = None
        if _should_save_sample(save_samples, i):
            tag = f"{row['selection_tag']}_{_safe_name(row.get('Name', 'source'))}"
            sample_file = OUT_DIR / f"{tag}_samples.csv.gz"
            _atomic_to_csv_gz(samples, sample_file)
            if return_samples_to_notebook:
                sample_item = (str(row["selection_tag"]), samples)
            if make_corner:
                try:
                    make_corner_plot(samples, row, OUT_DIR)
                except Exception as exc:
                    warnings.warn(f"corner plot failed for {row.get('Name', '')}: {exc}")
            try:
                make_sed_mag_plot(row, summary_row, _WORKER_MODEL, _WORKER_ZPS, sed_bands, sed_obs, sed_err, OUT_DIR)
            except Exception as exc:
                warnings.warn(f"SED plot failed for {row.get('Name', '')}: {exc}")
        result["sample_item"] = sample_item
        for key in ("sed_bands", "sed_obs", "sed_err", "model_pred"):
            result.pop(key, None)
        return result
    except Exception as exc:
        failure = _clip_failure(row, str(exc))
        return {
            "ok": False,
            "summary": None,
            "initial_summary": None,
            "failure": failure,
            "history": [],
            "sample_item": None,
        }


def run_mw_sed_pcprior_method2_distanceprior_emcee_direct(
    n_stars=30,
    quality_mode="primary",
    nwalkers=32,
    nsteps=2500,
    burnin=800,
    thin=10,
    random_seed=42,
    save_samples="diagnostic",
    make_corner=True,
    progress=False,
    n_workers=1,
    return_samples_to_notebook=True,
    iterative_residual_clipping=False,
    residual_clip_sigma=3.0,
    min_bands_after_clip=6,
):
    """Run direct emcee SED+PC-prior fits for selected MW Cepheids.

    Parameters
    ----------
    n_stars : int or None
        Number of selected sources. ``None`` means all eligible sources.
    quality_mode : {"primary", "secondary", "all"}
        Passed to the existing sample-tier selector.
    save_samples : {"diagnostic", "all", False, True}
        "diagnostic" saves/returns samples only for the first five selected
        stars. "all" or True saves/returns all thinned chains.
    progress : bool
        Passed to emcee. Keeping this False avoids noisy notebook output.
    """
    if int(burnin) >= int(nsteps):
        raise ValueError("burnin must be smaller than nsteps")
    if int(thin) <= 0:
        raise ValueError("thin must be positive")
    if not np.isfinite(float(residual_clip_sigma)) or float(residual_clip_sigma) <= 0:
        raise ValueError("residual_clip_sigma must be finite and positive")
    if int(min_bands_after_clip) < 1:
        raise ValueError("min_bands_after_clip must be positive")

    df = mapbase.read_merged_input()
    df = df.copy()
    df["row_id"] = np.arange(len(df), dtype=int)
    df["quality_tier"] = df.apply(mapbase.assign_quality_tier, axis=1)
    selected = build_selected_table(df, quality_mode=quality_mode, n_stars=n_stars, random_seed=random_seed)
    selected.to_csv(SELECTED_OUT, index=False)

    summaries = []
    initial_summaries = []
    failures = []
    clipping_history = []
    sample_store = {}

    if int(n_workers) > 1:
        tasks = [
            (
                row.to_dict(),
                i,
                nwalkers,
                nsteps,
                burnin,
                thin,
                random_seed,
                save_samples,
                make_corner,
                progress,
                return_samples_to_notebook,
                iterative_residual_clipping,
                residual_clip_sigma,
                min_bands_after_clip,
            )
            for i, (_, row) in enumerate(selected.iterrows())
        ]
        if str(save_samples).lower() == "all" or save_samples is True:
            warnings.warn("Parallel mode with save_samples='all' can return large sample tables to notebook memory.")
        executor = ProcessPoolExecutor(max_workers=int(n_workers), initializer=_init_direct_worker)
        futures = []
        try:
            futures = [executor.submit(_fit_direct_worker, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"MW SED+PC direct emcee x{int(n_workers)}"):
                result = fut.result()
                if result.get("initial_summary") is not None:
                    initial_summaries.append(result["initial_summary"])
                clipping_history.extend(result.get("history", []))
                if result["ok"]:
                    summaries.append(result["summary"])
                    if result.get("sample_item") is not None:
                        key, samples = result["sample_item"]
                        sample_store[key] = samples
                else:
                    failures.append(result["failure"])
        except BaseException:
            _terminate_executor(executor, futures)
            raise
        else:
            executor.shutdown(wait=True)
    else:
        zps = mapbase.load_zps()
        pc_coeff, pc_cov_long = mapbase.load_pc_tables()
        PreconvolvedBOSZModel = mapbase.sedbase._import_model()
        model = PreconvolvedBOSZModel(cache_file=mapbase.sedbase.CACHE_FILE)
        bounds = mapbase.bounds_for_model(model)
        iterator = tqdm(selected.iterrows(), total=len(selected), desc="MW SED+PC direct emcee")
        try:
            for i, (_, row) in enumerate(iterator):
                try:
                    result = _fit_source_with_optional_clipping(
                        row=row,
                        model=model,
                        zps=zps,
                        pc_coeff=pc_coeff,
                        pc_cov_long=pc_cov_long,
                        bounds=bounds,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        burnin=burnin,
                        thin=thin,
                        random_seed=random_seed + i,
                        progress=progress,
                        iterative_residual_clipping=iterative_residual_clipping,
                        residual_clip_sigma=residual_clip_sigma,
                        min_bands_after_clip=min_bands_after_clip,
                    )
                    if result.get("initial_summary") is not None:
                        initial_summaries.append(result["initial_summary"])
                    clipping_history.extend(result.get("history", []))
                    if not result["ok"]:
                        failures.append(result["failure"])
                        continue
                    summary_row = result["summary"]
                    samples = result["samples"]
                    sed_bands = result["sed_bands"]
                    sed_obs = result["sed_obs"]
                    sed_err = result["sed_err"]
                    summaries.append(summary_row)

                    if _should_save_sample(save_samples, i):
                        tag = f"{row['selection_tag']}_{_safe_name(row.get('Name', 'source'))}"
                        sample_file = OUT_DIR / f"{tag}_samples.csv.gz"
                        _atomic_to_csv_gz(samples, sample_file)
                        if return_samples_to_notebook:
                            sample_store[str(row["selection_tag"])] = samples
                        if make_corner:
                            try:
                                make_corner_plot(samples, row, OUT_DIR)
                            except Exception as exc:
                                warnings.warn(f"corner plot failed for {row.get('Name', '')}: {exc}")
                        try:
                            make_sed_mag_plot(row, summary_row, model, zps, sed_bands, sed_obs, sed_err, OUT_DIR)
                        except Exception as exc:
                            warnings.warn(f"SED plot failed for {row.get('Name', '')}: {exc}")
                except Exception as exc:
                    failures.append(_clip_failure(row, str(exc)))
        finally:
            model.close()

    emcee_summary = pd.DataFrame(summaries)
    initial_summary_df = pd.DataFrame(initial_summaries)
    failures_df = pd.DataFrame(failures)
    clipping_history_df = pd.DataFrame(clipping_history)
    if "selection_tag" in emcee_summary.columns:
        emcee_summary = emcee_summary.sort_values("selection_tag").reset_index(drop=True)
    if "selection_tag" in failures_df.columns:
        failures_df = failures_df.sort_values("selection_tag").reset_index(drop=True)
    if "selection_tag" in initial_summary_df.columns:
        initial_summary_df = initial_summary_df.sort_values("selection_tag").reset_index(drop=True)
    if not clipping_history_df.empty:
        clipping_history_df = clipping_history_df.sort_values(
            ["selection_tag", "iteration", "band"]
        ).reset_index(drop=True)
    emcee_summary.to_csv(SUMMARY_OUT, index=False)
    failures_df.to_csv(FAIL_OUT, index=False)
    initial_summary_df.to_csv(OUT_DIR / "initial_fit_summary.csv", index=False)
    clipping_history_df.to_csv(OUT_DIR / "photometry_clipping_history.csv", index=False)

    excluded = clipping_history_df.loc[
        clipping_history_df.get("action", pd.Series(dtype=str)).eq("clip_and_refit")
        & clipping_history_df.get("exceeds_clip_threshold", pd.Series(dtype=bool)).astype(bool)
    ].copy() if not clipping_history_df.empty else pd.DataFrame()
    excluded.to_csv(OUT_DIR / "excluded_photometry_by_source.csv", index=False)
    rejected = failures_df.loc[
        failures_df.get("fail_reason", pd.Series(dtype=str)).astype(str).str.startswith("residual_clipping")
    ].copy() if not failures_df.empty else pd.DataFrame()
    rejected.to_csv(OUT_DIR / "residual_clipping_rejected_sources.csv", index=False)

    clipping_stats = pd.DataFrame(
        [
            {"metric": "n_selected", "value": int(len(selected))},
            {"metric": "n_success", "value": int(len(emcee_summary))},
            {"metric": "n_rejected_by_residual_clipping", "value": int(len(rejected))},
            {"metric": "n_sources_refit", "value": int(excluded["selection_tag"].nunique()) if not excluded.empty else 0},
            {"metric": "n_excluded_measurements", "value": int(len(excluded))},
        ]
    )
    clipping_stats.to_csv(OUT_DIR / "residual_clipping_statistics.csv", index=False)

    runinfo = {
        "n_selected": int(len(selected)),
        "n_success": int(len(emcee_summary)),
        "n_failed": int(len(failures_df)),
        "quality_mode": quality_mode,
        "nwalkers": int(max(nwalkers, 2 * len(PARAM_NAMES) + 2)),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "thin": int(thin),
        "random_seed": int(random_seed),
        "save_samples": str(save_samples),
        "n_workers": int(n_workers),
        "return_samples_to_notebook": bool(return_samples_to_notebook),
        "iterative_residual_clipping": bool(iterative_residual_clipping),
        "residual_clip_sigma": float(residual_clip_sigma),
        "residual_clip_reference": "posterior_median",
        "min_bands_after_clip": int(min_bands_after_clip),
        "n_rejected_by_residual_clipping": int(len(rejected)),
        "n_sources_refit": int(excluded["selection_tag"].nunique()) if not excluded.empty else 0,
        "output_dir": str(OUT_DIR),
        "method": "direct emcee; no MAP/least_squares initialization",
    }
    with open(RUNINFO_OUT, "w", encoding="utf-8") as f:
        json.dump(runinfo, f, ensure_ascii=False, indent=2)

    return selected, emcee_summary, sample_store


print("Loaded direct-emcee MW SED + intrinsic-color PC prior + Wang25 distance-prior fitter.")
print("Call: selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_method2_distanceprior_emcee_direct(...)")
