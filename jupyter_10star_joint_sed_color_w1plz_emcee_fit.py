#!/usr/bin/env python3
"""Joint SED + multi-band color + Wang2025 W1-PLZ MCMC test.

This script combines:
  1. direct broad-band SED likelihood in Vega magnitudes;
  2. previous X-RP color-excess likelihood;
  3. W1 PLZ distance prior from Wang & Chen (2025), ApJ 981, 179.

Jupyter usage:

    %run -i "jupyter_10star_joint_sed_color_w1plz_emcee_fit.py"

    selected, joint_summary, joint_samples, comparison = run_10star_joint_sed_color_w1plz_emcee(
        n_stars=10,
        include_w1_in_sed=True,
        include_color_likelihood=True,
        nwalkers=48,
        nsteps=3000,
        burnin=800,
        thin=5,
        random_seed=42,
    )

High-quality batch usage:

    selected_hq, joint_summary_hq, joint_samples_hq, comparison_hq = run_high_quality_joint_sed_color_w1plz_emcee(
        max_stars=None,
        include_w1_in_sed=True,
        include_color_likelihood=True,
        color_weight=1.0,
        nwalkers=48,
        nsteps=3000,
        burnin=800,
        thin=5,
        resume=True,
        make_plots=False,
        save_samples=True,
        keep_samples_in_memory=False,
    )
"""

from __future__ import annotations

import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR
POST_DIR = OUT_DIR / "joint_sed_color_w1plz_emcee_10star"
POST_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(OUT_DIR))
import jupyter_10star_sed_w1plz_emcee_fit as sedbase


COLOR_BANDS = ["BP", "G", "J", "H", "K", "W1"]
PARAM_NAMES = sedbase.PARAM_NAMES

# Wang & Chen 2025, Table 1, W1 F+1O PLZ relation.
# M_W1 = alpha * (log P_F - 1) + beta + gamma * [Fe/H]
WANG2025_W1_PLZ = {
    "alpha": -3.049,
    "alpha_err": 0.026,
    "beta": -5.833,
    "beta_err": 0.009,
    "gamma": -0.360,
    "gamma_err": 0.042,
    "sigma_int": 0.263,
    "mode": "F+1O",
    "reference": "Wang & Chen 2025 ApJ 981 179 Table 1",
}


def band_d_interpolated(model, teff, logg, mh):
    vertices, weights = model._simplex_weights(float(teff), float(logg), float(mh))
    if not hasattr(model, "_sed_d_vertex_cache"):
        model._sed_d_vertex_cache = {}
    d_interp = np.zeros(len(model.bands), dtype=float)
    for vertex, weight in zip(vertices, weights):
        vertex = int(vertex)
        if vertex not in model._sed_d_vertex_cache:
            model._sed_d_vertex_cache[vertex] = model.h5["D"][vertex, :].astype(np.float32)
        d_interp += float(weight) * model._sed_d_vertex_cache[vertex].astype(float)
    if np.any(~np.isfinite(d_interp)) or np.any(d_interp <= 0):
        raise ValueError("non-positive interpolated intrinsic band integral")
    return d_interp


def predict_joint(theta, model, zps, sed_bands, color_bands):
    teff, logg, mh, av, rv, logd, logr = theta
    n = sedbase.band_n_interpolated(model, teff, logg, mh, rv, av)
    d = band_d_interpolated(model, teff, logg, mh)

    radius_cm = (10.0**logr) * sedbase.R_SUN_CM
    distance_cm = (10.0**logd) * sedbase.KPC_CM
    scale = (radius_cm / distance_cm) ** 2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bad R/D scale")

    mags = []
    for band in sed_bands:
        i = model.band_index[band]
        mags.append(float(-2.5 * np.log10(n[i]) + zps[band] - 2.5 * np.log10(scale)))

    a_band = -2.5 * np.log10(n / d)
    a_rp = a_band[model.band_index["RP"]]
    colors = []
    for band in color_bands:
        colors.append(float(a_band[model.band_index[band]] - a_rp))

    return np.asarray(mags, dtype=float), np.asarray(colors, dtype=float), {
        band: float(a_band[model.band_index[band]]) for band in model.bands
    }


def observed_color_excess_vector(row, include_color_likelihood=True):
    if not include_color_likelihood:
        return [], np.asarray([], dtype=float), np.asarray([], dtype=float)
    bands, obs, err = [], [], []
    for band in COLOR_BANDS:
        val = sedbase._finite(row.get(f"E_{band}_RP", np.nan))
        e = sedbase._finite(row.get(f"E_{band}_RP_error", np.nan))
        if not np.isfinite(e):
            e = sedbase._finite(row.get(f"E_{band}_RP_error_diag", np.nan))
        if np.isfinite(val) and np.isfinite(e) and e > 0:
            bands.append(band)
            obs.append(val)
            err.append(e)
    return bands, np.asarray(obs, dtype=float), np.asarray(err, dtype=float)


def logp_fundamentalized(row):
    lp = sedbase._finite(row.get("logP_F_plot_Pilecki2024_MW", np.nan))
    if np.isfinite(lp):
        return lp
    period = sedbase._finite(row.get("period_used", np.nan))
    if np.isfinite(period) and period > 0:
        return float(np.log10(period))
    return np.nan


def w1_plz_absolute_mag(row):
    logp = logp_fundamentalized(row)
    feh = sedbase._finite(row.get("[Fe/H]", np.nan))
    if not np.isfinite(logp) or not np.isfinite(feh):
        return np.nan
    c = WANG2025_W1_PLZ
    return c["alpha"] * (logp - 1.0) + c["beta"] + c["gamma"] * feh


def w1_plz_sigma(row):
    logp = logp_fundamentalized(row)
    feh = sedbase._finite(row.get("[Fe/H]", np.nan))
    w1err = sedbase._finite(row.get("W1_magerr", np.nan))
    c = WANG2025_W1_PLZ
    terms = [c["sigma_int"] ** 2, c["beta_err"] ** 2]
    if np.isfinite(logp):
        terms.append(((logp - 1.0) * c["alpha_err"]) ** 2)
    if np.isfinite(feh):
        terms.append((feh * c["gamma_err"]) ** 2)
        terms.append((c["gamma"] * sedbase.FEH_PRIOR_SIGMA) ** 2)
    if np.isfinite(w1err) and w1err > 0:
        terms.append(w1err**2)
    return float(np.sqrt(np.sum(terms)))


def prior_residuals_joint(theta, row, a_band):
    teff, logg, mh, _av, _rv, logd, _logr = theta
    res = []

    mu_logg = sedbase.logg_prior_mu(teff, row)
    if np.isfinite(mu_logg):
        res.append((logg - mu_logg) / sedbase.LOGG_PRIOR_SIGMA)

    mu_mh = sedbase._finite(row.get("[Fe/H]", np.nan))
    if np.isfinite(mu_mh):
        res.append((mh - mu_mh) / sedbase.FEH_PRIOR_SIGMA)

    w1 = sedbase._finite(row.get("W1_mag", np.nan))
    mw1 = w1_plz_absolute_mag(row)
    sig = w1_plz_sigma(row)
    if np.isfinite(w1) and np.isfinite(mw1) and np.isfinite(sig) and sig > 0:
        mu_dist = 5.0 * logd + 10.0
        mu_plz = w1 - a_band["W1"] - mw1
        res.append((mu_dist - mu_plz) / sig)

    return np.asarray(res, dtype=float)


def residual_joint(theta, row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, lower, upper, color_weight=1.0):
    theta = np.asarray(theta, dtype=float)
    n_bad = len(sed_obs) + len(color_obs) + 3
    if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
        return np.full(n_bad, 1e6)
    try:
        pred_mag, pred_color, a_band = predict_joint(theta, model, zps, sed_bands, color_bands)
    except Exception:
        return np.full(n_bad, 1e6)

    sed_res = (sed_obs - pred_mag) / sed_err
    color_res = (color_obs - pred_color) / color_err
    if color_weight != 1.0:
        color_res = np.sqrt(float(color_weight)) * color_res
    prior_res = prior_residuals_joint(theta, row, a_band)
    return np.concatenate([sed_res, color_res, prior_res])


def log_prob_joint(theta, row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, lower, upper, color_weight):
    r = residual_joint(theta, row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, lower, upper, color_weight)
    if np.any(~np.isfinite(r)) or np.any(np.abs(r) > 1e5):
        return -np.inf
    return -0.5 * float(np.sum(r * r))


def fit_map_joint(row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, bounds, random_seed=42, color_weight=1.0, n_start=24):
    rng = np.random.default_rng(random_seed)
    lower, upper = bounds["lower"], bounds["upper"]
    starts = [np.clip(sedbase.initial_guess(row, model), lower + 1e-6, upper - 1e-6)]
    for teff in [5000.0, 5800.0, 6500.0]:
        x = starts[0].copy()
        x[0] = teff
        x[1] = np.clip(sedbase.logg_prior_mu(teff, row), lower[1], upper[1])
        starts.append(np.clip(x, lower + 1e-6, upper - 1e-6))
    for _ in range(max(0, n_start - len(starts))):
        starts.append(lower + rng.random(len(lower)) * (upper - lower))

    best = None
    rows = []
    for x0 in starts:
        try:
            res = least_squares(
                residual_joint,
                x0=x0,
                bounds=(lower, upper),
                args=(row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, lower, upper, color_weight),
                max_nfev=1400,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            chi2 = float(np.sum(res.fun**2))
            rows.append(dict(zip(PARAM_NAMES, res.x)) | {"chi2": chi2, "success": bool(res.success)})
            if best is None or chi2 < best[0]:
                best = (chi2, res)
        except Exception as exc:
            rows.append({"chi2": np.inf, "success": False, "message": str(exc)})
    if best is None:
        raise RuntimeError("all least_squares starts failed")
    return best[1], pd.DataFrame(rows)


def run_emcee_joint(row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, bounds, x_map, nwalkers, nsteps, burnin, thin, random_seed, color_weight=1.0):
    import emcee

    lower, upper = bounds["lower"], bounds["upper"]
    rng = np.random.default_rng(random_seed)
    ndim = len(PARAM_NAMES)
    nwalkers = int(max(nwalkers, 2 * ndim + 2))
    scale = np.array([70.0, 0.035, 0.035, 0.06, 0.05, 0.006, 0.020])
    p0 = []
    while len(p0) < nwalkers:
        p = np.asarray(x_map, dtype=float) + rng.normal(0.0, scale)
        if np.all(np.isfinite(p)) and np.all(p > lower) and np.all(p < upper):
            p0.append(p)
    p0 = np.asarray(p0)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_joint,
        args=(row, model, zps, sed_bands, sed_obs, sed_err, color_bands, color_obs, color_err, lower, upper, color_weight),
    )
    sampler.run_mcmc(p0, int(nsteps), progress=False)
    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    logp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
    samples = pd.DataFrame(flat, columns=PARAM_NAMES)
    samples["D_kpc"] = 10.0 ** samples["logD"]
    samples["R_Rsun"] = 10.0 ** samples["logR"]
    samples["log_prob"] = logp
    return sampler, samples


def summarize_joint(row, samples, sed_bands, color_bands, sampler, map_res):
    out = sedbase.summarize_samples(row, samples, sed_bands, sampler, map_res)
    out["colors_fit"] = ",".join([f"E_{b}_RP" for b in color_bands])
    out["ncolor_fit"] = len(color_bands)
    out["fit_method"] = "SED_mag + X_RP_color_excess + Wang2025_W1_PLZ_distance_prior"
    out["W1_PLZ_alpha"] = WANG2025_W1_PLZ["alpha"]
    out["W1_PLZ_beta"] = WANG2025_W1_PLZ["beta"]
    out["W1_PLZ_gamma"] = WANG2025_W1_PLZ["gamma"]
    out["W1_PLZ_sigma_int"] = WANG2025_W1_PLZ["sigma_int"]
    out["W1_PLZ_M_W1_prior"] = w1_plz_absolute_mag(row)
    out["W1_PLZ_sigma_mu"] = w1_plz_sigma(row)
    return out


def _high_quality_mask(
    df,
    include_w1_in_sed=True,
    include_color_likelihood=True,
    require_w1_plz_distance=True,
    require_all_sed_bands=True,
    require_all_color_bands=True,
    av_min=1.0,
    bp_max=20.3,
    max_distance_frac_err=None,
    allowed_modes=None,
):
    sed_bands = sedbase.BANDS_ALL if include_w1_in_sed else [b for b in sedbase.BANDS_ALL if b != "W1"]
    color_bands = list(COLOR_BANDS)

    mask = np.ones(len(df), dtype=bool)

    n_sed = np.zeros(len(df), dtype=int)
    for band in sed_bands:
        good = (
            np.isfinite(df[sedbase.MAG_COL[band]].to_numpy(float))
            & np.isfinite(df[sedbase.MAGERR_COL[band]].to_numpy(float))
            & (df[sedbase.MAGERR_COL[band]].to_numpy(float) > 0)
        )
        n_sed += good.astype(int)
        if require_all_sed_bands:
            mask &= good

    n_color = np.zeros(len(df), dtype=int)
    if include_color_likelihood:
        for band in color_bands:
            err = df.get(f"E_{band}_RP_error", pd.Series(np.nan, index=df.index))
            err_diag = df.get(f"E_{band}_RP_error_diag", pd.Series(np.nan, index=df.index))
            err_use = pd.to_numeric(err, errors="coerce").fillna(pd.to_numeric(err_diag, errors="coerce"))
            good = (
                np.isfinite(pd.to_numeric(df.get(f"E_{band}_RP", np.nan), errors="coerce").to_numpy(float))
                & np.isfinite(err_use.to_numpy(float))
                & (err_use.to_numpy(float) > 0)
            )
            n_color += good.astype(int)
            if require_all_color_bands:
                mask &= good

    if require_w1_plz_distance:
        mask &= np.isfinite(df["D_w1"].to_numpy(float))
        mask &= np.isfinite(df["D_w1_error"].to_numpy(float))
        mask &= df["D_w1"].to_numpy(float) > 0
        mask &= df["D_w1_error"].to_numpy(float) > 0

    mask &= np.isfinite(df["[Fe/H]"].to_numpy(float))
    mask &= np.isfinite(df["period_used"].to_numpy(float))
    mask &= df["period_used"].to_numpy(float) > 0

    if av_min is not None and "Av_p50" in df.columns:
        mask &= df["Av_p50"].fillna(-np.inf).to_numpy(float) >= float(av_min)
    if bp_max is not None and "BP_mag" in df.columns:
        mask &= df["BP_mag"].fillna(np.inf).to_numpy(float) < float(bp_max)
    if "riello_cstar_bad_3sigma" in df.columns:
        mask &= ~df["riello_cstar_bad_3sigma"].to_numpy(bool)

    if max_distance_frac_err is not None:
        frac = df["D_w1_error"].to_numpy(float) / df["D_w1"].to_numpy(float)
        mask &= np.isfinite(frac) & (frac <= float(max_distance_frac_err))

    if allowed_modes is not None and "Mode" in df.columns:
        allowed = {str(x).strip().upper() for x in allowed_modes}
        mode = df["Mode"].astype(str).str.strip().str.upper()
        mask &= mode.isin(allowed).to_numpy(bool)

    return mask, n_sed, n_color


def select_high_quality_sources(
    df,
    include_w1_in_sed=True,
    include_color_likelihood=True,
    require_w1_plz_distance=True,
    require_all_sed_bands=True,
    require_all_color_bands=True,
    av_min=1.0,
    bp_max=20.3,
    max_distance_frac_err=None,
    allowed_modes=None,
    max_stars=None,
):
    """Select high-quality MW Cepheids for batch joint SED+color MCMC.

    Default cuts are intentionally close to the quality cuts used in the
    previous color-fit analysis: complete BP/G/RP/J/H/K/W1 photometry,
    complete X-RP colors, Av >= 1, BP < 20.3, and Riello C* good.
    """

    mask, n_sed, n_color = _high_quality_mask(
        df,
        include_w1_in_sed=include_w1_in_sed,
        include_color_likelihood=include_color_likelihood,
        require_w1_plz_distance=require_w1_plz_distance,
        require_all_sed_bands=require_all_sed_bands,
        require_all_color_bands=require_all_color_bands,
        av_min=av_min,
        bp_max=bp_max,
        max_distance_frac_err=max_distance_frac_err,
        allowed_modes=allowed_modes,
    )

    cand = df.loc[mask].copy()
    cand["n_sed_bands_available"] = n_sed[mask]
    cand["n_color_bands_available"] = n_color[mask]
    cand["logP_select"] = np.where(
        np.isfinite(cand.get("logP_F_plot_Pilecki2024_MW", np.nan)),
        cand["logP_F_plot_Pilecki2024_MW"],
        np.log10(cand["period_used"]),
    )
    cand["distance_frac_err"] = cand["D_w1_error"] / cand["D_w1"]
    cand = cand.sort_values(["logP_select", "distance_frac_err", "BP_mag", "Name"]).reset_index(drop=True)
    if max_stars is not None:
        cand = cand.head(int(max_stars)).copy()
    if "selection_tag" in cand.columns:
        cand = cand.drop(columns=["selection_tag"])
    cand.insert(0, "selection_tag", [f"HQ{i+1:04d}" for i in range(len(cand))])
    cand["W1_double_used"] = bool(include_w1_in_sed)
    cand["high_quality_av_min"] = np.nan if av_min is None else float(av_min)
    cand["high_quality_bp_max"] = np.nan if bp_max is None else float(bp_max)
    cand["high_quality_max_distance_frac_err"] = np.nan if max_distance_frac_err is None else float(max_distance_frac_err)
    return cand


def _failed_summary_row(row, reason, include_color_likelihood, color_weight):
    return {
        "selection_tag": row.get("selection_tag", ""),
        "Name": str(row.get("Name", "")),
        "GaiaDR3": str(row.get("GaiaDR3", "")),
        "gaia_key": row.get("gaia_key", ""),
        "fit_failed": True,
        "failure_reason": str(reason),
        "include_color_likelihood": bool(include_color_likelihood),
        "color_weight": float(color_weight),
    }


def _write_incremental_summary(rows, failed_rows, outdir, prefix):
    summary = pd.DataFrame(rows + failed_rows)
    summary.to_csv(outdir / f"{prefix}_emcee_summary_incremental.csv", index=False)
    return summary


_WORKER_MODEL = None
_WORKER_FILTERS = None
_WORKER_LAMBDA_EFF = None
_WORKER_ZPS = None
_WORKER_BOUNDS = None


def _init_high_quality_joint_worker():
    global _WORKER_MODEL, _WORKER_FILTERS, _WORKER_LAMBDA_EFF, _WORKER_ZPS, _WORKER_BOUNDS
    PreconvolvedBOSZModel = sedbase._import_model()
    _WORKER_MODEL = PreconvolvedBOSZModel(cache_file=sedbase.CACHE_FILE)
    _WORKER_FILTERS, _WORKER_LAMBDA_EFF = sedbase.load_pyphot_filters()
    _WORKER_ZPS, _ = sedbase.calibrate_band_zero_points(_WORKER_MODEL, _WORKER_FILTERS)
    _WORKER_BOUNDS = sedbase.bounds_for_model(_WORKER_MODEL)


def _close_high_quality_joint_worker():
    global _WORKER_MODEL
    if _WORKER_MODEL is not None:
        _WORKER_MODEL.close()
        _WORKER_MODEL = None


def _fit_high_quality_joint_worker(task):
    (
        row_dict,
        i,
        include_w1_in_sed,
        include_color_likelihood,
        color_weight,
        nwalkers,
        nsteps,
        burnin,
        thin,
        random_seed,
        make_plots,
        save_samples,
        keep_samples_in_memory,
        outdir_str,
    ) = task

    if _WORKER_MODEL is None:
        _init_high_quality_joint_worker()

    outdir = Path(outdir_str)
    row = pd.Series(row_dict)
    tag = f"{row['selection_tag']}_{sedbase.safe_name(row['Name'])}"

    try:
        sed_bands, sed_obs, sed_err = sedbase.observed_mag_vector(row, include_w1_in_sed=include_w1_in_sed)
        color_bands, color_obs, color_err = observed_color_excess_vector(row, include_color_likelihood=include_color_likelihood)
        map_res, starts = fit_map_joint(
            row,
            _WORKER_MODEL,
            _WORKER_ZPS,
            sed_bands,
            sed_obs,
            sed_err,
            color_bands,
            color_obs,
            color_err,
            _WORKER_BOUNDS,
            random_seed=random_seed + i,
            color_weight=color_weight,
        )
        sampler, samples = run_emcee_joint(
            row,
            _WORKER_MODEL,
            _WORKER_ZPS,
            sed_bands,
            sed_obs,
            sed_err,
            color_bands,
            color_obs,
            color_err,
            _WORKER_BOUNDS,
            map_res.x,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            random_seed=random_seed + 1000 + i,
            color_weight=color_weight,
        )
        samples.insert(0, "selection_tag", row["selection_tag"])
        samples.insert(1, "Name", str(row["Name"]))
        if save_samples:
            samples.to_csv(outdir / f"{tag}_samples.csv.gz", index=False)
        starts.to_csv(outdir / f"{tag}_least_squares_starts.csv", index=False)

        summary = summarize_joint(row, samples, sed_bands, color_bands, sampler, map_res)
        summary["corner_file"] = ""
        summary["sed_plot_file"] = ""
        if make_plots:
            summary["corner_file"] = str(sedbase.make_corner(samples, row, outdir))
            summary["sed_plot_file"] = str(
                sedbase.make_sed_plot(
                    row,
                    samples,
                    _WORKER_MODEL,
                    _WORKER_ZPS,
                    _WORKER_FILTERS,
                    _WORKER_LAMBDA_EFF,
                    sed_bands,
                    sed_obs,
                    sed_err,
                    outdir,
                )
            )
        summary["gaia_key"] = row["gaia_key"]
        summary["include_color_likelihood"] = bool(include_color_likelihood)
        summary["color_weight"] = float(color_weight)
        summary["make_plots"] = bool(make_plots)
        summary["save_samples"] = bool(save_samples)
        sample_item = (str(row["Name"]), samples) if keep_samples_in_memory else None
        return {"ok": True, "summary": summary, "sample_item": sample_item}
    except Exception as exc:
        return {
            "ok": False,
            "summary": _failed_summary_row(row, exc, include_color_likelihood, color_weight),
            "sample_item": None,
        }


def run_high_quality_joint_sed_color_w1plz_emcee(
    include_w1_in_sed=True,
    include_color_likelihood=True,
    color_weight=1.0,
    require_w1_plz_distance=True,
    require_all_sed_bands=True,
    require_all_color_bands=True,
    av_min=1.0,
    bp_max=20.3,
    max_distance_frac_err=None,
    allowed_modes=None,
    max_stars=None,
    start_index=0,
    nwalkers=48,
    nsteps=3000,
    burnin=800,
    thin=5,
    random_seed=42,
    resume=True,
    rerun_failed=False,
    make_plots=False,
    save_samples=True,
    keep_samples_in_memory=False,
    n_workers=1,
    outdir=OUT_DIR / "joint_sed_color_w1plz_emcee_high_quality",
):
    """Run the joint SED+X_RP-color+W1-PLZ MCMC for all high-quality sources.

    This is the scalable version of ``run_10star_joint_sed_color_w1plz_emcee``.
    It writes results after each source, supports resume, and disables
    per-source corner/SED plots by default to avoid thousands of image files.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "MW_high_quality_joint_SED_color_W1PLZ"

    full = sedbase.read_input_tables()
    selected = select_high_quality_sources(
        full,
        include_w1_in_sed=include_w1_in_sed,
        include_color_likelihood=include_color_likelihood,
        require_w1_plz_distance=require_w1_plz_distance,
        require_all_sed_bands=require_all_sed_bands,
        require_all_color_bands=require_all_color_bands,
        av_min=av_min,
        bp_max=bp_max,
        max_distance_frac_err=max_distance_frac_err,
        allowed_modes=allowed_modes,
        max_stars=max_stars,
    )
    if start_index:
        selected = selected.iloc[int(start_index) :].copy().reset_index(drop=True)
    selected.to_csv(outdir / f"{prefix}_selected.csv", index=False)

    summary_file = outdir / f"{prefix}_emcee_summary.csv"
    incremental_file = outdir / f"{prefix}_emcee_summary_incremental.csv"
    existing = pd.DataFrame()
    if resume:
        for candidate in [incremental_file, summary_file]:
            if candidate.exists():
                existing = pd.read_csv(candidate, dtype={"GaiaDR3": str, "gaia_key": str})
                break

    done_tags = set()
    summaries, failed_rows = [], []
    if len(existing) > 0:
        fit_failed = existing.get("fit_failed", False).astype(str).str.lower().isin(["true", "1", "yes"])
        if rerun_failed:
            keep_existing = ~fit_failed
        else:
            keep_existing = np.ones(len(existing), dtype=bool)
        kept = existing.loc[keep_existing].copy()
        summaries = kept.to_dict("records")
        done_tags = set(kept["selection_tag"].astype(str))

    PreconvolvedBOSZModel = sedbase._import_model()
    model = PreconvolvedBOSZModel(cache_file=sedbase.CACHE_FILE)
    filters, lambda_eff = sedbase.load_pyphot_filters()
    zps, zp_info = sedbase.calibrate_band_zero_points(model, filters)
    zp_info.to_csv(outdir / "SED_cache_to_pyphot_vega_zero_points.csv", index=False)
    pd.DataFrame([WANG2025_W1_PLZ]).to_csv(outdir / "Wang2025_W1_PLZ_relation_used.csv", index=False)
    bounds = sedbase.bounds_for_model(model)
    if int(n_workers) > 1:
        model.close()

    sample_dict = {}
    if int(n_workers) > 1:
        tasks = []
        for i, (_, row) in enumerate(selected.iterrows()):
            if resume and str(row["selection_tag"]) in done_tags:
                continue
            tasks.append(
                (
                    row.to_dict(),
                    i,
                    include_w1_in_sed,
                    include_color_likelihood,
                    color_weight,
                    nwalkers,
                    nsteps,
                    burnin,
                    thin,
                    random_seed,
                    make_plots,
                    save_samples,
                    keep_samples_in_memory,
                    str(outdir),
                )
            )

        if keep_samples_in_memory:
            warnings.warn("Parallel mode can return large sample tables; keep_samples_in_memory=False is recommended.")

        with ProcessPoolExecutor(max_workers=int(n_workers), initializer=_init_high_quality_joint_worker) as exe:
            futures = [exe.submit(_fit_high_quality_joint_worker, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="high-quality joint SED+color emcee"):
                result = fut.result()
                if result["ok"]:
                    summaries.append(result["summary"])
                else:
                    failed_rows.append(result["summary"])
                if result.get("sample_item") is not None:
                    name, samples = result["sample_item"]
                    sample_dict[name] = samples
                _write_incremental_summary(summaries, failed_rows, outdir, prefix)
    else:
        iterator = tqdm(selected.iterrows(), total=len(selected), desc="high-quality joint SED+color emcee")
        for i, (_, row) in enumerate(iterator):
            tag = f"{row['selection_tag']}_{sedbase.safe_name(row['Name'])}"
            if resume and str(row["selection_tag"]) in done_tags:
                continue

            sed_bands, sed_obs, sed_err = sedbase.observed_mag_vector(row, include_w1_in_sed=include_w1_in_sed)
            color_bands, color_obs, color_err = observed_color_excess_vector(row, include_color_likelihood=include_color_likelihood)
            try:
                map_res, starts = fit_map_joint(
                    row,
                    model,
                    zps,
                    sed_bands,
                    sed_obs,
                    sed_err,
                    color_bands,
                    color_obs,
                    color_err,
                    bounds,
                    random_seed=random_seed + i,
                    color_weight=color_weight,
                )
                sampler, samples = run_emcee_joint(
                    row,
                    model,
                    zps,
                    sed_bands,
                    sed_obs,
                    sed_err,
                    color_bands,
                    color_obs,
                    color_err,
                    bounds,
                    map_res.x,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    burnin=burnin,
                    thin=thin,
                    random_seed=random_seed + 1000 + i,
                    color_weight=color_weight,
                )
                samples.insert(0, "selection_tag", row["selection_tag"])
                samples.insert(1, "Name", str(row["Name"]))
                if save_samples:
                    samples.to_csv(outdir / f"{tag}_samples.csv.gz", index=False)
                starts.to_csv(outdir / f"{tag}_least_squares_starts.csv", index=False)

                summary = summarize_joint(row, samples, sed_bands, color_bands, sampler, map_res)
                summary["corner_file"] = ""
                summary["sed_plot_file"] = ""
                if make_plots:
                    summary["corner_file"] = str(sedbase.make_corner(samples, row, outdir))
                    summary["sed_plot_file"] = str(
                        sedbase.make_sed_plot(row, samples, model, zps, filters, lambda_eff, sed_bands, sed_obs, sed_err, outdir)
                    )
                summary["gaia_key"] = row["gaia_key"]
                summary["include_color_likelihood"] = bool(include_color_likelihood)
                summary["color_weight"] = float(color_weight)
                summary["make_plots"] = bool(make_plots)
                summary["save_samples"] = bool(save_samples)
                summaries.append(summary)
                if keep_samples_in_memory:
                    sample_dict[str(row["Name"])] = samples
            except Exception as exc:
                warnings.warn(f"joint high-quality fit failed for {row.get('Name')}: {exc}")
                failed_rows.append(_failed_summary_row(row, exc, include_color_likelihood, color_weight))

            _write_incremental_summary(summaries, failed_rows, outdir, prefix)
        model.close()

    joint_summary = pd.DataFrame(summaries + failed_rows)
    joint_summary.to_csv(summary_file, index=False)
    comparison = sedbase.build_comparison(joint_summary, full)
    comparison.to_csv(outdir / f"{prefix}_vs_colorfit_comparison.csv", index=False)
    if len(joint_summary) > 0:
        good = joint_summary[joint_summary.get("fit_failed", False) != True].copy()
        sedbase.make_summary_plots(good, comparison, outdir)
    return selected, joint_summary, sample_dict, comparison


def run_10star_joint_sed_color_w1plz_emcee(
    n_stars=10,
    include_w1_in_sed=True,
    include_color_likelihood=True,
    color_weight=1.0,
    nwalkers=48,
    nsteps=3000,
    burnin=800,
    thin=5,
    random_seed=42,
    outdir=POST_DIR,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full = sedbase.read_input_tables()
    selected = sedbase.select_10_sources(full, n_stars=n_stars, include_w1_in_sed=include_w1_in_sed)
    selected.to_csv(outdir / "MW_10star_joint_SED_color_W1PLZ_selected.csv", index=False)

    PreconvolvedBOSZModel = sedbase._import_model()
    model = PreconvolvedBOSZModel(cache_file=sedbase.CACHE_FILE)
    filters, lambda_eff = sedbase.load_pyphot_filters()
    zps, zp_info = sedbase.calibrate_band_zero_points(model, filters)
    zp_info.to_csv(outdir / "SED_cache_to_pyphot_vega_zero_points.csv", index=False)
    pd.DataFrame([WANG2025_W1_PLZ]).to_csv(outdir / "Wang2025_W1_PLZ_relation_used.csv", index=False)
    bounds = sedbase.bounds_for_model(model)

    summaries, failed_rows, sample_dict = [], [], {}
    try:
        for i, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected), desc="joint SED+color emcee stars")):
            sed_bands, sed_obs, sed_err = sedbase.observed_mag_vector(row, include_w1_in_sed=include_w1_in_sed)
            color_bands, color_obs, color_err = observed_color_excess_vector(row, include_color_likelihood=include_color_likelihood)
            try:
                map_res, starts = fit_map_joint(
                    row,
                    model,
                    zps,
                    sed_bands,
                    sed_obs,
                    sed_err,
                    color_bands,
                    color_obs,
                    color_err,
                    bounds,
                    random_seed=random_seed + i,
                    color_weight=color_weight,
                )
                sampler, samples = run_emcee_joint(
                    row,
                    model,
                    zps,
                    sed_bands,
                    sed_obs,
                    sed_err,
                    color_bands,
                    color_obs,
                    color_err,
                    bounds,
                    map_res.x,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    burnin=burnin,
                    thin=thin,
                    random_seed=random_seed + 1000 + i,
                    color_weight=color_weight,
                )
                samples.insert(0, "selection_tag", row["selection_tag"])
                samples.insert(1, "Name", str(row["Name"]))
                tag = f"{row['selection_tag']}_{sedbase.safe_name(row['Name'])}"
                samples.to_csv(outdir / f"{tag}_samples.csv.gz", index=False)
                starts.to_csv(outdir / f"{tag}_least_squares_starts.csv", index=False)

                summary = summarize_joint(row, samples, sed_bands, color_bands, sampler, map_res)
                summary["corner_file"] = str(sedbase.make_corner(samples, row, outdir))
                summary["sed_plot_file"] = str(sedbase.make_sed_plot(row, samples, model, zps, filters, lambda_eff, sed_bands, sed_obs, sed_err, outdir))
                summary["gaia_key"] = row["gaia_key"]
                summary["include_color_likelihood"] = bool(include_color_likelihood)
                summary["color_weight"] = float(color_weight)
                summaries.append(summary)
                sample_dict[str(row["Name"])] = samples
            except Exception as exc:
                warnings.warn(f"joint fit failed for {row.get('Name')}: {exc}")
                failed_rows.append(
                    {
                        "selection_tag": row["selection_tag"],
                        "Name": str(row["Name"]),
                        "GaiaDR3": str(row["GaiaDR3"]),
                        "gaia_key": row["gaia_key"],
                        "fit_failed": True,
                        "failure_reason": str(exc),
                    }
                )
    finally:
        model.close()

    joint_summary = pd.DataFrame(summaries + failed_rows)
    joint_summary.to_csv(outdir / "MW_10star_joint_SED_color_W1PLZ_emcee_summary.csv", index=False)
    comparison = sedbase.build_comparison(joint_summary, full)
    comparison.to_csv(outdir / "MW_10star_joint_SED_color_vs_colorfit_comparison.csv", index=False)
    good = joint_summary[joint_summary.get("fit_failed", False) != True].copy()
    sedbase.make_summary_plots(good, comparison, outdir)
    return selected, joint_summary, sample_dict, comparison


print("Loaded joint SED + X_RP color + Wang2025 W1 PLZ emcee helpers.")
print("Call: selected, joint_summary, joint_samples, comparison = run_10star_joint_sed_color_w1plz_emcee(...)")
print("Batch call: selected, joint_summary, joint_samples, comparison = run_high_quality_joint_sed_color_w1plz_emcee(...)")
