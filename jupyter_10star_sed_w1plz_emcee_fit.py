#!/usr/bin/env python3
"""10-star MW Cepheid SED fit with W1-PLZ distance priors.

Jupyter usage:

    %run -i "jupyter_10star_sed_w1plz_emcee_fit.py"

    selected, sed_summary, sed_samples, comparison = run_10star_sed_w1plz_emcee(
        n_stars=10,
        include_w1_in_sed=True,
        nwalkers=48,
        nsteps=3000,
        burnin=800,
        thin=5,
        random_seed=42,
    )
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import pyphot
except ModuleNotFoundError:
    pyphot = None
from scipy.optimize import least_squares
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR
SCRIPT_DIR = OUT_DIR
POST_DIR = OUT_DIR / "sed_w1plz_emcee_10star"
POST_DIR.mkdir(parents=True, exist_ok=True)

MW_PHOT_FILE = OUT_DIR / "MW_3640_PC_color_excess_from_LMC_PC.csv"
WANG25_FILE = OUT_DIR / "wang25x.csv"
COLORFIT_FILE = OUT_DIR / "MW_Rv_environment_master_table.csv"
CACHE_FILE = OUT_DIR / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_preconvolved_ND.h5"
WAVE_FILE = OUT_DIR / "r500/m+0.00/bosz2024_wave_r500.txt"

BANDS_ALL = ["BP", "G", "RP", "J", "H", "K", "W1"]
PARAM_NAMES = ["Teff", "logg", "MH", "Av", "Rv", "logD", "logR"]

MAG_COL = {
    "BP": "BP_mag",
    "G": "G_mag",
    "RP": "RP_mag",
    "J": "J_mean_mag",
    "H": "H_mean_mag",
    "K": "Ks_mean_mag",
    "W1": "W1_mag",
}

MAGERR_COL = {
    "BP": "BP_magerr",
    "G": "G_magerr",
    "RP": "RP_magerr",
    "J": "J_mean_magerr",
    "H": "H_mean_magerr",
    "K": "Ks_mean_magerr",
    "W1": "W1_magerr",
}

FILTERS = {
    "BP": (OUT_DIR / "filters/GAIA_GAIA3.Gbp.dat", "photon"),
    "G": (OUT_DIR / "filters/GAIA_GAIA3.G.dat", "photon"),
    "RP": (OUT_DIR / "filters/GAIA_GAIA3.Grp.dat", "photon"),
    "J": (OUT_DIR / "filters/2MASS_2MASS.J.dat", "photon"),
    "H": (OUT_DIR / "filters/2MASS_2MASS.H.dat", "photon"),
    "K": (OUT_DIR / "filters/2MASS_2MASS.Ks.dat", "photon"),
    "W1": (OUT_DIR / "filters/WISE_WISE.W1.dat", "energy"),
}

MAG_FLOOR = {"BP": 0.005, "G": 0.005, "RP": 0.005, "J": 0.020, "H": 0.020, "K": 0.020, "W1": 0.020}

LOGG_PRIOR_SIGMA = 0.108
FEH_PRIOR_SIGMA = 0.16
R_SUN_CM = 6.957e10
KPC_CM = 3.0856775814913673e21


def _import_model():
    sys.path.insert(0, str(SCRIPT_DIR))
    from bosz_preconvolved_interpolated_fit import PreconvolvedBOSZModel

    return PreconvolvedBOSZModel


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _finite(x):
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def _id_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_pyphot_filters():
    if pyphot is None:
        raise ModuleNotFoundError("pyphot is required only for recomputing Vega zero points; current MCMC runs can use the cached zero-point CSV.")
    filters = {}
    lambda_eff = {}
    for band, (path, dtype) in FILTERS.items():
        dat = np.loadtxt(path)
        filters[band] = pyphot.Filter(dat[:, 0], dat[:, 1], name=f"{band}_local", dtype=dtype, unit="AA")
        good = np.isfinite(dat[:, 0]) & np.isfinite(dat[:, 1]) & (dat[:, 1] > 0)
        lambda_eff[band] = float(np.average(dat[good, 0], weights=dat[good, 1]))
    return filters, lambda_eff


def pyphot_mag(filter_obj, wave_aa, flux):
    f_band = filter_obj.get_flux(
        wave_aa * u.AA,
        flux * u.erg / u.s / u.cm**2 / u.AA,
    )
    return float(-2.5 * np.log10(f_band.value) - filter_obj.Vega_zero_mag)


def band_n_interpolated(model, teff, logg, mh, rv, av):
    vertices, weights = model._simplex_weights(float(teff), float(logg), float(mh))
    ir0, ir1, tr = model._bracket(model.rv_grid, float(rv))
    ia0, ia1, ta = model._bracket(model.av_grid, float(av))
    n_interp = np.zeros(len(model.bands), dtype=float)
    if not hasattr(model, "_sed_n_vertex_cache"):
        model._sed_n_vertex_cache = {}

    for vertex, weight in zip(vertices, weights):
        vertex = int(vertex)
        if vertex not in model._sed_n_vertex_cache:
            # One vertex is about a few MB as float32. Caching the visited
            # simplex vertices makes emcee much faster than repeated HDF5 reads.
            model._sed_n_vertex_cache[vertex] = model.h5["N"][vertex, :, :, :].astype(np.float32)
        n_v = model._sed_n_vertex_cache[vertex]
        n00 = n_v[ir0, ia0, :].astype(float)
        n10 = n_v[ir1, ia0, :].astype(float)
        n01 = n_v[ir0, ia1, :].astype(float)
        n11 = n_v[ir1, ia1, :].astype(float)
        n_rv_av = (
            (1.0 - tr) * (1.0 - ta) * n00
            + tr * (1.0 - ta) * n10
            + (1.0 - tr) * ta * n01
            + tr * ta * n11
        )
        n_interp += float(weight) * n_rv_av

    if np.any(~np.isfinite(n_interp)) or np.any(n_interp <= 0):
        raise ValueError("non-positive interpolated band integral")
    return n_interp


def calibrate_band_zero_points(model, filters, ref_teff=6000.0, ref_logg=2.0, ref_mh=0.0):
    """Calibrate -2.5 log10(cache integral) to pyphot Vega magnitudes."""
    wave_aa = np.loadtxt(WAVE_FILE)
    params = model.params
    dist2 = (
        ((params[:, 0] - ref_teff) / 250.0) ** 2
        + ((params[:, 1] - ref_logg) / 0.5) ** 2
        + ((params[:, 2] - ref_mh) / 0.25) ** 2
    )
    idx = int(np.nanargmin(dist2))
    ref_params = params[idx]
    raw_spec_file = model.h5["file"][idx].decode("utf-8")
    spec_file = Path(raw_spec_file)
    if not spec_file.exists():
        candidates = [
            OUT_DIR / "r500" / spec_file.parent.name / spec_file.name,
            OUT_DIR / "r500/m+0.00" / spec_file.name,
            OUT_DIR / spec_file.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                spec_file = candidate
                break
        else:
            raise FileNotFoundError(f"Reference spectrum not found: {raw_spec_file}")
    flux = np.loadtxt(spec_file)
    if flux.ndim > 1:
        flux = flux[:, 0]

    d = model.h5["D"][idx, :].astype(float)
    zps = {}
    info_rows = []
    for band in BANDS_ALL:
        py_mag = pyphot_mag(filters[band], wave_aa, flux)
        band_idx = model.band_index[band]
        zps[band] = float(py_mag + 2.5 * np.log10(d[band_idx]))
        info_rows.append(
            {
                "band": band,
                "pyphot_surface_mag": py_mag,
                "cache_D": d[band_idx],
                "cache_to_vega_zp": zps[band],
            }
        )
    info = pd.DataFrame(info_rows)
    info["ref_Teff"] = float(ref_params[0])
    info["ref_logg"] = float(ref_params[1])
    info["ref_MH"] = float(ref_params[2])
    info["ref_spec_file"] = str(spec_file)
    return zps, info


def read_input_tables():
    phot = pd.read_csv(MW_PHOT_FILE, dtype={"GaiaDR3": "string", "Name": "string"})
    wang = pd.read_csv(WANG25_FILE, dtype={"source_id": "string"})
    phot["gaia_key"] = phot["GaiaDR3"].map(_id_key)
    wang["gaia_key"] = wang["source_id"].map(_id_key)
    wang_keep = [
        "gaia_key",
        "source_id",
        "D_w1",
        "D_w1_error",
        "[Fe/H]",
        "<d>",
        "<d_error>",
        "l[deg]",
        "b[deg]",
    ]
    wang_keep = [c for c in wang_keep if c in wang.columns]
    df = phot.merge(wang[wang_keep].drop_duplicates("gaia_key"), on="gaia_key", how="left", suffixes=("", "_wang25"))

    if COLORFIT_FILE.exists():
        color = pd.read_csv(COLORFIT_FILE, dtype={"GaiaDR3_match": "string", "Name": "string"})
        color["gaia_key"] = color["GaiaDR3_match"].map(_id_key)
        color_keep = [
            "gaia_key",
            "Name",
            "Teff_p50",
            "Teff_err_minus",
            "Teff_err_plus",
            "Av_p50",
            "Av_err_minus",
            "Av_err_plus",
            "Rv_p50",
            "Rv_err_minus",
            "Rv_err_plus",
            "logg_p50",
            "MH_p50",
        ]
        color_keep = [c for c in color_keep if c in color.columns]
        color = color[color_keep].drop_duplicates("gaia_key")
        df = df.merge(color, on="gaia_key", how="left", suffixes=("", "_colorfit"))

    for band in BANDS_ALL:
        df[MAG_COL[band]] = _num(df[MAG_COL[band]])
        df[MAGERR_COL[band]] = _num(df[MAGERR_COL[band]])
    for c in ["D_w1", "D_w1_error", "[Fe/H]", "period_used", "logP", "logP_F_plot_Pilecki2024_MW", "BP_mag"]:
        if c in df.columns:
            df[c] = _num(df[c])
    for c in ["Teff_p50", "Av_p50", "Rv_p50", "Teff_err_minus", "Teff_err_plus", "Av_err_minus", "Av_err_plus", "Rv_err_minus", "Rv_err_plus"]:
        if c in df.columns:
            df[c] = _num(df[c])
    if "riello_cstar_bad_3sigma" in df.columns:
        df["riello_cstar_bad_3sigma"] = df["riello_cstar_bad_3sigma"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["riello_cstar_bad_3sigma"] = False
    return df


def observed_mag_vector(row, include_w1_in_sed=True):
    bands = BANDS_ALL if include_w1_in_sed else [b for b in BANDS_ALL if b != "W1"]
    obs = []
    err = []
    used = []
    for band in bands:
        m = _finite(row.get(MAG_COL[band], np.nan))
        e = _finite(row.get(MAGERR_COL[band], np.nan))
        if np.isfinite(m) and np.isfinite(e) and e > 0:
            used.append(band)
            obs.append(m)
            err.append(math.sqrt(e * e + MAG_FLOOR[band] ** 2))
    return used, np.asarray(obs, dtype=float), np.asarray(err, dtype=float)


def select_10_sources(df, n_stars=10, include_w1_in_sed=True):
    quality = np.ones(len(df), dtype=bool)
    for band in (BANDS_ALL if include_w1_in_sed else [b for b in BANDS_ALL if b != "W1"]):
        quality &= np.isfinite(df[MAG_COL[band]].to_numpy(float))
        quality &= np.isfinite(df[MAGERR_COL[band]].to_numpy(float))
        quality &= df[MAGERR_COL[band]].to_numpy(float) > 0
    quality &= np.isfinite(df["D_w1"].to_numpy(float))
    quality &= np.isfinite(df["D_w1_error"].to_numpy(float))
    quality &= df["D_w1"].to_numpy(float) > 0
    quality &= df["D_w1_error"].to_numpy(float) > 0
    quality &= np.isfinite(df["[Fe/H]"].to_numpy(float))
    quality &= np.isfinite(df["period_used"].to_numpy(float))

    if "Av_p50" in df.columns:
        quality &= df["Av_p50"].fillna(-np.inf).to_numpy(float) > 1.0
    quality &= df["BP_mag"].fillna(np.inf).to_numpy(float) < 20.3
    quality &= ~df["riello_cstar_bad_3sigma"].to_numpy(bool)

    cand = df.loc[quality].copy()
    cand["logP_select"] = np.where(
        np.isfinite(cand.get("logP_F_plot_Pilecki2024_MW", np.nan)),
        cand["logP_F_plot_Pilecki2024_MW"],
        np.log10(cand["period_used"]),
    )
    cand["distance_frac_err"] = cand["D_w1_error"] / cand["D_w1"]
    cand = cand.sort_values(["distance_frac_err", "BP_mag", "logP_select"]).reset_index(drop=True)

    bins = [(-np.inf, 0.6, max(1, n_stars // 3)), (0.6, 1.0, max(1, n_stars // 3)), (1.0, np.inf, n_stars)]
    chosen = []
    used_idx = set()
    for lo, hi, quota in bins:
        sub = cand[cand["logP_select"].between(lo, hi, inclusive="left")]
        if len(sub) == 0:
            continue
        take_n = min(quota, len(sub), n_stars - len(chosen))
        positions = np.linspace(0, len(sub) - 1, take_n).round().astype(int)
        for pos in positions:
            idx = int(sub.index[pos])
            if idx not in used_idx and len(chosen) < n_stars:
                chosen.append(idx)
                used_idx.add(idx)
    if len(chosen) < n_stars:
        for idx in cand.index:
            if int(idx) not in used_idx:
                chosen.append(int(idx))
                used_idx.add(int(idx))
            if len(chosen) >= n_stars:
                break
    selected = cand.loc[chosen].copy().reset_index(drop=True)
    selected.insert(0, "selection_tag", [f"SED{i+1:02d}" for i in range(len(selected))])
    selected["W1_double_used"] = bool(include_w1_in_sed)
    return selected


def bounds_for_model(model):
    return {
        "lower": np.array([4000.0, 0.5, -1.5, 0.0, max(2.1, float(model.rv_grid[0])), np.log10(0.1), np.log10(3.0)]),
        "upper": np.array([7500.0, 3.5, 0.5, 20.0, min(5.6, float(model.rv_grid[-1])), np.log10(30.0), np.log10(600.0)]),
    }


def model_mags(theta, model, zps, used_bands):
    teff, logg, mh, av, rv, logd, logr = theta
    n = band_n_interpolated(model, teff, logg, mh, rv, av)
    radius_cm = (10.0**logr) * R_SUN_CM
    distance_cm = (10.0**logd) * KPC_CM
    scale = (radius_cm / distance_cm) ** 2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bad R/D scale")
    mags = {}
    for band in used_bands:
        i = model.band_index[band]
        mags[band] = float(-2.5 * np.log10(n[i]) + zps[band] - 2.5 * np.log10(scale))
    return np.asarray([mags[b] for b in used_bands], dtype=float)


def logg_prior_mu(teff, row):
    period = _finite(row.get("period_used", np.nan))
    if not np.isfinite(period) or period <= 0:
        return np.nan
    return 6.483 * np.log10(teff / 5800.0) - 0.775 * np.log10(period) + 2.475


def prior_residuals(theta, row):
    teff, logg, mh, _av, _rv, logd, _logr = theta
    res = []
    mu_logg = logg_prior_mu(teff, row)
    if np.isfinite(mu_logg):
        res.append((logg - mu_logg) / LOGG_PRIOR_SIGMA)
    mu_mh = _finite(row.get("[Fe/H]", np.nan))
    if np.isfinite(mu_mh):
        res.append((mh - mu_mh) / FEH_PRIOR_SIGMA)
    d_w1 = _finite(row.get("D_w1", np.nan))
    d_w1_err = _finite(row.get("D_w1_error", np.nan))
    d_model = 10.0**logd
    if np.isfinite(d_w1) and np.isfinite(d_w1_err) and d_w1_err > 0:
        res.append((d_model - d_w1) / d_w1_err)
    return np.asarray(res, dtype=float)


def residual(theta, row, model, zps, used_bands, obs, err, lower, upper):
    theta = np.asarray(theta, dtype=float)
    n_bad = len(obs) + 3
    if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
        return np.full(n_bad, 1e6)
    try:
        pred = model_mags(theta, model, zps, used_bands)
    except Exception:
        return np.full(n_bad, 1e6)
    phot_res = (obs - pred) / err
    return np.concatenate([phot_res, prior_residuals(theta, row)])


def log_prob(theta, row, model, zps, used_bands, obs, err, lower, upper):
    r = residual(theta, row, model, zps, used_bands, obs, err, lower, upper)
    if np.any(~np.isfinite(r)) or np.any(np.abs(r) > 1e5):
        return -np.inf
    return -0.5 * float(np.sum(r * r))


def initial_guess(row, model):
    teff = _finite(row.get("Teff_p50", np.nan))
    av = _finite(row.get("Av_p50", np.nan))
    rv = _finite(row.get("Rv_p50", np.nan))
    mh = _finite(row.get("[Fe/H]", np.nan))
    d = _finite(row.get("D_w1", np.nan))
    period = _finite(row.get("period_used", np.nan))
    logp = np.log10(period) if np.isfinite(period) and period > 0 else 1.0
    radius = np.clip(35.0 * (10.0**logp) ** 0.70, 3.0, 600.0)
    if not np.isfinite(teff):
        teff = 5800.0
    if not np.isfinite(av):
        av = 2.0
    if not np.isfinite(rv):
        rv = 3.1
    if not np.isfinite(mh):
        mh = 0.0
    if not np.isfinite(d) or d <= 0:
        d = 5.0
    logg = logg_prior_mu(teff, row)
    if not np.isfinite(logg):
        logg = 2.0
    return np.array([teff, logg, mh, av, rv, np.log10(d), np.log10(radius)], dtype=float)


def fit_map(row, model, zps, used_bands, obs, err, bounds, random_seed=42, n_start=24):
    rng = np.random.default_rng(random_seed)
    lower = bounds["lower"]
    upper = bounds["upper"]
    starts = [np.clip(initial_guess(row, model), lower + 1e-6, upper - 1e-6)]
    for teff in [5000.0, 5800.0, 6500.0]:
        x = starts[0].copy()
        x[0] = teff
        x[1] = np.clip(logg_prior_mu(teff, row), lower[1], upper[1])
        starts.append(np.clip(x, lower + 1e-6, upper - 1e-6))
    for _ in range(max(0, n_start - len(starts))):
        starts.append(lower + rng.random(len(lower)) * (upper - lower))

    best = None
    rows = []
    for x0 in starts:
        try:
            res = least_squares(
                residual,
                x0=x0,
                bounds=(lower, upper),
                args=(row, model, zps, used_bands, obs, err, lower, upper),
                max_nfev=1200,
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


def run_emcee(row, model, zps, used_bands, obs, err, bounds, x_map, nwalkers, nsteps, burnin, thin, random_seed):
    import emcee

    lower = bounds["lower"]
    upper = bounds["upper"]
    rng = np.random.default_rng(random_seed)
    ndim = len(PARAM_NAMES)
    nwalkers = int(max(nwalkers, 2 * ndim + 2))
    scale = np.array([80.0, 0.04, 0.04, 0.08, 0.06, 0.008, 0.025])
    p0 = []
    while len(p0) < nwalkers:
        p = np.asarray(x_map, dtype=float) + rng.normal(0.0, scale)
        if np.all(np.isfinite(p)) and np.all(p > lower) and np.all(p < upper):
            p0.append(p)
    p0 = np.asarray(p0)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob,
        args=(row, model, zps, used_bands, obs, err, lower, upper),
    )
    sampler.run_mcmc(p0, int(nsteps), progress=False)
    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    logp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
    samples = pd.DataFrame(flat, columns=PARAM_NAMES)
    samples["D_kpc"] = 10.0 ** samples["logD"]
    samples["R_Rsun"] = 10.0 ** samples["logR"]
    samples["log_prob"] = logp
    return sampler, samples


def summarize_samples(row, samples, used_bands, sampler, map_res):
    out = {
        "selection_tag": row["selection_tag"],
        "Name": str(row["Name"]),
        "GaiaDR3": str(row["GaiaDR3"]),
        "Mode": row.get("Mode", ""),
        "period_used": _finite(row.get("period_used", np.nan)),
        "logP": np.log10(_finite(row.get("period_used", np.nan))) if _finite(row.get("period_used", np.nan)) > 0 else np.nan,
        "bands_fit": ",".join(used_bands),
        "nband_fit": len(used_bands),
        "W1_double_used": bool(row.get("W1_double_used", True)),
        "D_w1": _finite(row.get("D_w1", np.nan)),
        "D_w1_error": _finite(row.get("D_w1_error", np.nan)),
        "FeH_prior": _finite(row.get("[Fe/H]", np.nan)),
        "chi2_map": float(np.sum(map_res.fun**2)),
        "acceptance_fraction": float(np.nanmedian(sampler.acceptance_fraction)),
        "fit_failed": False,
    }
    for p in PARAM_NAMES + ["D_kpc", "R_Rsun"]:
        x = samples[p].to_numpy(float)
        p16, p50, p84 = np.nanpercentile(x, [16, 50, 84])
        out[f"{p}_p16"] = float(p16)
        out[f"{p}_p50"] = float(p50)
        out[f"{p}_p84"] = float(p84)
        out[f"{p}_err_minus"] = float(p50 - p16)
        out[f"{p}_err_plus"] = float(p84 - p50)
    out["D_prior_pull"] = (out["D_kpc_p50"] - out["D_w1"]) / out["D_w1_error"] if out["D_w1_error"] > 0 else np.nan
    return out


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))[:80]


def make_corner(samples, row, outdir):
    import corner

    labels = [r"$T_{\rm eff}$", r"$\log g$", r"$[M/H]$", r"$A_V$", r"$R_V$", r"$D$ kpc", r"$R/R_\odot$"]
    plot_cols = ["Teff", "logg", "MH", "Av", "Rv", "D_kpc", "R_Rsun"]
    fig = corner.corner(
        samples[plot_cols],
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 9},
    )
    out = outdir / f"{row['selection_tag']}_{safe_name(row['Name'])}_corner.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def flux_from_mag(mag, filt):
    return 10.0 ** (-0.4 * (mag + filt.Vega_zero_mag))


def make_sed_plot(row, samples, model, zps, filters, lambda_eff, used_bands, obs, err, outdir):
    theta = np.array(
        [
            samples["Teff"].median(),
            samples["logg"].median(),
            samples["MH"].median(),
            samples["Av"].median(),
            samples["Rv"].median(),
            samples["logD"].median(),
            samples["logR"].median(),
        ],
        dtype=float,
    )
    pred_mag = model_mags(theta, model, zps, used_bands)
    lam = np.asarray([lambda_eff[b] for b in used_bands], dtype=float) / 1e4
    obs_flux = np.asarray([flux_from_mag(m, filters[b]) for m, b in zip(obs, used_bands)], dtype=float)
    pred_flux = np.asarray([flux_from_mag(m, filters[b]) for m, b in zip(pred_mag, used_bands)], dtype=float)
    obs_ferr = obs_flux * np.log(10.0) * 0.4 * err

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(lam, obs_flux, yerr=obs_ferr, fmt="o", color="k", label="observed")
    ax.plot(lam, pred_flux, "s-", color="tab:red", label="SED model p50")
    for x, y, b in zip(lam, obs_flux, used_bands):
        ax.text(x, y * 1.08, b, ha="center", va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective wavelength (micron)")
    ax.set_ylabel("band-integrated flux proxy")
    ax.set_title(f"{row['selection_tag']} {row['Name']} SED")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = outdir / f"{row['selection_tag']}_{safe_name(row['Name'])}_sed_flux_fit.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_comparison(summary, full_input):
    color_cols = [
        "gaia_key",
        "Teff_p50",
        "Teff_err_minus",
        "Teff_err_plus",
        "Av_p50",
        "Av_err_minus",
        "Av_err_plus",
        "Rv_p50",
        "Rv_err_minus",
        "Rv_err_plus",
    ]
    color_cols = [c for c in color_cols if c in full_input.columns]
    color = full_input[color_cols].drop_duplicates("gaia_key")
    rename = {c: f"{c}_color" for c in color.columns if c != "gaia_key"}
    color = color.rename(columns=rename)
    comp = summary.merge(color, on="gaia_key", how="left")
    for p in ["Teff", "Av", "Rv"]:
        sed_col = f"{p}_p50"
        color_col = f"{p}_p50_color"
        if sed_col in comp.columns and color_col in comp.columns:
            comp[f"Delta_{p}_SED_minus_color"] = comp[sed_col] - comp[color_col]
    return comp


def make_summary_plots(summary, comparison, outdir):
    if len(summary) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.errorbar(summary["logP"], summary["Teff_p50"], yerr=[summary["Teff_err_minus"], summary["Teff_err_plus"]], fmt="o")
    ax.invert_yaxis()
    ax.set_xlabel("log P")
    ax.set_ylabel("SED Teff (K)")
    ax.set_title("10-star SED Teff-logP")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "SED_W1PLZ_Teff_logP_10star.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.errorbar(summary["logP"], summary["R_Rsun_p50"], yerr=[summary["R_Rsun_err_minus"], summary["R_Rsun_err_plus"]], fmt="o")
    ax.set_xlabel("log P")
    ax.set_ylabel(r"SED radius ($R_\odot$)")
    ax.set_title("10-star SED radius-period")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "SED_W1PLZ_radius_period_10star.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, p, lab in zip(axes, ["Teff", "Av", "Rv"], [r"$T_{\rm eff}$", r"$A_V$", r"$R_V$"]):
        xcol = f"{p}_p50_color"
        ycol = f"{p}_p50"
        if xcol not in comparison.columns or ycol not in comparison.columns:
            ax.text(0.5, 0.5, f"missing {p} comparison", transform=ax.transAxes, ha="center")
            continue
        good = np.isfinite(comparison[xcol]) & np.isfinite(comparison[ycol])
        ax.scatter(comparison.loc[good, xcol], comparison.loc[good, ycol], s=35)
        if good.any():
            mn = min(comparison.loc[good, xcol].min(), comparison.loc[good, ycol].min())
            mx = max(comparison.loc[good, xcol].max(), comparison.loc[good, ycol].max())
            ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel(f"color-fit {lab}")
        ax.set_ylabel(f"SED {lab}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "SED_vs_colorfit_Teff_Av_Rv_comparison.png", dpi=180)
    plt.close(fig)


def run_10star_sed_w1plz_emcee(
    n_stars=10,
    include_w1_in_sed=True,
    nwalkers=48,
    nsteps=3000,
    burnin=800,
    thin=5,
    random_seed=42,
    outdir=POST_DIR,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full = read_input_tables()
    selected = select_10_sources(full, n_stars=n_stars, include_w1_in_sed=include_w1_in_sed)
    selected.to_csv(outdir / "MW_10star_SED_W1PLZ_selected.csv", index=False)

    PreconvolvedBOSZModel = _import_model()
    model = PreconvolvedBOSZModel(cache_file=CACHE_FILE)
    filters, lambda_eff = load_pyphot_filters()
    zps, zp_info = calibrate_band_zero_points(model, filters)
    zp_info.to_csv(outdir / "SED_cache_to_pyphot_vega_zero_points.csv", index=False)
    bounds = bounds_for_model(model)

    summaries = []
    sample_dict = {}
    failed_rows = []
    try:
        for i, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected), desc="SED emcee stars")):
            used_bands, obs, err = observed_mag_vector(row, include_w1_in_sed=include_w1_in_sed)
            try:
                map_res, starts = fit_map(
                    row,
                    model,
                    zps,
                    used_bands,
                    obs,
                    err,
                    bounds,
                    random_seed=random_seed + i,
                )
                sampler, samples = run_emcee(
                    row,
                    model,
                    zps,
                    used_bands,
                    obs,
                    err,
                    bounds,
                    map_res.x,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    burnin=burnin,
                    thin=thin,
                    random_seed=random_seed + 1000 + i,
                )
                samples.insert(0, "selection_tag", row["selection_tag"])
                samples.insert(1, "Name", str(row["Name"]))
                samples.to_csv(outdir / f"{row['selection_tag']}_{safe_name(row['Name'])}_samples.csv.gz", index=False)
                starts.to_csv(outdir / f"{row['selection_tag']}_{safe_name(row['Name'])}_least_squares_starts.csv", index=False)

                summary = summarize_samples(row, samples, used_bands, sampler, map_res)
                summary["corner_file"] = str(make_corner(samples, row, outdir))
                summary["sed_plot_file"] = str(make_sed_plot(row, samples, model, zps, filters, lambda_eff, used_bands, obs, err, outdir))
                summary["gaia_key"] = row["gaia_key"]
                summaries.append(summary)
                sample_dict[str(row["Name"])] = samples
            except Exception as exc:
                warnings.warn(f"SED fit failed for {row.get('Name')}: {exc}")
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

    sed_summary = pd.DataFrame(summaries + failed_rows)
    sed_summary.to_csv(outdir / "MW_10star_SED_W1PLZ_emcee_summary.csv", index=False)
    comparison = build_comparison(sed_summary, full)
    comparison.to_csv(outdir / "MW_10star_SED_vs_colorfit_comparison.csv", index=False)
    make_summary_plots(sed_summary[sed_summary.get("fit_failed", False) != True].copy(), comparison, outdir)

    return selected, sed_summary, sample_dict, comparison


print("Loaded 10-star SED+W1 PLZ emcee helpers.")
print("Call: selected, sed_summary, sed_samples, comparison = run_10star_sed_w1plz_emcee(...)")
