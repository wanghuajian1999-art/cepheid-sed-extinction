#!/usr/bin/env python3
"""MW Cepheid SED magnitude fit with intrinsic-color PC prior and Wang25 distance prior.

Jupyter usage
-------------

    %run -i "jupyter_all_mw_sed_pcprior_method2_distanceprior.py"

    df_fit, residuals_long, summary = run_all_mw_sed_pcprior_method2_distanceprior(
        max_stars=20,          # None = all available rows
        quality_mode="all",    # "primary", "secondary", "all"
        overwrite=True,
        make_plots=True,
    )

This implements method 2:
  - observational likelihood is only broad-band apparent magnitudes;
  - LMC PC information is used as a prior on model intrinsic colors
    C_syn(X-RP | Teff, logg, [M/H]);
  - Wang25 <d>/<d_error> is used as a distance prior;
  - no extra magnitude floor is added here.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR

MW_PHOT_FILE = SCRIPT_DIR / "MW_3640_PC_color_excess_from_LMC_PC.csv"
WANG25_FILE = SCRIPT_DIR / "wang25x.csv"
ZP_FILE = SCRIPT_DIR / "SED_pyphot_vega_fixed_zero_points.csv"
PC_COEFF_FILE = SCRIPT_DIR / "LMC_X_RP_PC_main_quality_noI_coefficients.csv"
PC_COV_FILE = SCRIPT_DIR / "LMC_X_RP_PC_main_quality_noI_residual_covariance.csv"

FIT_OUT = SCRIPT_DIR / "MW_3640_SED_PCprior_method2_distanceprior_MAP.csv"
RESID_OUT = SCRIPT_DIR / "MW_3640_SED_PCprior_method2_distanceprior_residuals_long.csv"
FAIL_OUT = SCRIPT_DIR / "MW_3640_SED_PCprior_method2_distanceprior_failures.csv"
SUMMARY_OUT = SCRIPT_DIR / "MW_3640_SED_PCprior_method2_distanceprior_summary.json"
PLOT_PREFIX = SCRIPT_DIR / "MW_3640_SED_PCprior_method2_distanceprior"

sys.path.insert(0, str(SCRIPT_DIR))
import jupyter_10star_sed_w1plz_emcee_fit as sedbase
import jupyter_10star_joint_sed_color_w1plz_emcee_fit as jointbase


BANDS_ALL = ["BP", "G", "RP", "J", "H", "K", "W1"]
REQUIRED_OPTICAL = ["BP", "G", "RP"]
NIR_MIR_BANDS = ["J", "H", "K", "W1"]
PC_COLOR_BANDS = ["BP", "G", "J", "H", "K", "W1"]
PARAM_NAMES = ["Teff", "logg", "MH", "Av", "Rv", "logD", "logR"]
PARAM_DERIVED = ["D_kpc", "R_Rsun"]

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

LOGG_PRIOR_SIGMA = 0.108
FEH_PRIOR_SIGMA = 0.13
PC_COV_JITTER = 1.0e-8


def _id_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _finite(x):
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def safe_name(x) -> str:
    return sedbase.safe_name(str(x))


def read_merged_input() -> pd.DataFrame:
    phot = pd.read_csv(MW_PHOT_FILE, dtype={"GaiaDR3": "string", "Name": "string"})
    wang = pd.read_csv(WANG25_FILE, dtype={"source_id": "string"})

    phot["gaia_key"] = phot["GaiaDR3"].map(_id_key)
    wang["gaia_key"] = wang["source_id"].map(_id_key)

    wang_keep = [
        "gaia_key",
        "source_id",
        "Cepheid",
        "<d>",
        "<d_error>",
        "[Fe/H]",
        "l[deg]",
        "b[deg]",
        "D_w1",
        "D_w1_error",
        "weighted_D",
        "D_error",
    ]
    wang_keep = [c for c in wang_keep if c in wang.columns]
    df = phot.merge(
        wang[wang_keep].drop_duplicates("gaia_key"),
        on="gaia_key",
        how="left",
        suffixes=("", "_wang25"),
    )

    numeric_cols = (
        list(MAG_COL.values())
        + list(MAGERR_COL.values())
        + ["period_used", "logP_F_plot_Pilecki2024_MW", "<d>", "<d_error>", "[Fe/H]"]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_zps() -> dict[str, float]:
    zp = pd.read_csv(ZP_FILE)
    if "ZP_X_cache_raw_integral" not in zp.columns:
        raise ValueError(f"{ZP_FILE} lacks ZP_X_cache_raw_integral")
    return {str(r["band"]): float(r["ZP_X_cache_raw_integral"]) for _, r in zp.iterrows()}


def load_pc_tables():
    coeff = pd.read_csv(PC_COEFF_FILE)
    cov_long = pd.read_csv(PC_COV_FILE)
    return coeff, cov_long


def has_band(row, band: str) -> bool:
    m = _finite(row.get(MAG_COL[band], np.nan))
    e = _finite(row.get(MAGERR_COL[band], np.nan))
    return np.isfinite(m) and np.isfinite(e) and e > 0


def assign_quality_tier(row) -> str:
    has = {band: has_band(row, band) for band in BANDS_ALL}
    if all(has[b] for b in BANDS_ALL):
        return "primary_7band"
    if all(has[b] for b in REQUIRED_OPTICAL) and any(has[b] for b in NIR_MIR_BANDS):
        return "secondary_4plus"
    return "not_fit"


def base_fail_reason(row, quality_mode: str) -> str:
    tier = row.get("quality_tier", "not_fit")
    if quality_mode == "primary" and tier != "primary_7band":
        return "not_primary_7band"
    if quality_mode in ("secondary", "all") and tier == "not_fit":
        return "missing_GBPRP_or_no_JHKW1"

    for col in ["<d>", "<d_error>", "[Fe/H]", "period_used"]:
        val = _finite(row.get(col, np.nan))
        if not np.isfinite(val):
            return f"missing_{col}"
    if _finite(row.get("<d>")) <= 0 or _finite(row.get("<d_error>")) <= 0:
        return "bad_distance_prior"
    if _finite(row.get("period_used")) <= 0:
        return "bad_period"
    return ""


def row_sed_vector(row):
    used, obs, err = [], [], []
    for band in BANDS_ALL:
        if has_band(row, band):
            used.append(band)
            obs.append(float(row[MAG_COL[band]]))
            err.append(float(row[MAGERR_COL[band]]))
    return used, np.asarray(obs, dtype=float), np.asarray(err, dtype=float)


def pc_mode_for_row(row) -> str:
    mode = str(row.get("Mode", "F")).strip().upper()
    return "1O" if mode.startswith("1O") else "F"


def logp_for_pc(row) -> float:
    logp = _finite(row.get("logP_F_plot_Pilecki2024_MW", np.nan))
    if np.isfinite(logp):
        return logp
    period = _finite(row.get("period_used", np.nan))
    return np.log10(period) if np.isfinite(period) and period > 0 else np.nan


def build_pc_prior(row, pc_coeff: pd.DataFrame, pc_cov_long: pd.DataFrame):
    mode_pc = pc_mode_for_row(row)
    logp = logp_for_pc(row)
    if not np.isfinite(logp):
        raise ValueError("missing logP for PC prior")

    rows = []
    colors = []
    for band in PC_COLOR_BANDS:
        color = f"E_{band}_RP"
        sub = pc_coeff[
            (pc_coeff["mode_for_PC"].astype(str).str.upper() == mode_pc)
            & (pc_coeff["color"].astype(str) == color)
        ]
        if len(sub) != 1:
            raise ValueError(f"missing PC coefficient for {mode_pc} {color}")
        r = sub.iloc[0]
        c_pc = float(r["intercept"]) + float(r["slope"]) * logp
        coeff_var = float(r.get("intercept_err", 0.0)) ** 2 + (logp * float(r.get("slope_err", 0.0))) ** 2
        rows.append(
            {
                "mode_for_PC": mode_pc,
                "band": band,
                "color": color,
                "logP_used": logp,
                "C_PC": c_pc,
                "sigma_mag": float(r["sigma_mag"]),
                "coeff_var": coeff_var,
                "intercept": float(r["intercept"]),
                "slope": float(r["slope"]),
            }
        )
        colors.append(color)

    cov = np.full((len(colors), len(colors)), np.nan, dtype=float)
    cov_sub = pc_cov_long[pc_cov_long["mode_for_PC"].astype(str).str.upper() == mode_pc]
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors):
            hit = cov_sub[(cov_sub["color1"].astype(str) == c1) & (cov_sub["color2"].astype(str) == c2)]
            if len(hit) == 1:
                cov[i, j] = float(hit.iloc[0]["covariance"])
    if np.any(~np.isfinite(cov)):
        # Fall back to diagonal intrinsic scatter from coefficient table.
        sig = np.asarray([r["sigma_mag"] for r in rows], dtype=float)
        cov = np.diag(sig * sig)

    coeff_diag = np.asarray([r["coeff_var"] for r in rows], dtype=float)
    cov = cov + np.diag(coeff_diag + PC_COV_JITTER)

    pc_prior = pd.DataFrame(rows)
    pc_prior["sigma_PC_diag"] = np.sqrt(np.diag(cov))
    return pc_prior, cov


def stable_whiten(delta: np.ndarray, cov: np.ndarray):
    cov = np.asarray(cov, dtype=float)
    delta = np.asarray(delta, dtype=float)
    for jitter in [0.0, 1e-8, 1e-6, 1e-4]:
        try:
            L = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter)
            return np.linalg.solve(L, delta), f"cholesky_jitter_{jitter:g}"
        except np.linalg.LinAlgError:
            continue
    inv_cov = np.linalg.pinv(cov)
    # Return a vector whose squared norm equals delta^T inv_cov delta.
    val = float(delta.T @ inv_cov @ delta)
    if not np.isfinite(val) or val < 0:
        return np.full_like(delta, 1e6), "pinv_failed"
    out = np.zeros_like(delta)
    out[0] = math.sqrt(val)
    return out, "pinv_scalar"


def intrinsic_color_syn(theta, model, zps, color_bands):
    teff, logg, mh, _av, _rv, _logd, _logr = theta
    d = jointbase.band_d_interpolated(model, teff, logg, mh)
    rp_i = model.band_index["RP"]
    m_rp0 = -2.5 * np.log10(d[rp_i]) + zps["RP"]
    colors = []
    for band in color_bands:
        i = model.band_index[band]
        m_x0 = -2.5 * np.log10(d[i]) + zps[band]
        colors.append(float(m_x0 - m_rp0))
    return np.asarray(colors, dtype=float)


def pc_prior_residuals(theta, model, zps, pc_prior, pc_cov):
    c_syn = intrinsic_color_syn(theta, model, zps, pc_prior["band"].tolist())
    delta = c_syn - pc_prior["C_PC"].to_numpy(float)
    return stable_whiten(delta, pc_cov)


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

    d_prior = _finite(row.get("<d>", np.nan))
    d_err = _finite(row.get("<d_error>", np.nan))
    d_model = 10.0 ** logd
    if np.isfinite(d_prior) and np.isfinite(d_err) and d_prior > 0 and d_err > 0:
        res.append((d_model - d_prior) / d_err)

    return np.asarray(res, dtype=float)


def bounds_for_model(model):
    lower = np.array([4000.0, 0.5, -1.5, 0.0, max(2.3, float(model.rv_grid[0])), np.log10(0.05), np.log10(3.0)])
    upper = np.array([7500.0, 3.5, 0.5, 20.0, min(5.6, float(model.rv_grid[-1])), np.log10(50.0), np.log10(1000.0)])
    return {"lower": lower, "upper": upper}


def model_mags(theta, model, zps, sed_bands):
    teff, logg, mh, av, rv, logd, logr = theta
    n = sedbase.band_n_interpolated(model, teff, logg, mh, rv, av)
    radius_cm = (10.0 ** logr) * sedbase.R_SUN_CM
    distance_cm = (10.0 ** logd) * sedbase.KPC_CM
    scale = (radius_cm / distance_cm) ** 2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bad R/D scale")
    out = []
    for band in sed_bands:
        i = model.band_index[band]
        out.append(float(-2.5 * np.log10(n[i]) + zps[band] - 2.5 * np.log10(scale)))
    return np.asarray(out, dtype=float)


def residual(theta, row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds):
    theta = np.asarray(theta, dtype=float)
    lower, upper = bounds["lower"], bounds["upper"]
    n_bad = len(sed_obs) + len(PC_COLOR_BANDS) + 3
    if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
        return np.full(n_bad, 1e6)
    try:
        sed_pred = model_mags(theta, model, zps, sed_bands)
        sed_res = (sed_obs - sed_pred) / sed_err
        pc_res, _ = pc_prior_residuals(theta, model, zps, pc_prior, pc_cov)
        prior_res = prior_residuals(theta, row)
        return np.concatenate([sed_res, pc_res, prior_res])
    except Exception:
        return np.full(n_bad, 1e6)


def initial_guess(row, model):
    teff = _finite(row.get("Teff_p50", np.nan))
    av = _finite(row.get("Av_p50", np.nan))
    rv = _finite(row.get("Rv_p50", np.nan))
    mh = _finite(row.get("[Fe/H]", np.nan))
    d = _finite(row.get("<d>", np.nan))
    period = _finite(row.get("period_used", np.nan))
    logp = np.log10(period) if np.isfinite(period) and period > 0 else 1.0
    radius = np.clip(35.0 * (10.0 ** logp) ** 0.70, 3.0, 1000.0)

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


def fit_map(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds, random_seed=42, n_start=10):
    rng = np.random.default_rng(random_seed)
    lower, upper = bounds["lower"], bounds["upper"]
    starts = [np.clip(initial_guess(row, model), lower + 1e-6, upper - 1e-6)]

    for teff in [4800.0, 5400.0, 6000.0, 6600.0]:
        x = starts[0].copy()
        x[0] = teff
        x[1] = np.clip(logg_prior_mu(teff, row), lower[1], upper[1])
        starts.append(np.clip(x, lower + 1e-6, upper - 1e-6))

    for _ in range(max(0, n_start - len(starts))):
        x = lower + rng.random(len(lower)) * (upper - lower)
        # Draw distance and metallicity starts closer to the priors.
        d0 = _finite(row.get("<d>", np.nan))
        if np.isfinite(d0) and d0 > 0:
            x[5] = np.log10(np.clip(d0, 0.05, 50.0))
        mh0 = _finite(row.get("[Fe/H]", np.nan))
        if np.isfinite(mh0):
            x[2] = np.clip(mh0 + rng.normal(0.0, 0.15), lower[2], upper[2])
        starts.append(np.clip(x, lower + 1e-6, upper - 1e-6))

    best = None
    start_rows = []
    for x0 in starts:
        try:
            res = least_squares(
                residual,
                x0=x0,
                bounds=(lower, upper),
                args=(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds),
                max_nfev=1600,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            chi2 = float(np.sum(res.fun**2))
            start_rows.append(dict(zip(PARAM_NAMES, res.x)) | {"chi2": chi2, "success": bool(res.success), "message": res.message})
            if best is None or chi2 < best[0]:
                best = (chi2, res)
        except Exception as exc:
            start_rows.append({"chi2": np.inf, "success": False, "message": str(exc)})
    if best is None:
        raise RuntimeError("all least_squares starts failed")
    return best[1], pd.DataFrame(start_rows)


def parameter_errors_from_jacobian(res, n_resid):
    n_param = len(PARAM_NAMES)
    out = {f"{p}_err": np.nan for p in PARAM_NAMES}
    out.update({"D_kpc_err": np.nan, "R_Rsun_err": np.nan})
    out["param_err_reliable"] = False
    out["hessian_condition"] = np.nan

    try:
        jac = np.asarray(res.jac, dtype=float)
        jtj = jac.T @ jac
        cond = float(np.linalg.cond(jtj))
        dof = max(1, int(n_resid) - n_param)
        scale = max(1.0, float(np.sum(res.fun**2)) / dof)
        cov = np.linalg.pinv(jtj) * scale
        sig = np.sqrt(np.diag(cov))
        if np.all(np.isfinite(sig)) and cond < 1e14:
            for p, s in zip(PARAM_NAMES, sig):
                out[f"{p}_err"] = float(s)
            logd, logr = res.x[5], res.x[6]
            out["D_kpc_err"] = float(np.log(10.0) * (10.0 ** logd) * sig[5])
            out["R_Rsun_err"] = float(np.log(10.0) * (10.0 ** logr) * sig[6])
            out["param_err_reliable"] = True
        out["hessian_condition"] = cond
    except Exception:
        pass
    return out


def fit_one_row(row, model, zps, pc_coeff, pc_cov_long, bounds, random_seed=42, n_start=10):
    out = {
        "row_id": row.get("row_id", np.nan),
        "Name": row.get("Name", ""),
        "GaiaDR3": row.get("GaiaDR3", ""),
        "Mode": row.get("Mode", ""),
        "period_used": row.get("period_used", np.nan),
        "quality_tier": row.get("quality_tier", "not_fit"),
        "fit_success": False,
        "fail_reason": "",
    }
    fail = base_fail_reason(row, "all")
    if fail:
        out["fail_reason"] = fail
        return out, [], None

    try:
        sed_bands, sed_obs, sed_err = row_sed_vector(row)
        if len(sed_bands) < 4:
            out["fail_reason"] = "n_sed_bands_lt4"
            return out, [], None
        pc_prior, pc_cov = build_pc_prior(row, pc_coeff, pc_cov_long)
        res, starts = fit_map(
            row,
            model,
            zps,
            sed_bands,
            sed_obs,
            sed_err,
            pc_prior,
            pc_cov,
            bounds,
            random_seed=random_seed,
            n_start=n_start,
        )
        theta = np.asarray(res.x, dtype=float)
        sed_pred = model_mags(theta, model, zps, sed_bands)
        sed_res = (sed_obs - sed_pred) / sed_err
        pc_res, pc_cov_method = pc_prior_residuals(theta, model, zps, pc_prior, pc_cov)
        prior_res = prior_residuals(theta, row)

        out.update(dict(zip([f"{p}_fit" for p in PARAM_NAMES], theta)))
        out["D_kpc_fit"] = float(10.0 ** theta[5])
        out["R_Rsun_fit"] = float(10.0 ** theta[6])
        out["fit_success"] = bool(res.success and np.all(np.isfinite(theta)))
        out["fail_reason"] = "" if out["fit_success"] else str(res.message)
        out["sed_bands_used"] = ",".join(sed_bands)
        out["n_sed_bands"] = len(sed_bands)
        out["pc_mode_used"] = pc_mode_for_row(row)
        out["pc_colors_used"] = ",".join(pc_prior["color"].astype(str))
        out["n_pc_colors"] = len(pc_prior)
        out["pc_cov_method"] = pc_cov_method
        out["chi2_sed"] = float(np.sum(sed_res**2))
        out["chi2_pc"] = float(np.sum(pc_res**2))
        out["chi2_prior"] = float(np.sum(prior_res**2))
        out["chi2_total"] = float(np.sum(res.fun**2))
        out["n_residual"] = int(len(res.fun))
        out["n_param"] = len(PARAM_NAMES)
        out["chi2_red_like"] = out["chi2_total"] / max(1, out["n_residual"] - out["n_param"])
        out["distance_prior_source"] = "<d>"
        out["distance_prior_kpc"] = float(row["<d>"])
        out["distance_prior_err_kpc"] = float(row["<d_error>"])
        out["MH_prior"] = float(row["[Fe/H]"])
        out["MH_prior_sigma"] = FEH_PRIOR_SIGMA
        out["ZP_source"] = "pyphot_Vega_cache_raw_integral"
        out.update(parameter_errors_from_jacobian(res, len(res.fun)))

        residual_rows = []
        for band, obs, pred, err, rr in zip(sed_bands, sed_obs, sed_pred, sed_err, sed_res):
            residual_rows.append(
                {
                    "row_id": row.get("row_id", np.nan),
                    "Name": row.get("Name", ""),
                    "band": band,
                    "obs_mag": float(obs),
                    "model_mag": float(pred),
                    "mag_error": float(err),
                    "residual_mag": float(obs - pred),
                    "residual_sigma": float(rr),
                    "quality_tier": out["quality_tier"],
                }
            )
        return out, residual_rows, starts
    except Exception as exc:
        out["fail_reason"] = str(exc)
        return out, [], None


def build_work_table(df: pd.DataFrame, quality_mode: str, max_stars):
    if quality_mode not in {"primary", "secondary", "all"}:
        raise ValueError("quality_mode must be 'primary', 'secondary', or 'all'")

    df = df.copy()
    df["quality_tier"] = df.apply(assign_quality_tier, axis=1)

    if quality_mode == "primary":
        mask = df["quality_tier"] == "primary_7band"
    elif quality_mode == "secondary":
        mask = df["quality_tier"].isin(["primary_7band", "secondary_4plus"])
    else:
        mask = np.ones(len(df), dtype=bool)

    if max_stars is None:
        return df.loc[mask].copy()

    eligible = df.loc[mask & (df["quality_tier"].isin(["primary_7band", "secondary_4plus"]))].head(int(max_stars)).copy()
    return eligible


def make_summary_and_plots(df_fit: pd.DataFrame, residuals_long: pd.DataFrame, make_plots=True):
    summary = {
        "n_rows": int(len(df_fit)),
        "n_fit_success": int(df_fit.get("fit_success", pd.Series(dtype=bool)).fillna(False).sum()),
        "quality_tier_counts": df_fit.get("quality_tier", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "failure_counts": df_fit.loc[~df_fit.get("fit_success", False).astype(bool), "fail_reason"].value_counts(dropna=False).to_dict()
        if "fail_reason" in df_fit.columns
        else {},
    }

    if "fit_success" in df_fit.columns:
        good = df_fit["fit_success"].astype(bool)
        for p in ["Teff", "Av", "Rv", "logg", "MH", "D_kpc", "R_Rsun"]:
            col = f"{p}_fit"
            if col in df_fit.columns and good.any():
                vals = pd.to_numeric(df_fit.loc[good, col], errors="coerce")
                summary[f"{p}_median"] = float(np.nanmedian(vals))
                summary[f"{p}_p16"] = float(np.nanpercentile(vals, 16))
                summary[f"{p}_p84"] = float(np.nanpercentile(vals, 84))

    if len(residuals_long) > 0:
        by_band = residuals_long.groupby("band")["residual_sigma"].agg(["count", "median", "std"]).reset_index()
        summary["residual_sigma_by_band"] = by_band.to_dict(orient="records")

    if make_plots and len(df_fit) > 0:
        try:
            good = df_fit["fit_success"].astype(bool)
            fig, ax = plt.subplots(figsize=(7, 5))
            for mode, color in [("F", "tab:blue"), ("1O", "tab:orange")]:
                m = good & df_fit["Mode"].astype(str).str.upper().str.startswith(mode)
                if mode == "F":
                    m = good & ~df_fit["Mode"].astype(str).str.upper().str.startswith("1O")
                ax.scatter(
                    np.log10(pd.to_numeric(df_fit.loc[m, "period_used"], errors="coerce")),
                    np.log10(pd.to_numeric(df_fit.loc[m, "Teff_fit"], errors="coerce")),
                    s=8,
                    alpha=0.6,
                    label=mode,
                    color=color,
                )
            ax.set_xlabel("log P")
            ax.set_ylabel("log Teff")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(f"{PLOT_PREFIX}_logTeff_logP.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            warnings.warn(f"failed Teff-logP plot: {exc}")

        try:
            if len(residuals_long) > 0:
                bands = [b for b in BANDS_ALL if b in set(residuals_long["band"])]
                fig, axes = plt.subplots(len(bands), 1, figsize=(6, max(2, 1.45 * len(bands))), sharex=True)
                if len(bands) == 1:
                    axes = [axes]
                for ax, band in zip(axes, bands):
                    vals = residuals_long.loc[residuals_long["band"] == band, "residual_sigma"].to_numpy(float)
                    ax.hist(vals[np.isfinite(vals)], bins=40, histtype="stepfilled", alpha=0.65)
                    ax.axvline(0, color="k", lw=0.7)
                    ax.set_ylabel(band)
                axes[-1].set_xlabel("(obs-model)/sigma")
                fig.tight_layout()
                fig.savefig(f"{PLOT_PREFIX}_sed_residual_histograms.png", dpi=180)
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"failed residual plot: {exc}")

    return summary


def run_all_mw_sed_pcprior_method2_distanceprior(
    max_stars=None,
    quality_mode="all",
    overwrite=True,
    make_plots=True,
    random_seed=42,
    n_start=10,
):
    """Fit MW Cepheids with SED magnitudes + intrinsic-color PC prior.

    Parameters
    ----------
    max_stars : int or None
        If None, process the selected full table. If an integer, process the
        first ``max_stars`` eligible rows only; useful for smoke tests.
    quality_mode : {"primary", "secondary", "all"}
        "primary" selects only 7-band rows. "secondary" selects primary plus
        secondary_4plus. "all" keeps all rows for full output when max_stars is
        None; non-fit rows receive fail_reason.
    overwrite : bool
        If False and full-run outputs exist, read and return them.
    make_plots : bool
        Save basic diagnostic plots.
    """

    full_run = max_stars is None
    if full_run and (not overwrite) and FIT_OUT.exists() and RESID_OUT.exists() and SUMMARY_OUT.exists():
        df_fit = pd.read_csv(FIT_OUT)
        residuals_long = pd.read_csv(RESID_OUT)
        with open(SUMMARY_OUT, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return df_fit, residuals_long, summary

    df = read_merged_input()
    work = build_work_table(df, quality_mode=quality_mode, max_stars=max_stars)
    zps = load_zps()
    pc_coeff, pc_cov_long = load_pc_tables()

    PreconvolvedBOSZModel = sedbase._import_model()
    model = PreconvolvedBOSZModel(cache_file=sedbase.CACHE_FILE)
    bounds = bounds_for_model(model)

    result_rows = []
    residual_rows_all = []
    try:
        iterator = tqdm(work.iterrows(), total=len(work), desc="MW SED+PC MAP")
        for k, (_, row) in enumerate(iterator):
            out, residual_rows, _starts = fit_one_row(
                row,
                model,
                zps,
                pc_coeff,
                pc_cov_long,
                bounds,
                random_seed=random_seed + k,
                n_start=n_start,
            )
            result_rows.append(out)
            residual_rows_all.extend(residual_rows)
    finally:
        model.close()

    df_fit = pd.DataFrame(result_rows)
    residuals_long = pd.DataFrame(residual_rows_all)
    summary = make_summary_and_plots(df_fit, residuals_long, make_plots=make_plots)

    if full_run:
        df_fit.to_csv(FIT_OUT, index=False)
        residuals_long.to_csv(RESID_OUT, index=False)
        failures = df_fit[~df_fit["fit_success"].astype(bool)].copy()
        failures.to_csv(FAIL_OUT, index=False)
        with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        debug_prefix = SCRIPT_DIR / f"MW_debug{len(df_fit)}_SED_PCprior_method2_distanceprior"
        df_fit.to_csv(f"{debug_prefix}_MAP.csv", index=False)
        residuals_long.to_csv(f"{debug_prefix}_residuals_long.csv", index=False)
        with open(f"{debug_prefix}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return df_fit, residuals_long, summary


print("Loaded MW all-sample SED + intrinsic-color PC prior + Wang25 <d> distance-prior fitter.")
print("Call: df_fit, residuals_long, summary = run_all_mw_sed_pcprior_method2_distanceprior(...)")
