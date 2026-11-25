#!/usr/bin/env python3
"""Latest MW pure-photometry SED fit: no-quality LMC PC + loggF + RGC Fe prior.

This is a wrapper around the existing direct-emcee seven-band SED fitter.  It
keeps the old fitter intact, but replaces the scientific priors at runtime:

* PC prior: latest LMC ``temperature_six`` no-quality photometry version
  (finite values + 3 sigma clipping only), using BP-RP, RP-J, RP-H, RP-K, V-I.
* logg prior: flux-weighted gravity relation.
* metallicity prior: C-MetaLL Fe gradient as a function of R_GC, sigma=0.11 dex.

The main SED likelihood remains BP,G,RP,J,H,K,W1.  V and I are only used for
the synthetic V-I PC prior.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PHOT_FILE = SCRIPT_DIR / "MW_3640_10band_GBPRPVIJHKW1S36_phase_corrected_Bras2025_RPonly.csv"
DEFAULT_OLD_PHOT_FILE = SCRIPT_DIR / "MW_3640_PC_color_excess_from_LMC_PC.csv"
DEFAULT_WANG25_FILE = SCRIPT_DIR / "wang25x.csv"
DEFAULT_ZP_FILE = SCRIPT_DIR / "SED_pyphot_vega_fixed_zero_points_V_I_W2.csv"
DEFAULT_CACHE_FILE = SCRIPT_DIR / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_preconvolved_ND.h5"
DEFAULT_VI_CACHE_FILE = SCRIPT_DIR / "bosz_r500_msmp_a0_c0_v4_G23_V_I_preconvolved_ND.h5"

DEFAULT_PC_COEFF_FILE = SCRIPT_DIR / "LMC_temperature_six_PC_WC2023_Rv3p4_convolved_lambda_coefficients.csv"
DEFAULT_PC_COV_FILE = SCRIPT_DIR / "LMC_temperature_six_PC_WC2023_Rv3p4_convolved_lambda_residual_covariance.csv"

DEFAULT_OUT_DIR = SCRIPT_DIR / "sed_latest_noqualityPC_loggF_RGCFeH011_distanceprior_emcee_direct"

R0_KPC = 8.277
FE_GRAD_ALPHA = -0.064
FE_GRAD_BETA = 0.530
FE_GRAD_SIGMA = 0.11
LOGGF_PRIOR_SIGMA = 0.16
PC_COV_JITTER = 1.0e-8

SED_BANDS = ["BP", "G", "RP", "J", "H", "K", "W1"]
PC_COLOR_DEFS = [
    ("BP-RP", "BP", "RP"),
    ("RP-J", "RP", "J"),
    ("RP-H", "RP", "H"),
    ("RP-K", "RP", "K"),
    ("V-I", "V", "I"),
]

MAG_CANDIDATES = {
    "BP": [("BP_mag", "BP_magerr")],
    "G": [("G_mag", "G_magerr")],
    "RP": [("RP_mag", "RP_magerr")],
    "J": [
        ("J_mag_recommended_phasecorr_rp", "J_magerr_recommended_phasecorr_rp"),
        ("J_mean_mag", "J_mean_magerr"),
        ("J_mag", "J_magerr"),
    ],
    "H": [
        ("H_mag_recommended_phasecorr_rp", "H_magerr_recommended_phasecorr_rp"),
        ("H_mean_mag", "H_mean_magerr"),
        ("H_mag", "H_magerr"),
    ],
    "K": [
        ("K_mag_recommended_phasecorr_rp", "K_magerr_recommended_phasecorr_rp"),
        ("Ks_mean_mag", "Ks_mean_magerr"),
        ("K_mag", "K_magerr"),
    ],
    "W1": [("W1_mag", "W1_magerr")],
}


sys.path.insert(0, str(SCRIPT_DIR))
import jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct as directbase

mapbase = directbase.mapbase
VI_MODEL = None
VI_CACHE_FILE_FOR_PC = DEFAULT_VI_CACHE_FILE


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


def _calc_rgc_kpc(l_deg, b_deg, d_kpc, r0=R0_KPC) -> float:
    l = np.deg2rad(_finite(l_deg))
    b = np.deg2rad(_finite(b_deg))
    d = _finite(d_kpc)
    if not (np.isfinite(l) and np.isfinite(b) and np.isfinite(d) and d > 0):
        return np.nan
    dc = d * np.cos(b)
    return float(np.sqrt(r0 * r0 + dc * dc - 2.0 * r0 * dc * np.cos(l)))


def pc_mode_for_row_latest(row) -> str:
    mode = str(row.get("Mode", row.get("mode", "F"))).strip().upper()
    return "1O" if mode.startswith("1O") else "F"


def logp_for_pc_latest(row) -> float:
    period = _finite(row.get("period_used", np.nan))
    return float(np.log10(period)) if np.isfinite(period) and period > 0 else np.nan


def logp0_for_loggF_latest(row) -> tuple[float, str]:
    logp0 = _finite(row.get("logP_F_plot_Pilecki2024_MW", np.nan))
    if np.isfinite(logp0):
        return float(logp0), "logP_F_plot_Pilecki2024_MW"
    period = _finite(row.get("period_used", np.nan))
    if not (np.isfinite(period) and period > 0):
        return np.nan, "missing_period"
    logp = float(np.log10(period))
    mode = pc_mode_for_row_latest(row)
    if mode == "1O":
        return float(logp + 0.127), "fallback_1O_fundamentalized_logP_plus_0p127"
    return logp, "period_used_F"


def loggF_prior_mu_latest(row) -> float:
    logp0, _source = logp0_for_loggF_latest(row)
    return float(-0.80 * logp0 + 3.43) if np.isfinite(logp0) else np.nan


def logg_prior_mu_latest(teff, row) -> float:
    mu_loggF = loggF_prior_mu_latest(row)
    teff = _finite(teff)
    if not (np.isfinite(mu_loggF) and np.isfinite(teff) and teff > 0):
        return np.nan
    return float(mu_loggF + 4.0 * np.log10(teff / 1.0e4))


def read_merged_input_latest(
    phot_file=DEFAULT_PHOT_FILE,
    wang25_file=DEFAULT_WANG25_FILE,
    old_phot_file=DEFAULT_OLD_PHOT_FILE,
) -> pd.DataFrame:
    phot_file = Path(phot_file)
    if not phot_file.exists():
        raise FileNotFoundError(f"latest photometry file not found: {phot_file}")
    phot = pd.read_csv(phot_file, dtype={"GaiaDR3": "string", "gaia_dr3_source_id": "string", "Name": "string"})
    if "GaiaDR3" not in phot.columns and "gaia_dr3_source_id" in phot.columns:
        phot = pd.concat([phot, pd.DataFrame({"GaiaDR3": phot["gaia_dr3_source_id"]}, index=phot.index)], axis=1)
    phot = pd.concat([phot, pd.DataFrame({"gaia_key": phot["GaiaDR3"].map(_id_key)}, index=phot.index)], axis=1).copy()

    old_phot_file = Path(old_phot_file)
    if old_phot_file.exists():
        old = pd.read_csv(old_phot_file, dtype={"GaiaDR3": "string"})
        old = pd.concat([old, pd.DataFrame({"gaia_key": old["GaiaDR3"].map(_id_key)}, index=old.index)], axis=1).copy()
        keep = [c for c in ["gaia_key", "logP_F_plot_Pilecki2024_MW", "P_F_plot_Pilecki2024_MW"] if c in old.columns]
        if len(keep) > 1:
            phot = phot.merge(old[keep].drop_duplicates("gaia_key"), on="gaia_key", how="left", suffixes=("", "_oldpc"))
            for c in ["logP_F_plot_Pilecki2024_MW", "P_F_plot_Pilecki2024_MW"]:
                alt = f"{c}_oldpc"
                if alt in phot.columns:
                    if c not in phot.columns:
                        phot = pd.concat([phot, pd.DataFrame({c: phot[alt]}, index=phot.index)], axis=1)
                    else:
                        phot[c] = phot[c].where(pd.to_numeric(phot[c], errors="coerce").notna(), phot[alt])
            phot = phot.copy()

    wang = pd.read_csv(wang25_file, dtype={"source_id": "string"})
    wang = pd.concat([wang, pd.DataFrame({"gaia_key": wang["source_id"].map(_id_key)}, index=wang.index)], axis=1).copy()
    wang_keep = [
        "gaia_key",
        "source_id",
        "Cepheid",
        "<d>",
        "<d_error>",
        "l[deg]",
        "b[deg]",
        "D_w1",
        "D_w1_error",
        "weighted_D",
        "D_error",
    ]
    if "[Fe/H]" in wang.columns:
        wang_keep.append("[Fe/H]")
    wang_keep = [c for c in wang_keep if c in wang.columns]
    out = phot.merge(wang[wang_keep].drop_duplicates("gaia_key"), on="gaia_key", how="left", suffixes=("", "_wang25")).copy()

    numeric_cols = [
        "period_used",
        "logP_F_plot_Pilecki2024_MW",
        "<d>",
        "<d_error>",
        "l[deg]",
        "b[deg]",
    ]
    for pairs in MAG_CANDIDATES.values():
        for mcol, ecol in pairs:
            numeric_cols.extend([mcol, ecol])
    numeric_present = [col for col in sorted(set(numeric_cols)) if col in out.columns]
    if numeric_present:
        converted = pd.DataFrame(
            {col: pd.to_numeric(out[col], errors="coerce") for col in numeric_present},
            index=out.index,
        )
        out = pd.concat([out.drop(columns=numeric_present), converted], axis=1).copy()

    rgc_wang25_kpc = [
        _calc_rgc_kpc(l, b, d) for l, b, d in zip(out.get("l[deg]", np.nan), out.get("b[deg]", np.nan), out.get("<d>", np.nan))
    ]
    rgc_wang25_kpc = pd.Series(rgc_wang25_kpc, index=out.index, dtype=float)
    feh_prior_mu = FE_GRAD_ALPHA * rgc_wang25_kpc + FE_GRAD_BETA
    feh_wang25_original = pd.to_numeric(out.get("[Fe/H]", np.nan), errors="coerce")
    extra = pd.DataFrame(
        {
            "RGC_Wang25_kpc": rgc_wang25_kpc,
            "FeH_prior_mu_RGC_gradient": feh_prior_mu,
            "FeH_prior_sigma": FE_GRAD_SIGMA,
            "FeH_gradient_alpha": FE_GRAD_ALPHA,
            "FeH_gradient_beta": FE_GRAD_BETA,
            "FeH_gradient_rms": FE_GRAD_SIGMA,
            "FeH_Wang25_original": feh_wang25_original,
            "[Fe/H]": feh_prior_mu,
        },
        index=out.index,
    )
    replace_cols = [c for c in extra.columns if c in out.columns]
    out = pd.concat([out.drop(columns=replace_cols), extra], axis=1).copy()

    return out


def load_zps_latest(zp_file=DEFAULT_ZP_FILE) -> dict[str, float]:
    zp_file = Path(zp_file)
    if not zp_file.exists():
        raise FileNotFoundError(f"zero-point file not found: {zp_file}")
    zp = pd.read_csv(zp_file)
    required = {"band", "ZP_X_cache_raw_integral"}
    missing = required - set(zp.columns)
    if missing:
        raise ValueError(f"{zp_file} lacks {sorted(missing)}")
    zps = {str(r["band"]): float(r["ZP_X_cache_raw_integral"]) for _, r in zp.iterrows()}
    needed = set(SED_BANDS + ["V", "I"])
    missing_bands = sorted(needed - set(zps))
    if missing_bands:
        raise ValueError(f"{zp_file} missing zero points for {missing_bands}; V-I PC prior cannot be evaluated")
    return zps


def _band_value(row, band: str):
    for mcol, ecol in MAG_CANDIDATES[band]:
        mag = _finite(row.get(mcol, np.nan))
        err = _finite(row.get(ecol, np.nan))
        if np.isfinite(mag) and np.isfinite(err) and err > 0:
            return mcol, ecol, mag, err
    return "", "", np.nan, np.nan


def has_band_latest(row, band: str) -> bool:
    return np.isfinite(_band_value(row, band)[2])


def assign_quality_tier_latest(row) -> str:
    has = {band: has_band_latest(row, band) for band in SED_BANDS}
    if all(has[b] for b in SED_BANDS):
        return "primary_7band"
    if all(has[b] for b in ["BP", "G", "RP"]) and any(has[b] for b in ["J", "H", "K", "W1"]):
        return "secondary_4plus"
    return "not_fit"


def row_sed_vector_latest(row):
    used, obs, err = [], [], []
    for band in SED_BANDS:
        _mcol, _ecol, mag, magerr = _band_value(row, band)
        if np.isfinite(mag) and np.isfinite(magerr) and magerr > 0:
            used.append(band)
            obs.append(mag)
            err.append(magerr)
    return used, np.asarray(obs, dtype=float), np.asarray(err, dtype=float)


def base_fail_reason_latest(row, quality_mode: str) -> str:
    tier = row.get("quality_tier", "not_fit")
    if quality_mode == "primary" and tier != "primary_7band":
        return "not_primary_7band"
    if quality_mode in ("secondary", "all") and tier == "not_fit":
        return "missing_GBPRP_or_no_JHKW1"
    for col in ["<d>", "<d_error>", "period_used", "RGC_Wang25_kpc", "FeH_prior_mu_RGC_gradient"]:
        val = _finite(row.get(col, np.nan))
        if not np.isfinite(val):
            return f"missing_{col}"
    if _finite(row.get("<d>")) <= 0 or _finite(row.get("<d_error>")) <= 0:
        return "bad_distance_prior"
    if _finite(row.get("period_used")) <= 0:
        return "bad_period"
    return ""


def load_pc_tables_latest(pc_coeff_file=DEFAULT_PC_COEFF_FILE, pc_cov_file=DEFAULT_PC_COV_FILE):
    coeff = pd.read_csv(pc_coeff_file)
    cov = pd.read_csv(pc_cov_file)
    return coeff, cov


def build_pc_prior_latest(row, pc_coeff: pd.DataFrame, pc_cov_long: pd.DataFrame):
    mode_pc = pc_mode_for_row_latest(row)
    logp = logp_for_pc_latest(row)
    if not np.isfinite(logp):
        raise ValueError("missing logP for PC prior")

    rows = []
    colors = []
    for color, band1, band2 in PC_COLOR_DEFS:
        sub = pc_coeff[
            (pc_coeff["sample"].astype(str).str.upper() == mode_pc)
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
                "sample": mode_pc,
                "band1": band1,
                "band2": band2,
                "band": band1,
                "color": color,
                "logP_used": logp,
                "logP_pc_used": logp,
                "C_PC": c_pc,
                "sigma_mag": float(r["sigma_mag"]),
                "coeff_var": coeff_var,
                "intercept": float(r["intercept"]),
                "slope": float(r["slope"]),
            }
        )
        colors.append(color)

    cov = np.full((len(colors), len(colors)), np.nan, dtype=float)
    cov_sub = pc_cov_long[pc_cov_long["sample"].astype(str).str.upper() == mode_pc]
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors):
            hit = cov_sub[(cov_sub["color_i"].astype(str) == c1) & (cov_sub["color_j"].astype(str) == c2)]
            if len(hit) == 1:
                cov[i, j] = float(hit.iloc[0]["covariance"])
    if np.any(~np.isfinite(cov)):
        sig = np.asarray([r["sigma_mag"] for r in rows], dtype=float)
        cov = np.diag(sig * sig)

    coeff_diag = np.asarray([r["coeff_var"] for r in rows], dtype=float)
    cov = cov + np.diag(coeff_diag + PC_COV_JITTER)
    pc_prior = pd.DataFrame(rows)
    pc_prior["sigma_PC_diag"] = np.sqrt(np.diag(cov))
    return pc_prior, cov


def intrinsic_color_syn_latest(theta, model, zps, pc_prior):
    teff, logg, mh, _av, _rv, _logd, _logr = theta
    used_main = sorted(
        b
        for b in set(pc_prior["band1"].astype(str)).union(set(pc_prior["band2"].astype(str)))
        if b in model.band_index
    )
    d_main = band_d_interpolated_latest(model, teff, logg, mh, used_bands=used_main) if used_main else None
    d_vi = None
    colors = []
    for _, r in pc_prior.iterrows():
        b1 = str(r["band1"])
        b2 = str(r["band2"])

        def mag_for_band(band):
            nonlocal d_vi
            if band not in zps:
                raise KeyError(f"zero-point table lacks band {band}; cannot evaluate {r['color']}")
            if band in model.band_index:
                return -2.5 * np.log10(d_main[model.band_index[band]]) + zps[band]
            if band in {"V", "I"}:
                vi_model = get_vi_model_latest()
                if band not in vi_model.band_index:
                    raise KeyError(f"V/I cache lacks band {band}; cannot evaluate {r['color']}")
                if d_vi is None:
                    d_vi = band_d_interpolated_latest(vi_model, teff, logg, mh, used_bands=["V", "I"])
                return -2.5 * np.log10(d_vi[vi_model.band_index[band]]) + zps[band]
            raise KeyError(f"model cache lacks band {band}; cannot evaluate {r['color']}")

        m1 = mag_for_band(b1)
        m2 = mag_for_band(b2)
        colors.append(float(m1 - m2))
    return np.asarray(colors, dtype=float)


def get_vi_model_latest():
    global VI_MODEL
    if VI_MODEL is None:
        PreconvolvedBOSZModel = mapbase.sedbase._import_model()
        VI_MODEL = PreconvolvedBOSZModel(cache_file=VI_CACHE_FILE_FOR_PC)
    return VI_MODEL


def pc_prior_residuals_latest(theta, model, zps, pc_prior, pc_cov):
    c_syn = intrinsic_color_syn_latest(theta, model, zps, pc_prior)
    delta = c_syn - pc_prior["C_PC"].to_numpy(float)
    return mapbase.stable_whiten(delta, pc_cov)


def band_d_interpolated_latest(model, teff, logg, mh, used_bands=None):
    vertices, weights = model._simplex_weights(float(teff), float(logg), float(mh))
    if not hasattr(model, "_sed_d_vertex_cache"):
        model._sed_d_vertex_cache = {}
    d_interp = np.zeros(len(model.bands), dtype=float)
    for vertex, weight in zip(vertices, weights):
        vertex = int(vertex)
        if vertex not in model._sed_d_vertex_cache:
            model._sed_d_vertex_cache[vertex] = model.h5["D"][vertex, :].astype(np.float32)
        d_interp += float(weight) * model._sed_d_vertex_cache[vertex].astype(float)
    if used_bands is None:
        check = d_interp
    else:
        idx = [model.band_index[b] for b in used_bands]
        check = d_interp[idx]
    if np.any(~np.isfinite(check)) or np.any(check <= 0):
        raise ValueError("non-positive interpolated intrinsic band integral in used bands")
    return d_interp


def band_n_interpolated_latest(model, teff, logg, mh, rv, av, used_bands=None):
    vertices, weights = model._simplex_weights(float(teff), float(logg), float(mh))
    ir0, ir1, tr = model._bracket(model.rv_grid, float(rv))
    ia0, ia1, ta = model._bracket(model.av_grid, float(av))
    n_interp = np.zeros(len(model.bands), dtype=float)
    if not hasattr(model, "_sed_n_vertex_cache"):
        model._sed_n_vertex_cache = {}

    for vertex, weight in zip(vertices, weights):
        vertex = int(vertex)
        if vertex not in model._sed_n_vertex_cache:
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
    if used_bands is None:
        check = n_interp
    else:
        idx = [model.band_index[b] for b in used_bands]
        check = n_interp[idx]
    if np.any(~np.isfinite(check)) or np.any(check <= 0):
        raise ValueError("non-positive interpolated extincted band integral in used bands")
    return n_interp


def model_mags_latest(theta, model, zps, sed_bands):
    teff, logg, mh, av, rv, logd, logr = theta
    n = band_n_interpolated_latest(model, teff, logg, mh, rv, av, used_bands=sed_bands)
    radius_cm = (10.0**logr) * mapbase.sedbase.R_SUN_CM
    distance_cm = (10.0**logd) * mapbase.sedbase.KPC_CM
    scale = (radius_cm / distance_cm) ** 2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bad R/D scale")
    out = []
    for band in sed_bands:
        i = model.band_index[band]
        out.append(float(-2.5 * np.log10(n[i]) + zps[band] - 2.5 * np.log10(scale)))
    return np.asarray(out, dtype=float)


def prior_residuals_latest(theta, row):
    teff, logg, mh, _av, _rv, logd, _logr = theta
    res = []

    mu_loggF = loggF_prior_mu_latest(row)
    if np.isfinite(mu_loggF) and np.isfinite(teff) and teff > 0:
        loggF = logg - 4.0 * np.log10(teff / 1.0e4)
        res.append((loggF - mu_loggF) / LOGGF_PRIOR_SIGMA)

    mu_mh = _finite(row.get("FeH_prior_mu_RGC_gradient", row.get("[Fe/H]", np.nan)))
    if np.isfinite(mu_mh):
        res.append((mh - mu_mh) / FE_GRAD_SIGMA)

    d_prior = _finite(row.get("<d>", np.nan))
    d_err = _finite(row.get("<d_error>", np.nan))
    d_model = 10.0 ** logd
    if np.isfinite(d_prior) and np.isfinite(d_err) and d_prior > 0 and d_err > 0:
        res.append((d_model - d_prior) / d_err)

    return np.asarray(res, dtype=float)


def residual_latest(theta, row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds):
    theta = np.asarray(theta, dtype=float)
    lower, upper = bounds["lower"], bounds["upper"]
    n_bad = len(sed_obs) + len(PC_COLOR_DEFS) + 3
    if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
        return np.full(n_bad, 1e6)
    try:
        sed_pred = model_mags_latest(theta, model, zps, sed_bands)
        sed_res = (sed_obs - sed_pred) / sed_err
        pc_res, _ = pc_prior_residuals_latest(theta, model, zps, pc_prior, pc_cov)
        prior_res = prior_residuals_latest(theta, row)
        return np.concatenate([sed_res, pc_res, prior_res])
    except Exception:
        return np.full(n_bad, 1e6)


def summarize_samples_latest(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
    out = directbase.summarize_samples_original(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin)
    logp0, logp0_source = logp0_for_loggF_latest(row)
    out.update(
        {
            "pc_prior_version": "LMC_temperature_six_WC2023_Rv3p4_no_quality_3sigma",
            "pc_quality_filter": "finite_only_plus_3sigma_clip",
            "mh_prior_mode": "RGC_Fe_gradient_CMetaLL_Table4",
            "mh_prior_sigma": FE_GRAD_SIGMA,
            "MH_prior": _finite(row.get("FeH_prior_mu_RGC_gradient", np.nan)),
            "MH_prior_sigma": FE_GRAD_SIGMA,
            "FeH_Wang25_original": _finite(row.get("FeH_Wang25_original", np.nan)),
            "RGC_Wang25_kpc": _finite(row.get("RGC_Wang25_kpc", np.nan)),
            "FeH_prior_mu_RGC_gradient": _finite(row.get("FeH_prior_mu_RGC_gradient", np.nan)),
            "FeH_gradient_alpha": FE_GRAD_ALPHA,
            "FeH_gradient_beta": FE_GRAD_BETA,
            "FeH_gradient_rms": FE_GRAD_SIGMA,
            "logP_pc_used": logp_for_pc_latest(row),
            "logP0_used": logp0,
            "logP0_source": logp0_source,
            "loggF_prior_mu": loggF_prior_mu_latest(row),
            "loggF_prior_sigma": LOGGF_PRIOR_SIGMA,
            "logg_prior_sigma": LOGGF_PRIOR_SIGMA,
            "uses_latest_PC_prior": True,
            "uses_loggF_prior": True,
            "uses_RGC_FeH_prior": True,
            "uses_distance_prior": True,
        }
    )
    teff = out.get("Teff_p50", np.nan)
    logg = out.get("logg_p50", np.nan)
    if np.isfinite(teff) and teff > 0 and np.isfinite(logg):
        loggF = logg - 4.0 * np.log10(teff / 1.0e4)
        out["loggF_p50"] = float(loggF)
        out["loggF_prior_delta_sigma"] = float((loggF - out["loggF_prior_mu"]) / LOGGF_PRIOR_SIGMA)
    else:
        out["loggF_p50"] = np.nan
        out["loggF_prior_delta_sigma"] = np.nan
    mh = out.get("MH_p50", np.nan)
    mu_mh = out.get("FeH_prior_mu_RGC_gradient", np.nan)
    out["MH_prior_delta_sigma"] = float((mh - mu_mh) / FE_GRAD_SIGMA) if np.isfinite(mh) and np.isfinite(mu_mh) else np.nan
    return out


def _apply_latest_patches(
    out_dir,
    phot_file,
    wang25_file,
    old_phot_file,
    zp_file,
    cache_file,
    vi_cache_file,
    pc_coeff_file,
    pc_cov_file,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch mapbase data/model settings.
    mapbase.MW_PHOT_FILE = Path(phot_file)
    mapbase.WANG25_FILE = Path(wang25_file)
    mapbase.ZP_FILE = Path(zp_file)
    mapbase.PC_COEFF_FILE = Path(pc_coeff_file)
    mapbase.PC_COV_FILE = Path(pc_cov_file)
    mapbase.BANDS_ALL = list(SED_BANDS)
    mapbase.REQUIRED_OPTICAL = ["BP", "G", "RP"]
    mapbase.NIR_MIR_BANDS = ["J", "H", "K", "W1"]
    mapbase.PC_COLOR_BANDS = [c for c, _b1, _b2 in PC_COLOR_DEFS]
    mapbase.LOGG_PRIOR_SIGMA = LOGGF_PRIOR_SIGMA
    mapbase.FEH_PRIOR_SIGMA = FE_GRAD_SIGMA
    mapbase.MAG_COL = {b: MAG_CANDIDATES[b][0][0] for b in SED_BANDS}
    mapbase.MAGERR_COL = {b: MAG_CANDIDATES[b][0][1] for b in SED_BANDS}
    cache_file = Path(cache_file)
    vi_cache_file = Path(vi_cache_file)
    mapbase.sedbase.CACHE_FILE = cache_file
    global VI_MODEL, VI_CACHE_FILE_FOR_PC
    if any(color == "V-I" for color, _, _ in PC_COLOR_DEFS) and not vi_cache_file.exists():
        raise FileNotFoundError(
            f"V/I pre-convolved cache is required for the V-I PC prior but was not found: {vi_cache_file}"
        )
    VI_CACHE_FILE_FOR_PC = vi_cache_file
    VI_MODEL = None

    mapbase.read_merged_input = lambda: read_merged_input_latest(phot_file, wang25_file, old_phot_file)
    mapbase.load_zps = lambda: load_zps_latest(zp_file)
    mapbase.load_pc_tables = lambda: load_pc_tables_latest(pc_coeff_file, pc_cov_file)
    mapbase.pc_mode_for_row = pc_mode_for_row_latest
    mapbase.logp_for_pc = logp_for_pc_latest
    mapbase.logg_prior_mu = logg_prior_mu_latest
    mapbase.prior_residuals = prior_residuals_latest
    mapbase.row_sed_vector = row_sed_vector_latest
    mapbase.has_band = has_band_latest
    mapbase.assign_quality_tier = assign_quality_tier_latest
    mapbase.base_fail_reason = base_fail_reason_latest
    mapbase.build_pc_prior = build_pc_prior_latest
    mapbase.intrinsic_color_syn = intrinsic_color_syn_latest
    mapbase.pc_prior_residuals = pc_prior_residuals_latest
    mapbase.model_mags = model_mags_latest
    mapbase.residual = residual_latest

    # Patch direct-emcee output globals.
    directbase.OUT_DIR = out_dir
    directbase.SELECTED_OUT = out_dir / "MW_SED_latest_noqualityPC_loggF_RGCFeH011_selected.csv"
    directbase.SUMMARY_OUT = out_dir / "MW_SED_latest_noqualityPC_loggF_RGCFeH011_summary.csv"
    directbase.FAIL_OUT = out_dir / "MW_SED_latest_noqualityPC_loggF_RGCFeH011_failures.csv"
    directbase.RUNINFO_OUT = out_dir / "MW_SED_latest_noqualityPC_loggF_RGCFeH011_runinfo.json"
    if not hasattr(directbase, "summarize_samples_original"):
        directbase.summarize_samples_original = directbase.summarize_samples
    directbase.summarize_samples = summarize_samples_latest
    directbase._WORKER_MODEL = None
    directbase._WORKER_ZPS = None
    directbase._WORKER_PC_COEFF = None
    directbase._WORKER_PC_COV_LONG = None
    directbase._WORKER_BOUNDS = None


def run_mw_sed_latest_pc_loggF_rgcFeH_distanceprior_emcee_direct(
    n_stars=30,
    quality_mode="primary",
    pc_coeff_file=DEFAULT_PC_COEFF_FILE,
    pc_cov_file=DEFAULT_PC_COV_FILE,
    phot_file=DEFAULT_PHOT_FILE,
    wang25_file=DEFAULT_WANG25_FILE,
    old_phot_file=DEFAULT_OLD_PHOT_FILE,
    zp_file=DEFAULT_ZP_FILE,
    cache_file=DEFAULT_CACHE_FILE,
    vi_cache_file=DEFAULT_VI_CACHE_FILE,
    out_dir=DEFAULT_OUT_DIR,
    mh_prior_mode="rgc_fe_gradient",
    mh_prior_sigma=FE_GRAD_SIGMA,
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
    """Run the latest pure-photometry SED fit.

    ``make_plots`` and ``resume`` are accepted for notebook-interface
    compatibility.  The underlying direct-emcee fitter always writes summary
    CSVs and saves per-source SED plots for saved diagnostic samples.
    """
    if str(mh_prior_mode) != "rgc_fe_gradient":
        raise ValueError("This latest method only supports mh_prior_mode='rgc_fe_gradient'")
    if abs(float(mh_prior_sigma) - FE_GRAD_SIGMA) > 1.0e-12:
        raise ValueError("This latest method fixes mh_prior_sigma=0.11 dex")

    _apply_latest_patches(
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

    runinfo_path = Path(out_dir) / "MW_SED_latest_noqualityPC_loggF_RGCFeH011_runinfo_extra.json"
    runinfo = {
        "pc_prior_version": "LMC_temperature_six_WC2023_Rv3p4_no_quality_3sigma",
        "pc_quality_filter": "finite_only_plus_3sigma_clip",
        "pc_coeff_file": str(pc_coeff_file),
        "pc_cov_file": str(pc_cov_file),
        "phot_file": str(phot_file),
        "zp_file": str(zp_file),
        "cache_file": str(cache_file),
        "vi_cache_file": str(vi_cache_file),
        "mh_prior_mode": "RGC_Fe_gradient_CMetaLL_Table4",
        "mh_prior_sigma": FE_GRAD_SIGMA,
        "loggF_prior_sigma": LOGGF_PRIOR_SIGMA,
        "sed_bands": SED_BANDS,
        "pc_color_defs": PC_COLOR_DEFS,
    }
    with open(runinfo_path, "w", encoding="utf-8") as f:
        json.dump(runinfo, f, ensure_ascii=False, indent=2)

    return selected, summary, samples


print("Loaded latest MW pure-photometry SED fitter: no-quality LMC PC + loggF + RGC Fe prior.")
print("Call: selected, latest_summary, latest_samples = run_mw_sed_latest_pc_loggF_rgcFeH_distanceprior_emcee_direct(...)")
