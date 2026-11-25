#!/usr/bin/env python3
"""Filtered latest MW pure-photometry SED fitter with source-dependent floors.

This wrapper extends
``jupyter_mw_sed_latest_pc_source_dependent_floor_distanceprior_emcee_direct.py``
with source/band quality filters and a missing-Wang25 fallback:

* BP/G/RP can be restricted by BP magnitude and Gaia source type.
* JHK 2MASS-based measurements are filtered per band by ph_qual and cc_flg.
* AllWISE/WISE-AllSky W1 profile-fit measurements are filtered by W1 quality.
* Sources without Wang25 distances are allowed; they use no distance prior and
  a solar metallicity prior center, [M/H]=0.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import jupyter_mw_sed_latest_pc_source_dependent_floor_distanceprior_emcee_direct as sfloor


DEFAULT_OUT_ROOT = SCRIPT_DIR / "outputs" / "sed_latest_source_dependent_floor_filtered"

SED_BANDS = list(sfloor.SED_BANDS)
NEW_QUALITY_MODE = "GBPRP_plus_3of4_JHKW1"

DEFAULT_EXTINCTION_LAW = "G23"
DEFAULT_EXTINCTION_LAW_CACHE_MAP = {
    "G23": SCRIPT_DIR / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_preconvolved_ND.h5",
    "WC2019": SCRIPT_DIR / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_WC2019_preconvolved_ND.h5",
    "F19": SCRIPT_DIR / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_F19_preconvolved_ND.h5",
    "F99": SCRIPT_DIR / "bosz_r500_msmp_a0_c0_v4_Rv2p0_6p0_F99_preconvolved_ND.h5",
}
DEFAULT_A550_R550_CACHE_MAP = {
    "G23": SCRIPT_DIR / "bosz_r500_G23_A550_R550_preconvolved_ND.h5",
    "WC2019": SCRIPT_DIR / "bosz_r500_WC2019_A550_R550_preconvolved_ND.h5",
    "F19": SCRIPT_DIR / "bosz_r500_F19_A550_R550_preconvolved_ND.h5",
    "F99": SCRIPT_DIR / "bosz_r500_F99_A550_R550_preconvolved_ND.h5",
}
EXTINCTION_LAW_ALIASES = {
    "G23": "G23",
    "GORDON23": "G23",
    "GORDON2023": "G23",
    "WC19": "WC2019",
    "WC2019": "WC2019",
    "WANGCHEN2019": "WC2019",
    "F19": "F19",
    "FITZPATRICK19": "F19",
    "FITZPATRICK2019": "F19",
    "F99": "F99",
    "FITZPATRICK99": "F99",
    "FITZPATRICK1999": "F99",
}

ORIGINAL_BAND_VALUE = None
ORIGINAL_BUILD_WORK_TABLE = None
ORIGINAL_PRIOR_RESIDUALS = None
ORIGINAL_SUMMARIZE_WITH_MISSING = None
ORIGINAL_DRAW_ONE_PRIOR_POSITION = None
ORIGINAL_INITIALIZE_WALKERS = None
ACTIVE_FILTER_CONFIG = {}
ACTIVE_USE_DISTANCE_PRIOR = True
ACTIVE_PC_PRIOR_ENABLED = True
ACTIVE_PC_PRIOR_SCALE = 1.0
ACTIVE_PC_PRIOR_MODE = "1sigma"
ACTIVE_EXTINCTION_PARAMETERIZATION = "legacy_Av_Rv"
ACTIVE_BAND_EXTINCTION_DRAWS = 512
ACTIVE_SAVE_BAND_EXTINCTION_DRAWS = False
ACTIVE_COMPUTE_BAND_EXTINCTION = True
NATIVE_EXTINCTION_PARAMETERIZATIONS = {"A550_R550", "Av_Rv"}


def _validate_native_g23_cache(cache_file, required_dust_extinction_version="1.7"):
    """Validate that a cache is the expected native G23 A550/R550 product."""
    cache_file = Path(cache_file).expanduser().resolve()
    with h5py.File(cache_file, "r") as h5:
        attrs = {str(key): h5.attrs[key] for key in h5.attrs.keys()}
        datasets = set(h5.keys())

    law = str(attrs.get("extinction_law", "")).strip().upper()
    normalization = str(attrs.get("normalization", "")).strip()
    try:
        versions = json.loads(str(attrs.get("software_versions_json", "{}")))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid software_versions_json in native cache {cache_file}: {exc}"
        ) from exc
    actual_version = str(versions.get("dust_extinction", "")).strip()
    required = None if required_dust_extinction_version is None else str(required_dust_extinction_version).strip()

    errors = []
    if law != "G23":
        errors.append(f"extinction_law={law!r}, expected 'G23'")
    if normalization != "A(lambda)/A550":
        errors.append(
            f"normalization={normalization!r}, expected 'A(lambda)/A550'"
        )
    missing = {"a550_grid", "r550_grid"} - datasets
    if missing:
        errors.append(f"missing datasets: {sorted(missing)}")
    if {"av_grid", "rv_grid"} & datasets:
        errors.append("legacy av_grid/rv_grid datasets are present")
    if required and actual_version != required:
        errors.append(
            f"dust-extinction={actual_version!r}, expected {required!r}"
        )
    if errors:
        raise ValueError(
            f"native G23 cache validation failed for {cache_file}: " + "; ".join(errors)
        )
    return {
        "cache_file": str(cache_file),
        "extinction_law": law,
        "normalization": normalization,
        "dust_extinction_version": actual_version,
        "required_dust_extinction_version": required,
        "software_versions": versions,
        "validated": True,
    }


def _normalize_extinction_law(extinction_law):
    key = str(extinction_law).strip().upper().replace("_", "").replace("-", "").replace(" ", "")
    if key in EXTINCTION_LAW_ALIASES:
        return EXTINCTION_LAW_ALIASES[key]
    raise ValueError(
        "Unknown extinction_law="
        f"{extinction_law!r}. Available built-in laws: "
        f"{', '.join(sorted(DEFAULT_EXTINCTION_LAW_CACHE_MAP))}. "
        "Use extinction_law_cache_map to add a custom law, or cache_file to override directly."
    )


def _normalize_extinction_parameterization(value="A550_R550"):
    key = str(value).strip().lower().replace("-", "_").replace("/", "_")
    aliases = {
        "legacy_av_rv": "legacy_Av_Rv",
        "legacy": "legacy_Av_Rv",
        "av_rv": "Av_Rv",
        "avrv": "Av_Rv",
        "native_av_rv": "Av_Rv",
        "g23_av_rv": "Av_Rv",
        "a550_r550": "A550_R550",
        "a550r550": "A550_R550",
        "native_a550_r550": "A550_R550",
    }
    if key not in aliases:
        raise ValueError(
            "extinction_parameterization must be 'A550_R550', 'Av_Rv', "
            "or the backward-compatible 'legacy_Av_Rv'"
        )
    return aliases[key]


def _validate_extinction_cache_file(path, required_bands=None, extinction_parameterization="A550_R550"):
    """Fail fast when a preconvolved cache is incomplete or contains zero integrals."""
    import h5py

    path = Path(path)
    required_bands = list(SED_BANDS if required_bands is None else required_bands)
    parameterization = _normalize_extinction_parameterization(extinction_parameterization)
    uses_native_cache = parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS
    grid_names = ("r550_grid", "a550_grid") if uses_native_cache else ("rv_grid", "av_grid")
    required_datasets = {"params", "D", "N", grid_names[0], grid_names[1], "bands"}
    if uses_native_cache:
        required_datasets.add("native_shape_for_r550")
    try:
        with h5py.File(path, "r") as h5:
            missing_datasets = sorted(required_datasets.difference(h5.keys()))
            if missing_datasets:
                raise ValueError(f"missing datasets: {missing_datasets}")
            if uses_native_cache:
                forbidden = sorted({"rv_grid", "av_grid"}.intersection(h5.keys()))
                if forbidden:
                    raise ValueError(
                        "native G23 cache contains forbidden legacy datasets: "
                        f"{forbidden}"
                    )

            bands = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["bands"][:]]
            missing_bands = [band for band in required_bands if band not in bands]
            if missing_bands:
                raise ValueError(f"missing required SED bands: {missing_bands}; cache bands={bands}")

            band_indices = [bands.index(band) for band in required_bands]
            d_values = np.asarray(h5["D"][:, band_indices], dtype=float)
            if d_values.size == 0 or np.any(~np.isfinite(d_values)) or np.any(d_values <= 0):
                n_bad = int(np.sum((~np.isfinite(d_values)) | (d_values <= 0)))
                raise ValueError(
                    f"intrinsic D integrals are not finite and positive in required bands; "
                    f"bad={n_bad}/{d_values.size}, min={np.nanmin(d_values):.6g}"
                )

            n_shape = h5["N"].shape
            if len(n_shape) != 4 or n_shape[0] != h5["D"].shape[0] or n_shape[3] != len(bands):
                raise ValueError(f"inconsistent N shape: N={n_shape}, D={h5['D'].shape}, bands={len(bands)}")

            vertex_indices = sorted({0, n_shape[0] // 2, n_shape[0] - 1})
            rv_indices = sorted({0, n_shape[1] // 2, n_shape[1] - 1})
            av_indices = sorted({0, n_shape[2] // 2, n_shape[2] - 1})
            for iv in vertex_indices:
                for irv in rv_indices:
                    for iav in av_indices:
                        for ib in band_indices:
                            value = float(h5["N"][iv, irv, iav, ib])
                            if not np.isfinite(value) or value <= 0:
                                raise ValueError(
                                    "extincted N integral is not finite and positive at "
                                    f"vertex={iv}, rv_index={irv}, av_index={iav}, band={bands[ib]}: {value}"
                                )
    except Exception as exc:
        raise ValueError(f"invalid preconvolved extinction cache {path}: {exc}") from exc
    return path


def _resolve_extinction_cache_file(
    extinction_law=DEFAULT_EXTINCTION_LAW,
    cache_file=None,
    extinction_law_cache_map=None,
    extinction_parameterization="A550_R550",
    build_native_cache_if_missing=True,
    native_cache_build_kwargs=None,
):
    parameterization = _normalize_extinction_parameterization(extinction_parameterization)
    law = _normalize_extinction_law(extinction_law)
    if cache_file is not None:
        path = Path(cache_file)
        if not path.exists():
            raise FileNotFoundError(f"explicit cache_file does not exist: {path}")
        _validate_extinction_cache_file(path, extinction_parameterization=parameterization)
        return str(extinction_law), law, path, "explicit_cache_file"

    uses_native_cache = parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS
    cache_map = dict(DEFAULT_A550_R550_CACHE_MAP if uses_native_cache else DEFAULT_EXTINCTION_LAW_CACHE_MAP)
    if extinction_law_cache_map is not None:
        for key, value in dict(extinction_law_cache_map).items():
            cache_map[_normalize_extinction_law(key) if str(key).upper() in EXTINCTION_LAW_ALIASES else str(key).strip()] = Path(value)
    if law not in cache_map:
        raise ValueError(f"extinction_law={law!r} has no cache path in extinction_law_cache_map")
    path = Path(cache_map[law])
    if not path.exists() and uses_native_cache and build_native_cache_if_missing:
        from bosz_a550_r550_preconvolved import build_native_a550_r550_cache

        kwargs = {} if native_cache_build_kwargs is None else dict(native_cache_build_kwargs)
        forbidden = {"law", "cache_file", "overwrite"}.intersection(kwargs)
        if forbidden:
            raise ValueError(f"native_cache_build_kwargs cannot override {sorted(forbidden)}")
        print(f"Native A550/R550 cache is missing; building and validating it now: {path}")
        build_native_a550_r550_cache(law=law, cache_file=path, overwrite=False, **kwargs)
    if not path.exists():
        hint = " Set build_native_cache_if_missing=True to build it on first use." if uses_native_cache else ""
        raise FileNotFoundError(f"cache for extinction_law={law!r} does not exist: {path}.{hint}")
    _validate_extinction_cache_file(path, extinction_parameterization=parameterization)
    return str(extinction_law), law, path, "extinction_law_map"


def _extinction_law_slug(extinction_law=DEFAULT_EXTINCTION_LAW, cache_file=None):
    if cache_file is not None:
        return f"extCustom{hashlib.sha1(str(cache_file).encode('utf-8')).hexdigest()[:8]}"
    try:
        return f"ext{_slug_value(_normalize_extinction_law(extinction_law))}"
    except Exception:
        return f"ext{_slug_value(extinction_law)}"


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


def _first_flag_char(x) -> str:
    s = _clean(x)
    return s[0] if s else ""


def _as_allowed_tuple(values):
    if values is None:
        return None
    return tuple(str(v).strip() for v in values)


def _resolve_pc_prior_config(pc_prior_mode="1sigma", pc_prior_scale=None):
    """Return (enabled, sigma_multiplier, normalized_mode)."""
    if pc_prior_scale is not None:
        scale = float(pc_prior_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("pc_prior_scale must be a positive finite number")
        return True, scale, f"{scale:g}sigma"

    mode = str(pc_prior_mode).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "nopc": "none",
        "nopr": "none",
        "1": "1sigma",
        "1sigma": "1sigma",
        "sigma1": "1sigma",
        "onesigma": "1sigma",
        "2": "2sigma",
        "2sigma": "2sigma",
        "sigma2": "2sigma",
        "twosigma": "2sigma",
    }
    norm = aliases.get(mode)
    if norm is None:
        raise ValueError("pc_prior_mode must be one of 'none', '1sigma', '2sigma', or set pc_prior_scale")
    if norm == "none":
        return False, np.inf, "none"
    if norm == "1sigma":
        return True, 1.0, "1sigma"
    if norm == "2sigma":
        return True, 2.0, "2sigma"
    raise RuntimeError(f"unhandled pc_prior_mode={pc_prior_mode!r}")


def _set_active_pc_prior_config(pc_prior_mode="1sigma", pc_prior_scale=None):
    global ACTIVE_PC_PRIOR_ENABLED, ACTIVE_PC_PRIOR_SCALE, ACTIVE_PC_PRIOR_MODE
    enabled, scale, mode = _resolve_pc_prior_config(pc_prior_mode, pc_prior_scale)
    ACTIVE_PC_PRIOR_ENABLED = bool(enabled)
    ACTIVE_PC_PRIOR_SCALE = float(scale)
    ACTIVE_PC_PRIOR_MODE = str(mode)


def _pc_prior_slug(pc_prior_mode="1sigma", pc_prior_scale=None) -> str:
    enabled, _scale, mode = _resolve_pc_prior_config(pc_prior_mode, pc_prior_scale)
    if not enabled:
        return "pcNone"
    return f"pc{_slug_value(mode)}"


def _patch_pc_prior_strength(latest_module, mapbase=None):
    """Patch latest PC residuals to allow no-PC or wider-sigma PC priors."""
    if not hasattr(latest_module, "_pc_prior_residuals_latest_unscaled"):
        latest_module._pc_prior_residuals_latest_unscaled = latest_module.pc_prior_residuals_latest
    original_pc = latest_module._pc_prior_residuals_latest_unscaled

    def pc_prior_residuals_latest_scaled(theta, model, zps, pc_prior, pc_cov):
        if not ACTIVE_PC_PRIOR_ENABLED:
            return np.zeros(0, dtype=float), "pc_prior_disabled"
        cov = np.asarray(pc_cov, dtype=float) * (float(ACTIVE_PC_PRIOR_SCALE) ** 2)
        return original_pc(theta, model, zps, pc_prior, cov)

    latest_module.pc_prior_residuals_latest = pc_prior_residuals_latest_scaled
    if mapbase is not None:
        mapbase.pc_prior_residuals = pc_prior_residuals_latest_scaled


def _has_valid_distance(row) -> bool:
    d = _finite(row.get("<d>", np.nan))
    e = _finite(row.get("<d_error>", np.nan))
    return bool(np.isfinite(d) and np.isfinite(e) and d > 0 and e > 0)


def _source_uses_distance_prior(row) -> bool:
    return bool(ACTIVE_USE_DISTANCE_PRIOR and _has_valid_distance(row))


SCALE_PARAMETERIZATION_ALIASES = {
    "logr_logd": "logR_logD",
    "logrlogd": "logR_logD",
    "old": "logR_logD",
    "default": "logR_logD",
    "logr_over_d_logd": "logRoverD_logD",
    "logroverd_logd": "logRoverD_logD",
    "logr_d_logd": "logRoverD_logD",
    "roverd_logd": "logRoverD_logD",
    "logr_over_d_logr": "logRoverD_logR",
    "logroverd_logr": "logRoverD_logR",
    "logr_d_logr": "logRoverD_logR",
    "roverd_logr": "logRoverD_logR",
    "logroverd_only": "logRoverD_only",
    "logr_over_d_only": "logRoverD_only",
    "roverd_only": "logRoverD_only",
}
ACTIVE_SCALE_PARAMETERIZATION = "logR_logD"


def _normalize_scale_parameterization(scale_parameterization="logR_logD") -> str:
    key = str(scale_parameterization).strip().replace(" ", "").replace("-", "_").lower()
    key = key.replace("/", "_over_").replace("(", "").replace(")", "")
    if key in SCALE_PARAMETERIZATION_ALIASES:
        return SCALE_PARAMETERIZATION_ALIASES[key]
    allowed = "logR_logD, logRoverD_logD, logRoverD_logR, logRoverD_only"
    raise ValueError(f"scale_parameterization must be one of {allowed}; got {scale_parameterization!r}")


def _scale_parameterization_slug(scale_parameterization="logR_logD") -> str:
    mode = _normalize_scale_parameterization(scale_parameterization)
    return {
        "logR_logD": "scaleLogRLogD",
        "logRoverD_logD": "scaleLogRoverDLogD",
        "logRoverD_logR": "scaleLogRoverDLogR",
        "logRoverD_only": "scaleLogRoverDOnly",
    }[mode]


def _scale_sample_names(old_names, mode: str):
    old_names = list(old_names)
    if mode == "logR_logD":
        return old_names
    core = [name for name in old_names if name not in {"logD", "logR"}]
    if mode == "logRoverD_logD":
        return core + ["logR_over_D", "logD"]
    if mode == "logRoverD_logR":
        return core + ["logR_over_D", "logR"]
    raise ValueError(f"unsupported scale parameterization: {mode}")


def _theta_dict(theta, names):
    return {name: float(value) for name, value in zip(list(names), np.asarray(theta, dtype=float))}


def _sample_theta_to_old(theta, sample_names, old_names, mode: str):
    if mode == "logR_logD":
        return np.asarray(theta, dtype=float)
    d = _theta_dict(theta, sample_names)
    if mode == "logRoverD_logD":
        q = d["logR_over_D"]
        logd = d["logD"]
        d["logR"] = q + logd
    elif mode == "logRoverD_logR":
        q = d["logR_over_D"]
        logr = d["logR"]
        d["logD"] = logr - q
    else:
        raise ValueError(f"unsupported scale parameterization: {mode}")
    return np.asarray([d[name] for name in old_names], dtype=float)


def _old_theta_to_sample(theta_old, old_names, sample_names, mode: str):
    if mode == "logR_logD":
        return np.asarray(theta_old, dtype=float)
    d = _theta_dict(theta_old, old_names)
    d["logR_over_D"] = d["logR"] - d["logD"]
    return np.asarray([d[name] for name in sample_names], dtype=float)


def _augment_scale_columns(samples: pd.DataFrame) -> pd.DataFrame:
    if "logR_over_D" not in samples.columns and {"logR", "logD"}.issubset(samples.columns):
        samples["logR_over_D"] = samples["logR"] - samples["logD"]
    if "logR" not in samples.columns and {"logR_over_D", "logD"}.issubset(samples.columns):
        samples["logR"] = samples["logR_over_D"] + samples["logD"]
    if "logD" not in samples.columns and {"logR_over_D", "logR"}.issubset(samples.columns):
        samples["logD"] = samples["logR"] - samples["logR_over_D"]
    if "logD" in samples.columns:
        samples["D_kpc"] = 10.0 ** samples["logD"]
    if "logR" in samples.columns:
        samples["R_Rsun"] = 10.0 ** samples["logR"]
    if "logR_over_D" in samples.columns:
        samples["R_over_D_Rsun_per_kpc"] = 10.0 ** samples["logR_over_D"]
    return samples


def _sample_bounds_from_old_bounds(old_bounds, old_names, sample_names, mode: str):
    if mode == "logR_logD":
        return old_bounds
    lower_old = np.asarray(old_bounds["lower"], dtype=float)
    upper_old = np.asarray(old_bounds["upper"], dtype=float)
    lower_by_name = {name: lower_old[i] for i, name in enumerate(old_names)}
    upper_by_name = {name: upper_old[i] for i, name in enumerate(old_names)}
    lower = []
    upper = []
    for name in sample_names:
        if name == "logR_over_D":
            lower.append(lower_by_name["logR"] - upper_by_name["logD"])
            upper.append(upper_by_name["logR"] - lower_by_name["logD"])
        else:
            lower.append(lower_by_name[name])
            upper.append(upper_by_name[name])
    return {"lower": np.asarray(lower, dtype=float), "upper": np.asarray(upper, dtype=float)}


def _patch_scale_parameterization(directbase, mapbase, latest_module, scale_parameterization="logR_logD"):
    """Patch the direct-emcee scale parameters without changing old defaults.

    Supported modes:
      - logR_logD: original parameterization.
      - logRoverD_logD: sample log(R/D) and logD; derive logR.
      - logRoverD_logR: sample log(R/D) and logR; derive logD.

    The MCMC still has only two independent scale parameters.  The third of
    logR_over_D, logR, and logD is written as a derived sample/summary column.
    """
    global ACTIVE_SCALE_PARAMETERIZATION
    mode = _normalize_scale_parameterization(scale_parameterization)
    ACTIVE_SCALE_PARAMETERIZATION = mode
    if mode == "logR_logD":
        directbase.scale_parameterization = mode
        return

    old_param_names = list(directbase.PARAM_NAMES)
    if "logD" not in old_param_names or "logR" not in old_param_names:
        raise ValueError(f"Cannot apply {mode}: current PARAM_NAMES lacks logD/logR: {old_param_names}")
    sample_param_names = _scale_sample_names(old_param_names, mode)
    old_derived_names = list(getattr(directbase, "DERIVED_NAMES", []))
    sample_derived_names = []
    for name in ["logR_over_D", "logD", "logR", "D_kpc", "R_Rsun", "R_over_D_Rsun_per_kpc"]:
        if name not in sample_param_names and name not in sample_derived_names:
            sample_derived_names.append(name)
    for name in old_derived_names:
        if name not in sample_param_names and name not in sample_derived_names:
            sample_derived_names.append(name)

    old_bounds_for_model = mapbase.bounds_for_model
    old_model_mags = mapbase.model_mags
    old_pc_prior_residuals = mapbase.pc_prior_residuals
    old_prior_residuals = mapbase.prior_residuals
    old_direct_center = directbase._direct_center
    old_draw_one = directbase._draw_one_prior_position

    def to_old(theta_sample):
        return _sample_theta_to_old(theta_sample, sample_param_names, old_param_names, mode)

    def bounds_for_model_sample(model):
        old = old_bounds_for_model(model)
        _patch_scale_parameterization._last_model = model
        _patch_scale_parameterization._last_old_bounds = old
        return _sample_bounds_from_old_bounds(old, old_param_names, sample_param_names, mode)

    def model_mags_sample(theta, model, zps, sed_bands):
        return old_model_mags(to_old(theta), model, zps, sed_bands)

    def pc_prior_residuals_sample(theta, model, zps, pc_prior, pc_cov):
        return old_pc_prior_residuals(to_old(theta), model, zps, pc_prior, pc_cov)

    def prior_residuals_sample(theta, row):
        return old_prior_residuals(to_old(theta), row)

    def residual_sample(theta, row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds):
        theta = np.asarray(theta, dtype=float)
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)
        n_bad = len(sed_obs) + len(getattr(latest_module, "PC_COLOR_DEFS", [])) + 3
        if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
            return np.full(n_bad, 1.0e6)
        try:
            old_theta = to_old(theta)
            old_bounds = _patch_scale_parameterization._last_old_bounds
            old_lower = np.asarray(old_bounds["lower"], dtype=float)
            old_upper = np.asarray(old_bounds["upper"], dtype=float)
            if np.any(old_theta < old_lower) or np.any(old_theta > old_upper):
                return np.full(n_bad, 1.0e6)
            sed_pred = old_model_mags(old_theta, model, zps, sed_bands)
            sed_res = (np.asarray(sed_obs, dtype=float) - np.asarray(sed_pred, dtype=float)) / np.asarray(sed_err, dtype=float)
            pc_res, _method = old_pc_prior_residuals(old_theta, model, zps, pc_prior, pc_cov)
            prior_res = old_prior_residuals(old_theta, row)
            return np.concatenate([sed_res, pc_res, prior_res])
        except Exception:
            return np.full(n_bad, 1.0e6)

    def direct_center_sample(row, bounds):
        old_bounds = _patch_scale_parameterization._last_old_bounds
        old_center = old_direct_center(row, old_bounds)
        center = _old_theta_to_sample(old_center, old_param_names, sample_param_names, mode)
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)
        return np.clip(center, lower + 1.0e-6, upper - 1.0e-6)

    def draw_one_prior_position_sample(row, bounds, rng):
        old_bounds = _patch_scale_parameterization._last_old_bounds
        # The filtered missing-distance draw wrapper inspects directbase.PARAM_NAMES
        # at call time. Temporarily restore the old names so its logD index refers
        # to the old theta vector, not the sample-space vector.
        saved_names = list(directbase.PARAM_NAMES)
        directbase.PARAM_NAMES = list(old_param_names)
        try:
            old_p = old_draw_one(row, old_bounds, rng)
        finally:
            directbase.PARAM_NAMES = saved_names
        p = _old_theta_to_sample(old_p, old_param_names, sample_param_names, mode)
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)
        return np.clip(p, lower + 1.0e-6, upper - 1.0e-6)

    def initialize_walkers_direct_sample(row, bounds, nwalkers, random_seed):
        rng = np.random.default_rng(random_seed)
        ndim = len(sample_param_names)
        nwalkers_eff = int(max(nwalkers, 2 * ndim + 2))
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)
        p0 = []
        n_try = 0
        while len(p0) < nwalkers_eff and n_try < 10000:
            n_try += 1
            p = directbase._draw_one_prior_position(row, bounds, rng)
            if np.all(np.isfinite(p)) and np.all(p > lower) and np.all(p < upper):
                old_p = to_old(p)
                old_bounds = _patch_scale_parameterization._last_old_bounds
                if np.all(old_p > old_bounds["lower"]) and np.all(old_p < old_bounds["upper"]):
                    p0.append(p)
        if len(p0) < nwalkers_eff:
            center = np.asarray(directbase._direct_center(row, bounds), dtype=float)
            scale = np.full(ndim, 0.03, dtype=float)
            for name, value in {
                "Teff": 50.0,
                "logg": 0.02,
                "MH": 0.02,
                "Av": 0.05,
                "Rv": 0.03,
                "logD": 0.005,
                "logR": 0.02,
                "logR_over_D": 0.02,
            }.items():
                if name in sample_param_names:
                    scale[sample_param_names.index(name)] = value
            if (not _source_uses_distance_prior(row)) and "logD" in sample_param_names:
                scale[sample_param_names.index("logD")] = 0.35
            while len(p0) < nwalkers_eff:
                p = center + rng.normal(0.0, scale)
                p = np.clip(p, lower + 1.0e-6, upper - 1.0e-6)
                old_p = to_old(p)
                old_bounds = _patch_scale_parameterization._last_old_bounds
                if np.all(old_p > old_bounds["lower"]) and np.all(old_p < old_bounds["upper"]):
                    p0.append(p)
        return np.asarray(p0, dtype=float)

    def run_emcee_direct_sample(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds, nwalkers, nsteps, burnin, thin, random_seed, progress=False):
        import emcee

        p0 = directbase.initialize_walkers_direct(row, bounds, nwalkers, random_seed=random_seed)
        ndim = len(sample_param_names)
        sampler = emcee.EnsembleSampler(
            p0.shape[0],
            ndim,
            directbase.log_probability,
            args=(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds),
        )
        sampler.run_mcmc(p0, int(nsteps), progress=bool(progress))
        flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
        logp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
        samples = pd.DataFrame(flat, columns=sample_param_names)
        samples = _augment_scale_columns(samples)
        samples["log_prob"] = logp
        return sampler, samples

    def make_corner_plot_sample(samples, row, out_dir):
        import matplotlib.pyplot as plt
        import corner

        cols = [c for c in ["Teff", "logg", "MH", "Av", "Rv", "logR_over_D", "logD", "logR"] if c in samples.columns]
        labels = {
            "Teff": r"$T_{\rm eff}$",
            "logg": r"$\log g$",
            "MH": r"$[M/H]$",
            "Av": r"$A_V$",
            "Rv": r"$R_V$",
            "logR_over_D": r"$\log(R/D)$",
            "logD": r"$\log D$",
            "logR": r"$\log R$",
        }
        fig = corner.corner(
            samples[cols],
            labels=[labels[c] for c in cols],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
            title_kwargs={"fontsize": 9},
        )
        out = Path(out_dir) / f"{row['selection_tag']}_{directbase._safe_name(row.get('Name', 'source'))}_corner.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return out

    mapbase.bounds_for_model = bounds_for_model_sample
    mapbase.model_mags = model_mags_sample
    mapbase.pc_prior_residuals = pc_prior_residuals_sample
    mapbase.prior_residuals = prior_residuals_sample
    mapbase.residual = residual_sample
    directbase.PARAM_NAMES = list(sample_param_names)
    directbase.DERIVED_NAMES = list(sample_derived_names)
    directbase.scale_parameterization = mode
    directbase._scale_old_param_names = list(old_param_names)
    directbase._direct_center = direct_center_sample
    directbase._draw_one_prior_position = draw_one_prior_position_sample
    directbase.initialize_walkers_direct = initialize_walkers_direct_sample
    directbase.run_emcee_direct = run_emcee_direct_sample
    directbase.make_corner_plot = make_corner_plot_sample


_patch_scale_parameterization._last_model = None
_patch_scale_parameterization._last_old_bounds = None


def _patch_native_a550_r550_logroverd_only(
    directbase,
    mapbase,
    latest_module,
    *,
    extinction_parameterization="A550_R550",
    fix_logg_mh=False,
    fixed_logg=2.0,
    fixed_mh=0.0,
    band_extinction_draws=512,
    band_extinction_batch_size=512,
    compute_band_extinction_posterior=True,
    save_band_extinction_draws=False,
    native_vertex_cache_size=128,
):
    """Install a native-cache likelihood sampled in A550/R550 or Av/Rv."""
    from collections import OrderedDict
    from bosz_a550_r550_preconvolved import PreconvolvedBOSZModel, _evaluate_law
    from scipy.interpolate import PchipInterpolator

    global ACTIVE_EXTINCTION_PARAMETERIZATION
    global ACTIVE_BAND_EXTINCTION_DRAWS, ACTIVE_SAVE_BAND_EXTINCTION_DRAWS
    global ACTIVE_COMPUTE_BAND_EXTINCTION
    extinction_parameterization = _normalize_extinction_parameterization(
        extinction_parameterization
    )
    if extinction_parameterization not in NATIVE_EXTINCTION_PARAMETERIZATIONS:
        raise ValueError(
            "native-cache likelihood requires A550_R550 or Av_Rv parameterization"
        )
    sample_in_av_rv = extinction_parameterization == "Av_Rv"
    ACTIVE_EXTINCTION_PARAMETERIZATION = extinction_parameterization
    ACTIVE_BAND_EXTINCTION_DRAWS = int(band_extinction_draws)
    ACTIVE_SAVE_BAND_EXTINCTION_DRAWS = bool(save_band_extinction_draws)
    ACTIVE_COMPUTE_BAND_EXTINCTION = bool(compute_band_extinction_posterior)
    band_extinction_batch_size = int(band_extinction_batch_size)
    native_vertex_cache_size = int(native_vertex_cache_size)
    if ACTIVE_BAND_EXTINCTION_DRAWS <= 0:
        raise ValueError("band_extinction_draws must be positive")
    if band_extinction_batch_size <= 0:
        raise ValueError("band_extinction_batch_size must be positive")
    if native_vertex_cache_size <= 0:
        raise ValueError("native_vertex_cache_size must be positive")

    fix_logg_mh = bool(fix_logg_mh)
    fixed_logg = float(fixed_logg)
    fixed_mh = float(fixed_mh)
    if fix_logg_mh and (not np.isfinite(fixed_logg) or not np.isfinite(fixed_mh)):
        raise ValueError("fixed_logg and fixed_mh must be finite when fix_logg_mh=True")

    extinction_names = ["Av", "Rv"] if sample_in_av_rv else ["A550", "R550"]
    sample_names = (
        ["Teff", *extinction_names, "logR_over_D"]
        if fix_logg_mh
        else ["Teff", "logg", "MH", *extinction_names, "logR_over_D"]
    )
    old_pc_residuals = mapbase.pc_prior_residuals
    old_prior_residuals = mapbase.prior_residuals
    old_log_probability = directbase.log_probability
    previous_summarize = directbase.summarize_samples
    old_param_names = ["Teff", "logg", "MH", "Av", "Rv", "logD", "logR"]
    old_derived_names = ["D_kpc", "R_Rsun"]

    mapbase.sedbase._import_model = lambda: PreconvolvedBOSZModel

    def _full_theta(theta):
        values = np.asarray(theta, dtype=float)
        if fix_logg_mh:
            teff, amount, shape, q = values
            return np.asarray(
                [teff, fixed_logg, fixed_mh, amount, shape, q], dtype=float
            )
        return values

    def _native_mapping_arrays(model):
        if hasattr(model, "_avrv_native_mapping"):
            return model._avrv_native_mapping
        native_rv = np.asarray(model.h5["native_shape_for_r550"][:], dtype=float)
        r550 = np.asarray(model.r550_grid, dtype=float)
        order = np.argsort(native_rv)
        native_rv = native_rv[order]
        r550 = r550[order]
        extinction_law = model.h5.attrs.get("extinction_law", "G23")
        if isinstance(extinction_law, bytes):
            extinction_law = extinction_law.decode("utf-8")
        k550 = np.asarray(
            [
                float(np.asarray(_evaluate_law(str(extinction_law), float(rv), [0.550])).reshape(-1)[0])
                for rv in native_rv
            ],
            dtype=float,
        )
        r550_of_rv = PchipInterpolator(native_rv, r550, extrapolate=False)
        k550_of_rv = PchipInterpolator(native_rv, k550, extrapolate=False)
        r550_order = np.argsort(r550)
        rv_of_r550 = PchipInterpolator(
            r550[r550_order],
            native_rv[r550_order],
            extrapolate=False,
        )
        model._avrv_native_mapping = (
            native_rv,
            r550,
            k550,
            r550_of_rv,
            k550_of_rv,
            rv_of_r550,
        )
        return model._avrv_native_mapping

    def _av_rv_to_native(model, av, rv):
        (
            native_rv,
            _r550_grid_for_rv,
            _k550_for_rv,
            r550_of_rv,
            k550_of_rv,
            _rv_of_r550,
        ) = _native_mapping_arrays(model)
        rv_values = np.asarray(rv, dtype=float)
        av_values = np.asarray(av, dtype=float)
        if (
            np.any(~np.isfinite(rv_values))
            or np.any(rv_values < native_rv[0])
            or np.any(rv_values > native_rv[-1])
        ):
            raise ValueError(
                f"Rv outside native extinction-law range [{native_rv[0]}, {native_rv[-1]}]"
            )
        r550_values = np.asarray(r550_of_rv(rv_values), dtype=float)
        k550_values = np.asarray(k550_of_rv(rv_values), dtype=float)
        a550_values = av_values * k550_values
        if (
            np.any(~np.isfinite(a550_values))
            or np.any(a550_values < model.a550_grid[0])
            or np.any(a550_values > model.a550_grid[-1])
        ):
            raise ValueError(
                f"Av maps outside native A550 range "
                f"[{model.a550_grid[0]}, {model.a550_grid[-1]}]"
            )
        return a550_values, r550_values

    def _native_to_av_rv(model, a550, r550):
        (
            _native_rv,
            _r550_grid_for_rv,
            _k550_for_rv,
            _r550_of_rv,
            k550_of_rv,
            rv_of_r550,
        ) = _native_mapping_arrays(model)
        r550_values = np.asarray(r550, dtype=float)
        a550_values = np.asarray(a550, dtype=float)
        rv_values = np.asarray(rv_of_r550(r550_values), dtype=float)
        k550_values = np.asarray(k550_of_rv(rv_values), dtype=float)
        av_values = a550_values / k550_values
        if (
            np.any(~np.isfinite(av_values))
            or np.any(~np.isfinite(rv_values))
            or np.any(k550_values <= 0)
        ):
            raise ValueError("invalid native A550/R550 to Av/Rv conversion")
        return av_values, rv_values

    def _log_avrv_to_native_jacobian(model, rv):
        (
            native_rv,
            _r550_grid_for_rv,
            _k550_for_rv,
            r550_of_rv,
            k550_of_rv,
            _rv_of_r550,
        ) = _native_mapping_arrays(model)
        rv = float(rv)
        if not np.isfinite(rv) or rv < native_rv[0] or rv > native_rv[-1]:
            return -np.inf
        k550 = float(k550_of_rv(rv))
        dr550_drv = float(r550_of_rv.derivative()(rv))
        determinant = k550 * abs(dr550_drv)
        if not np.isfinite(determinant) or determinant <= 0:
            return -np.inf
        return float(np.log(determinant))

    def _native_extinction_values(model, amount, shape):
        if sample_in_av_rv:
            a550, r550 = _av_rv_to_native(model, amount, shape)
            return float(np.asarray(a550)), float(np.asarray(r550))
        return float(amount), float(shape)

    def _old_theta(theta):
        teff, logg, mh, _amount, _shape, q = _full_theta(theta)
        return np.asarray([teff, logg, mh, 0.0, 3.1, 0.0, q], dtype=float)

    def _native_bounds_from_model(model):
        p = np.asarray(model.params, dtype=float)
        teff_lower = float(np.min(p[:, 0]))
        teff_upper = float(np.max(p[:, 0]))
        if not fix_logg_mh:
            return teff_lower, teff_upper

        candidate_teff = np.unique(p[:, 0])
        probe = np.column_stack(
            [candidate_teff, np.full_like(candidate_teff, fixed_logg), np.full_like(candidate_teff, fixed_mh)]
        )
        valid = np.asarray(model.tri.find_simplex(probe), dtype=int) >= 0
        if not np.any(valid):
            raise ValueError(
                "fixed_logg/fixed_mh lies outside the native BOSZ interpolation hull: "
                f"logg={fixed_logg}, [M/H]={fixed_mh}"
            )
        return float(np.min(candidate_teff[valid])), float(np.max(candidate_teff[valid]))

    def _vertex_n(model, vertex):
        if not hasattr(model, "_native_n_lru"):
            model._native_n_lru = OrderedDict()
        cache = model._native_n_lru
        vertex = int(vertex)
        if vertex in cache:
            value = cache.pop(vertex)
            cache[vertex] = value
            return value
        value = np.asarray(model.h5["N"][vertex, :, :, :], dtype=np.float32)
        cache[vertex] = value
        while len(cache) > native_vertex_cache_size:
            cache.popitem(last=False)
        return value

    def band_n_native(model, teff, logg, mh, r550, a550, used_bands=None):
        vertices, weights = model._simplex_weights(float(teff), float(logg), float(mh))
        ir0, ir1, tr = model._bracket(model.r550_grid, float(r550))
        ia0, ia1, ta = model._bracket(model.a550_grid, float(a550))
        n_interp = np.zeros(len(model.bands), dtype=float)
        for vertex, weight in zip(vertices, weights):
            n_v = _vertex_n(model, vertex)
            n_interp += float(weight) * (
                (1.0 - tr) * (1.0 - ta) * n_v[ir0, ia0, :]
                + tr * (1.0 - ta) * n_v[ir1, ia0, :]
                + (1.0 - tr) * ta * n_v[ir0, ia1, :]
                + tr * ta * n_v[ir1, ia1, :]
            )
        check = n_interp if used_bands is None else n_interp[[model.band_index[b] for b in used_bands]]
        if np.any(~np.isfinite(check)) or np.any(check <= 0):
            raise ValueError("non-positive interpolated native A550/R550 band integral")
        return n_interp

    def band_d_native(model, teff, logg, mh):
        vertices, weights = model._simplex_weights(float(teff), float(logg), float(mh))
        d_interp = np.zeros(len(model.bands), dtype=float)
        for vertex, weight in zip(vertices, weights):
            d_interp += float(weight) * np.asarray(model.h5["D"][int(vertex), :], dtype=float)
        if np.any(~np.isfinite(d_interp)) or np.any(d_interp <= 0):
            raise ValueError("non-positive interpolated intrinsic band integral")
        return d_interp

    def _simplex_weights_batch(model, points):
        """Vectorized Delaunay lookup and barycentric weights."""
        points = np.asarray(points, dtype=float)
        simplices = np.asarray(model.tri.find_simplex(points), dtype=int)
        if np.any(simplices < 0):
            bad = points[simplices < 0][0]
            raise ValueError(
                "Point outside BOSZ convex hull: "
                f"Teff={bad[0]}, logg={bad[1]}, [M/H]={bad[2]}"
            )
        transform = model.tri.transform[simplices]
        bary = np.einsum(
            "nij,nj->ni",
            transform[:, :3, :],
            points - transform[:, 3, :],
            optimize=True,
        )
        weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
        vertices = model.tri.simplices[simplices].astype(int, copy=False)
        return vertices, weights

    def _bracket_batch(grid, values):
        """Vectorized equivalent of model._bracket."""
        grid = np.asarray(grid, dtype=float)
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < grid[0]) or np.any(values > grid[-1]):
            raise ValueError(f"values outside grid [{grid[0]}, {grid[-1]}]")
        hi = np.searchsorted(grid, values, side="right")
        hi = np.clip(hi, 1, len(grid) - 1)
        lo = hi - 1
        at_upper = values >= grid[-1]
        lo[at_upper] = len(grid) - 1
        hi[at_upper] = len(grid) - 1
        denominator = grid[hi] - grid[lo]
        fraction = np.divide(
            values - grid[lo],
            denominator,
            out=np.zeros_like(values, dtype=float),
            where=denominator != 0,
        )
        return lo, hi, fraction

    def band_extinction_batch_native(model, sample_table, chunk_size=None):
        """Compatibility adapter to the shared vectorized model method."""
        return model.band_extinction_batch(
            sample_table,
            bands=SED_BANDS,
            chunk_size=int(chunk_size or band_extinction_batch_size),
        )

    def bounds_native(model):
        if not getattr(model, "is_native_a550_r550", False):
            raise ValueError(
                f"{extinction_parameterization} fitting requires a native cache, "
                f"got {model.cache_file}"
            )
        p = np.asarray(model.params, dtype=float)
        teff_lower, teff_upper = _native_bounds_from_model(model)
        if sample_in_av_rv:
            (
                native_rv,
                _r550_for_rv,
                k550_for_rv,
                _r550_interp,
                _k550_interp,
                _rv_interp,
            ) = _native_mapping_arrays(model)
            amount_lower = 0.0
            amount_upper = (
                float(model.a550_grid[-1]) / float(np.min(k550_for_rv))
            )
            shape_lower = float(native_rv[0])
            shape_upper = float(native_rv[-1])
        else:
            amount_lower = float(model.a550_grid[0])
            amount_upper = float(model.a550_grid[-1])
            shape_lower = float(model.r550_grid[0])
            shape_upper = float(model.r550_grid[-1])
        if fix_logg_mh:
            lower = np.asarray([
                teff_lower, amount_lower, shape_lower,
                np.log10(3.0) - np.log10(50.0),
            ], dtype=float)
            upper = np.asarray([
                teff_upper, amount_upper, shape_upper,
                np.log10(1000.0) - np.log10(0.05),
            ], dtype=float)
        else:
            lower = np.asarray([
                teff_lower, np.min(p[:, 1]), np.min(p[:, 2]),
                amount_lower, shape_lower,
                np.log10(3.0) - np.log10(50.0),
            ], dtype=float)
            upper = np.asarray([
                teff_upper, np.max(p[:, 1]), np.max(p[:, 2]),
                amount_upper, shape_upper,
                np.log10(1000.0) - np.log10(0.05),
            ], dtype=float)
        _patch_native_a550_r550_logroverd_only._active_model = model
        return {"lower": lower, "upper": upper}

    def model_mags_native(theta, model, zps, sed_bands):
        teff, logg, mh, amount, shape, q = _full_theta(theta)
        a550, r550 = _native_extinction_values(model, amount, shape)
        n = band_n_native(model, teff, logg, mh, r550, a550, used_bands=sed_bands)
        scale = (10.0**q * mapbase.sedbase.R_SUN_CM / mapbase.sedbase.KPC_CM) ** 2
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("bad R/D scale")
        return np.asarray([
            -2.5 * np.log10(n[model.band_index[band]]) + zps[band] - 2.5 * np.log10(scale)
            for band in sed_bands
        ], dtype=float)

    def pc_residuals_native(theta, model, zps, pc_prior, pc_cov):
        return old_pc_residuals(_old_theta(theta), model, zps, pc_prior, pc_cov)

    def prior_residuals_native(theta, row):
        if fix_logg_mh:
            return np.zeros(0, dtype=float)
        return old_prior_residuals(_old_theta(theta), row)

    def residual_native(theta, row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds):
        theta = np.asarray(theta, dtype=float)
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)
        n_bad = len(sed_obs) + len(pc_prior) + (0 if fix_logg_mh else 3)
        if np.any(~np.isfinite(theta)) or np.any(theta < lower) or np.any(theta > upper):
            return np.full(n_bad, 1.0e6)
        try:
            sed_pred = model_mags_native(theta, model, zps, sed_bands)
            sed_res = (np.asarray(sed_obs, dtype=float) - sed_pred) / np.asarray(sed_err, dtype=float)
            pc_output = pc_residuals_native(theta, model, zps, pc_prior, pc_cov)
            pc_res = pc_output[0] if isinstance(pc_output, tuple) else pc_output
            prior_res = prior_residuals_native(theta, row)
            return np.concatenate([sed_res, np.asarray(pc_res, dtype=float), np.asarray(prior_res, dtype=float)])
        except Exception:
            return np.full(n_bad, 1.0e6)

    def log_probability_native(
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
    ):
        value = old_log_probability(
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
        if not np.isfinite(value) or not sample_in_av_rv:
            return value
        rv = float(np.asarray(theta, dtype=float)[sample_names.index("Rv")])
        log_jacobian = _log_avrv_to_native_jacobian(model, rv)
        if not np.isfinite(log_jacobian):
            return -np.inf
        return float(value + log_jacobian)

    def center_native(row, bounds):
        teff = float(np.clip(directbase._teff_center_from_period(row), bounds["lower"][0], bounds["upper"][0]))
        a550 = directbase._av_center(row)
        d = _finite(row.get("<d>", np.nan))
        d = d if np.isfinite(d) and d > 0 else 5.0
        q = np.log10(directbase._radius_seed_from_period(row)) - np.log10(d)
        if fix_logg_mh:
            center = np.asarray([teff, a550, 3.1, q], dtype=float)
        else:
            logg = latest_module.logg_prior_mu_latest(teff, row)
            if not np.isfinite(logg):
                logg = 2.0
            mh = _finite(row.get("[Fe/H]", np.nan))
            if not np.isfinite(mh):
                mh = 0.0
            center = np.asarray([teff, logg, mh, a550, 3.1, q], dtype=float)
        return np.clip(center, bounds["lower"] + 1.0e-6, bounds["upper"] - 1.0e-6)

    def draw_native(row, bounds, rng):
        center = center_native(row, bounds)
        teff = np.clip(rng.normal(center[0], 350.0), bounds["lower"][0] + 1e-6, bounds["upper"][0] - 1e-6)
        if fix_logg_mh:
            values = np.asarray([
                teff,
                rng.normal(center[1], 0.6),
                rng.normal(center[2], 0.35),
                rng.normal(center[3], 0.30),
            ], dtype=float)
            return np.clip(values, bounds["lower"] + 1.0e-6, bounds["upper"] - 1.0e-6)
        logg_mu = latest_module.logg_prior_mu_latest(teff, row)
        if not np.isfinite(logg_mu):
            logg_mu = center[1]
        mh_mu = _mh_prior_mu_used(row)[0]
        values = np.asarray([
            teff,
            rng.normal(logg_mu, latest_module.LOGGF_PRIOR_SIGMA),
            rng.normal(mh_mu, latest_module.FE_GRAD_SIGMA),
            rng.normal(center[3], 0.6),
            rng.normal(center[4], 0.35),
            rng.normal(center[5], 0.30),
        ], dtype=float)
        return np.clip(values, bounds["lower"] + 1.0e-6, bounds["upper"] - 1.0e-6)

    def initialize_native(row, bounds, nwalkers, random_seed):
        rng = np.random.default_rng(random_seed)
        nwalkers = int(max(nwalkers, 2 * len(sample_names) + 2))
        walkers = []
        while len(walkers) < nwalkers:
            p = draw_native(row, bounds, rng)
            if np.all(np.isfinite(p)) and np.all(p > bounds["lower"]) and np.all(p < bounds["upper"]):
                walkers.append(p)
        return np.asarray(walkers, dtype=float)

    def run_emcee_native(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds,
                         nwalkers, nsteps, burnin, thin, random_seed, progress=False):
        import emcee

        p0 = initialize_native(row, bounds, nwalkers, random_seed)
        sampler = emcee.EnsembleSampler(
            p0.shape[0], len(sample_names), directbase.log_probability,
            args=(row, model, zps, sed_bands, sed_obs, sed_err, pc_prior, pc_cov, bounds),
        )
        sampler.run_mcmc(p0, int(nsteps), progress=bool(progress))
        flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
        logp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
        samples = pd.DataFrame(flat, columns=sample_names)
        if fix_logg_mh:
            samples.insert(1, "logg", fixed_logg)
            samples.insert(2, "MH", fixed_mh)
        if sample_in_av_rv:
            a550, r550 = _av_rv_to_native(
                model,
                samples["Av"].to_numpy(dtype=float),
                samples["Rv"].to_numpy(dtype=float),
            )
            samples["A550"] = np.asarray(a550, dtype=float)
            samples["R550"] = np.asarray(r550, dtype=float)
        else:
            av, rv = _native_to_av_rv(
                model,
                samples["A550"].to_numpy(dtype=float),
                samples["R550"].to_numpy(dtype=float),
            )
            samples["Av"] = np.asarray(av, dtype=float)
            samples["Rv"] = np.asarray(rv, dtype=float)
        samples["R_over_D_Rsun_per_kpc"] = 10.0 ** samples["logR_over_D"]
        samples["log_prob"] = logp
        if ACTIVE_COMPUTE_BAND_EXTINCTION and ACTIVE_SAVE_BAND_EXTINCTION_DRAWS:
            extinction = band_extinction_batch_native(model, samples)
            for band_index, band in enumerate(SED_BANDS):
                samples[f"A_band_{band}"] = extinction[:, band_index]
        return sampler, samples

    def band_extinction_for_sample(model, sample):
        d = band_d_native(model, sample["Teff"], sample["logg"], sample["MH"])
        n = band_n_native(model, sample["Teff"], sample["logg"], sample["MH"], sample["R550"], sample["A550"])
        values = -2.5 * np.log10(n / d)
        return {band: float(values[model.band_index[band]]) for band in SED_BANDS}

    def _quantiles(out, name, values, best_value):
        p16, p50, p84 = np.nanpercentile(np.asarray(values, dtype=float), [16, 50, 84])
        out[f"{name}_p16"] = float(p16)
        out[f"{name}_p50"] = float(p50)
        out[f"{name}_p84"] = float(p84)
        out[f"{name}_err_minus"] = float(p50 - p16)
        out[f"{name}_err_plus"] = float(p84 - p50)
        out[f"{name}_map_sample"] = float(best_value)

    def summarize_native(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
        legacy = samples.copy()
        if not sample_in_av_rv:
            legacy["Av"] = legacy["A550"]
            legacy["Rv"] = legacy["R550"]
        legacy["logD"] = 0.0
        legacy["logR"] = legacy["logR_over_D"]
        legacy["D_kpc"] = 1.0
        legacy["R_Rsun"] = 10.0 ** legacy["logR_over_D"]
        saved_params = list(directbase.PARAM_NAMES)
        saved_derived = list(directbase.DERIVED_NAMES)
        saved_model_mags = mapbase.model_mags
        def legacy_model_mags_bridge(theta_old, model, zps, bands):
            teff, logg, mh, amount, shape, logd, logr = np.asarray(theta_old, dtype=float)
            return model_mags_native(
                [teff, logg, mh, amount, shape, logr - logd],
                model,
                zps,
                bands,
            )
        directbase.PARAM_NAMES = list(old_param_names)
        directbase.DERIVED_NAMES = list(old_derived_names)
        mapbase.model_mags = legacy_model_mags_bridge
        try:
            out = previous_summarize(row, legacy, sampler, sed_bands, pc_prior, nsteps, burnin, thin)
        finally:
            directbase.PARAM_NAMES = saved_params
            directbase.DERIVED_NAMES = saved_derived
            mapbase.model_mags = saved_model_mags
        removable_prefixes = ["logD_", "logR_", "D_kpc_", "R_Rsun_"]
        if not sample_in_av_rv:
            removable_prefixes.extend(["Av_", "Rv_"])
        for prefix in removable_prefixes:
            for key in [k for k in out if k.startswith(prefix) and not k.startswith("logR_over_D_")]:
                out.pop(key, None)

        best_idx = int(np.nanargmax(samples["log_prob"].to_numpy(float)))
        summary_names = list(
            dict.fromkeys(
                sample_names
                + (["A550", "R550"] if sample_in_av_rv else ["Av", "Rv"])
                + ["R_over_D_Rsun_per_kpc"]
            )
        )
        for name in summary_names:
            _quantiles(out, name, samples[name], samples.iloc[best_idx][name])
        if fix_logg_mh:
            for name, value in [("logg", fixed_logg), ("MH", fixed_mh)]:
                _quantiles(out, name, samples[name], value)

        model = _patch_native_a550_r550_logroverd_only._active_model
        if not ACTIVE_COMPUTE_BAND_EXTINCTION:
            out["band_extinction_computed"] = False
            out["band_extinction_posterior_draws"] = 0
            out["band_extinction_batch_size"] = 0
            out["extinction_parameterization"] = extinction_parameterization
            out["scale_parameterization"] = "logRoverD_only"
            out["uses_distance_prior"] = False
            out["uses_distance_prior_for_source"] = False
            return out
        n_draw = min(ACTIVE_BAND_EXTINCTION_DRAWS, len(samples))
        draw_indices = np.linspace(0, len(samples) - 1, n_draw, dtype=int)
        saved_extinction_columns = [f"A_band_{band}" for band in SED_BANDS]
        if all(column in samples.columns for column in saved_extinction_columns):
            extinction_draws = samples.iloc[draw_indices][saved_extinction_columns].to_numpy(dtype=float)
            best_extinction = samples.iloc[best_idx][saved_extinction_columns].to_numpy(dtype=float)
        else:
            extinction_draws = band_extinction_batch_native(model, samples.iloc[draw_indices])
            best_extinction = band_extinction_batch_native(model, samples.iloc[[best_idx]])[0]
        for band_index, band in enumerate(SED_BANDS):
            _quantiles(
                out,
                f"A_band_{band}",
                extinction_draws[:, band_index],
                best_extinction[band_index],
            )
        out["extinction_parameterization"] = extinction_parameterization
        out["scale_parameterization"] = "logRoverD_only"
        out["band_extinction_posterior_draws"] = int(n_draw)
        out["band_extinction_batch_size"] = int(band_extinction_batch_size)
        out["band_extinction_computed"] = True
        out["uses_distance_prior"] = False
        out["uses_distance_prior_for_source"] = False
        return out

    def corner_native(samples, row, out_dir):
        import corner
        import matplotlib.pyplot as plt

        extinction_labels = (
            [r"$A_V$", r"$R_V$"]
            if sample_in_av_rv
            else [r"$A_{550}$", r"$R_{550}$"]
        )
        labels = (
            [r"$T_{\rm eff}$", *extinction_labels, r"$\log(R/D)$"]
            if fix_logg_mh
            else [
                r"$T_{\rm eff}$",
                r"$\log g$",
                r"$[M/H]$",
                *extinction_labels,
                r"$\log(R/D)$",
            ]
        )
        fig = corner.corner(samples[sample_names], labels=labels, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt=".3g", title_kwargs={"fontsize": 9})
        out = Path(out_dir) / f"{row['selection_tag']}_{directbase._safe_name(row.get('Name', 'source'))}_corner.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return out

    mapbase.bounds_for_model = bounds_native
    mapbase.model_mags = model_mags_native
    mapbase.pc_prior_residuals = pc_residuals_native
    mapbase.prior_residuals = prior_residuals_native
    mapbase.residual = residual_native
    latest_module.band_n_interpolated_latest = band_n_native
    latest_module.model_mags_latest = model_mags_native
    directbase.PARAM_NAMES = list(sample_names)
    directbase.DERIVED_NAMES = ["R_over_D_Rsun_per_kpc"]
    directbase.scale_parameterization = "logRoverD_only"
    directbase._direct_center = center_native
    directbase._draw_one_prior_position = draw_native
    directbase.initialize_walkers_direct = initialize_native
    directbase.log_probability = log_probability_native
    directbase.run_emcee_direct = run_emcee_native
    directbase.summarize_samples = summarize_native
    directbase.make_corner_plot = corner_native
    directbase._WORKER_MODEL = None
    directbase._WORKER_ZPS = None
    directbase._WORKER_PC_COEFF = None
    directbase._WORKER_PC_COV_LONG = None
    directbase._WORKER_BOUNDS = None


_patch_native_a550_r550_logroverd_only._active_model = None


def _write_native_mapping_and_band_extinction_summary(cache_file, out_dir, summary, write_band_summary=True):
    import h5py

    out_dir = Path(out_dir)
    with h5py.File(cache_file, "r") as h5:
        mapping = pd.DataFrame({
            "R550": np.asarray(h5["r550_grid"][:], dtype=float),
            "native_shape_parameter": np.asarray(h5["native_shape_for_r550"][:], dtype=float),
        })
        mapping["extinction_law"] = str(h5.attrs.get("extinction_law", ""))
    mapping.to_csv(out_dir / "extinction_A550_R550_mapping.csv", index=False)

    if not write_band_summary:
        return mapping, pd.DataFrame()

    rows = []
    for _, source in summary.iterrows():
        for band in SED_BANDS:
            rows.append({
                "selection_tag": source.get("selection_tag", ""),
                "Name": source.get("Name", ""),
                "GaiaDR3": source.get("GaiaDR3", ""),
                "band": band,
                "A_band_p16": source.get(f"A_band_{band}_p16", np.nan),
                "A_band_p50": source.get(f"A_band_{band}_p50", np.nan),
                "A_band_p84": source.get(f"A_band_{band}_p84", np.nan),
                "A_band_err_minus": source.get(f"A_band_{band}_err_minus", np.nan),
                "A_band_err_plus": source.get(f"A_band_{band}_err_plus", np.nan),
            })
    long_table = pd.DataFrame(rows)
    long_table.to_csv(out_dir / "MW_SED_band_extinction_posterior_summary_long.csv", index=False)
    return mapping, long_table


def _mh_prior_mu_used(row) -> tuple[float, str]:
    if _has_valid_distance(row):
        mu = _finite(row.get("FeH_prior_mu_RGC_gradient", np.nan))
        if np.isfinite(mu):
            return float(mu), "RGC_Fe_gradient_CMetaLL_Table4"
    return 0.0, "solar_zero_missing_Wang25"


def _prepare_missing_wang25_columns(df: pd.DataFrame, latest_module) -> pd.DataFrame:
    out = df.copy()
    has_dist = []
    mu_used = []
    source_used = []
    for _, row in out.iterrows():
        has = _has_valid_distance(row)
        mu, src = _mh_prior_mu_used(row)
        has_dist.append(has)
        mu_used.append(mu)
        source_used.append(src)

    extra = pd.DataFrame(
        {
            "has_Wang25_distance": np.asarray(has_dist, dtype=bool),
            "uses_distance_prior_for_source": np.asarray([bool(ACTIVE_USE_DISTANCE_PRIOR and h) for h in has_dist], dtype=bool),
            "mh_prior_source": source_used,
            "mh_prior_mu_used": np.asarray(mu_used, dtype=float),
            "mh_prior_sigma": float(latest_module.FE_GRAD_SIGMA),
            "[Fe/H]": np.asarray(mu_used, dtype=float),
        },
        index=out.index,
    )
    replace = [c for c in extra.columns if c in out.columns]
    return pd.concat([out.drop(columns=replace), extra], axis=1).copy()


def _bp_passes(row) -> bool:
    bp_max = ACTIVE_FILTER_CONFIG.get("bp_max", None)
    bp = _finite(row.get("BP_mag", np.nan))
    if not np.isfinite(bp):
        return False
    if bp_max is None:
        return True
    return bool(bp <= float(bp_max))


def _gaia_band_passes(row, band: str) -> bool:
    if band not in {"BP", "G", "RP"}:
        return True
    if not _bp_passes(row):
        return False
    if ACTIVE_FILTER_CONFIG.get("require_gaia_vari_cepheid", False):
        return _clean(row.get(f"{band}_source", "")) == "Gaia_DR3_vari_cepheid"
    return True


def _jhk_band_passes(row, band: str) -> bool:
    if band not in {"J", "H", "K"}:
        return True

    group, _label = sfloor._jhk_floor_group(row, band)
    if group == "jhk_literature_intensity_mean" and not ACTIVE_FILTER_CONFIG.get(
        "apply_jhk_quality_to_literature_mean", False
    ):
        return True

    ph_allowed = ACTIVE_FILTER_CONFIG.get("jhk_phqual_allowed", None)
    cc_allowed = ACTIVE_FILTER_CONFIG.get("jhk_cc_flg_allowed", None)
    key = "Ks" if band == "K" else band
    ph = _clean(row.get(f"{key}_2mass_ph_qual_band", ""))
    cc = _clean(row.get(f"{key}_2mass_cc_flg_band", ""))
    ok_ph = True if ph_allowed is None else ph in ph_allowed
    ok_cc = True if cc_allowed is None else cc in cc_allowed
    return bool(ok_ph and ok_cc)


def _w1_band_passes(row) -> bool:
    group, _label = sfloor._w1_floor_group(row)
    if group == "w1_unwise":
        return True
    if not ACTIVE_FILTER_CONFIG.get("require_wise_profile_quality", True):
        return True

    ph_allowed = ACTIVE_FILTER_CONFIG.get("wise_phqual_allowed", None)
    cc_allowed = ACTIVE_FILTER_CONFIG.get("wise_cc_flags_allowed", None)
    ph = _first_flag_char(row.get("W1_profile_ph_qual", ""))
    cc = _first_flag_char(row.get("W1_profile_cc_flags", ""))
    ok_ph = True if ph_allowed is None else ph in ph_allowed
    ok_cc = True if cc_allowed is None else cc in cc_allowed
    return bool(ok_ph and ok_cc)


def _band_value_filtered(row, band: str):
    mcol, ecol, mag, err = ORIGINAL_BAND_VALUE(row, band)
    if not (np.isfinite(mag) and np.isfinite(err) and err > 0):
        return "", "", np.nan, np.nan
    if band in {"BP", "G", "RP"} and not _gaia_band_passes(row, band):
        return "", "", np.nan, np.nan
    if band in {"J", "H", "K"} and not _jhk_band_passes(row, band):
        return "", "", np.nan, np.nan
    if band == "W1" and not _w1_band_passes(row):
        return "", "", np.nan, np.nan
    return mcol, ecol, mag, err


def _has_band_filtered(row, band: str) -> bool:
    return np.isfinite(_band_value_filtered(row, band)[2])


def _row_sed_vector_filtered(row):
    used, obs, err = [], [], []
    for band in SED_BANDS:
        _mcol, _ecol, mag, magerr = _band_value_filtered(row, band)
        if np.isfinite(mag) and np.isfinite(magerr) and magerr > 0:
            used.append(band)
            obs.append(mag)
            err.append(magerr)
    return used, np.asarray(obs, dtype=float), np.asarray(err, dtype=float)


def _assign_quality_tier_filtered(row) -> str:
    has = {band: _has_band_filtered(row, band) for band in SED_BANDS}
    has_gbprp = all(has[b] for b in ["BP", "G", "RP"])
    n_red = int(sum(has[b] for b in ["J", "H", "K", "W1"]))
    if has_gbprp and n_red == 4:
        return "primary_7band"
    if has_gbprp and n_red >= 3:
        return NEW_QUALITY_MODE
    if has_gbprp and n_red >= 1:
        return "secondary_4plus"
    return "not_fit"


def _base_fail_reason_filtered(row, quality_mode: str) -> str:
    tier = row.get("quality_tier", "not_fit")
    if quality_mode == "primary" and tier != "primary_7band":
        return "not_primary_7band"
    if quality_mode == NEW_QUALITY_MODE and tier not in {"primary_7band", NEW_QUALITY_MODE}:
        return "not_GBPRP_plus_3of4_JHKW1"
    if quality_mode in ("secondary", "all") and tier == "not_fit":
        return "missing_GBPRP_or_no_JHKW1"
    if quality_mode == "all" and tier not in {"primary_7band", NEW_QUALITY_MODE, "secondary_4plus"}:
        return "missing_GBPRP_or_no_JHKW1"

    period = _finite(row.get("period_used", np.nan))
    if not np.isfinite(period):
        return "missing_period_used"
    if period <= 0:
        return "bad_period"

    if not np.isfinite(_finite(row.get("mh_prior_mu_used", np.nan))):
        return "missing_mh_prior_mu_used"
    return ""


def _build_work_table_filtered(df: pd.DataFrame, quality_mode: str, max_stars):
    allowed = {"primary", "secondary", "all", NEW_QUALITY_MODE}
    if quality_mode not in allowed:
        raise ValueError(f"quality_mode must be one of {sorted(allowed)}")

    df = df.copy()
    df["quality_tier"] = df.apply(_assign_quality_tier_filtered, axis=1)
    if quality_mode == "primary":
        mask = df["quality_tier"] == "primary_7band"
    elif quality_mode == NEW_QUALITY_MODE:
        mask = df["quality_tier"].isin(["primary_7band", NEW_QUALITY_MODE])
    elif quality_mode == "secondary":
        mask = df["quality_tier"].isin(["primary_7band", NEW_QUALITY_MODE, "secondary_4plus"])
    else:
        mask = np.ones(len(df), dtype=bool)

    out = df.loc[mask].copy()
    if max_stars is not None:
        out = out.head(int(max_stars)).copy()
    return out


def _prior_residuals_missing_wang25(theta, row, latest_module):
    teff, logg, mh, _av, _rv, logd, _logr = np.asarray(theta, dtype=float)
    res = []

    mu_loggF = latest_module.loggF_prior_mu_latest(row)
    if np.isfinite(mu_loggF) and np.isfinite(teff) and teff > 0:
        loggF = logg - 4.0 * np.log10(teff / 1.0e4)
        res.append((loggF - mu_loggF) / latest_module.LOGGF_PRIOR_SIGMA)

    mu_mh = _finite(row.get("mh_prior_mu_used", np.nan))
    sig_mh = _finite(row.get("mh_prior_sigma", latest_module.FE_GRAD_SIGMA))
    if np.isfinite(mu_mh) and np.isfinite(sig_mh) and sig_mh > 0:
        res.append((mh - mu_mh) / sig_mh)

    if bool(row.get("uses_distance_prior_for_source", False)):
        d_prior = _finite(row.get("<d>", np.nan))
        d_err = _finite(row.get("<d_error>", np.nan))
        d_model = 10.0 ** logd
        if np.isfinite(d_prior) and np.isfinite(d_err) and d_prior > 0 and d_err > 0:
            res.append((d_model - d_prior) / d_err)
    return np.asarray(res, dtype=float)


def _make_summarize_missing_wang25(previous_summarize, latest_module, fix_logg_mh: bool):
    def summarize(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
        out = previous_summarize(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin)
        has_dist = _has_valid_distance(row)
        uses_dist = _source_uses_distance_prior(row)
        mu_mh, mh_src = _mh_prior_mu_used(row)
        out["has_Wang25_distance"] = bool(has_dist)
        out["uses_distance_prior_for_source"] = bool(uses_dist)
        out["distance_prior_source"] = "<d>" if uses_dist else ("disabled_by_user" if has_dist else "none_missing_Wang25")
        out["distance_prior_kpc"] = _finite(row.get("<d>", np.nan)) if has_dist else np.nan
        out["distance_prior_err_kpc"] = _finite(row.get("<d_error>", np.nan)) if has_dist else np.nan
        out["mh_prior_source"] = "fixed_MH0_no_prior" if fix_logg_mh else mh_src
        out["mh_prior_mu_used"] = np.nan if fix_logg_mh else float(mu_mh)
        out["mh_prior_sigma"] = np.nan if fix_logg_mh else float(latest_module.FE_GRAD_SIGMA)
        out["MH_prior"] = np.nan if fix_logg_mh else float(mu_mh)
        out["MH_prior_sigma"] = np.nan if fix_logg_mh else float(latest_module.FE_GRAD_SIGMA)
        out["uses_distance_prior"] = bool(uses_dist)
        if not fix_logg_mh:
            out["uses_mh_prior"] = True
            out["uses_RGC_FeH_prior"] = bool(has_dist)
            mh = out.get("MH_p50", np.nan)
            out["MH_prior_delta_sigma"] = (
                float((mh - mu_mh) / latest_module.FE_GRAD_SIGMA)
                if np.isfinite(mh) and np.isfinite(mu_mh)
                else np.nan
            )
        if not uses_dist:
            out["D_prior_pull"] = np.nan
        return out

    return summarize


def _patch_filtered_behavior(latest_module, directbase, mapbase, *, fix_logg_mh: bool):
    global ORIGINAL_BAND_VALUE, ORIGINAL_BUILD_WORK_TABLE
    global ORIGINAL_PRIOR_RESIDUALS, ORIGINAL_SUMMARIZE_WITH_MISSING

    ORIGINAL_BAND_VALUE = latest_module._band_value
    ORIGINAL_BUILD_WORK_TABLE = mapbase.build_work_table
    ORIGINAL_PRIOR_RESIDUALS = mapbase.prior_residuals

    original_read = mapbase.read_merged_input

    def read_merged_input_filtered():
        df = original_read()
        return _prepare_missing_wang25_columns(df, latest_module)

    mapbase.read_merged_input = read_merged_input_filtered
    mapbase.has_band = _has_band_filtered
    mapbase.row_sed_vector = _row_sed_vector_filtered
    mapbase.assign_quality_tier = _assign_quality_tier_filtered
    mapbase.base_fail_reason = _base_fail_reason_filtered
    mapbase.build_work_table = _build_work_table_filtered

    if not fix_logg_mh:
        mapbase.prior_residuals = lambda theta, row: _prior_residuals_missing_wang25(theta, row, latest_module)
        latest_module.prior_residuals_latest = mapbase.prior_residuals

    previous_summarize = directbase.summarize_samples
    directbase.summarize_samples = _make_summarize_missing_wang25(previous_summarize, latest_module, fix_logg_mh)
    ORIGINAL_SUMMARIZE_WITH_MISSING = directbase.summarize_samples


def _patch_missing_distance_initializer(directbase):
    """Avoid degenerate walker starts when a source has no distance prior."""
    global ORIGINAL_DRAW_ONE_PRIOR_POSITION, ORIGINAL_INITIALIZE_WALKERS
    ORIGINAL_DRAW_ONE_PRIOR_POSITION = directbase._draw_one_prior_position
    ORIGINAL_INITIALIZE_WALKERS = directbase.initialize_walkers_direct

    def draw_one_prior_position_filtered(row, bounds, rng):
        p = ORIGINAL_DRAW_ONE_PRIOR_POSITION(row, bounds, rng)
        if _source_uses_distance_prior(row):
            return p
        names = list(directbase.PARAM_NAMES)
        if "logD" not in names:
            return p
        idx = names.index("logD")
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)
        center = np.asarray(directbase._direct_center(row, bounds), dtype=float)
        # With no distance prior the old draw kept logD fixed at the center,
        # which can make emcee reject the initial ensemble.  A 0.35 dex scatter
        # is broad but still comfortably inside the model bounds for most stars.
        p[idx] = rng.normal(center[idx], 0.35)
        p[idx] = np.clip(p[idx], lower[idx] + 1.0e-6, upper[idx] - 1.0e-6)
        return p

    directbase._draw_one_prior_position = draw_one_prior_position_filtered

    def initialize_walkers_direct_filtered(row, bounds, nwalkers, random_seed):
        rng = np.random.default_rng(random_seed)
        ndim = len(directbase.PARAM_NAMES)
        nwalkers_eff = int(max(nwalkers, 2 * ndim + 2))
        lower = np.asarray(bounds["lower"], dtype=float)
        upper = np.asarray(bounds["upper"], dtype=float)

        p0 = []
        n_try = 0
        while len(p0) < nwalkers_eff and n_try < 10000:
            n_try += 1
            p = directbase._draw_one_prior_position(row, bounds, rng)
            if np.all(np.isfinite(p)) and np.all(p > lower) and np.all(p < upper):
                p0.append(p)

        if len(p0) < nwalkers_eff:
            center = np.asarray(directbase._direct_center(row, bounds), dtype=float)
            scale = np.full(ndim, 0.03, dtype=float)
            names = list(directbase.PARAM_NAMES)
            for name, value in {"Teff": 50.0, "Av": 0.05, "Rv": 0.03, "logD": 0.005, "logR": 0.02}.items():
                if name in names:
                    scale[names.index(name)] = value
            if (not _source_uses_distance_prior(row)) and "logD" in names:
                scale[names.index("logD")] = 0.35
            while len(p0) < nwalkers_eff:
                p = center + rng.normal(0.0, scale)
                p = np.clip(p, lower + 1.0e-6, upper - 1.0e-6)
                p0.append(p)
        return np.asarray(p0, dtype=float)

    directbase.initialize_walkers_direct = initialize_walkers_direct_filtered


def _set_active_filter_config(
    *,
    bp_max,
    require_gaia_vari_cepheid,
    jhk_phqual_allowed,
    jhk_cc_flg_allowed,
    apply_jhk_quality_to_literature_mean,
    require_wise_profile_quality,
    wise_phqual_allowed,
    wise_cc_flags_allowed,
):
    global ACTIVE_FILTER_CONFIG
    ACTIVE_FILTER_CONFIG = {
        "bp_max": None if bp_max is None else float(bp_max),
        "require_gaia_vari_cepheid": bool(require_gaia_vari_cepheid),
        "jhk_phqual_allowed": _as_allowed_tuple(jhk_phqual_allowed),
        "jhk_cc_flg_allowed": _as_allowed_tuple(jhk_cc_flg_allowed),
        "apply_jhk_quality_to_literature_mean": bool(apply_jhk_quality_to_literature_mean),
        "require_wise_profile_quality": bool(require_wise_profile_quality),
        "wise_phqual_allowed": _as_allowed_tuple(wise_phqual_allowed),
        "wise_cc_flags_allowed": _as_allowed_tuple(wise_cc_flags_allowed),
    }


def _theta_from_summary(summary_row, param_names):
    return np.asarray([summary_row.get(f"{p}_p50", np.nan) for p in param_names], dtype=float)


def _robust_scatter(vals):
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    med = np.nanmedian(arr)
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))


def _summarize_residual_groups(long_df: pd.DataFrame):
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame()
    group_specs = [
        ["band"],
        ["band", "floor_group"],
        ["band", "source_label"],
        ["band", "Mode"],
        ["band", "has_Wang25_distance"],
        ["band", "floor_group", "Mode"],
    ]
    rows = []
    for keys in group_specs:
        for key_vals, sub in long_df.groupby(keys, dropna=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            entry = {"group_keys": ",".join(keys), "N": int(len(sub))}
            for key, val in zip(keys, key_vals):
                entry[key] = val
            vals = pd.to_numeric(sub["normalized_residual_with_floor"], errors="coerce")
            entry["median_normalized_residual"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
            entry["normalized_residual_robust_scatter"] = _robust_scatter(vals)
            entry["p16_normalized_residual"] = float(np.nanpercentile(vals, 16)) if np.isfinite(vals).any() else np.nan
            entry["p84_normalized_residual"] = float(np.nanpercentile(vals, 84)) if np.isfinite(vals).any() else np.nan
            rows.append(entry)
    return pd.DataFrame(rows)


def _build_residual_tables(selected, summary, directbase, mapbase, out_dir: Path):
    if selected is None or summary is None or len(selected) == 0 or len(summary) == 0:
        long_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        long_df.to_csv(out_dir / "normalized_residual_by_band_source.csv", index=False)
        long_df.to_csv(out_dir / "final_normalized_residual_by_band_source.csv", index=False)
        summary_df.to_csv(out_dir / "normalized_residual_robust_scatter_summary.csv", index=False)
        return long_df, summary_df

    model, zps = sfloor._summary_model_and_zps(directbase, mapbase)
    selected_by_tag = selected.set_index("selection_tag", drop=False)
    rows = []
    for _, srow in summary.iterrows():
        if not bool(srow.get("fit_success", True)):
            continue
        tag = srow.get("selection_tag", "")
        if tag not in selected_by_tag.index:
            continue
        row = selected_by_tag.loc[tag]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            sed_bands, sed_obs, sed_err_cat = sfloor.ORIGINAL_ROW_SED_VECTOR(row)
            floors, groups, labels = sfloor.source_floor_vector(row, sed_bands)
            final_band_text = str(srow.get("bands_final", "")).strip()
            if final_band_text:
                final_band_set = {
                    band.strip() for band in final_band_text.split(",") if band.strip()
                }
                keep = np.asarray([band in final_band_set for band in sed_bands], dtype=bool)
                sed_bands = [band for band, use in zip(sed_bands, keep) if use]
                sed_obs = np.asarray(sed_obs, dtype=float)[keep]
                sed_err_cat = np.asarray(sed_err_cat, dtype=float)[keep]
                floors = np.asarray(floors, dtype=float)[keep]
                groups = np.asarray(groups, dtype=object)[keep]
                labels = np.asarray(labels, dtype=object)[keep]
                if not sed_bands:
                    raise ValueError("no final retained SED bands")
            sed_err_eff = np.sqrt(np.asarray(sed_err_cat, dtype=float) ** 2 + floors**2)
            theta = _theta_from_summary(srow, directbase.PARAM_NAMES)
            pred = mapbase.model_mags(theta, model, zps, sed_bands)
        except Exception as exc:
            rows.append(
                {
                    "selection_tag": tag,
                    "Name": row.get("Name", ""),
                    "band": "",
                    "residual_warning": str(exc),
                }
            )
            continue
        for i, band in enumerate(sed_bands):
            rows.append(
                {
                    "selection_tag": tag,
                    "Name": row.get("Name", ""),
                    "GaiaDR3": row.get("GaiaDR3", ""),
                    "Mode": row.get("Mode", ""),
                    "quality_tier": row.get("quality_tier", ""),
                    "has_Wang25_distance": bool(row.get("has_Wang25_distance", False)),
                    "uses_distance_prior_for_source": bool(row.get("uses_distance_prior_for_source", False)),
                    "band": band,
                    "obs_mag": float(sed_obs[i]),
                    "model_mag": float(pred[i]),
                    "residual_mag": float(sed_obs[i] - pred[i]),
                    "catalog_magerr": float(sed_err_cat[i]),
                    "floor_mag": float(floors[i]),
                    "effective_magerr": float(sed_err_eff[i]),
                    "normalized_residual_with_floor": float((sed_obs[i] - pred[i]) / sed_err_eff[i]),
                    "floor_group": groups[i],
                    "source_label": labels[i],
                    "BP_source": row.get("BP_source", ""),
                    "G_source": row.get("G_source", ""),
                    "RP_source": row.get("RP_source", ""),
                    "W1_source": row.get("W1_source", ""),
                }
            )
    long_df = pd.DataFrame(rows)
    summary_df = _summarize_residual_groups(long_df)
    long_df.to_csv(out_dir / "normalized_residual_by_band_source.csv", index=False)
    long_df.to_csv(out_dir / "final_normalized_residual_by_band_source.csv", index=False)
    summary_df.to_csv(out_dir / "normalized_residual_robust_scatter_summary.csv", index=False)
    try:
        model.close()
    except Exception:
        pass
    return long_df, summary_df


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
    text = str(value)
    return (
        text.strip()
        .replace(" ", "")
        .replace(".", "p")
        .replace("-", "m")
        .replace("/", "")
        .replace(",", "")
        .replace("=", "")
    )


def _allowed_slug(values, prefix: str) -> str:
    vals = _as_allowed_tuple(values)
    if vals is None:
        return f"{prefix}any"
    compact = "".join(str(v) for v in vals)
    compact = compact.replace("0", "0")
    return f"{prefix}{_slug_value(compact)}"


def _floor_config_hash(floor_config, use_source_dependent_floor: bool) -> str:
    if not use_source_dependent_floor:
        return "nofloor"
    cfg = sfloor.DEFAULT_FLOOR_CONFIG if floor_config is None else floor_config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _config_slug(
    fix_logg_mh: bool,
    use_vi_pc_prior: bool,
    *,
    fixed_logg=2.0,
    fixed_mh=0.0,
    quality_mode: str = NEW_QUALITY_MODE,
    bp_max=20.3,
    require_gaia_vari_cepheid: bool = False,
    use_distance_prior: bool = True,
    scale_parameterization="logR_logD",
    pc_prior_mode="1sigma",
    pc_prior_scale=None,
    extinction_law=DEFAULT_EXTINCTION_LAW,
    extinction_parameterization="A550_R550",
    band_extinction_draws=512,
    cache_file=None,
    zp_file=None,
    iterative_residual_clipping=False,
    residual_clip_sigma=3.0,
    min_bands_after_clip=6,
    jhk_phqual_allowed=("A", "B", "C", "D"),
    jhk_cc_flg_allowed=("0",),
    apply_jhk_quality_to_literature_mean: bool = False,
    require_wise_profile_quality: bool = True,
    wise_phqual_allowed=("A", "B", "C"),
    wise_cc_flags_allowed=("0",),
    use_source_dependent_floor: bool = True,
    floor_config=None,
) -> str:
    if fix_logg_mh:
        base = (
            f"fixedLogg{_slug_value(fixed_logg)}_MH{_slug_value(fixed_mh)}_withVI"
            if use_vi_pc_prior
            else f"fixedLogg{_slug_value(fixed_logg)}_MH{_slug_value(fixed_mh)}_noVI"
        )
    else:
        base = "variableLoggMH_withVI" if use_vi_pc_prior else "variableLoggMH_noVI"
    parts = [
        f"{base}_sourceFloor_filtered",
        _slug_value(quality_mode),
        f"bp{_slug_value(bp_max)}",
        "gaiaVar" if require_gaia_vari_cepheid else "gaiaAny",
        "withDprior" if use_distance_prior else "noDprior",
        _scale_parameterization_slug(scale_parameterization),
        _pc_prior_slug(pc_prior_mode, pc_prior_scale),
        _extinction_law_slug(extinction_law, cache_file),
        (
            "nativeA550R550"
            if _normalize_extinction_parameterization(extinction_parameterization) == "A550_R550"
            else "nativeAvRv"
            if _normalize_extinction_parameterization(extinction_parameterization) == "Av_Rv"
            else "legacyAvRv"
        ),
        (
            f"bandExt{int(band_extinction_draws)}"
            if _normalize_extinction_parameterization(extinction_parameterization)
            in NATIVE_EXTINCTION_PARAMETERIZATIONS
            else "bandExtNA"
        ),
        "VegaCALSPEC012" if "alpha_lyr_stis_012" in Path(str(zp_file or "")).name.lower() else "VegaDefault",
        (
            f"iterClip{_slug_value(residual_clip_sigma)}_minBand{int(min_bands_after_clip)}"
            if iterative_residual_clipping
            else "noResidualClip"
        ),
        _allowed_slug(jhk_phqual_allowed, "jhkph"),
        _allowed_slug(jhk_cc_flg_allowed, "jhkcc"),
        "jhkLitQ" if apply_jhk_quality_to_literature_mean else "jhkLitNoQ",
        "wiseQ" if require_wise_profile_quality else "wiseNoQ",
        _allowed_slug(wise_phqual_allowed, "wiseph"),
        _allowed_slug(wise_cc_flags_allowed, "wisecc"),
        f"floor{_floor_config_hash(floor_config, use_source_dependent_floor)}",
    ]
    return "_".join(parts)


def run_mw_sed_latest_pc_source_dependent_floor_filtered_distanceprior_emcee_direct(
    n_stars=30,
    quality_mode=NEW_QUALITY_MODE,
    bp_max=20.3,
    require_gaia_vari_cepheid=False,
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
    extinction_law=DEFAULT_EXTINCTION_LAW,
    extinction_parameterization="A550_R550",
    extinction_law_cache_map=None,
    build_native_cache_if_missing=True,
    native_cache_build_kwargs=None,
    compute_band_extinction_posterior=True,
    band_extinction_draws=512,
    band_extinction_batch_size=512,
    save_band_extinction_draws=False,
    native_vertex_cache_size=128,
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
    return_samples_to_notebook=True,
    iterative_residual_clipping=False,
    residual_clip_sigma=3.0,
    min_bands_after_clip=6,
    required_g23_dust_extinction_version="1.7",
):
    """Run the filtered source-dependent-floor SED fit."""
    _set_active_filter_config(
        bp_max=bp_max,
        require_gaia_vari_cepheid=require_gaia_vari_cepheid,
        jhk_phqual_allowed=jhk_phqual_allowed,
        jhk_cc_flg_allowed=jhk_cc_flg_allowed,
        apply_jhk_quality_to_literature_mean=apply_jhk_quality_to_literature_mean,
        require_wise_profile_quality=require_wise_profile_quality,
        wise_phqual_allowed=wise_phqual_allowed,
        wise_cc_flags_allowed=wise_cc_flags_allowed,
    )

    global ACTIVE_USE_DISTANCE_PRIOR
    fix_logg_mh = bool(fix_logg_mh)
    use_vi_pc_prior = bool(use_vi_pc_prior)
    ACTIVE_USE_DISTANCE_PRIOR = bool(use_distance_prior)
    scale_parameterization = _normalize_scale_parameterization(scale_parameterization)
    extinction_parameterization = _normalize_extinction_parameterization(extinction_parameterization)
    if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS:
        if use_distance_prior:
            raise ValueError("scale_parameterization='logRoverD_only' requires use_distance_prior=False")
        if scale_parameterization != "logRoverD_only":
            raise ValueError(
                f"extinction_parameterization={extinction_parameterization!r} "
                "requires scale_parameterization='logRoverD_only'"
            )
    elif scale_parameterization == "logRoverD_only":
        raise ValueError(
            "scale_parameterization='logRoverD_only' is available with "
            "extinction_parameterization='A550_R550' or 'Av_Rv'"
        )
    _set_active_pc_prior_config(pc_prior_mode, pc_prior_scale)
    extinction_law_requested, extinction_law_used, resolved_cache_file, extinction_cache_source = _resolve_extinction_cache_file(
        extinction_law=extinction_law,
        cache_file=cache_file,
        extinction_law_cache_map=extinction_law_cache_map,
        extinction_parameterization=extinction_parameterization,
        build_native_cache_if_missing=build_native_cache_if_missing,
        native_cache_build_kwargs=native_cache_build_kwargs,
    )
    cache_validation = None
    if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS and extinction_law_used == "G23":
        cache_validation = _validate_native_g23_cache(
            resolved_cache_file,
            required_dust_extinction_version=required_g23_dust_extinction_version,
        )
    custom_out_dir_supplied = out_dir is not None
    config_slug = _config_slug(
        fix_logg_mh,
        use_vi_pc_prior,
        fixed_logg=fixed_logg,
        fixed_mh=fixed_mh,
        quality_mode=quality_mode,
        bp_max=bp_max,
        require_gaia_vari_cepheid=require_gaia_vari_cepheid,
        use_distance_prior=use_distance_prior,
        scale_parameterization=scale_parameterization,
        pc_prior_mode=pc_prior_mode,
        pc_prior_scale=pc_prior_scale,
        extinction_law=extinction_law_used,
        extinction_parameterization=extinction_parameterization,
        band_extinction_draws=band_extinction_draws,
        cache_file=resolved_cache_file if extinction_cache_source == "explicit_cache_file" else None,
        zp_file=zp_file,
        iterative_residual_clipping=iterative_residual_clipping,
        residual_clip_sigma=residual_clip_sigma,
        min_bands_after_clip=min_bands_after_clip,
        jhk_phqual_allowed=jhk_phqual_allowed,
        jhk_cc_flg_allowed=jhk_cc_flg_allowed,
        apply_jhk_quality_to_literature_mean=apply_jhk_quality_to_literature_mean,
        require_wise_profile_quality=require_wise_profile_quality,
        wise_phqual_allowed=wise_phqual_allowed,
        wise_cc_flags_allowed=wise_cc_flags_allowed,
        use_source_dependent_floor=use_source_dependent_floor,
        floor_config=floor_config,
    )
    out_dir = Path(out_dir) if out_dir is not None else Path(out_root) / config_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if fix_logg_mh and extinction_parameterization not in NATIVE_EXTINCTION_PARAMETERIZATIONS:
        fixed_module = sfloor._load_fixed_module()
        latest_module = fixed_module.latest
        directbase = fixed_module.directbase
        mapbase = fixed_module.mapbase
        sfloor._set_pc_colors(latest_module, use_vi_pc_prior)
        latest_module._apply_latest_patches(
            out_dir=out_dir,
            phot_file=phot_file or latest_module.DEFAULT_PHOT_FILE,
            wang25_file=wang25_file or latest_module.DEFAULT_WANG25_FILE,
            old_phot_file=old_phot_file or latest_module.DEFAULT_OLD_PHOT_FILE,
            zp_file=zp_file or latest_module.DEFAULT_ZP_FILE,
            cache_file=resolved_cache_file,
            vi_cache_file=vi_cache_file or latest_module.DEFAULT_VI_CACHE_FILE,
            pc_coeff_file=pc_coeff_file or latest_module.DEFAULT_PC_COEFF_FILE,
            pc_cov_file=pc_cov_file or latest_module.DEFAULT_PC_COV_FILE,
        )
        fixed_module._apply_fixed_patches(out_dir)
        directbase = fixed_module.directbase
        mapbase = fixed_module.mapbase
        _patch_pc_prior_strength(latest_module, mapbase)
    else:
        _mapbase0, _directbase0, latest_module = sfloor._load_modules()
        directbase = latest_module.directbase
        mapbase = latest_module.mapbase
        sfloor._set_pc_colors(latest_module, use_vi_pc_prior)
        latest_module._apply_latest_patches(
            out_dir=out_dir,
            phot_file=phot_file or latest_module.DEFAULT_PHOT_FILE,
            wang25_file=wang25_file or latest_module.DEFAULT_WANG25_FILE,
            old_phot_file=old_phot_file or latest_module.DEFAULT_OLD_PHOT_FILE,
            zp_file=zp_file or latest_module.DEFAULT_ZP_FILE,
            cache_file=resolved_cache_file,
            vi_cache_file=vi_cache_file or latest_module.DEFAULT_VI_CACHE_FILE,
            pc_coeff_file=pc_coeff_file or latest_module.DEFAULT_PC_COEFF_FILE,
            pc_cov_file=pc_cov_file or latest_module.DEFAULT_PC_COV_FILE,
        )
        _patch_pc_prior_strength(latest_module, mapbase)

    _patch_filtered_behavior(latest_module, directbase, mapbase, fix_logg_mh=fix_logg_mh)
    _patch_missing_distance_initializer(directbase)

    sfloor._apply_source_floor_patches(
        directbase=directbase,
        mapbase=mapbase,
        out_dir=out_dir,
        use_source_dependent_floor=use_source_dependent_floor,
        floor_config=floor_config,
    )
    if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS:
        _patch_native_a550_r550_logroverd_only(
            directbase,
            mapbase,
            latest_module,
            extinction_parameterization=extinction_parameterization,
            fix_logg_mh=fix_logg_mh,
            fixed_logg=fixed_logg,
            fixed_mh=fixed_mh,
            band_extinction_draws=band_extinction_draws,
            band_extinction_batch_size=band_extinction_batch_size,
            compute_band_extinction_posterior=compute_band_extinction_posterior,
            save_band_extinction_draws=save_band_extinction_draws,
            native_vertex_cache_size=native_vertex_cache_size,
        )
    else:
        _patch_scale_parameterization(directbase, mapbase, latest_module, scale_parameterization)

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
        return_samples_to_notebook=return_samples_to_notebook,
        iterative_residual_clipping=iterative_residual_clipping,
        residual_clip_sigma=residual_clip_sigma,
        min_bands_after_clip=min_bands_after_clip,
    )

    if len(summary) > 0:
        summary = summary.copy()
        for key, val in ACTIVE_FILTER_CONFIG.items():
            summary[key] = json.dumps(val) if isinstance(val, tuple) else val
        summary["fit_configuration"] = config_slug
        summary["auto_config_slug"] = config_slug
        summary["quality_mode_requested"] = quality_mode
        summary["fix_logg_mh"] = bool(fix_logg_mh)
        summary["fixed_logg_value"] = float(fixed_logg) if fix_logg_mh else np.nan
        summary["fixed_MH_value"] = float(fixed_mh) if fix_logg_mh else np.nan
        summary["sampled_parameter_names"] = json.dumps(
            (
                ["Teff", "Av", "Rv", "logR_over_D"]
                if extinction_parameterization == "Av_Rv"
                else ["Teff", "A550", "R550", "logR_over_D"]
            )
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS and fix_logg_mh
            else list(directbase.PARAM_NAMES)
        )
        summary["fixed_parameter_names"] = json.dumps(["logg", "MH"] if fix_logg_mh else [])
        summary["use_vi_pc_prior"] = bool(use_vi_pc_prior)
        summary["scale_parameterization"] = scale_parameterization
        summary["pc_prior_color_set"] = "with_VI" if use_vi_pc_prior else "no_VI"
        summary["pc_prior_mode"] = ACTIVE_PC_PRIOR_MODE
        summary["pc_prior_enabled"] = bool(ACTIVE_PC_PRIOR_ENABLED)
        summary["pc_prior_scale"] = float(ACTIVE_PC_PRIOR_SCALE) if ACTIVE_PC_PRIOR_ENABLED else np.nan
        summary["uses_pc_prior"] = bool(ACTIVE_PC_PRIOR_ENABLED)
        summary["extinction_law_requested"] = extinction_law_requested
        summary["extinction_law_used"] = extinction_law_used
        summary["extinction_parameterization"] = extinction_parameterization
        summary["extinction_sampling_coordinates"] = extinction_parameterization
        summary["extinction_physical_prior"] = (
            "uniform_in_native_A550_R550"
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS
            else "legacy_parameterization_prior"
        )
        summary["extinction_coordinate_jacobian_applied"] = bool(
            extinction_parameterization == "Av_Rv"
        )
        summary["extinction_cache_file"] = str(resolved_cache_file)
        summary["extinction_cache_source"] = extinction_cache_source
        summary["vega_zero_point_file"] = str(
            zp_file or latest_module.DEFAULT_ZP_FILE
        )
        summary["iterative_residual_clipping"] = bool(iterative_residual_clipping)
        summary["residual_clip_sigma"] = float(residual_clip_sigma)
        summary["min_bands_after_clip"] = int(min_bands_after_clip)
        summary["uses_fixed_band_floor"] = bool(use_source_dependent_floor)
        summary["floor_config_name"] = "default_source_dependent" if floor_config is None else "custom_source_dependent"
        summary["floor_config_json"] = json.dumps(sfloor.ACTIVE_FLOOR_CONFIG, sort_keys=True)
        summary.to_csv(directbase.SUMMARY_OUT, index=False)

    if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS:
        _write_native_mapping_and_band_extinction_summary(
            resolved_cache_file,
            out_dir,
            summary,
            write_band_summary=compute_band_extinction_posterior,
        )

    long_residuals, residual_summary = _build_residual_tables(selected, summary, directbase, mapbase, out_dir)

    clipping_stats_path = out_dir / "residual_clipping_statistics.csv"
    if clipping_stats_path.exists():
        clipping_stats_frame = pd.read_csv(clipping_stats_path)
        clipping_stats = {
            str(row["metric"]): int(row["value"])
            for _, row in clipping_stats_frame.iterrows()
        }
    else:
        clipping_stats = {}

    runinfo = {
        "method": "latest pure-photometry SED direct emcee with filters and fixed source-dependent floors",
        "out_dir": str(out_dir),
        "fit_configuration": config_slug,
        "auto_config_slug": config_slug,
        "custom_out_dir_supplied": bool(custom_out_dir_supplied),
        "n_selected": int(len(selected)),
        "n_success": int(summary["fit_success"].fillna(False).astype(bool).sum())
        if "fit_success" in summary.columns
        else int(len(summary)),
        "n_fit_failed": int((~summary["fit_success"].fillna(False).astype(bool)).sum())
        if "fit_success" in summary.columns
        else 0,
        "quality_mode": quality_mode,
        "filter_config": copy.deepcopy(ACTIVE_FILTER_CONFIG),
        "fix_logg_mh": bool(fix_logg_mh),
        "fixed_logg": float(fixed_logg) if fix_logg_mh else None,
        "fixed_MH": float(fixed_mh) if fix_logg_mh else None,
        "sampled_parameter_names": (
            (
                ["Teff", "Av", "Rv", "logR_over_D"]
                if extinction_parameterization == "Av_Rv"
                else ["Teff", "A550", "R550", "logR_over_D"]
            )
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS and fix_logg_mh
            else list(directbase.PARAM_NAMES)
        ),
        "fixed_parameter_names": ["logg", "MH"] if fix_logg_mh else [],
        "use_vi_pc_prior": bool(use_vi_pc_prior),
        "use_distance_prior": bool(use_distance_prior),
        "scale_parameterization": scale_parameterization,
        "pc_prior_mode": ACTIVE_PC_PRIOR_MODE,
        "pc_prior_enabled": bool(ACTIVE_PC_PRIOR_ENABLED),
        "pc_prior_scale": float(ACTIVE_PC_PRIOR_SCALE) if ACTIVE_PC_PRIOR_ENABLED else None,
        "extinction_law_requested": extinction_law_requested,
        "extinction_law_used": extinction_law_used,
        "extinction_parameterization": extinction_parameterization,
        "extinction_sampling_coordinates": extinction_parameterization,
        "extinction_derived_coordinates": (
            ["Av", "Rv"]
            if extinction_parameterization == "A550_R550"
            else ["A550", "R550"]
            if extinction_parameterization == "Av_Rv"
            else []
        ),
        "extinction_physical_prior": (
            "uniform_in_native_A550_R550"
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS
            else "legacy_parameterization_prior"
        ),
        "extinction_coordinate_jacobian_applied": bool(
            extinction_parameterization == "Av_Rv"
        ),
        "extinction_coordinate_transform": (
            {
                "A550": "Av * k550(Rv)",
                "R550": "k550(Rv) / (k440(Rv) - k550(Rv))",
                "jacobian": "k550(Rv) * abs(dR550/dRv)",
                "mapping_source": "native_shape_for_r550 and extinction_law in the native cache",
            }
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS
            else None
        ),
        "extinction_cache_file": str(resolved_cache_file),
        "extinction_cache_source": extinction_cache_source,
        "extinction_cache_validation": cache_validation,
        "required_g23_dust_extinction_version": required_g23_dust_extinction_version,
        "vega_zero_point_file": str(zp_file or latest_module.DEFAULT_ZP_FILE),
        "iterative_residual_clipping": bool(iterative_residual_clipping),
        "residual_clip_sigma": float(residual_clip_sigma),
        "residual_clip_reference": "posterior_median",
        "min_bands_after_clip": int(min_bands_after_clip),
        "residual_clipping_statistics": clipping_stats,
        "available_extinction_laws": {k: str(v) for k, v in DEFAULT_EXTINCTION_LAW_CACHE_MAP.items()},
        "available_A550_R550_caches": {k: str(v) for k, v in DEFAULT_A550_R550_CACHE_MAP.items()},
        "A550_definition": "monochromatic extinction at 550 nm" if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None,
        "R550_definition": "A550 / (A440 - A550)" if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None,
        "Av_definition": f"native {extinction_law_used} A(V) amplitude" if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None,
        "Rv_definition": f"native {extinction_law_used} R(V) shape parameter" if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None,
        "band_extinction_posterior_draws": (
            int(band_extinction_draws)
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS and compute_band_extinction_posterior
            else 0 if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None
        ),
        "band_extinction_batch_size": (
            int(band_extinction_batch_size)
            if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS and compute_band_extinction_posterior
            else 0 if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None
        ),
        "band_extinction_computed": bool(compute_band_extinction_posterior) if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else False,
        "native_vertex_cache_size": int(native_vertex_cache_size) if extinction_parameterization in NATIVE_EXTINCTION_PARAMETERIZATIONS else None,
        "save_band_extinction_draws": bool(save_band_extinction_draws),
        "build_native_cache_if_missing": bool(build_native_cache_if_missing),
        "native_cache_build_kwargs": native_cache_build_kwargs,
        "pc_color_defs": sfloor._pc_color_defs(use_vi_pc_prior),
        "sed_bands": SED_BANDS,
        "uses_pc_prior": bool(ACTIVE_PC_PRIOR_ENABLED),
        "uses_distance_prior": "per_source_if_Wang25_distance_available" if use_distance_prior else "disabled_by_user",
        "missing_Wang25_distance_policy": {
            "uses_distance_prior_for_source": False,
            "mh_prior_mu_used": 0.0,
            "mh_prior_sigma": 0.11,
        },
        "distance_prior_disabled_policy": {
            "uses_distance_prior_for_source": False,
            "distance_prior_kpc_preserved_for_diagnostics": True,
        } if not use_distance_prior else None,
        "uses_fixed_band_floor": bool(use_source_dependent_floor),
        "floor_config_name": "default_source_dependent" if floor_config is None else "custom_source_dependent",
        "floor_config": sfloor.ACTIVE_FLOOR_CONFIG,
        "floor_formula": "sigma_eff^2 = sigma_cat^2 + sigma_floor^2",
        "n_residual_rows": int(len(long_residuals)),
        "nwalkers_requested": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "thin": int(thin),
        "random_seed": int(random_seed),
        "n_workers": int(n_workers),
        "return_samples_to_notebook": bool(return_samples_to_notebook),
        "save_samples": str(save_samples),
        "make_corner": bool(make_corner),
        "make_plots": bool(make_plots),
        "progress": bool(progress),
        "resume_argument": bool(resume),
    }
    sfloor._write_standard_outputs(out_dir, selected, summary, runinfo, directbase.FAIL_OUT)
    return selected, summary, samples, residual_summary


print("Loaded filtered latest MW pure-photometry SED fitter with source-dependent floors.")
print(
    "Call: selected, summary, samples, residual_summary = "
    "run_mw_sed_latest_pc_source_dependent_floor_filtered_distanceprior_emcee_direct(...)"
)
