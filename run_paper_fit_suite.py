#!/usr/bin/env python3
"""Run the six production MCMC configurations used by the Cepheid paper.

The suite is deliberately self-contained at the orchestration layer.  It does
not edit the historical fitting modules.  Two runtime patches add (1) exact
Gaia DR3 source selection for recovery runs and (2) autocorrelation diagnostics
computed from the unthinned post-burn-in chain.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FIT_MODULE = "jupyter_mw_sed_latest_pc_dynamic_phase_floor_filtered_distanceprior_emcee_direct"
FIT_SCRIPT = BASE_DIR / f"{FIT_MODULE}.py"
RUNTIME_DEPENDENCY_FILES = (
    BASE_DIR / "jupyter_mw_sed_latest_pc_source_dependent_floor_filtered_distanceprior_emcee_direct.py",
    BASE_DIR / "bosz_a550_r550_preconvolved.py",
)
SUITE_ROOT = (
    BASE_DIR
    / "outputs"
    / "paper_fit_suite_PCvariCepheidOnly_include2MASS_H_steps10000"
)
PC_DIR = BASE_DIR / "pc_calibration_variCepheidOnly_include2MASS_H"
PC_COEFF_FILE = PC_DIR / "PC_coefficients.csv"
PC_COV_FILE = PC_DIR / "PC_residual_covariance.csv"
ZP_FILE = BASE_DIR / "SED_CALSPEC_alpha_lyr_stis_012_zero_points.csv"
VEGA_FILE = BASE_DIR / "calibration" / "alpha_lyr_stis_012.fits"
DEFAULT_PHOT_FILE = BASE_DIR / "MW_3640_10band_GBPRPVIJHKW1S36_phase_corrected_Bras2025_RPonly.csv"

CACHE_FILES = {
    "G23": BASE_DIR / "bosz_r500_G23_A550_R550_preconvolved_ND.h5",
    "WC2019": BASE_DIR / "bosz_r500_WC2019_A550_R550_preconvolved_ND.h5",
    "F19": BASE_DIR / "bosz_r500_F19_A550_R550_preconvolved_ND.h5",
}

OLD_FIDUCIAL_SUMMARY = (
    BASE_DIR
    / "outputs"
    / "sed_latest_dynamic_phase_floor_filtered"
    / (
        "variableLoggMH_noVI_dynamicPhaseFloor_filtered_"
        "GBPRP_plus_3of4_JHKW1_bp20p3_gaiaVar_withMultiF1O-1O2O_"
        "A550R550_scaleLogRoverDOnly_bandExt512_VegaCALSPEC012_"
        "iterClip3p0_minBand6_cfga65a9de1b4_run636ce0d74992"
    )
    / "summary.csv"
)

SMOKE_SOURCE_IDS = (
    "2168964501736155776",  # F, six bands
    "2018900951456670208",  # F, seven bands
    "4301612233202961024",  # 1O, seven bands
    "5891313563729348480",  # F1O, six bands
    "5601418217705666560",  # 1O2O, seven bands
)

PC_COLORS = ("BP-RP", "RP-J", "RP-H", "RP-K")
PC_COMPLETE_CASE_COUNTS = {"F": 1792, "1O": 1360}
TAU_MIN_CHAIN_MULTIPLES = 50.0

BASE_CHAIN = {"nwalkers": 32, "nsteps": 10000, "burnin": 1000, "thin": 10}
RECOVERY_CHAIN = {"nwalkers": 32, "nsteps": 20000, "burnin": 2000, "thin": 10}
SMOKE_CHAIN = {"nwalkers": 32, "nsteps": 600, "burnin": 100, "thin": 5}

BRANCHES = (
    {"index": 1, "name": "fiducial_G23_pc1sigma", "law": "G23", "pc_mode": "1sigma", "fix": False},
    {"index": 2, "name": "G23_pc2sigma", "law": "G23", "pc_mode": "2sigma", "fix": False},
    {"index": 3, "name": "G23_noPC", "law": "G23", "pc_mode": "none", "fix": False},
    {"index": 4, "name": "G23_fixedLogg2_MH0", "law": "G23", "pc_mode": "1sigma", "fix": True},
    {"index": 5, "name": "WC2019_pc1sigma", "law": "WC2019", "pc_mode": "1sigma", "fix": False},
    {"index": 6, "name": "F19_pc1sigma", "law": "F19", "pc_mode": "1sigma", "fix": False},
)

ACTIVE_SOURCE_IDS: set[str] | None = None
PATCH_GENERATION = "paper-fit-suite-runtime-patch-v6"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_inputs(laws) -> dict:
    caches = {law: CACHE_FILES[law] for law in sorted(set(laws))}
    required = [
        FIT_SCRIPT,
        PC_COEFF_FILE,
        PC_COV_FILE,
        ZP_FILE,
        VEGA_FILE,
        DEFAULT_PHOT_FILE,
        *RUNTIME_DEPENDENCY_FILES,
        *caches.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing suite inputs: {missing}")

    coefficients = pd.read_csv(PC_COEFF_FILE)
    covariance = pd.read_csv(PC_COV_FILE)
    expected = {(family, color) for family in ("F", "1O") for color in PC_COLORS}
    actual = set(zip(coefficients["sample"], coefficients["color"]))
    if len(coefficients) != 8 or actual != expected or not coefficients["fit_success"].astype(bool).all():
        raise ValueError("PC coefficient validation failed")

    covariance_checks = {}
    for family, expected_n in PC_COMPLETE_CASE_COUNTS.items():
        block = covariance.loc[covariance["sample"] == family]
        matrix = (
            block.pivot(index="color_i", columns="color_j", values="covariance")
            .reindex(index=PC_COLORS, columns=PC_COLORS)
            .to_numpy(float)
        )
        if matrix.shape != (4, 4) or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
            raise ValueError(f"{family} PC covariance is incomplete or asymmetric")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues <= 0):
            raise ValueError(f"{family} PC covariance is not positive definite")
        counts = set(block["n_common_all_colors"].astype(int))
        if counts != {expected_n}:
            raise ValueError(f"Unexpected {family} complete-case count: {counts}")
        covariance_checks[family] = {"n_complete": expected_n, "eigenvalues": eigenvalues.tolist()}

    return {
        "fit_script": str(FIT_SCRIPT),
        "fit_script_sha256": sha256(FIT_SCRIPT),
        "runtime_dependencies": {
            str(path): sha256(path) for path in RUNTIME_DEPENDENCY_FILES
        },
        "pc_coeff_file": str(PC_COEFF_FILE),
        "pc_coeff_sha256": sha256(PC_COEFF_FILE),
        "pc_cov_file": str(PC_COV_FILE),
        "pc_cov_sha256": sha256(PC_COV_FILE),
        "pc_covariance_checks": covariance_checks,
        "zp_file": str(ZP_FILE),
        "zp_sha256": sha256(ZP_FILE),
        "vega_file": str(VEGA_FILE),
        "vega_sha256": sha256(VEGA_FILE),
        "phot_file": str(DEFAULT_PHOT_FILE),
        "phot_sha256": sha256(DEFAULT_PHOT_FILE),
        "cache_files": {law: {"path": str(path), "sha256": sha256(path)} for law, path in caches.items()},
        "runtime_patch": PATCH_GENERATION,
    }


def configure_process_environment() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["PYTHONWARNINGS"] = (
        "ignore:invalid value encountered in scalar subtract:RuntimeWarning:emcee.moves.red_blue"
    )


def _sampled_parameter_names(sampler, fallback_names) -> list[str]:
    ndim = int(sampler.get_chain().shape[-1])
    if ndim == 6:
        return ["Teff", "logg", "MH", "A550", "R550", "logR_over_D"]
    if ndim == 4:
        return ["Teff", "A550", "R550", "logR_over_D"]
    fallback = list(fallback_names)
    return fallback[:ndim] if len(fallback) >= ndim else [f"parameter_{i}" for i in range(ndim)]


def _add_tau_diagnostics(out: dict, sampler, param_names, nsteps: int, burnin: int) -> dict:
    post_burn_steps = int(nsteps) - int(burnin)
    out["autocorr_post_burn_steps"] = post_burn_steps
    out["autocorr_method"] = "emcee.get_autocorr_time(discard=burnin,thin=1,tol=0)"
    out["autocorr_error"] = ""
    try:
        tau = np.asarray(
            sampler.get_autocorr_time(discard=int(burnin), thin=1, tol=0, quiet=True),
            dtype=float,
        )
        if len(tau) != len(param_names):
            raise ValueError(f"tau length {len(tau)} != sampled parameter count {len(param_names)}")
    except Exception as exc:
        tau = np.full(len(param_names), np.nan)
        out["autocorr_error"] = f"{type(exc).__name__}: {exc}"

    ratios = []
    for name, value in zip(param_names, tau):
        value = float(value) if np.isfinite(value) and value > 0 else np.nan
        ratio = post_burn_steps / value if np.isfinite(value) else np.nan
        ess = len(sampler.acceptance_fraction) * ratio if np.isfinite(ratio) else np.nan
        out[f"tau_{name}"] = value
        out[f"chain_over_tau_{name}"] = ratio
        out[f"effective_samples_{name}"] = ess
        ratios.append(ratio)
    finite_tau = tau[np.isfinite(tau) & (tau > 0)]
    finite_ratios = np.asarray(ratios, dtype=float)
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
    out["tau_max"] = float(np.max(finite_tau)) if len(finite_tau) == len(param_names) else np.nan
    out["chain_over_tau_min"] = (
        float(np.min(finite_ratios)) if len(finite_ratios) == len(param_names) else np.nan
    )
    out["autocorr_pass_50tau"] = bool(
        len(finite_ratios) == len(param_names)
        and np.all(finite_ratios >= TAU_MIN_CHAIN_MULTIPLES)
    )
    return out


def patch_directbase(directbase) -> None:
    if (
        getattr(directbase.summarize_samples, "_paper_suite_patch_generation", None)
        == PATCH_GENERATION
    ):
        return

    original_summarize = directbase.summarize_samples

    def summarize_with_tau(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin):
        out = original_summarize(row, samples, sampler, sed_bands, pc_prior, nsteps, burnin, thin)
        names = _sampled_parameter_names(sampler, directbase.PARAM_NAMES)
        return _add_tau_diagnostics(out, sampler, names, nsteps, burnin)

    summarize_with_tau._paper_suite_patch_generation = PATCH_GENERATION
    directbase.summarize_samples = summarize_with_tau

    original_clip_failure = directbase._clip_failure

    def clip_failure_with_traceback(row, reason, *args, **kwargs):
        failure = original_clip_failure(row, reason, *args, **kwargs)
        formatted = traceback.format_exc()
        failure["exception_traceback"] = "" if formatted == "NoneType: None\n" else formatted
        return failure

    directbase._clip_failure = clip_failure_with_traceback


def install_runtime_patches(dynamic_module) -> None:
    sfloor = dynamic_module.sfloor
    original_load_modules = sfloor._load_modules
    original_load_fixed = sfloor._load_fixed_module

    def load_modules_patched():
        mapbase, directbase, latest = original_load_modules()
        patch_directbase(directbase)
        return mapbase, directbase, latest

    def load_fixed_patched():
        fixed = original_load_fixed()
        original_apply_fixed = fixed._apply_fixed_patches

        def apply_fixed_then_patch(out_dir):
            original_apply_fixed(out_dir)
            patch_directbase(fixed.directbase)

        fixed._apply_fixed_patches = apply_fixed_then_patch
        return fixed

    sfloor._load_modules = load_modules_patched
    sfloor._load_fixed_module = load_fixed_patched


def load_runner():
    for module_name in list(sys.modules):
        if module_name.startswith(("jupyter_mw_sed_", "jupyter_all_mw_sed_", "run_latest_calspec012")):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(BASE_DIR))
    dynamic = importlib.import_module(FIT_MODULE)
    install_runtime_patches(dynamic)
    return dynamic.run_mw_sed_latest_pc_dynamic_phase_floor_filtered_distanceprior_emcee_direct


def common_fit_kwargs(
    branch: dict,
    output_dir: Path,
    chain: dict,
    n_workers: int,
    phot_file: Path,
) -> dict:
    return {
        "n_stars": None,
        "quality_mode": "GBPRP_plus_3of4_JHKW1",
        "include_multimode": True,
        "allowed_multimode_modes": ("F1O", "1O2O"),
        "bp_max": 20.3,
        "require_gaia_vari_cepheid": True,
        "use_distance_prior": False,
        "pc_prior_mode": branch["pc_mode"],
        "pc_prior_scale": None,
        "pc_coeff_file": str(PC_COEFF_FILE),
        "pc_cov_file": str(PC_COV_FILE),
        "extinction_law": branch["law"],
        "extinction_parameterization": "A550_R550",
        "scale_parameterization": "logRoverD_only",
        "cache_file": str(CACHE_FILES[branch["law"]]),
        "build_native_cache_if_missing": False,
        "native_cache_build_kwargs": None,
        "required_g23_dust_extinction_version": "1.7" if branch["law"] == "G23" else None,
        "zp_file": str(ZP_FILE),
        "phot_file": str(phot_file),
        "iterative_residual_clipping": True,
        "residual_clip_sigma": 3.0,
        "min_bands_after_clip": 6,
        "jhk_phqual_allowed": ("A", "B", "C", "D"),
        "jhk_cc_flg_allowed": ("0",),
        "apply_jhk_quality_to_literature_mean": False,
        "require_wise_profile_quality": True,
        "wise_phqual_allowed": ("A", "B", "C"),
        "wise_cc_flags_allowed": ("0",),
        "fix_logg_mh": bool(branch["fix"]),
        "fixed_logg": 2.0,
        "fixed_mh": 0.0,
        "use_vi_pc_prior": False,
        "gaia_vari_floor": {"BP": 0.002, "G": 0.002, "RP": 0.002},
        "non_gaia_vari_floor": {"BP": 0.060, "G": 0.010, "RP": 0.030},
        "ai_over_arp": 0.95,
        "ax_over_ai_F_by_logp": {
            "logP_break": 1.3,
            "short": {"J": 0.63, "H": 0.50, "K": 0.49},
            "long": {"J": 0.70, "H": 0.63, "K": 0.62},
        },
        "ax_over_ai_1o": {"J": 0.628, "H": 0.417, "K": 0.398},
        "floor_F_phasecorr_JHK": {"J": 0.040, "H": 0.030, "K": 0.020},
        "floor_literature_JHK": {"J": 0.0, "H": 0.0, "K": 0.0},
        "arp_clip_max_by_family": {"F": 0.7335, "1O": 0.3210},
        "a_w1_over_abp": 0.32,
        "lmc_arp_over_abp_median": 0.5978,
        "a_w1_over_arp": None,
        "w1_epoch_coeff": {
            "unWISE": 0.11,
            "AllWISE": 0.25,
            "WISE_AllSky": 1.0 / (2.0 * 2.0**0.5),
        },
        "compute_band_extinction_posterior": False,
        "band_extinction_draws": 512,
        "band_extinction_batch_size": 512,
        "save_band_extinction_draws": False,
        "native_vertex_cache_size": 128,
        **chain,
        "random_seed": 42,
        "n_workers": int(n_workers),
        "save_samples": "all",
        "return_samples_to_notebook": False,
        "make_corner": False,
        "make_plots": False,
        "progress": False,
        "resume": False,
        "out_dir": str(output_dir),
        "overwrite": False,
        "cleanup_incomplete_run": False,
    }


def branch_manifest(branch: dict, chain: dict, input_manifest: dict, run_kind: str) -> dict:
    payload = {
        "suite_version": 1,
        "runtime_patch": PATCH_GENERATION,
        "branch": branch,
        "run_kind": run_kind,
        "chain": chain,
        "common_selection": {
            "quality_mode": "GBPRP_plus_3of4_JHKW1",
            "bp_max": 20.3,
            "require_gaia_vari_cepheid": True,
            "include_multimode": True,
            "allowed_multimode_modes": ["F1O", "1O2O"],
            "minimum_final_bands": 6,
        },
        "input_hashes": input_manifest,
    }
    payload["configuration_hash"] = canonical_hash(payload)
    return payload


def archive_incomplete(path: Path) -> None:
    if not path.exists() or not any(path.iterdir()):
        return
    diagnostic_root = path.parent / "diagnostic_incomplete"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    shutil.move(str(path), str(diagnostic_root / f"{path.name}_{timestamp}"))


def posterior_file_index(output_dir: Path, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        tag = str(row["selection_tag"])
        candidates = sorted(output_dir.glob(f"{tag}_*_samples.csv.gz"))
        rows.append(
            {
                "GaiaDR3": normalize_id(row["GaiaDR3"]),
                "selection_tag": tag,
                "Name": str(row.get("Name", "")),
                "posterior_file": str(candidates[0]) if len(candidates) == 1 else "",
                "posterior_file_count": len(candidates),
            }
        )
    return pd.DataFrame(rows)


def make_final_products(branch_dir: Path, base_dir: Path, recovery_dir: Path | None) -> tuple[pd.DataFrame, dict]:
    base_summary = pd.read_csv(base_dir / "summary.csv", dtype={"GaiaDR3": str})
    base_selected = pd.read_csv(base_dir / "selected.csv", dtype={"GaiaDR3": str})
    if "GaiaDR3" not in base_summary.columns:
        failure_path = base_dir / "failures.csv"
        raise RuntimeError(
            f"No successful fits were produced in {base_dir}; "
            f"inspect {failure_path if failure_path.exists() else base_dir}"
        )
    base_summary["GaiaDR3"] = base_summary["GaiaDR3"].map(normalize_id)
    base_index = posterior_file_index(base_dir, base_selected)
    base_summary = base_summary.merge(base_index[["GaiaDR3", "posterior_file", "posterior_file_count"]], on="GaiaDR3", how="left")
    base_summary["result_source"] = "base"
    base_summary["recovery_attempted"] = False
    base_summary["recovery_succeeded"] = False

    recovery_summary = pd.DataFrame()
    recovery_requested: set[str] = set()
    if recovery_dir is not None and (recovery_dir / "requested_source_ids.txt").exists():
        recovery_requested = {
            normalize_id(line) for line in (recovery_dir / "requested_source_ids.txt").read_text().splitlines() if line.strip()
        }
    if recovery_dir is not None and (recovery_dir / "summary.csv").exists():
        recovery_summary = pd.read_csv(recovery_dir / "summary.csv", dtype={"GaiaDR3": str})
        recovery_selected = pd.read_csv(recovery_dir / "selected.csv", dtype={"GaiaDR3": str})
        if "GaiaDR3" in recovery_summary.columns:
            recovery_summary["GaiaDR3"] = recovery_summary["GaiaDR3"].map(normalize_id)
            recovery_index = posterior_file_index(recovery_dir, recovery_selected)
            recovery_summary = recovery_summary.merge(
                recovery_index[["GaiaDR3", "posterior_file", "posterior_file_count"]], on="GaiaDR3", how="left"
            )
            recovery_summary["result_source"] = "recovery"
            recovery_summary["recovery_attempted"] = True
            recovery_summary["recovery_succeeded"] = True
        else:
            recovery_summary = pd.DataFrame()

    if recovery_requested:
        base_summary.loc[base_summary["GaiaDR3"].isin(recovery_requested), "recovery_attempted"] = True
    if not recovery_summary.empty:
        final = pd.concat(
            [base_summary.loc[~base_summary["GaiaDR3"].isin(set(recovery_summary["GaiaDR3"]))], recovery_summary],
            ignore_index=True,
            sort=False,
        )
    else:
        final = base_summary.copy()

    final["autocorr_final_pass_50tau"] = final.get("autocorr_pass_50tau", False).fillna(False).astype(bool)
    failed_recovery = recovery_requested.difference(set(recovery_summary.get("GaiaDR3", pd.Series(dtype=str))))
    final["recovery_failed"] = final["GaiaDR3"].isin(failed_recovery)
    final = final.sort_values("GaiaDR3").reset_index(drop=True)
    atomic_csv(branch_dir / "final_summary.csv", final)
    atomic_csv(
        branch_dir / "final_posterior_index.csv",
        final[["GaiaDR3", "Name", "result_source", "posterior_file", "posterior_file_count"]],
    )
    stats = {
        "n_final_success": int(len(final)),
        "n_recovery_requested": int(len(recovery_requested)),
        "n_recovery_success": int(len(recovery_summary)),
        "n_recovery_failed": int(len(failed_recovery)),
        "n_final_pass_50tau": int(final["autocorr_final_pass_50tau"].sum()),
        "n_final_fail_50tau": int((~final["autocorr_final_pass_50tau"]).sum()),
    }
    atomic_json(branch_dir / "final_statistics.json", stats)
    return final, stats


def make_source_subset_photometry(output_dir: Path, source_ids) -> Path:
    requested = {normalize_id(value) for value in source_ids}
    full = pd.read_csv(DEFAULT_PHOT_FILE, dtype={"GaiaDR3": str, "source_id": str, "gaia_dr3_source_id": str})
    id_column = next(
        (name for name in ("GaiaDR3", "gaia_dr3_source_id", "source_id") if name in full.columns),
        None,
    )
    if id_column is None:
        raise KeyError("No Gaia DR3 source-ID column found in the photometry input")
    ids = full[id_column].map(normalize_id)
    subset = full.loc[ids.isin(requested)].copy()
    found = set(subset[id_column].map(normalize_id))
    missing = sorted(requested.difference(found))
    if missing or len(found) != len(requested):
        raise ValueError(f"Could not create exact source subset; missing={missing}, found={len(found)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "requested_photometry_subset.csv"
    subset.to_csv(path, index=False)
    (output_dir / "requested_source_ids.txt").write_text(
        "\n".join(sorted(requested)) + "\n", encoding="utf-8"
    )
    return path


def run_one_fit(runner, branch: dict, output_dir: Path, chain: dict, n_workers: int, source_ids=None):
    global ACTIVE_SOURCE_IDS
    ACTIVE_SOURCE_IDS = None if source_ids is None else {normalize_id(value) for value in source_ids}
    phot_file = (
        DEFAULT_PHOT_FILE
        if source_ids is None
        else make_source_subset_photometry(output_dir, ACTIVE_SOURCE_IDS)
    )
    kwargs = common_fit_kwargs(branch, output_dir, chain, n_workers, phot_file)
    selected, summary, _samples, residual_summary = runner(**kwargs)
    if source_ids is None and len(selected) != 2591:
        raise RuntimeError(f"Expected 2591 selected sources, found {len(selected)}")
    if source_ids is not None and len(selected) != len(ACTIVE_SOURCE_IDS):
        raise RuntimeError(f"Expected {len(ACTIVE_SOURCE_IDS)} recovery sources, found {len(selected)}")
    return selected, summary, residual_summary


def run_branch(runner, branch: dict, mode: str, input_manifest: dict, n_workers: int) -> dict:
    prefix = f"{branch['index']:02d}_{branch['name']}"
    branch_dir = SUITE_ROOT / ("smoke" if mode == "smoke" else "production") / prefix
    base_dir = branch_dir / "base"
    chain = SMOKE_CHAIN if mode == "smoke" else BASE_CHAIN
    manifest = branch_manifest(branch, chain, input_manifest, mode)

    completed = branch_dir / "completed.json"
    if completed.exists() and (branch_dir / "final_summary.csv").exists():
        old = json.loads(completed.read_text())
        if old.get("configuration_hash") == manifest["configuration_hash"]:
            return {**old, "skipped": True}

    if base_dir.exists():
        archive_incomplete(base_dir)
    branch_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(branch_dir / "branch_manifest.json", manifest)
    started = now_iso()
    selected, base_summary, _residual = run_one_fit(
        runner,
        branch,
        base_dir,
        chain,
        n_workers,
        source_ids=SMOKE_SOURCE_IDS if mode == "smoke" else None,
    )

    recovery_dir = None
    recovery_ids = []
    if mode == "full" and not base_summary.empty:
        passed = base_summary.get("autocorr_pass_50tau", pd.Series(False, index=base_summary.index)).fillna(False).astype(bool)
        recovery_ids = base_summary.loc[~passed, "GaiaDR3"].map(normalize_id).tolist()
        if recovery_ids:
            recovery_dir = branch_dir / "recovery"
            if recovery_dir.exists():
                archive_incomplete(recovery_dir)
            recovery_dir.mkdir(parents=True, exist_ok=True)
            (recovery_dir / "requested_source_ids.txt").write_text("\n".join(recovery_ids) + "\n", encoding="utf-8")
            run_one_fit(runner, branch, recovery_dir, RECOVERY_CHAIN, n_workers, source_ids=recovery_ids)

    final, stats = make_final_products(branch_dir, base_dir, recovery_dir)
    result = {
        "branch": branch["name"],
        "branch_index": branch["index"],
        "configuration_hash": manifest["configuration_hash"],
        "status": "completed",
        "started_at": started,
        "completed_at": now_iso(),
        "n_selected_base": int(len(selected)),
        "n_success_base": int(len(base_summary)),
        **stats,
        "branch_dir": str(branch_dir),
        "skipped": False,
    }
    atomic_json(completed, result)
    return result


def uncertainty(frame: pd.DataFrame, parameter: str) -> pd.Series:
    return 0.5 * (frame[f"{parameter}_err_minus"].astype(float) + frame[f"{parameter}_err_plus"].astype(float))


def robust_scatter(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def build_impact_audit() -> None:
    production = SUITE_ROOT / "production"
    summaries = {}
    for branch in BRANCHES:
        path = production / f"{branch['index']:02d}_{branch['name']}" / "final_summary.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"GaiaDR3": str})
            frame["GaiaDR3"] = frame["GaiaDR3"].map(normalize_id)
            summaries[branch["name"]] = frame
    if "fiducial_G23_pc1sigma" not in summaries:
        return

    fid = summaries["fiducial_G23_pc1sigma"]
    parameters = [p for p in ("Teff", "logg", "MH", "A550", "R550", "logR_over_D") if f"{p}_p50" in fid]
    rows = []
    common_root = SUITE_ROOT / "impact_audit"
    common_root.mkdir(parents=True, exist_ok=True)
    for name, other in summaries.items():
        if name == "fiducial_G23_pc1sigma":
            continue
        common = fid.merge(other, on="GaiaDR3", suffixes=("_fid", "_alt"))
        atomic_csv(common_root / f"common_fiducial_vs_{name}.csv", common)
        for parameter in parameters:
            delta = common[f"{parameter}_p50_alt"] - common[f"{parameter}_p50_fid"]
            sig_fid = 0.5 * (
                common[f"{parameter}_err_minus_fid"] + common[f"{parameter}_err_plus_fid"]
            )
            sig_alt = 0.5 * (
                common[f"{parameter}_err_minus_alt"] + common[f"{parameter}_err_plus_alt"]
            )
            rows.append(
                {
                    "comparison": name,
                    "parameter": parameter,
                    "n_common": len(common),
                    "median_alt_minus_fid": float(np.nanmedian(delta)),
                    "robust_scatter_alt_minus_fid": robust_scatter(delta),
                    "median_uncertainty_ratio_alt_over_fid": float(np.nanmedian(sig_alt / sig_fid)),
                }
            )
    atomic_csv(common_root / "sensitivity_comparison_statistics.csv", pd.DataFrame(rows))

    rerr = uncertainty(fid, "R550")
    high = fid.loc[rerr < 0.1, ["GaiaDR3", "Name", "A550_p50", "R550_p50"]].copy()
    high["sigma_R550"] = rerr.loc[high.index]
    atomic_csv(common_root / "fiducial_high_precision_R550_sample.csv", high)

    if OLD_FIDUCIAL_SUMMARY.exists():
        old = pd.read_csv(OLD_FIDUCIAL_SUMMARY, dtype={"GaiaDR3": str})
        old["GaiaDR3"] = old["GaiaDR3"].map(normalize_id)
        common_old = old.merge(fid, on="GaiaDR3", suffixes=("_old", "_new"))
        atomic_csv(common_root / "common_old_vs_new_fiducial.csv", common_old)
        old_rows = []
        for parameter in parameters:
            delta = common_old[f"{parameter}_p50_new"] - common_old[f"{parameter}_p50_old"]
            old_rows.append(
                {
                    "parameter": parameter,
                    "n_common": len(common_old),
                    "median_new_minus_old": float(np.nanmedian(delta)),
                    "robust_scatter_new_minus_old": robust_scatter(delta),
                }
            )
        atomic_csv(common_root / "new_vs_old_fiducial_statistics.csv", pd.DataFrame(old_rows))


def write_suite_status(results: list[dict], current=None, error=None) -> None:
    payload = {
        "suite_root": str(SUITE_ROOT),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "updated_at": now_iso(),
        "current_branch": current,
        "error": error,
        "branches": results,
    }
    atomic_json(SUITE_ROOT / "suite_status.json", payload)
    rows = []
    by_name = {row.get("branch"): row for row in results}
    for branch in BRANCHES:
        item = by_name.get(branch["name"], {})
        rows.append(
            {
                "index": branch["index"],
                "branch": branch["name"],
                "status": item.get("status", "running" if current == branch["name"] else "pending"),
                "n_selected_base": item.get("n_selected_base", ""),
                "n_success_base": item.get("n_success_base", ""),
                "n_recovery_requested": item.get("n_recovery_requested", ""),
                "n_final_success": item.get("n_final_success", ""),
                "n_final_pass_50tau": item.get("n_final_pass_50tau", ""),
                "branch_dir": item.get("branch_dir", ""),
            }
        )
    atomic_csv(SUITE_ROOT / "suite_status.csv", pd.DataFrame(rows))


@contextlib.contextmanager
def suite_lock():
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = SUITE_ROOT / "suite.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another paper-fit suite process holds {lock_path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} host={socket.gethostname()} started={now_iso()}\n")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def run_suite(mode: str, n_workers: int, branch_names: set[str] | None = None) -> None:
    configure_process_environment()
    input_manifest = validate_inputs(b["law"] for b in BRANCHES if branch_names is None or b["name"] in branch_names)
    atomic_json(SUITE_ROOT / "suite_input_manifest.json", input_manifest)
    results = []
    write_suite_status(results)
    selected_branches = [
        branch for branch in BRANCHES
        if branch_names is None or branch["name"] in branch_names
    ]
    if not selected_branches:
        raise ValueError(f"No branches matched {sorted(branch_names or [])}")
    for branch in selected_branches:
        write_suite_status(results, current=branch["name"])
        try:
            runner = load_runner()
            result = run_branch(runner, branch, mode, input_manifest, n_workers)
            results.append(result)
            write_suite_status(results)
        except Exception as exc:
            failure = {
                "branch": branch["name"],
                "branch_index": branch["index"],
                "status": "failed",
                "failed_at": now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            results.append(failure)
            write_suite_status(results, error=failure["error"])
            raise
    if mode == "full":
        build_impact_audit()
    write_suite_status(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--branch", action="append", choices=[branch["name"] for branch in BRANCHES])
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    with suite_lock():
        run_suite(args.mode, args.workers, None if not args.branch else set(args.branch))


if __name__ == "__main__":
    main()
