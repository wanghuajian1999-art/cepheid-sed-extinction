#!/usr/bin/env python3
"""Complete one paper-fit branch to 50 autocorrelation times.

The existing production branch is read-only.  Only sources that failed the
branch-specific 50-tau criterion are rerun, using their accepted final
passbands and posterior-centred walker initialization.  Converged products are
written to a separate directory and merged only when every successful source
passes the unchanged threshold.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SUITE_ROOT = BASE_DIR / "outputs" / "paper_fit_suite_PCvariCepheidOnly_include2MASS_H_steps10000"
PHOTOMETRY_FILE = BASE_DIR / "MW_3640_10band_GBPRPVIJHKW1S36_phase_corrected_Bras2025_RPonly.csv"

TAU_THRESHOLD = 50.0
AUDIT_PARAMETERS = ("Teff", "logg", "MH", "A550", "R550", "logR_over_D")
FIT_BANDS = ("BP", "G", "RP", "J", "H", "K", "W1")
MAG_COLUMNS = {band: (f"{band}_mag", f"{band}_magerr") for band in FIT_BANDS}
BRANCH_CONFIGS = {
    "fixed_g23": {
        "directory": "04_G23_fixedLogg2_MH0",
        "expected_total": 2400,
        "expected_targets": 21,
        "sampled_parameters": ("Teff", "A550", "R550", "logR_over_D"),
        "fit_branch": {
            "index": 4,
            "name": "G23_fixedLogg2_MH0",
            "law": "G23",
            "pc_mode": "1sigma",
            "fix": True,
        },
    },
    "wc2019": {
        "directory": "05_WC2019_pc1sigma",
        "expected_total": 2413,
        "expected_targets": 142,
        "sampled_parameters": AUDIT_PARAMETERS,
        "fit_branch": {
            "index": 5,
            "name": "WC2019_pc1sigma",
            "law": "WC2019",
            "pc_mode": "1sigma",
            "fix": False,
        },
    },
    "f19": {
        "directory": "06_F19_pc1sigma",
        "expected_total": 2212,
        "expected_targets": 113,
        "sampled_parameters": AUDIT_PARAMETERS,
        "fit_branch": {
            "index": 6,
            "name": "F19_pc1sigma",
            "law": "F19",
            "pc_mode": "1sigma",
            "fix": False,
        },
    },
}

ACTIVE_BRANCH_KEY = ""
ACTIVE_CONFIG: dict = {}
BRANCH_DIR = Path()
COMPLETION_ROOT = Path()
ORIGINAL_SUMMARY = Path()
EXPECTED_TOTAL = 0
EXPECTED_TARGETS = 0
SAMPLED_PARAMETERS: tuple[str, ...] = ()
FIT_BRANCH: dict = {}

ROUND_CONFIGS = {
    1: {"nwalkers": 32, "nsteps": 50000, "burnin": 5000, "thin": 10, "move": "default"},
    2: {"nwalkers": 64, "nsteps": 100000, "burnin": 10000, "thin": 10, "move": "default"},
    3: {
        "nwalkers": 64,
        "nsteps": 200000,
        "burnin": 20000,
        "thin": 10,
        "move": "de_stretch_mixture",
    },
}
SMOKE_CONFIG = {"nwalkers": 32, "nsteps": 1500, "burnin": 300, "thin": 10, "move": "default"}

_ACTIVE_POSTERIOR_PATHS: dict[str, Path] = {}
_ACTIVE_ROUND = 0
_ACTIVE_MOVE = "default"


def configure_branch(branch_key: str) -> None:
    global ACTIVE_BRANCH_KEY, ACTIVE_CONFIG, BRANCH_DIR, COMPLETION_ROOT
    global ORIGINAL_SUMMARY, EXPECTED_TOTAL, EXPECTED_TARGETS
    global SAMPLED_PARAMETERS, FIT_BRANCH

    if branch_key not in BRANCH_CONFIGS:
        raise KeyError(f"Unknown branch {branch_key!r}; choose from {sorted(BRANCH_CONFIGS)}")
    config = BRANCH_CONFIGS[branch_key]
    ACTIVE_BRANCH_KEY = branch_key
    ACTIVE_CONFIG = config
    BRANCH_DIR = SUITE_ROOT / "production" / config["directory"]
    COMPLETION_ROOT = BRANCH_DIR / "convergence_completion_50tau"
    ORIGINAL_SUMMARY = BRANCH_DIR / "final_summary.csv"
    EXPECTED_TOTAL = int(config["expected_total"])
    EXPECTED_TARGETS = int(config["expected_targets"])
    SAMPLED_PARAMETERS = tuple(config["sampled_parameters"])
    FIT_BRANCH = dict(config["fit_branch"])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_convergence_targets(
    summary: pd.DataFrame,
    expected_total: int | None = None,
    expected_failed: int | None = None,
) -> pd.DataFrame:
    expected_total = EXPECTED_TOTAL if expected_total is None else int(expected_total)
    expected_failed = EXPECTED_TARGETS if expected_failed is None else int(expected_failed)
    required = {"GaiaDR3", "autocorr_final_pass_50tau", "bands_final", "posterior_file"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"Missing final-summary columns: {missing}")
    out = summary.copy()
    out["GaiaDR3"] = out["GaiaDR3"].map(normalize_id)
    if len(out) != expected_total:
        raise ValueError(f"Expected {expected_total} successful sources, found {len(out)}")
    if not out["GaiaDR3"].is_unique:
        raise ValueError("Expected unique GaiaDR3 identifiers in the final summary")
    passed = _bool_series(out["autocorr_final_pass_50tau"])
    targets = out.loc[~passed].copy().sort_values("GaiaDR3").reset_index(drop=True)
    if len(targets) != expected_failed:
        raise ValueError(f"Expected {expected_failed} convergence targets, found {len(targets)}")
    missing_posteriors = [value for value in targets["posterior_file"] if not Path(str(value)).is_file()]
    if missing_posteriors:
        raise FileNotFoundError(f"Missing target posterior files: {missing_posteriors[:5]}")
    return targets


def parse_bands(value) -> tuple[str, ...]:
    bands = tuple(part.strip() for part in str(value).split(",") if part.strip())
    unknown = sorted(set(bands).difference(FIT_BANDS))
    if unknown:
        raise ValueError(f"Unknown final passbands: {unknown}")
    if len(bands) < 6:
        raise ValueError(f"Each convergence target must retain at least six passbands; got {bands}")
    return bands


def build_fixed_band_photometry(
    photometry: pd.DataFrame,
    targets: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    id_column = next(
        (column for column in ("GaiaDR3", "gaia_dr3_source_id", "source_id") if column in photometry.columns),
        None,
    )
    if id_column is None:
        raise KeyError("No Gaia DR3 source-ID column found in the photometry input")
    target_bands = {
        normalize_id(row.GaiaDR3): parse_bands(row.bands_final)
        for row in targets[["GaiaDR3", "bands_final"]].itertuples(index=False)
    }
    ids = photometry[id_column].map(normalize_id)
    fixed = photometry.loc[ids.isin(target_bands)].copy()
    fixed[id_column] = fixed[id_column].map(normalize_id)
    if set(fixed[id_column]) != set(target_bands) or not fixed[id_column].is_unique:
        raise ValueError("Fixed-band photometry does not match the requested unique GaiaDR3 sources")
    # Keep the original measurements so the unchanged pre-fit quality selection
    # sees the same source.  A runtime row-vector patch applies this exact band
    # list only when constructing the likelihood data vector.
    fixed["convergence_bands_final"] = fixed[id_column].map(
        lambda value: ",".join(target_bands[normalize_id(value)])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_path, index=False)
    return fixed


def load_posterior_initial_pool(
    path: Path,
    parameter_names: list[str] | tuple[str, ...],
    min_rows: int,
    log_prob_quantile: float = 0.20,
) -> pd.DataFrame:
    columns = list(parameter_names) + ["log_prob"]
    posterior = pd.read_csv(path, usecols=columns)
    finite = np.isfinite(posterior[columns].to_numpy(dtype=float)).all(axis=1)
    posterior = posterior.loc[finite].copy()
    if len(posterior) < min_rows:
        raise ValueError(f"Posterior {path} has only {len(posterior)} finite rows; need {min_rows}")
    cutoff = posterior["log_prob"].quantile(float(log_prob_quantile))
    pool = posterior.loc[posterior["log_prob"] >= cutoff, list(parameter_names)].drop_duplicates()
    if len(pool) < min_rows:
        pool = posterior.nlargest(min_rows, "log_prob")[list(parameter_names)].drop_duplicates()
    if len(pool) < min_rows:
        raise ValueError(f"Posterior {path} has only {len(pool)} distinct initialization rows")
    return pool.reset_index(drop=True)


def deterministic_seed(source_id: str, round_index: int) -> int:
    digest = hashlib.sha256(f"{normalize_id(source_id)}:round:{round_index}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "little")


def install_convergence_runtime_patch(suite) -> None:
    original_patch = suite.patch_directbase
    if getattr(original_patch, "_convergence_completion_patch", False):
        return

    def patch_directbase_for_completion(directbase) -> None:
        original_patch(directbase)
        if getattr(directbase.initialize_walkers_direct, "_posterior_centered_completion", False):
            return
        original_initializer = directbase.initialize_walkers_direct

        def posterior_centered_initializer(row, bounds, nwalkers, random_seed):
            source_id = normalize_id(
                next(
                    value for value in (
                        row.get("GaiaDR3"), row.get("gaia_dr3_source_id"), row.get("source_id")
                    ) if value is not None and str(value).strip() not in {"", "nan"}
                )
            )
            posterior_path = _ACTIVE_POSTERIOR_PATHS.get(source_id)
            if posterior_path is None:
                return original_initializer(row, bounds, nwalkers, random_seed)
            names = list(directbase.PARAM_NAMES)
            pool = load_posterior_initial_pool(posterior_path, names, int(nwalkers))
            lower = np.asarray(bounds["lower"], dtype=float)
            upper = np.asarray(bounds["upper"], dtype=float)
            values = pool[names].to_numpy(dtype=float)
            valid = np.isfinite(values).all(axis=1) & (values > lower).all(axis=1) & (values < upper).all(axis=1)
            values = values[valid]
            if len(values) < int(nwalkers):
                return original_initializer(row, bounds, nwalkers, deterministic_seed(source_id, _ACTIVE_ROUND))
            rng = np.random.default_rng(deterministic_seed(source_id, _ACTIVE_ROUND))
            chosen = rng.choice(len(values), size=int(nwalkers), replace=False)
            return values[chosen]

        posterior_centered_initializer._posterior_centered_completion = True
        directbase.initialize_walkers_direct = posterior_centered_initializer

        original_run = directbase.run_emcee_direct

        def run_with_optional_moves(*args, **kwargs):
            if _ACTIVE_MOVE != "de_stretch_mixture":
                return original_run(*args, **kwargs)
            import emcee

            original_sampler = emcee.EnsembleSampler

            def sampler_with_moves(*sampler_args, **sampler_kwargs):
                sampler_kwargs.setdefault(
                    "moves",
                    [(emcee.moves.DEMove(), 0.8), (emcee.moves.StretchMove(), 0.2)],
                )
                return original_sampler(*sampler_args, **sampler_kwargs)

            emcee.EnsembleSampler = sampler_with_moves
            try:
                return original_run(*args, **kwargs)
            finally:
                emcee.EnsembleSampler = original_sampler

        run_with_optional_moves._convergence_move_patch = True
        directbase.run_emcee_direct = run_with_optional_moves

    patch_directbase_for_completion._convergence_completion_patch = True
    suite.patch_directbase = patch_directbase_for_completion


def install_fixed_band_vector_patch(runner) -> None:
    """Apply per-source final bands after the unchanged quality selection."""
    dynamic = sys.modules[runner.__module__]
    filtered = getattr(dynamic, "ffloor", dynamic)
    original_patch = filtered._patch_filtered_behavior
    if getattr(original_patch, "_fixed_convergence_bands_patch", False):
        return

    def patch_filtered_behavior_with_fixed_bands(latest_module, directbase, mapbase, *, fix_logg_mh):
        original_patch(latest_module, directbase, mapbase, fix_logg_mh=fix_logg_mh)
        original_vector = mapbase.row_sed_vector

        def fixed_vector(row):
            bands, observations, uncertainties = original_vector(row)
            retained_text = str(row.get("convergence_bands_final", "")).strip()
            if not retained_text or retained_text.lower() == "nan":
                return bands, observations, uncertainties
            retained = set(parse_bands(retained_text))
            keep = np.asarray([band in retained for band in bands], dtype=bool)
            fixed_bands = [band for band, use in zip(bands, keep) if use]
            if tuple(fixed_bands) != tuple(band for band in FIT_BANDS if band in retained):
                raise RuntimeError(
                    f"Could not construct requested fixed band vector for {normalize_id(row.get('GaiaDR3', ''))}: "
                    f"requested={retained_text}, available={bands}"
                )
            return fixed_bands, np.asarray(observations)[keep], np.asarray(uncertainties)[keep]

        mapbase.row_sed_vector = fixed_vector

    patch_filtered_behavior_with_fixed_bands._fixed_convergence_bands_patch = True
    filtered._patch_filtered_behavior = patch_filtered_behavior_with_fixed_bands


def attach_posterior_files(suite, output_dir: Path, selected: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    index = suite.posterior_file_index(output_dir, selected)
    out = summary.copy()
    out["GaiaDR3"] = out["GaiaDR3"].map(normalize_id)
    out = out.merge(index[["GaiaDR3", "posterior_file", "posterior_file_count"]], on="GaiaDR3", how="left")
    if not out["GaiaDR3"].is_unique or not out["posterior_file_count"].eq(1).all():
        raise RuntimeError(f"Posterior indexing failed in {output_dir}")
    return out


def merge_converged_results(original: pd.DataFrame, replacements: pd.DataFrame) -> pd.DataFrame:
    old = original.copy()
    new = replacements.copy()
    old["GaiaDR3"] = old["GaiaDR3"].map(normalize_id)
    new["GaiaDR3"] = new["GaiaDR3"].map(normalize_id)
    if not new["GaiaDR3"].is_unique:
        raise ValueError("Replacement GaiaDR3 identifiers are not unique")
    passed = _bool_series(new["autocorr_pass_50tau"])
    if not passed.all():
        raise RuntimeError(f"Cannot merge: not all pass 50 tau ({int(passed.sum())}/{len(new)})")
    new["autocorr_final_pass_50tau"] = passed
    new["result_source"] = new["convergence_round"].map(lambda value: f"convergence_round_{int(value)}")
    if "max_abs_residual_final" in new:
        new["post_convergence_residual_qc"] = new["max_abs_residual_final"].astype(float).le(3.0)
    replacement_ids = set(new["GaiaDR3"])
    unchanged = old.loc[~old["GaiaDR3"].isin(replacement_ids)].copy()
    unchanged["convergence_round"] = 0
    if "post_convergence_residual_qc" not in unchanged:
        unchanged["post_convergence_residual_qc"] = _bool_series(
            unchanged.get("passed_residual_clipping", pd.Series(True, index=unchanged.index))
        )
    merged = pd.concat([unchanged, new], ignore_index=True, sort=False).sort_values("GaiaDR3").reset_index(drop=True)
    if len(merged) != len(old) or not merged["GaiaDR3"].is_unique:
        raise RuntimeError("Merged converged summary changed the successful-source accounting")
    if not _bool_series(merged["autocorr_final_pass_50tau"]).all():
        raise RuntimeError("Merged converged summary still contains a source below 50 tau")
    return merged


def robust_scatter(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def build_parameter_audit(original: pd.DataFrame, converged: pd.DataFrame, target_ids: set[str]) -> pd.DataFrame:
    old = original.copy()
    new = converged.copy()
    old["GaiaDR3"] = old["GaiaDR3"].map(normalize_id)
    new["GaiaDR3"] = new["GaiaDR3"].map(normalize_id)
    common = old.loc[old["GaiaDR3"].isin(target_ids)].merge(
        new.loc[new["GaiaDR3"].isin(target_ids)], on="GaiaDR3", suffixes=("_original", "_converged")
    )
    rows = []
    for parameter in AUDIT_PARAMETERS:
        required = {f"{parameter}_p50_original", f"{parameter}_p50_converged"}
        if not required.issubset(common.columns):
            continue
        delta = common[f"{parameter}_p50_converged"] - common[f"{parameter}_p50_original"]
        rows.append(
            {
                "parameter": parameter,
                "n": int(delta.notna().sum()),
                "median_converged_minus_original": float(np.nanmedian(delta)),
                "robust_scatter_converged_minus_original": robust_scatter(delta),
                "p16_converged_minus_original": float(np.nanpercentile(delta, 16)),
                "p84_converged_minus_original": float(np.nanpercentile(delta, 84)),
            }
        )
    return pd.DataFrame(rows)


def choose_smoke_targets(targets: pd.DataFrame) -> pd.DataFrame:
    preferred = []
    if "recovery_failed" in targets:
        preferred.extend(targets.loc[_bool_series(targets["recovery_failed"]), "GaiaDR3"].head(2).tolist())
    finite = targets.loc[np.isfinite(targets.get("chain_over_tau_min", np.nan))].sort_values("chain_over_tau_min")
    preferred.extend(finite["GaiaDR3"].head(5).tolist())
    preferred.extend(targets["GaiaDR3"].head(5).tolist())
    ordered = []
    for source_id in preferred:
        source_id = normalize_id(source_id)
        if source_id not in ordered and source_id in set(targets["GaiaDR3"]):
            ordered.append(source_id)
    wanted = min(5, len(targets))
    ordered = ordered[:wanted]
    if len(ordered) != wanted:
        raise RuntimeError(f"Could not construct the planned smoke sample: {ordered}")
    return targets.set_index("GaiaDR3").loc[ordered].reset_index()


def prepare_round_input(
    targets: pd.DataFrame,
    posterior_paths: dict[str, Path],
    round_dir: Path,
) -> Path:
    photometry = pd.read_csv(
        PHOTOMETRY_FILE,
        dtype={"GaiaDR3": str, "gaia_dr3_source_id": str, "source_id": str},
    )
    path = round_dir / "fixed_band_photometry.csv"
    build_fixed_band_photometry(photometry, targets, path)
    atomic_csv(
        round_dir / "initialization_manifest.csv",
        pd.DataFrame(
            {
                "GaiaDR3": targets["GaiaDR3"].map(normalize_id),
                "bands_final": targets["bands_final"],
                "posterior_initialization_file": [str(posterior_paths[normalize_id(value)]) for value in targets["GaiaDR3"]],
                "posterior_initialization_sha256": [sha256(posterior_paths[normalize_id(value)]) for value in targets["GaiaDR3"]],
                "random_seed": [deterministic_seed(value, _ACTIVE_ROUND) for value in targets["GaiaDR3"]],
            }
        ),
    )
    return path


def run_round(suite, runner, targets: pd.DataFrame, posterior_paths: dict[str, Path], round_index: int, workers: int, smoke: bool = False) -> pd.DataFrame:
    global _ACTIVE_POSTERIOR_PATHS, _ACTIVE_ROUND, _ACTIVE_MOVE

    config = SMOKE_CONFIG if smoke else ROUND_CONFIGS[round_index]
    root = COMPLETION_ROOT / ("smoke" if smoke else f"round_{round_index}")
    completed = root / "completed.json"
    summary_path = root / "round_summary.csv"
    if completed.exists() and summary_path.exists():
        return pd.read_csv(summary_path, dtype={"GaiaDR3": str})
    if root.exists() and any(root.iterdir()):
        diagnostic = COMPLETION_ROOT / "diagnostic_incomplete"
        diagnostic.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(diagnostic / f"{root.name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"))
    root.mkdir(parents=True, exist_ok=True)

    _ACTIVE_ROUND = 0 if smoke else round_index
    _ACTIVE_MOVE = config["move"]
    _ACTIVE_POSTERIOR_PATHS = {normalize_id(key): Path(value) for key, value in posterior_paths.items()}
    fixed_photometry = prepare_round_input(targets, _ACTIVE_POSTERIOR_PATHS, root)
    source_ids = targets["GaiaDR3"].map(normalize_id).tolist()
    suite.ACTIVE_SOURCE_IDS = set(source_ids)
    chain = {key: config[key] for key in ("nwalkers", "nsteps", "burnin", "thin")}
    kwargs = suite.common_fit_kwargs(FIT_BRANCH, root, chain, workers, fixed_photometry)
    kwargs["iterative_residual_clipping"] = False
    kwargs["random_seed"] = 42 + _ACTIVE_ROUND * 1000
    selected, summary, _samples, _residual_summary = runner(**kwargs)
    if len(selected) != len(source_ids) or len(summary) != len(source_ids):
        raise RuntimeError(
            f"Round {_ACTIVE_ROUND} expected {len(source_ids)} selected/successful sources; "
            f"found selected={len(selected)}, summary={len(summary)}"
        )
    summary = attach_posterior_files(suite, root, selected, summary)
    summary["convergence_round"] = _ACTIVE_ROUND
    summary["result_source"] = "convergence_smoke" if smoke else f"convergence_round_{round_index}"
    summary["post_convergence_residual_qc"] = summary["max_abs_residual_final"].astype(float).le(3.0)
    expected_bands = targets.set_index("GaiaDR3")["bands_final"].to_dict()
    summary["expected_bands_final"] = summary["GaiaDR3"].map(expected_bands)
    if not summary["bands_final"].eq(summary["expected_bands_final"]).all():
        raise RuntimeError(f"Round {_ACTIVE_ROUND} changed at least one fixed final passband set")
    atomic_csv(summary_path, summary)
    atomic_csv(root / "passed_summary.csv", summary.loc[_bool_series(summary["autocorr_pass_50tau"])])
    atomic_csv(root / "failed_summary.csv", summary.loc[~_bool_series(summary["autocorr_pass_50tau"])])
    atomic_json(
        completed,
        {
            "completed_at": now_iso(),
            "round": _ACTIVE_ROUND,
            "smoke": smoke,
            "configuration": config,
            "n_requested": len(source_ids),
            "n_pass_50tau": int(_bool_series(summary["autocorr_pass_50tau"]).sum()),
            "n_residual_qc": int(summary["post_convergence_residual_qc"].sum()),
        },
    )
    return summary


def run_completion(workers: int, mode: str) -> None:
    if not ORIGINAL_SUMMARY.is_file() or not PHOTOMETRY_FILE.is_file():
        raise FileNotFoundError("The completed branch summary or source photometry is missing")
    original = pd.read_csv(ORIGINAL_SUMMARY, dtype={"GaiaDR3": str})
    targets = select_convergence_targets(original)
    if mode == "smoke":
        targets = choose_smoke_targets(targets)

    sys.path.insert(0, str(BASE_DIR))
    import run_paper_fit_suite as suite

    suite.configure_process_environment()
    input_manifest = suite.validate_inputs([FIT_BRANCH["law"]])
    install_convergence_runtime_patch(suite)
    runner = suite.load_runner()
    install_fixed_band_vector_patch(runner)
    COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        COMPLETION_ROOT / "input_manifest.json",
        {
            "created_at": now_iso(),
            "mode": mode,
            "original_summary": str(ORIGINAL_SUMMARY),
            "original_summary_sha256": sha256(ORIGINAL_SUMMARY),
            "photometry_file": str(PHOTOMETRY_FILE),
            "photometry_sha256": sha256(PHOTOMETRY_FILE),
            "fit_suite_inputs": input_manifest,
            "branch_key": ACTIVE_BRANCH_KEY,
            "branch_config": ACTIVE_CONFIG,
            "target_count": len(targets),
            "tau_threshold": TAU_THRESHOLD,
            "round_configs": ROUND_CONFIGS,
        },
    )

    posterior_paths = {
        normalize_id(row.GaiaDR3): Path(str(row.posterior_file))
        for row in targets[["GaiaDR3", "posterior_file"]].itertuples(index=False)
    }
    if mode == "smoke":
        smoke_summary = run_round(suite, runner, targets, posterior_paths, 0, workers, smoke=True)
        atomic_json(
            COMPLETION_ROOT / "smoke_status.json",
            {
                "completed_at": now_iso(),
                "n_requested": len(targets),
                "n_success": len(smoke_summary),
                "fixed_bands_preserved": bool(smoke_summary["bands_final"].eq(smoke_summary["expected_bands_final"]).all()),
                "posterior_files_unique": bool(smoke_summary["posterior_file_count"].eq(1).all()),
            },
        )
        return

    remaining = targets.copy()
    accepted = []
    for round_index in sorted(ROUND_CONFIGS):
        atomic_json(
            COMPLETION_ROOT / "status.json",
            {"status": "running", "updated_at": now_iso(), "round": round_index, "remaining": len(remaining)},
        )
        round_summary = run_round(suite, runner, remaining, posterior_paths, round_index, workers)
        passed = _bool_series(round_summary["autocorr_pass_50tau"])
        accepted.append(round_summary.loc[passed].copy())
        failed = round_summary.loc[~passed].copy()
        if failed.empty:
            remaining = failed
            break
        posterior_paths = {
            normalize_id(row.GaiaDR3): Path(str(row.posterior_file))
            for row in failed[["GaiaDR3", "posterior_file"]].itertuples(index=False)
        }
        remaining = targets.set_index("GaiaDR3").loc[failed["GaiaDR3"]].reset_index()

    replacements = pd.concat(accepted, ignore_index=True, sort=False) if accepted else pd.DataFrame()
    if not remaining.empty:
        atomic_csv(COMPLETION_ROOT / "manual_diagnostic_targets.csv", remaining)
        atomic_json(
            COMPLETION_ROOT / "status.json",
            {
                "status": "manual_diagnosis_required",
                "updated_at": now_iso(),
                "remaining": len(remaining),
                "n_passed": len(replacements),
            },
        )
        raise RuntimeError(f"{len(remaining)} sources remain below 50 tau after round 3")

    merged = merge_converged_results(original, replacements)
    if len(merged) != EXPECTED_TOTAL or not _bool_series(merged["autocorr_final_pass_50tau"]).all():
        raise RuntimeError(
            f"Final converged summary failed the {EXPECTED_TOTAL}-source acceptance check"
        )
    for parameter in SAMPLED_PARAMETERS:
        column = f"chain_over_tau_{parameter}"
        if column not in merged:
            raise KeyError(f"Missing convergence column {column}")
        values = pd.to_numeric(merged[column], errors="coerce")
        if not np.isfinite(values).all() or not values.ge(TAU_THRESHOLD).all():
            raise RuntimeError(f"Final converged summary failed {column} >= {TAU_THRESHOLD:g}")
    atomic_csv(COMPLETION_ROOT / "final_summary_converged.csv", merged)
    atomic_csv(
        COMPLETION_ROOT / "final_posterior_index_converged.csv",
        merged[["GaiaDR3", "Name", "result_source", "posterior_file", "posterior_file_count"]],
    )
    target_ids = set(targets["GaiaDR3"])
    audit = build_parameter_audit(original, merged, target_ids)
    atomic_csv(COMPLETION_ROOT / "original_vs_converged_parameter_audit.csv", audit)
    special = merged.loc[merged["GaiaDR3"].isin(target_ids) & ~_bool_series(merged["post_convergence_residual_qc"])].copy()
    atomic_csv(COMPLETION_ROOT / "post_convergence_residual_qc_failures.csv", special)
    atomic_json(
        COMPLETION_ROOT / "convergence_completion_statistics.json",
        {
            "completed_at": now_iso(),
            "n_final": len(merged),
            "n_final_pass_50tau": int(_bool_series(merged["autocorr_final_pass_50tau"]).sum()),
            "n_rerun": len(replacements),
            "n_post_convergence_residual_qc_fail": len(special),
            "round_counts": replacements["convergence_round"].value_counts().sort_index().to_dict(),
        },
    )
    atomic_json(
        COMPLETION_ROOT / "status.json",
        {"status": "completed", "updated_at": now_iso(), "n_final": len(merged), "n_pass_50tau": len(merged)},
    )


@contextlib.contextmanager
def completion_lock():
    COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
    path = COMPLETION_ROOT / "completion.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another convergence-completion process holds {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={now_iso()}\n")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=tuple(BRANCH_CONFIGS), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    configure_branch(args.branch)
    try:
        with completion_lock():
            run_completion(args.workers, args.mode)
    except Exception as exc:
        COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(
            COMPLETION_ROOT / "last_failure.json",
            {"failed_at": now_iso(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
