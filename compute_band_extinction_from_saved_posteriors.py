#!/usr/bin/env python3
"""Post-process saved native A550/R550 posterior tables."""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bosz_a550_r550_preconvolved import (
    DEFAULT_JOHNSON_BV_CACHE_FILES,
    PreconvolvedBOSZModel,
    build_johnson_bv_companion_cache,
)


BANDS = ("BP", "G", "RP", "J", "H", "K", "W1")
JOHNSON_BV_BANDS = ("B", "V")
REQUIRED = ("Teff", "logg", "MH", "A550", "R550")
_WORKER_MODEL = None
_WORKER_BV_MODEL = None


def _terminate_executor(executor, futures):
    """Cancel futures and terminate all still-running process workers."""
    for future in futures or ():
        future.cancel()
    processes = list(getattr(executor, "_processes", {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
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


def _close_worker_model():
    global _WORKER_MODEL, _WORKER_BV_MODEL
    if _WORKER_MODEL is not None:
        try:
            _WORKER_MODEL.close()
        finally:
            _WORKER_MODEL = None
    if _WORKER_BV_MODEL is not None:
        try:
            _WORKER_BV_MODEL.close()
        finally:
            _WORKER_BV_MODEL = None


def _init_postprocess_worker(cache_file, bv_cache_file=None):
    global _WORKER_MODEL, _WORKER_BV_MODEL
    _WORKER_MODEL = PreconvolvedBOSZModel(cache_file)
    _WORKER_BV_MODEL = (
        PreconvolvedBOSZModel(bv_cache_file) if bv_cache_file else None
    )
    atexit.register(_close_worker_model)


def _atomic_csv(frame: pd.DataFrame, path: Path, compression=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        frame.to_csv(tmp, index=False, compression=compression)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _select_draws(samples: pd.DataFrame, draws, seed: int):
    if draws == "all" or draws is None:
        return samples
    n = int(draws)
    if n <= 0:
        raise ValueError("band_extinction_draws must be positive or 'all'")
    if n >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(samples), size=n, replace=False))
    return samples.iloc[indices].reset_index(drop=True)


def _sample_tag(path: Path):
    match = re.match(r"(EMCEE\d+)", path.name)
    return match.group(1) if match else path.stem.split("_")[0]


def _postprocess_one(task):
    path, tag, name, gaia, draws, seed, batch_size, save_draws, output_dir, include_bv = task
    samples = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in samples.columns]
    if missing:
        raise ValueError(f"{path.name} missing posterior columns: {missing}")
    selected = _select_draws(samples, draws, seed)
    extinction = _WORKER_MODEL.band_extinction_batch(selected, bands=BANDS, chunk_size=batch_size)
    p16, p50, p84 = np.nanpercentile(extinction, [16, 50, 84], axis=0)
    row = {"selection_tag": tag, "Name": name, "GaiaDR3": gaia, "n_draws": len(selected)}
    long_rows = []
    for i, band in enumerate(BANDS):
        row[f"A_band_{band}_p16"] = float(p16[i])
        row[f"A_band_{band}_p50"] = float(p50[i])
        row[f"A_band_{band}_p84"] = float(p84[i])
        row[f"A_band_{band}_err_minus"] = float(p50[i] - p16[i])
        row[f"A_band_{band}_err_plus"] = float(p84[i] - p50[i])
        long_rows.append({
            "selection_tag": tag,
            "Name": name,
            "GaiaDR3": gaia,
            "band": band,
            "n_draws": len(selected),
            "A_band_p16": float(p16[i]),
            "A_band_p50": float(p50[i]),
            "A_band_p84": float(p84[i]),
            "A_band_err_minus": float(p50[i] - p16[i]),
            "A_band_err_plus": float(p84[i] - p50[i]),
        })
    if include_bv:
        if _WORKER_BV_MODEL is None:
            raise RuntimeError("Johnson B/V companion model is not initialized")
        bv_extinction = _WORKER_BV_MODEL.band_extinction_batch(
            selected, bands=JOHNSON_BV_BANDS, chunk_size=batch_size
        )
        bv_p16, bv_p50, bv_p84 = np.nanpercentile(
            bv_extinction, [16, 50, 84], axis=0
        )
        for i, band in enumerate(JOHNSON_BV_BANDS):
            row[f"A_band_{band}_p16"] = float(bv_p16[i])
            row[f"A_band_{band}_p50"] = float(bv_p50[i])
            row[f"A_band_{band}_p84"] = float(bv_p84[i])
            row[f"A_band_{band}_err_minus"] = float(bv_p50[i] - bv_p16[i])
            row[f"A_band_{band}_err_plus"] = float(bv_p84[i] - bv_p50[i])
            long_rows.append({
                "selection_tag": tag,
                "Name": name,
                "GaiaDR3": gaia,
                "band": band,
                "n_draws": len(selected),
                "A_band_p16": float(bv_p16[i]),
                "A_band_p50": float(bv_p50[i]),
                "A_band_p84": float(bv_p84[i]),
                "A_band_err_minus": float(bv_p50[i] - bv_p16[i]),
                "A_band_err_plus": float(bv_p84[i] - bv_p50[i]),
            })
        e440_550 = selected["A550"].to_numpy(float) / selected["R550"].to_numpy(float)
        e_bv = bv_extinction[:, 0] - bv_extinction[:, 1]
        for key, values in (("E_440_550", e440_550), ("E_BV", e_bv)):
            q16, q50, q84 = np.nanpercentile(values, [16, 50, 84])
            row[f"{key}_p16"] = float(q16)
            row[f"{key}_p50"] = float(q50)
            row[f"{key}_p84"] = float(q84)
            row[f"{key}_err_minus"] = float(q50 - q16)
            row[f"{key}_err_plus"] = float(q84 - q50)
        if save_draws:
            output = selected.copy()
            for i, band in enumerate(BANDS):
                output[f"A_band_{band}"] = extinction[:, i]
            for i, band in enumerate(JOHNSON_BV_BANDS):
                output[f"A_band_{band}"] = bv_extinction[:, i]
            output["E_440_550"] = e440_550
            output["E_BV"] = e_bv
            target = Path(output_dir) / f"{tag}_band_extinction_samples.csv.gz"
            _atomic_csv(output, target, compression="gzip")
    if save_draws:
        if not include_bv:
            output = selected.copy()
            for i, band in enumerate(BANDS):
                output[f"A_band_{band}"] = extinction[:, i]
            output["E_440_550"] = selected["A550"].to_numpy(float) / selected["R550"].to_numpy(float)
            target = Path(output_dir) / f"{tag}_band_extinction_samples.csv.gz"
            _atomic_csv(output, target, compression="gzip")
    return row, long_rows


def compute_extinction_products_from_saved_posteriors(
    fit_dir,
    band_extinction_draws="all",
    batch_size=512,
    n_workers=4,
    random_seed=42,
    include_johnson_bv=True,
    save_extinction_draws=False,
    overwrite=False,
):
    """Compute seven-band extinction and optional B/V color excesses.

    The MCMC posterior is read from ``fit_dir``.  The seven fitted bands use
    the native A550/R550 cache; Johnson B/V use a small companion cache that
    has the same BOSZ grid and extinction-law normalization.
    """
    fit_dir = Path(fit_dir).expanduser().resolve()
    with open(fit_dir / "runinfo.json", "r", encoding="utf-8") as handle:
        runinfo = json.load(handle)
    if str(runinfo.get("extinction_parameterization", "")) != "A550_R550":
        raise ValueError("fit_dir is not an A550/R550 fit")
    cache_file = Path(runinfo["extinction_cache_file"])
    if not cache_file.exists():
        raise FileNotFoundError(cache_file)
    sample_files = sorted(fit_dir.glob("EMCEE*_samples.csv.gz"))
    if not sample_files:
        raise FileNotFoundError(f"No EMCEE*_samples.csv.gz files under {fit_dir}")
    if band_extinction_draws == "all":
        draw_label = "all"
    else:
        draw_label = str(int(band_extinction_draws))
    output_dir = fit_dir / f"extinction_postprocess_{draw_label}"
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")
    work_dir = fit_dir / f".inprogress_band_extinction_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    summary_path = fit_dir / "summary.csv"
    if not summary_path.exists():
        candidates = sorted(fit_dir.glob("MW_SED*_summary.csv"))
        summary_path = candidates[0] if candidates else summary_path
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    metadata = summary.set_index("selection_tag", drop=False) if "selection_tag" in summary else pd.DataFrame()
    tasks = []
    bv_cache_file = None
    if include_johnson_bv:
        law = str(runinfo.get("extinction_law_used", runinfo.get("extinction_law_requested", "G23")))
        bv_cache_file = DEFAULT_JOHNSON_BV_CACHE_FILES.get(law)
        if bv_cache_file is None:
            raise ValueError(f"No Johnson B/V companion cache mapping for extinction law {law!r}")
        bv_cache_file = Path(bv_cache_file)
        if not bv_cache_file.exists():
            build_johnson_bv_companion_cache(law=law, cache_file=bv_cache_file, overwrite=False)
    for path in sample_files:
        tag = _sample_tag(path)
        info = metadata.loc[tag] if len(metadata) and tag in metadata.index else {}
        name = info.get("Name", "") if hasattr(info, "get") else ""
        gaia = info.get("GaiaDR3", "") if hasattr(info, "get") else ""
        tasks.append((str(path), tag, name, gaia, band_extinction_draws, random_seed, int(batch_size), bool(save_extinction_draws), str(work_dir), bool(include_johnson_bv)))

    wide_rows, long_rows, failures = [], [], []
    model = None
    try:
        if int(n_workers) > 1:
            executor = ProcessPoolExecutor(max_workers=int(n_workers), initializer=_init_postprocess_worker, initargs=(str(cache_file), str(bv_cache_file)))
            futures = []
            try:
                futures = [executor.submit(_postprocess_one, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Band extinction postprocess"):
                    try:
                        row, rows = future.result()
                        wide_rows.append(row)
                        long_rows.extend(rows)
                    except Exception as exc:
                        failures.append({"selection_tag": "unknown", "fail_reason": str(exc)})
            except BaseException:
                _terminate_executor(executor, futures)
                raise
            else:
                executor.shutdown(wait=True)
        else:
            model = PreconvolvedBOSZModel(cache_file)
            global _WORKER_BV_MODEL
            _WORKER_BV_MODEL = PreconvolvedBOSZModel(bv_cache_file) if bv_cache_file else None
            global _WORKER_MODEL
            _WORKER_MODEL = model
            for task in tqdm(tasks, desc="Band extinction postprocess"):
                try:
                    row, rows = _postprocess_one(task)
                    wide_rows.append(row)
                    long_rows.extend(rows)
                except Exception as exc:
                    failures.append({"selection_tag": task[1], "fail_reason": str(exc)})
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        if model is not None:
            model.close()
            _WORKER_MODEL = None
        if _WORKER_BV_MODEL is not None:
            _WORKER_BV_MODEL.close()
            _WORKER_BV_MODEL = None

    try:
        wide = pd.DataFrame(wide_rows).sort_values("selection_tag") if wide_rows else pd.DataFrame()
        long = pd.DataFrame(long_rows).sort_values(["selection_tag", "band"]) if long_rows else pd.DataFrame()
        _atomic_csv(wide, work_dir / "MW_SED_extinction_posterior_summary_wide.csv")
        _atomic_csv(long, work_dir / "MW_SED_extinction_posterior_summary_long.csv")
        _atomic_csv(pd.DataFrame(failures), work_dir / "extinction_postprocess_failures.csv")
        output_runinfo = {
            "fit_dir": str(fit_dir),
            "cache_file": str(cache_file),
            "johnson_bv_cache_file": str(bv_cache_file) if bv_cache_file else None,
            "extinction_law": runinfo.get("extinction_law_used", runinfo.get("extinction_law_requested")),
            "include_johnson_bv": bool(include_johnson_bv),
            "E_440_550_definition": "A550/R550 = A(440 nm)-A(550 nm)",
            "E_BV_definition": "synthetic energy-counting GROUND_JOHNSON_B minus V",
            "n_sample_files": len(sample_files),
            "n_success": len(wide),
            "n_failed": len(failures),
            "band_extinction_draws": band_extinction_draws,
            "batch_size": int(batch_size),
            "n_workers": int(n_workers),
            "random_seed": int(random_seed),
            "save_extinction_draws": bool(save_extinction_draws),
        }
        with open(work_dir / "extinction_postprocess_runinfo.json", "w", encoding="utf-8") as handle:
            json.dump(output_runinfo, handle, ensure_ascii=False, indent=2)
        if output_dir.exists() and overwrite:
            shutil.rmtree(output_dir)
        os.replace(work_dir, output_dir)
        return wide, long, pd.DataFrame(failures)
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def compute_band_extinction_from_saved_posteriors(
    fit_dir,
    band_extinction_draws="all",
    batch_size=512,
    n_workers=4,
    random_seed=42,
    save_band_extinction_draws=False,
    overwrite=False,
):
    """Backward-compatible wrapper including Johnson B/V products."""
    return compute_extinction_products_from_saved_posteriors(
        fit_dir=fit_dir,
        band_extinction_draws=band_extinction_draws,
        batch_size=batch_size,
        n_workers=n_workers,
        random_seed=random_seed,
        include_johnson_bv=True,
        save_extinction_draws=save_band_extinction_draws,
        overwrite=overwrite,
    )


print("Loaded independent A550/R550 posterior band-extinction postprocessor.")
