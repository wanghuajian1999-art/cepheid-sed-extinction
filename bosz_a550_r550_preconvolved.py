#!/usr/bin/env python3
"""Native monochromatic A550/R550 BOSZ pre-convolution and interpolation.

This module builds caches whose extinction coordinates are exactly

    A550 = A(550 nm)
    R550 = A550 / (A(440 nm) - A550)

The native shape parameter of an extinction-law implementation is used only
while constructing the cache.  Fitting code interpolates solely in A550 and
R550 and therefore remains agnostic to the law's original parameterization.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import astropy
import astropy.units as u
import h5py
import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay
from tqdm import tqdm

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


_COMPANION_BUILD_CONTEXT = None


def _init_companion_build_worker(wave_spec, filters, curves, a550_grid, bands):
    global _COMPANION_BUILD_CONTEXT
    _COMPANION_BUILD_CONTEXT = (wave_spec, filters, curves, a550_grid, bands)


def _companion_integrals_task(task):
    index, spec_path = task
    wave_spec, filters, curves, a550_grid, bands = _COMPANION_BUILD_CONTEXT
    d, n = _integrals_one_spectrum(
        spec_path, wave_spec, filters, curves, a550_grid, bands=bands
    )
    return index, d, n


SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_BASE_CANDIDATES = [
    SCRIPT_DIR,
    Path(__file__).resolve().parent,


]

ALL_BANDS = ["BP", "G", "RP", "J", "H", "K", "W1"]
JOHNSON_BV_BANDS = ["B", "V"]
FILTER_FILENAMES = {
    "BP": ("GAIA_GAIA3.Gbp.dat", True),
    "G": ("GAIA_GAIA3.G.dat", True),
    "RP": ("GAIA_GAIA3.Grp.dat", True),
    "J": ("2MASS_2MASS.J.dat", True),
    "H": ("2MASS_2MASS.H.dat", True),
    "K": ("2MASS_2MASS.Ks.dat", True),
    "W1": ("WISE_WISE.W1.dat", False),
}
JOHNSON_BV_FILTER_FILENAMES = {
    "B": ("B_pyphot_GROUND_JOHNSON_B.dat", False),
    "V": ("V_pyphot_GROUND_JOHNSON_V.dat", False),
}

LAW_NATIVE_RANGES = {
    "G23": (2.3, 5.6),
    "WC2019": (2.302, 5.582),
    "F19": (2.0, 6.0),
    "F99": (2.0, 6.0),
}

DEFAULT_CACHE_FILES = {
    law: SCRIPT_DIR / f"bosz_r500_{law}_A550_R550_preconvolved_ND.h5"
    for law in LAW_NATIVE_RANGES
}
DEFAULT_JOHNSON_BV_CACHE_FILES = {
    law: SCRIPT_DIR / f"bosz_r500_{law}_A550_R550_JOHNSON_BV_companion_ND.h5"
    for law in LAW_NATIVE_RANGES
}


def _asset_base(explicit=None) -> Path:
    candidates = ([Path(explicit)] if explicit is not None else []) + ASSET_BASE_CANDIDATES
    for candidate in candidates:
        if (
            (candidate / "filters/GAIA_GAIA3.Gbp.dat").exists()
            and (candidate / "filters/WISE_WISE.W1.dat").exists()
            and (candidate / "r500/m+0.00/bosz2024_wave_r500.txt").exists()
        ):
            return candidate
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find the filters and BOSZ R=500 library. Checked: {checked}")


def _law_name(value) -> str:
    key = str(value).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "G23": "G23", "GORDON23": "G23", "GORDON2023": "G23",
        "WC19": "WC2019", "WC2019": "WC2019", "WANGCHEN2019": "WC2019",
        "F19": "F19", "FITZPATRICK19": "F19", "FITZPATRICK2019": "F19",
        "F99": "F99", "FITZPATRICK99": "F99", "FITZPATRICK1999": "F99",
    }
    if key not in aliases:
        raise ValueError(f"Unknown extinction law {value!r}; choose from {sorted(LAW_NATIVE_RANGES)}")
    return aliases[key]


def _law_factory(law: str, native_shape: float):
    law = _law_name(law)
    if law == "WC2019":
        from jupyter_mw_sed_pcprior_WC2019_method2_distanceprior_emcee_direct import WC2019

        return WC2019(Rv=float(native_shape))
    if law == "F19":
        from jupyter_mw_sed_pcprior_F19_method2_distanceprior_emcee_direct import F19

        return F19(Rv=float(native_shape))
    if law == "F99":
        from jupyter_mw_sed_pcprior_F99_method2_distanceprior_emcee_direct import F99

        return F99(Rv=float(native_shape))
    from dust_extinction.parameter_averages import G23

    return G23(Rv=float(native_shape))


def _evaluate_law(law: str, native_shape: float, wavelength_micron) -> np.ndarray:
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    model = _law_factory(law, native_shape)
    return _evaluate_model(law, model, wavelength_micron)


def _evaluate_model(law: str, model, wavelength_micron) -> np.ndarray:
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    if _law_name(law) in {"WC2019", "F19", "F99"}:
        value = model((1.0 / wavelength_micron) / u.micron)
    else:
        value = model(wavelength_micron * u.micron)
    return np.asarray(value, dtype=np.float64)


def native_shape_to_r550(law: str, native_shape) -> np.ndarray:
    native_shape = np.atleast_1d(np.asarray(native_shape, dtype=float))
    out = np.empty(native_shape.size, dtype=float)
    for i, shape in enumerate(native_shape):
        k440, k550 = _evaluate_law(law, float(shape), [0.440, 0.550])
        denominator = k440 - k550
        if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(k550) or k550 <= 0:
            raise ValueError(
                f"Invalid monochromatic normalization for {law}, native shape={shape}: "
                f"k440={k440}, k550={k550}"
            )
        out[i] = k550 / denominator
    return out


def build_r550_mapping(law: str, dense_size: int = 20001, r550_step: float = 0.02):
    law = _law_name(law)
    native_min, native_max = LAW_NATIVE_RANGES[law]
    native_dense = np.linspace(native_min, native_max, int(dense_size), dtype=float)
    r550_dense = native_shape_to_r550(law, native_dense)
    delta = np.diff(r550_dense)
    if not np.all(np.isfinite(r550_dense)) or not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(f"{law}: native shape to R550 mapping is not strictly monotonic")
    if np.all(delta < 0):
        r550_dense = r550_dense[::-1]
        native_dense = native_dense[::-1]

    rmin = float(r550_dense[0])
    rmax = float(r550_dense[-1])
    interior_start = np.ceil(rmin / r550_step) * r550_step
    interior_end = np.floor(rmax / r550_step) * r550_step
    interior = np.arange(interior_start, interior_end + 0.5 * r550_step, r550_step, dtype=float)
    r550_grid = np.unique(np.r_[rmin, interior[(interior > rmin) & (interior < rmax)], rmax])
    inverse = PchipInterpolator(r550_dense, native_dense, extrapolate=False)
    native_grid = np.asarray(inverse(r550_grid), dtype=float)
    native_grid = np.clip(native_grid, native_min, native_max)
    native_grid[0] = native_dense[0]
    native_grid[-1] = native_dense[-1]

    r550_check = native_shape_to_r550(law, native_grid)
    max_error = float(np.max(np.abs(r550_check - r550_grid)))
    if max_error > 2.0e-7:
        raise ValueError(f"{law}: R550 inversion error {max_error:.3g} exceeds tolerance")
    return {
        "law": law,
        "native_dense": native_dense,
        "r550_dense": r550_dense,
        "r550_grid": r550_grid,
        "native_shape_for_r550": native_grid,
        "max_inverse_error": max_error,
    }


def _load_filters(asset_base: Path, bands=None):
    bands = list(FILTER_FILENAMES if bands is None else bands)
    filters = {}
    for band in bands:
        filename, photon_based = FILTER_FILENAMES[band]
        values = np.loadtxt(asset_base / "filters" / filename)
        filters[band] = (
            np.asarray(values[:, 0], dtype=np.float64),
            np.asarray(values[:, 1], dtype=np.float64),
            bool(photon_based),
        )
    return filters


def _load_johnson_bv_filters(asset_base: Path):
    filters = {}
    for band, (filename, photon_based) in JOHNSON_BV_FILTER_FILENAMES.items():
        values = np.loadtxt(asset_base / "filters" / filename)
        wave = np.asarray(values[:, 0], dtype=np.float64)
        response = np.clip(np.asarray(values[:, 1], dtype=np.float64), 0.0, None)
        # The pyphot Johnson-B file contains a disconnected, sub-percent
        # positive tail near 1 micron caused by tabulated throughput noise.
        # Retain the physical main passband and remove that numerical red leak.
        if band == "B" and np.nanmax(response) > 0:
            response[response < 0.005 * np.nanmax(response)] = 0.0
        positive = response > 0
        if np.any(positive):
            lo = max(0, int(np.flatnonzero(positive)[0]) - 1)
            hi = min(len(wave) - 1, int(np.flatnonzero(positive)[-1]) + 1)
            wave = wave[lo:hi + 1]
            response = response[lo:hi + 1]
        # The exported V curve is very finely sampled.  A uniform 1200-point
        # representation preserves its smooth response while avoiding a
        # 10,000-point x 1,000-mag-grid inner integration for every spectrum.
        if len(wave) > 1200:
            compressed_wave = np.linspace(wave[0], wave[-1], 1200)
            response = np.interp(compressed_wave, wave, response)
            wave = compressed_wave
        filters[band] = (
            wave,
            response,
            bool(photon_based),
        )
    return filters


def _scan_spectra(asset_base: Path):
    from bosz_preconvolved_interpolated_fit import scan_bosz_spectra

    return scan_bosz_spectra(spec_base_dir=asset_base / "r500")


def _normalized_curves_on_filters(law, native_grid, filters, bands=None):
    bands = list(filters if bands is None else bands)
    curves = []
    for native_shape in tqdm(native_grid, desc=f"{law} R550 curves"):
        model = _law_factory(law, native_shape)
        k550 = float(_evaluate_model(law, model, [0.550])[0])
        per_band = {}
        for band in bands:
            wave_angstrom = filters[band][0]
            curve = _evaluate_model(law, model, wave_angstrom * 1.0e-4) / k550
            if np.any(~np.isfinite(curve)) or np.any(curve < 0):
                raise ValueError(f"{law}: invalid A(lambda)/A550 curve for R550 shape={native_shape}, band={band}")
            per_band[band] = np.asarray(curve, dtype=np.float32)
        curves.append(per_band)
    return curves


def _integrals_one_spectrum(spec_path, wave_spec, filters, curves, a550_grid, bands=None):
    bands = list(filters if bands is None else bands)
    raw = np.loadtxt(spec_path)
    flux_spec = np.asarray(raw if raw.ndim == 1 else raw[:, 0], dtype=np.float64)
    d = np.full(len(bands), np.nan, dtype=np.float32)
    n = np.full((len(curves), len(a550_grid), len(bands)), np.nan, dtype=np.float32)
    kernels = {}
    for ib, band in enumerate(bands):
        wave_band, response, photon_based = filters[band]
        flux_band = np.interp(wave_band, wave_spec, flux_spec, left=0.0, right=0.0)
        kernel = flux_band * response * (wave_band if photon_based else 1.0)
        denominator = np.trapz(kernel, wave_band)
        if np.isfinite(denominator) and denominator > 0:
            d[ib] = denominator
            kernels[band] = (wave_band, kernel)

    for ir, curve_set in enumerate(curves):
        for ib, band in enumerate(bands):
            if band not in kernels:
                continue
            wave_band, kernel = kernels[band]
            curve = np.asarray(curve_set[band], dtype=np.float64)
            attenuation = 10.0 ** (-0.4 * a550_grid[:, None] * curve[None, :])
            numerator = np.trapz(attenuation * kernel[None, :], wave_band, axis=1)
            n[ir, :, ib] = numerator.astype(np.float32)
    return d, n


def _versions():
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
    }
    try:
        versions["dust_extinction"] = importlib.metadata.version("dust_extinction")
    except Exception:
        versions["dust_extinction"] = "unavailable"
    return versions


def validate_native_cache(cache_file, direct_checks=0, asset_base=None, tolerance_mag=1.0e-4):
    cache_file = Path(cache_file)
    required = {"params", "D", "N", "bands", "a550_grid", "r550_grid", "native_shape_for_r550"}
    with h5py.File(cache_file, "r") as h5:
        missing = sorted(required.difference(h5.keys()))
        forbidden = sorted({"av_grid", "rv_grid"}.intersection(h5.keys()))
        if missing or forbidden:
            raise ValueError(f"Invalid native cache: missing={missing}, forbidden={forbidden}")
        a550 = np.asarray(h5["a550_grid"][:], dtype=float)
        r550 = np.asarray(h5["r550_grid"][:], dtype=float)
        if not np.isclose(a550[0], 0.0):
            raise ValueError("A550 grid must start at zero")
        if np.any(np.diff(a550) <= 0) or np.any(np.diff(r550) <= 0):
            raise ValueError("A550 and R550 grids must be strictly increasing")
        n_shape = h5["N"].shape
        if n_shape != (h5["params"].shape[0], r550.size, a550.size, h5["bands"].shape[0]):
            raise ValueError(f"N shape mismatch: {n_shape}")
        probe_vertices = sorted({0, n_shape[0] // 2, n_shape[0] - 1})
        probe_r = sorted({0, n_shape[1] // 2, n_shape[1] - 1})
        for iv in probe_vertices:
            d = np.asarray(h5["D"][iv], dtype=float)
            for ir in probe_r:
                n0 = np.asarray(h5["N"][iv, ir, 0], dtype=float)
                rel = np.max(np.abs(n0 / d - 1.0))
                if not np.isfinite(rel) or rel > 3.0e-6:
                    raise ValueError(f"A550=0 validation failed at vertex={iv}, R550 index={ir}: {rel}")

    if int(direct_checks) > 0:
        compare_cache_to_direct_integration(
            cache_file,
            n_checks=int(direct_checks),
            asset_base=asset_base,
            tolerance_mag=tolerance_mag,
        )
    return cache_file


def compare_cache_to_direct_integration(cache_file, n_checks=8, asset_base=None, tolerance_mag=1.0e-4, seed=42):
    cache_file = Path(cache_file)
    asset_base = _asset_base(asset_base)
    rng = np.random.default_rng(seed)
    with h5py.File(cache_file, "r") as h5:
        law = str(h5.attrs["extinction_law"])
        a_grid = np.asarray(h5["a550_grid"][:], dtype=float)
        r_grid = np.asarray(h5["r550_grid"][:], dtype=float)
        native_grid = np.asarray(h5["native_shape_for_r550"][:], dtype=float)
        files = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["file"][:]]
        bands = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["bands"][:]]
        filters = _load_filters(asset_base, bands=bands) if set(bands).issubset(FILTER_FILENAMES) else _load_johnson_bv_filters(asset_base)
        for _ in range(int(n_checks)):
            iv = int(rng.integers(0, len(files)))
            ir = int(rng.integers(0, len(r_grid)))
            ia = int(rng.integers(0, len(a_grid)))
            raw = np.loadtxt(files[iv])
            flux = np.asarray(raw if raw.ndim == 1 else raw[:, 0], dtype=float)
            wave_spec = np.loadtxt(asset_base / "r500/m+0.00/bosz2024_wave_r500.txt")
            k550 = float(_evaluate_law(law, native_grid[ir], [0.550])[0])
            for ib, band in enumerate(bands):
                wave, response, photon = filters[band]
                kernel = np.interp(wave, wave_spec, flux, left=0.0, right=0.0) * response * (wave if photon else 1.0)
                d = np.trapz(kernel, wave)
                curve = _evaluate_law(law, native_grid[ir], wave * 1.0e-4) / k550
                direct_n = np.trapz(kernel * 10.0 ** (-0.4 * a_grid[ia] * curve), wave)
                cache_n = float(h5["N"][iv, ir, ia, ib])
                delta_mag = abs(-2.5 * np.log10(cache_n / direct_n))
                if not np.isfinite(delta_mag) or delta_mag > tolerance_mag:
                    raise ValueError(
                        f"Direct integration mismatch {delta_mag:.6g} mag for vertex={iv}, "
                        f"R550={r_grid[ir]:.6g}, A550={a_grid[ia]:.6g}, band={band}"
                    )
    return True


def build_native_a550_r550_cache(
    law="G23",
    cache_file=None,
    asset_base=None,
    a550_min=0.0,
    a550_max=20.0,
    a550_step=0.02,
    r550_step=0.02,
    mapping_dense_size=20001,
    overwrite=False,
    max_spectra=None,
    direct_validation_checks=8,
    n_workers=20,
):
    law = _law_name(law)
    asset_base = _asset_base(asset_base)
    cache_file = Path(DEFAULT_CACHE_FILES[law] if cache_file is None else cache_file)
    if cache_file.exists() and not overwrite:
        validate_native_cache(cache_file, direct_checks=0)
        print(f"Validated existing native cache: {cache_file}")
        return cache_file

    mapping = build_r550_mapping(law, dense_size=mapping_dense_size, r550_step=r550_step)
    a550_grid = np.arange(a550_min, a550_max + 0.5 * a550_step, a550_step, dtype=np.float64)
    a550_grid[0] = float(a550_min)
    a550_grid[-1] = float(a550_max)
    spectra = _scan_spectra(asset_base)
    if max_spectra is not None:
        spectra = spectra[: int(max_spectra)]
    if not spectra:
        raise RuntimeError(f"No matching BOSZ spectra found under {asset_base / 'r500'}")
    filters = _load_filters(asset_base)
    curves = _normalized_curves_on_filters(law, mapping["native_shape_for_r550"], filters)
    wave_spec = np.loadtxt(asset_base / "r500/m+0.00/bosz2024_wave_r500.txt").astype(np.float64)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_name(cache_file.name + f".tmp-{os.getpid()}")
    if tmp_file.exists():
        tmp_file.unlink()
    params = np.asarray([[s.teff, s.logg, s.mh] for s in spectra], dtype=np.float32)
    kinds = np.asarray([s.kind.encode("ascii") for s in spectra])
    files = np.asarray([str(s.path).encode("utf-8") for s in spectra])
    versions = _versions()
    try:
        with h5py.File(tmp_file, "w") as h5:
            h5.attrs["description"] = "BOSZ R500 native monochromatic A550/R550 pre-convolved D and N cache"
            h5.attrs["created_unix_time"] = time.time()
            h5.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h5.attrs["extinction_law"] = law
            h5.attrs["normalization"] = "A(lambda)/A550"
            h5.attrs["A550_reference_nm"] = 550.0
            h5.attrs["R550_blue_reference_nm"] = 440.0
            h5.attrs["R550_red_reference_nm"] = 550.0
            h5.attrs["law_native_shape_parameter"] = "Rv used only during cache construction"
            h5.attrs["extinction_law_implementation"] = (
                "dust_extinction.parameter_averages.G23"
                if law == "G23"
                else f"jupyter_mw_sed_pcprior_{law}_method2_distanceprior_emcee_direct.{law}"
            )
            h5.attrs["software_versions_json"] = json.dumps(versions, sort_keys=True)
            h5.attrs["filter_counting_json"] = json.dumps(
                {band: ("photon" if FILTER_FILENAMES[band][1] else "energy") for band in ALL_BANDS},
                sort_keys=True,
            )
            h5.attrs["asset_base"] = str(asset_base)
            h5.attrs["mapping_max_inverse_error"] = mapping["max_inverse_error"]
            h5.create_dataset("params", data=params)
            h5.create_dataset("kind", data=kinds)
            h5.create_dataset("file", data=files)
            h5.create_dataset("bands", data=np.asarray([b.encode("ascii") for b in ALL_BANDS]))
            h5.create_dataset("a550_grid", data=a550_grid.astype(np.float32))
            h5.create_dataset("r550_grid", data=mapping["r550_grid"].astype(np.float64))
            h5.create_dataset("native_shape_for_r550", data=mapping["native_shape_for_r550"].astype(np.float64))
            h5.create_dataset(
                "D", shape=(len(spectra), len(ALL_BANDS)), dtype="f4",
                chunks=(1, len(ALL_BANDS)), compression="gzip", compression_opts=4,
            )
            dset_n = h5.create_dataset(
                "N",
                shape=(len(spectra), len(mapping["r550_grid"]), len(a550_grid), len(ALL_BANDS)),
                dtype="f4",
                chunks=(1, 1, min(512, len(a550_grid)), len(ALL_BANDS)),
                compression="gzip",
                compression_opts=4,
            )
            for i, spectrum in enumerate(tqdm(spectra, desc=f"{law} BOSZ pre-convolution")):
                d, n = _integrals_one_spectrum(spectrum.path, wave_spec, filters, curves, a550_grid)
                h5["D"][i, :] = d
                dset_n[i, :, :, :] = n
            h5.flush()
        validate_native_cache(
            tmp_file,
            direct_checks=int(direct_validation_checks),
            asset_base=asset_base,
        )
        os.replace(tmp_file, cache_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    print(f"Native {law} A550/R550 cache written and validated: {cache_file}")
    return cache_file


def build_all_native_a550_r550_caches(laws=("G23", "WC2019", "F19", "F99"), **kwargs):
    return {law: build_native_a550_r550_cache(law=law, **kwargs) for law in laws}


def build_johnson_bv_companion_cache(
    law="G23",
    cache_file=None,
    asset_base=None,
    a550_min=0.0,
    a550_max=20.0,
    a550_step=0.02,
    r550_step=0.02,
    mapping_dense_size=20001,
    overwrite=False,
    max_spectra=None,
    direct_validation_checks=8,
    n_workers=20,
):
    """Build a small B/V-only companion cache for posterior color excesses."""
    law = _law_name(law)
    asset_base = _asset_base(asset_base)
    cache_file = Path(
        DEFAULT_JOHNSON_BV_CACHE_FILES[law] if cache_file is None else cache_file
    )
    if cache_file.exists() and not overwrite:
        validate_native_cache(cache_file, direct_checks=0)
        print(f"Validated existing Johnson B/V companion cache: {cache_file}")
        return cache_file

    mapping = build_r550_mapping(
        law, dense_size=mapping_dense_size, r550_step=r550_step
    )
    a550_grid = np.arange(
        a550_min, a550_max + 0.5 * a550_step, a550_step, dtype=np.float64
    )
    a550_grid[0] = float(a550_min)
    a550_grid[-1] = float(a550_max)
    spectra = _scan_spectra(asset_base)
    if max_spectra is not None:
        spectra = spectra[: int(max_spectra)]
    if not spectra:
        raise RuntimeError(f"No matching BOSZ spectra found under {asset_base / 'r500'}")
    bands = list(JOHNSON_BV_BANDS)
    filters = _load_johnson_bv_filters(asset_base)
    curves = _normalized_curves_on_filters(
        law, mapping["native_shape_for_r550"], filters, bands=bands
    )
    wave_spec = np.loadtxt(
        asset_base / "r500/m+0.00/bosz2024_wave_r500.txt"
    ).astype(np.float64)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_name(cache_file.name + f".tmp-{os.getpid()}")
    if tmp_file.exists():
        tmp_file.unlink()
    params = np.asarray([[s.teff, s.logg, s.mh] for s in spectra], dtype=np.float32)
    kinds = np.asarray([s.kind.encode("ascii") for s in spectra])
    files = np.asarray([str(s.path).encode("utf-8") for s in spectra])
    versions = _versions()
    try:
        with h5py.File(tmp_file, "w") as h5:
            h5.attrs["description"] = "BOSZ R500 native A550/R550 Johnson B/V companion cache"
            h5.attrs["created_unix_time"] = time.time()
            h5.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h5.attrs["extinction_law"] = law
            h5.attrs["cache_type"] = "johnson_bv_companion"
            h5.attrs["normalization"] = "A(lambda)/A550"
            h5.attrs["A550_reference_nm"] = 550.0
            h5.attrs["R550_blue_reference_nm"] = 440.0
            h5.attrs["R550_red_reference_nm"] = 550.0
            h5.attrs["johnson_filter_source"] = "pyphot GROUND_JOHNSON_B/V exported response tables"
            h5.attrs["software_versions_json"] = json.dumps(versions, sort_keys=True)
            h5.attrs["filter_counting_json"] = json.dumps(
                {band: "energy" for band in bands}, sort_keys=True
            )
            h5.attrs["asset_base"] = str(asset_base)
            h5.attrs["mapping_max_inverse_error"] = mapping["max_inverse_error"]
            h5.create_dataset("params", data=params)
            h5.create_dataset("kind", data=kinds)
            h5.create_dataset("file", data=files)
            h5.create_dataset("bands", data=np.asarray([b.encode("ascii") for b in bands]))
            h5.create_dataset("a550_grid", data=a550_grid.astype(np.float32))
            h5.create_dataset("r550_grid", data=mapping["r550_grid"].astype(np.float64))
            h5.create_dataset(
                "native_shape_for_r550",
                data=mapping["native_shape_for_r550"].astype(np.float64),
            )
            h5.create_dataset(
                "D", shape=(len(spectra), len(bands)), dtype="f4",
                chunks=(1, len(bands)), compression="gzip", compression_opts=4,
            )
            dset_n = h5.create_dataset(
                "N",
                shape=(len(spectra), len(mapping["r550_grid"]), len(a550_grid), len(bands)),
                dtype="f4",
                chunks=(1, 1, min(512, len(a550_grid)), len(bands)),
                compression="gzip",
                compression_opts=4,
            )
            if int(n_workers) > 1:
                executor = ProcessPoolExecutor(
                    max_workers=int(n_workers),
                    initializer=_init_companion_build_worker,
                    initargs=(wave_spec, filters, curves, a550_grid, bands),
                )
                futures = [
                    executor.submit(_companion_integrals_task, (i, str(s.path)))
                    for i, s in enumerate(spectra)
                ]
                try:
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"{law} Johnson B/V companion",
                    ):
                        i, d, n = future.result()
                        h5["D"][i, :] = d
                        dset_n[i, :, :, :] = n
                except BaseException:
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
            else:
                for i, spectrum in enumerate(tqdm(spectra, desc=f"{law} Johnson B/V companion")):
                    d, n = _integrals_one_spectrum(
                        spectrum.path, wave_spec, filters, curves, a550_grid, bands=bands
                    )
                    h5["D"][i, :] = d
                    dset_n[i, :, :, :] = n
            h5.flush()
        validate_native_cache(tmp_file, direct_checks=int(direct_validation_checks), asset_base=asset_base)
        os.replace(tmp_file, cache_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    print(f"Johnson B/V companion cache written and validated: {cache_file}")
    return cache_file


class PreconvolvedBOSZModel:
    """Read either a native A550/R550 cache or a legacy Av/Rv cache."""

    def __init__(self, cache_file):
        self.cache_file = Path(cache_file)
        self.h5 = h5py.File(self.cache_file, "r")
        self.params = np.asarray(self.h5["params"][:], dtype=float)
        self.bands = [x.decode() if isinstance(x, bytes) else str(x) for x in self.h5["bands"][:]]
        self.band_index = {band: i for i, band in enumerate(self.bands)}
        self.base_band = "RP"
        self.color_bands = [b for b in ["BP", "G", "J", "H", "K", "W1"] if b in self.band_index]
        self.is_native_a550_r550 = "a550_grid" in self.h5 and "r550_grid" in self.h5
        if self.is_native_a550_r550:
            self.a550_grid = np.asarray(self.h5["a550_grid"][:], dtype=float)
            self.r550_grid = np.asarray(self.h5["r550_grid"][:], dtype=float)
        else:
            self.av_grid = np.asarray(self.h5["av_grid"][:], dtype=float)
            self.rv_grid = np.asarray(self.h5["rv_grid"][:], dtype=float)
        self.tri = Delaunay(self.params)

    def close(self):
        self.h5.close()

    def _simplex_weights(self, teff, logg, mh):
        point = np.asarray([teff, logg, mh], dtype=float)
        simplex = self.tri.find_simplex(point)
        if simplex < 0:
            raise ValueError(f"Point outside BOSZ convex hull: Teff={teff}, logg={logg}, [M/H]={mh}")
        transform = self.tri.transform[simplex]
        bary = np.dot(transform[:3], point - transform[3])
        return self.tri.simplices[simplex].astype(int), np.r_[bary, 1.0 - bary.sum()].astype(float)

    @staticmethod
    def _bracket(grid, value):
        if value < grid[0] or value > grid[-1]:
            raise ValueError(f"value {value} outside grid [{grid[0]}, {grid[-1]}]")
        hi = int(np.searchsorted(grid, value, side="right"))
        if hi == 0:
            return 0, 0, 0.0
        if hi >= len(grid):
            return len(grid) - 1, len(grid) - 1, 0.0
        lo = hi - 1
        return lo, hi, float((value - grid[lo]) / (grid[hi] - grid[lo]))

    def band_integrals(self, teff, logg, mh, shape, amount):
        vertices, weights = self._simplex_weights(teff, logg, mh)
        shape_grid = self.r550_grid if self.is_native_a550_r550 else self.rv_grid
        amount_grid = self.a550_grid if self.is_native_a550_r550 else self.av_grid
        ir0, ir1, tr = self._bracket(shape_grid, shape)
        ia0, ia1, ta = self._bracket(amount_grid, amount)
        d_interp = np.zeros(len(self.bands), dtype=float)
        n_interp = np.zeros(len(self.bands), dtype=float)
        for vertex, weight in zip(vertices, weights):
            d_v = np.asarray(self.h5["D"][vertex, :], dtype=float)
            n00 = np.asarray(self.h5["N"][vertex, ir0, ia0, :], dtype=float)
            n10 = np.asarray(self.h5["N"][vertex, ir1, ia0, :], dtype=float)
            n01 = np.asarray(self.h5["N"][vertex, ir0, ia1, :], dtype=float)
            n11 = np.asarray(self.h5["N"][vertex, ir1, ia1, :], dtype=float)
            n_value = (
                (1.0 - tr) * (1.0 - ta) * n00 + tr * (1.0 - ta) * n10
                + (1.0 - tr) * ta * n01 + tr * ta * n11
            )
            d_interp += weight * d_v
            n_interp += weight * n_value
        if np.any(~np.isfinite(d_interp)) or np.any(d_interp <= 0):
            raise ValueError("non-positive interpolated intrinsic band integral")
        if np.any(~np.isfinite(n_interp)) or np.any(n_interp <= 0):
            raise ValueError("non-positive interpolated extincted band integral")
        return d_interp, n_interp

    def band_extinction(self, teff, logg, mh, r550=None, a550=None, rv=None, av=None):
        if self.is_native_a550_r550:
            shape, amount = r550, a550
        else:
            shape, amount = rv, av
        d, n = self.band_integrals(teff, logg, mh, float(shape), float(amount))
        values = -2.5 * np.log10(n / d)
        return {band: float(values[i]) for i, band in enumerate(self.bands)}

    def band_extinction_batch(self, sample_table, bands=None, chunk_size=512):
        """Compute band extinction for many A550/R550 posterior draws.

        The stellar-parameter simplex search, A550/R550 bracketing, and the
        four-vertex interpolation are vectorized. HDF5 vertex slabs are
        loaded once per batch and reused for all matching draws.
        """
        if not self.is_native_a550_r550:
            raise ValueError("band_extinction_batch requires a native A550/R550 cache")
        required = ["Teff", "logg", "MH", "R550", "A550"]
        if hasattr(sample_table, "__getitem__") and not isinstance(sample_table, np.ndarray):
            values = sample_table[required].to_numpy(dtype=float, copy=False)
        else:
            values = np.asarray(sample_table, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(required):
            raise ValueError(f"sample_table must contain columns {required}")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        output_bands = list(self.bands if bands is None else bands)
        output_indices = np.asarray([self.band_index[b] for b in output_bands], dtype=int)
        output = np.empty((len(values), len(output_bands)), dtype=float)
        d_all = np.asarray(self.h5["D"][:], dtype=np.float32)
        if not hasattr(self, "_batch_vertex_cache"):
            self._batch_vertex_cache = {}

        for start in range(0, len(values), int(chunk_size)):
            stop = min(start + int(chunk_size), len(values))
            block = values[start:stop]
            simplex = np.asarray(self.tri.find_simplex(block[:, :3]), dtype=int)
            if np.any(simplex < 0):
                raise ValueError("posterior point outside BOSZ convex hull")
            transform = self.tri.transform[simplex]
            bary = np.einsum(
                "nij,nj->ni", transform[:, :3, :],
                block[:, :3] - transform[:, 3, :], optimize=True
            )
            weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
            vertices = self.tri.simplices[simplex].astype(int, copy=False)

            def bracket(grid, x):
                if np.any(~np.isfinite(x)) or np.any(x < grid[0]) or np.any(x > grid[-1]):
                    raise ValueError("posterior value outside native cache grid")
                hi = np.clip(np.searchsorted(grid, x, side="right"), 1, len(grid) - 1)
                lo = hi - 1
                upper = x >= grid[-1]
                lo[upper] = len(grid) - 1
                hi[upper] = len(grid) - 1
                denom = grid[hi] - grid[lo]
                frac = np.divide(x - grid[lo], denom, out=np.zeros_like(x), where=denom != 0)
                return lo, hi, frac

            ir0, ir1, tr = bracket(self.r550_grid, block[:, 3])
            ia0, ia1, ta = bracket(self.a550_grid, block[:, 4])
            d_interp = np.einsum("nk,nkj->nj", weights, d_all[vertices], optimize=True)
            n_interp = np.zeros((len(block), len(self.bands)), dtype=float)

            for corner in range(vertices.shape[1]):
                corner_vertices = vertices[:, corner]
                for vertex in np.unique(corner_vertices):
                    mask = corner_vertices == vertex
                    if int(vertex) not in self._batch_vertex_cache:
                        self._batch_vertex_cache[int(vertex)] = np.asarray(
                            self.h5["N"][int(vertex), :, :, :], dtype=np.float32
                        )
                    else:
                        self._batch_vertex_cache[int(vertex)] = self._batch_vertex_cache.pop(int(vertex))
                    n_v = self._batch_vertex_cache[int(vertex)]
                    while len(self._batch_vertex_cache) > 128:
                        self._batch_vertex_cache.pop(next(iter(self._batch_vertex_cache)))
                    tr_m = tr[mask, None]
                    ta_m = ta[mask, None]
                    n00 = n_v[ir0[mask], ia0[mask], :]
                    n10 = n_v[ir1[mask], ia0[mask], :]
                    n01 = n_v[ir0[mask], ia1[mask], :]
                    n11 = n_v[ir1[mask], ia1[mask], :]
                    interpolated = (
                        (1.0 - tr_m) * (1.0 - ta_m) * n00
                        + tr_m * (1.0 - ta_m) * n10
                        + (1.0 - tr_m) * ta_m * n01
                        + tr_m * ta_m * n11
                    )
                    n_interp[mask] += weights[mask, corner, None] * interpolated

            d_used = d_interp[:, output_indices]
            n_used = n_interp[:, output_indices]
            if np.any(~np.isfinite(d_used)) or np.any(d_used <= 0) or np.any(~np.isfinite(n_used)) or np.any(n_used <= 0):
                raise ValueError("non-positive batch band integral")
            output[start:stop] = -2.5 * np.log10(n_used / d_used)
        return output

    def color_excess(self, teff, logg, mh, rv, av, color_bands=None):
        values = self.band_extinction(teff, logg, mh, rv=rv, av=av)
        if color_bands is None:
            color_bands = self.color_bands
        return {f"E_{band}_RP": values[band] - values["RP"] for band in color_bands}


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", action="append", choices=sorted(LAW_NATIVE_RANGES))
    parser.add_argument("--asset-base")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--a550-max", type=float, default=20.0)
    parser.add_argument("--a550-step", type=float, default=0.02)
    parser.add_argument("--r550-step", type=float, default=0.02)
    parser.add_argument("--direct-checks", type=int, default=8)
    args = parser.parse_args()
    laws = args.law or list(LAW_NATIVE_RANGES)
    output_dir = Path(args.output_dir)
    results = {}
    for law in laws:
        output = output_dir / f"bosz_r500_{law}_A550_R550_preconvolved_ND.h5"
        results[law] = str(build_native_a550_r550_cache(
            law=law,
            cache_file=output,
            asset_base=args.asset_base,
            a550_max=args.a550_max,
            a550_step=args.a550_step,
            r550_step=args.r550_step,
            overwrite=args.overwrite,
            max_spectra=args.max_spectra,
            direct_validation_checks=args.direct_checks,
        ))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()
