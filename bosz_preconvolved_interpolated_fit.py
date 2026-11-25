#!/usr/bin/env python3
"""
BOSZ/MARCS pre-convolved cache and interpolated Cepheid color-excess fitting.

This module is designed for the downloaded BOSZ R=500 subset:
    model type: ms/mp
    4000 <= Teff <= 7500 K
    0.5 <= logg <= 3.5
    -1.5 <= [M/H] <= 0.5
    a+0.00_c+0.00_v4

Main idea
---------
For each original BOSZ spectrum and each band, cache:

    D(model, band) = integral F(lambda) S_band(lambda) W_band(lambda) d lambda

    N(model, Rv, Av, band)
        = integral F(lambda) S_band(lambda) W_band(lambda)
          10^(-0.4 Av k(lambda, Rv)) d lambda

where W_band(lambda)=lambda for photon-counting systems, and 1 for
energy-counting systems such as WISE W1 here.

For arbitrary (Teff, logg, [M/H]), the code uses a Delaunay simplex in
the original parameter space and linearly interpolates D and N. This is
equivalent to interpolating the spectrum first and then doing the
bandpass convolution, but it avoids writing many interpolated spectra.

Usage in a notebook
-------------------
    from bosz_preconvolved_interpolated_fit import *

    build_preconvolved_cache()
    model = PreconvolvedBOSZModel()

    fit_res = cepheids3636.progress_apply(
        lambda row: fit_star_continuous(row, model),
        axis=1,
    )
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
from tqdm import tqdm

# If you want to use your own WC2019/WC2024 class, define/import it before
# calling build_preconvolved_cache and pass ExtLaw=WC2019.
try:
    from dust_extinction.parameter_averages import G23

    DEFAULT_EXT_LAW = G23
except Exception:
    DEFAULT_EXT_LAW = None


# =========================
# Paths and default settings
# =========================

BASE_DIR = Path(__file__).resolve().parent
SPEC_BASE_DIR = BASE_DIR / "r500"
WAVE_FILE = SPEC_BASE_DIR / "m+0.00" / "bosz2024_wave_r500.txt"

OUT_DIR = BASE_DIR / "codex 自动化文件"
CACHE_FILE = OUT_DIR / "bosz_r500_msmp_a0_c0_v4_preconvolved_ND.h5"

FILTER_PATHS = {
    "BP": (BASE_DIR / "数据/滤光片数据/GAIA_GAIA3.Gbp.dat", True),
    "G": (BASE_DIR / "数据/滤光片数据/GAIA_GAIA3.G.dat", True),
    "RP": (BASE_DIR / "数据/滤光片数据/GAIA_GAIA3.Grp.dat", True),
    "J": (BASE_DIR / "数据/滤光片数据/2MASS_2MASS.J.dat", True),
    "H": (BASE_DIR / "数据/滤光片数据/2MASS_2MASS.H.dat", True),
    "K": (BASE_DIR / "数据/滤光片数据/2MASS_2MASS.Ks.dat", True),
    # WISE relative spectral response is treated as energy-counting here.
    "W1": (BASE_DIR / "数据/滤光片数据/WISE_WISE.W1.dat", False),
}

BASE_BAND = "RP"
COLOR_BANDS = ["BP", "G", "J", "H", "K", "W1"]
ALL_BANDS = ["BP", "G", "RP", "J", "H", "K", "W1"]

SPEC_RE = re.compile(
    r"bosz2024_(?P<kind>ms|mp)_"
    r"t(?P<teff>\d+)_"
    r"g(?P<logg>[+-]\d+\.\d+)_"
    r"m(?P<mh>[+-]\d+\.\d+)_"
    r"a\+0\.00_c\+0\.00_v4_r500_resam\.txt\.gz$"
)


@dataclass
class SpectrumMeta:
    path: Path
    kind: str
    teff: float
    logg: float
    mh: float


def load_filter(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0].astype(float), data[:, 1].astype(float)


def scan_bosz_spectra(
    spec_base_dir: Path = SPEC_BASE_DIR,
    teff_range: tuple[float, float] = (4000.0, 7500.0),
    logg_range: tuple[float, float] = (0.5, 3.5),
    mh_range: tuple[float, float] = (-1.5, 0.5),
) -> list[SpectrumMeta]:
    """Find downloaded ms/mp, a0/c0/v4 BOSZ spectra in the selected range."""
    spectra: list[SpectrumMeta] = []

    for path in spec_base_dir.rglob("*.txt.gz"):
        m = SPEC_RE.match(path.name)
        if not m:
            continue

        teff = float(m.group("teff"))
        logg = float(m.group("logg"))
        mh = float(m.group("mh"))

        if not (teff_range[0] <= teff <= teff_range[1]):
            continue
        if not (logg_range[0] <= logg <= logg_range[1]):
            continue
        if not (mh_range[0] <= mh <= mh_range[1]):
            continue

        spectra.append(
            SpectrumMeta(
                path=path,
                kind=m.group("kind"),
                teff=teff,
                logg=logg,
                mh=mh,
            )
        )

    spectra.sort(key=lambda x: (x.mh, x.teff, x.logg, x.kind, str(x.path)))
    return spectra


def band_integrals_for_one_spectrum(
    spec_path: Path,
    w_spec: np.ndarray,
    filters: dict[str, tuple[np.ndarray, np.ndarray, bool]],
    ext_aav_cache: dict[float, dict[str, np.ndarray]],
    rv_grid: np.ndarray,
    av_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return D[band] and N[Rv, Av, band] for one spectrum.
    """
    spec = np.loadtxt(spec_path)
    f_spec = spec if spec.ndim == 1 else spec[:, 0]
    f_spec = np.asarray(f_spec, dtype=np.float64)

    n_rv = len(rv_grid)
    n_av = len(av_grid)
    n_band = len(ALL_BANDS)

    d = np.full(n_band, np.nan, dtype=np.float32)
    n = np.full((n_rv, n_av, n_band), np.nan, dtype=np.float32)

    # Cache kernels once per band for this spectrum.
    kernels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ib, band in enumerate(ALL_BANDS):
        w_band, s_band, photon_based = filters[band]
        f_on_band = np.interp(w_band, w_spec, f_spec, left=0.0, right=0.0)

        if photon_based:
            kernel = f_on_band * s_band * w_band
        else:
            kernel = f_on_band * s_band

        kernel = np.asarray(kernel, dtype=np.float64)
        den = np.trapz(kernel, w_band)

        if np.isfinite(den) and den > 0:
            d[ib] = den
            kernels[band] = (w_band, kernel)

    for irv, rv in enumerate(rv_grid):
        rv_key = float(np.round(rv, 3))
        for ib, band in enumerate(ALL_BANDS):
            if band not in kernels:
                continue

            w_band, kernel = kernels[band]
            aav = ext_aav_cache[rv_key][band].astype(np.float64)

            # Shape: (n_Av, n_wave)
            atten = 10.0 ** (-0.4 * av_grid[:, None] * aav[None, :])
            num = np.trapz(atten * kernel[None, :], w_band, axis=1)
            n[irv, :, ib] = num.astype(np.float32)

    return d, n


def build_preconvolved_cache(
    cache_file: Path = CACHE_FILE,
    spec_base_dir: Path = SPEC_BASE_DIR,
    wave_file: Path = WAVE_FILE,
    rv_grid: np.ndarray | None = None,
    av_grid: np.ndarray | None = None,
    ExtLaw=None,
    overwrite: bool = False,
) -> Path:
    """
    Build HDF5 cache containing pre-convolved D and N arrays.

    Notes
    -----
    - Default Av grid is coarser than your original code to keep the first
      cache manageable. Increase it if needed.
    - For production runs you can use:
          rv_grid=np.linspace(2.3, 5.6, 331)
          av_grid=np.linspace(0.0, 20.0, 2001)
    """
    if ExtLaw is None:
        if DEFAULT_EXT_LAW is None:
            raise RuntimeError("No extinction law available. Pass ExtLaw explicitly.")
        ExtLaw = DEFAULT_EXT_LAW

    if rv_grid is None:
        rv_grid = np.linspace(2.3, 5.6, 166)
    if av_grid is None:
        av_grid = np.linspace(0.0, 20.0, 1001)

    # Keep Rv in float64.  G23 validates Rv strictly against [2.3, 5.6];
    # float32 can turn 2.3 into 2.29999995 and fail at the lower boundary.
    rv_grid = np.clip(np.round(np.asarray(rv_grid, dtype=np.float64), 10), 2.3, 5.6)
    av_grid = np.asarray(av_grid, dtype=np.float32)

    if cache_file.exists() and not overwrite:
        print(f"Cache already exists: {cache_file}")
        return cache_file

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    spectra = scan_bosz_spectra(spec_base_dir=spec_base_dir)
    if len(spectra) == 0:
        raise RuntimeError(f"No matching spectra found under {spec_base_dir}")

    print(f"Found {len(spectra)} spectra for pre-convolution.")
    w_spec = np.loadtxt(wave_file).astype(np.float64)

    filters: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
    for band, (path, photon_based) in FILTER_PATHS.items():
        w, s = load_filter(path)
        filters[band] = (w, s, photon_based)

    print("Precomputing extinction k(lambda,Rv)=A(lambda)/Av on filter grids...")
    ext_aav_cache: dict[float, dict[str, np.ndarray]] = {}
    for rv in tqdm(rv_grid, desc="Rv"):
        rv_key = float(np.round(rv, 3))
        ext_model = ExtLaw(Rv=float(rv))
        ext_aav_cache[rv_key] = {}
        for band in ALL_BANDS:
            w_band = filters[band][0]
            x = (1.0 / (w_band * 1e-4)) / u.micron
            ext_aav_cache[rv_key][band] = np.asarray(ext_model(x), dtype=np.float32)

    params = np.array([[s.teff, s.logg, s.mh] for s in spectra], dtype=np.float32)
    kinds = np.array([s.kind.encode("ascii") for s in spectra])
    files = np.array([str(s.path).encode("utf-8") for s in spectra])

    if cache_file.exists():
        cache_file.unlink()

    with h5py.File(cache_file, "w") as h5:
        h5.attrs["description"] = "BOSZ R500 ms/mp a0 c0 v4 pre-convolved D and N cache"
        h5.attrs["created_unix_time"] = time.time()
        h5.attrs["ext_law"] = getattr(ExtLaw, "__name__", str(ExtLaw))
        h5.attrs["base_band"] = BASE_BAND
        h5.create_dataset("params", data=params)
        h5.create_dataset("kind", data=kinds)
        h5.create_dataset("file", data=files)
        h5.create_dataset("rv_grid", data=rv_grid)
        h5.create_dataset("av_grid", data=av_grid)
        h5.create_dataset("bands", data=np.array([b.encode("ascii") for b in ALL_BANDS]))

        dset_d = h5.create_dataset(
            "D",
            shape=(len(spectra), len(ALL_BANDS)),
            dtype="f4",
            chunks=(1, len(ALL_BANDS)),
            compression="gzip",
            compression_opts=4,
        )
        dset_n = h5.create_dataset(
            "N",
            shape=(len(spectra), len(rv_grid), len(av_grid), len(ALL_BANDS)),
            dtype="f4",
            chunks=(1, 1, min(512, len(av_grid)), len(ALL_BANDS)),
            compression="gzip",
            compression_opts=4,
        )

        for i, meta in enumerate(tqdm(spectra, desc="Pre-convolving spectra")):
            d, n = band_integrals_for_one_spectrum(
                spec_path=meta.path,
                w_spec=w_spec,
                filters=filters,
                ext_aav_cache=ext_aav_cache,
                rv_grid=rv_grid,
                av_grid=av_grid,
            )
            dset_d[i, :] = d
            dset_n[i, :, :, :] = n

    print(f"Cache written to: {cache_file}")
    return cache_file


class PreconvolvedBOSZModel:
    """Load pre-convolved cache and evaluate interpolated color excesses."""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.h5 = h5py.File(self.cache_file, "r")
        self.params = self.h5["params"][:].astype(float)
        self.rv_grid = self.h5["rv_grid"][:].astype(float)
        self.av_grid = self.h5["av_grid"][:].astype(float)
        self.bands = [b.decode("ascii") for b in self.h5["bands"][:]]
        self.band_index = {b: i for i, b in enumerate(self.bands)}
        self.color_bands = [b for b in COLOR_BANDS if b in self.band_index]
        self.base_band = BASE_BAND
        self.tri = Delaunay(self.params)

    def close(self) -> None:
        self.h5.close()

    def _simplex_weights(self, teff: float, logg: float, mh: float) -> tuple[np.ndarray, np.ndarray]:
        point = np.array([teff, logg, mh], dtype=float)
        simplex = self.tri.find_simplex(point)
        if simplex < 0:
            raise ValueError(f"Point outside BOSZ convex hull: Teff={teff}, logg={logg}, [M/H]={mh}")

        transform = self.tri.transform[simplex]
        delta = point - transform[3]
        bary = np.dot(transform[:3], delta)
        weights = np.r_[bary, 1.0 - bary.sum()]
        vertices = self.tri.simplices[simplex]
        return vertices.astype(int), weights.astype(float)

    @staticmethod
    def _bracket(grid: np.ndarray, value: float) -> tuple[int, int, float]:
        if value < grid[0] or value > grid[-1]:
            raise ValueError(f"value {value} outside grid [{grid[0]}, {grid[-1]}]")
        hi = int(np.searchsorted(grid, value, side="right"))
        if hi == 0:
            return 0, 0, 0.0
        if hi >= len(grid):
            return len(grid) - 1, len(grid) - 1, 0.0
        lo = hi - 1
        t = (value - grid[lo]) / (grid[hi] - grid[lo])
        return lo, hi, float(t)

    def band_extinction(
        self,
        teff: float,
        logg: float,
        mh: float,
        rv: float,
        av: float,
    ) -> dict[str, float]:
        """
        Return A_band for all cached bands at arbitrary parameters.
        """
        vertices, weights = self._simplex_weights(teff, logg, mh)
        ir0, ir1, tr = self._bracket(self.rv_grid, rv)
        ia0, ia1, ta = self._bracket(self.av_grid, av)

        d_interp = np.zeros(len(self.bands), dtype=float)
        n_interp = np.zeros(len(self.bands), dtype=float)

        for vertex, weight in zip(vertices, weights):
            d_v = self.h5["D"][vertex, :].astype(float)
            n00 = self.h5["N"][vertex, ir0, ia0, :].astype(float)
            n10 = self.h5["N"][vertex, ir1, ia0, :].astype(float)
            n01 = self.h5["N"][vertex, ir0, ia1, :].astype(float)
            n11 = self.h5["N"][vertex, ir1, ia1, :].astype(float)

            n_rv_av = (
                (1.0 - tr) * (1.0 - ta) * n00
                + tr * (1.0 - ta) * n10
                + (1.0 - tr) * ta * n01
                + tr * ta * n11
            )

            d_interp += weight * d_v
            n_interp += weight * n_rv_av

        a_band = -2.5 * np.log10(n_interp / d_interp)
        return {band: float(a_band[i]) for i, band in enumerate(self.bands)}

    def color_excess(
        self,
        teff: float,
        logg: float,
        mh: float,
        rv: float,
        av: float,
        color_bands: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Return E_band_RP = A_band - A_RP.
        """
        if color_bands is None:
            color_bands = self.color_bands
        a_band = self.band_extinction(teff, logg, mh, rv, av)
        a_rp = a_band[self.base_band]
        return {f"E_{band}_RP": a_band[band] - a_rp for band in color_bands}


def collect_observed_colors(
    row: pd.Series,
    color_bands: list[str] = COLOR_BANDS,
    err_suffix_candidates: tuple[str, ...] = ("_error", "_err"),
) -> tuple[list[str], np.ndarray, np.ndarray]:
    used_bands: list[str] = []
    obs_vals: list[float] = []
    obs_errs: list[float] = []

    for band in color_bands:
        col = f"E_{band}_RP"
        if col not in row.index or not np.isfinite(row.get(col, np.nan)):
            continue

        err = np.nan
        for suffix in err_suffix_candidates:
            err_col = f"E_{band}_RP{suffix}"
            if err_col in row.index and np.isfinite(row.get(err_col, np.nan)):
                err = float(row[err_col])
                break

        if not np.isfinite(err) or err <= 0:
            continue

        used_bands.append(band)
        obs_vals.append(float(row[col]))
        obs_errs.append(err)

    return used_bands, np.asarray(obs_vals, dtype=float), np.asarray(obs_errs, dtype=float)


def build_color_covariance(
    row: pd.Series,
    used_bands: list[str],
    color_err: np.ndarray,
    include_shared_rp: bool = True,
    pc_corr: float = 0.0,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """
    Build covariance matrix for colors E(X-RP).

    Diagonal terms use the already propagated color-excess errors
    E_X_RP_err / E_X_RP_error. Off-diagonal terms account for the shared
    RP magnitude error:

        Cov[(X-RP), (Y-RP)] = Var(RP),  X != Y

    The PC-relation scatter is included in the diagonal because it is already
    part of E_X_RP_err in the current MW color-excess table. Its off-diagonal
    covariance is not saved by the PC fitting pipeline, so pc_corr defaults to
    zero. If you want a sensitivity test, use e.g. pc_corr=0.5.
    """
    n = len(used_bands)
    cov = np.diag(np.asarray(color_err, dtype=float) ** 2)

    rp_err = row.get("RP_magerr", np.nan)
    rp_var = 0.0
    if include_shared_rp and np.isfinite(rp_err) and float(rp_err) > 0:
        rp_var = float(rp_err) ** 2

    pc_sigmas = []
    for band in used_bands:
        sig = row.get(f"E_{band}_RP_PC_sigma", np.nan)
        pc_sigmas.append(float(sig) if np.isfinite(sig) and float(sig) > 0 else 0.0)
    pc_sigmas = np.asarray(pc_sigmas, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            off = rp_var
            if pc_corr != 0.0:
                off += float(pc_corr) * pc_sigmas[i] * pc_sigmas[j]

            # Numerical safety: keep pairwise correlation below unity.
            max_off = 0.98 * np.sqrt(cov[i, i] * cov[j, j])
            if np.isfinite(max_off) and max_off > 0:
                off = np.clip(off, -max_off, max_off)

            cov[i, j] = off
            cov[j, i] = off

    meta = {
        "covariance_used": True,
        "shared_rp_used": bool(rp_var > 0),
        "rp_magerr_used": float(rp_err) if np.isfinite(rp_err) else np.nan,
        "pc_corr_assumed": float(pc_corr),
    }
    return cov, meta


def cholesky_with_jitter(cov: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """
    Return lower Cholesky factor. Add small jitter if needed.
    """
    cov = np.asarray(cov, dtype=float)
    scale = float(np.nanmedian(np.diag(cov))) if cov.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    for power in range(0, 10):
        jitter = 0.0 if power == 0 else scale * 10.0 ** (-12 + power)
        try:
            return np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter), jitter, True
        except np.linalg.LinAlgError:
            continue

    # Last-resort fallback to diagonal covariance.
    diag = np.diag(np.diag(cov))
    jitter = scale * 1e-6
    return np.linalg.cholesky(diag + np.eye(cov.shape[0]) * jitter), jitter, False


def initial_av_from_egrp(e_g_rp: float, rv: float = 3.1) -> float:
    """
    A harmless rough initial guess. The coefficient is deliberately broad;
    least_squares will refine it using the pre-convolved model.
    """
    if not np.isfinite(e_g_rp):
        return 2.0
    return float(np.clip(e_g_rp / 0.18, 0.0, 20.0))


def fit_star_continuous(
    row: pd.Series,
    model: PreconvolvedBOSZModel,
    color_bands: list[str] = COLOR_BANDS,
    min_bands: int = 3,
    use_log_params: bool = False,
    use_covariance: bool = True,
    include_shared_rp: bool = True,
    pc_corr: float = 0.0,
) -> pd.Series:
    """
    Fit Teff, logg, [M/H], Rv, Av using the pre-convolved interpolated model.

    Parameters
    ----------
    row : pandas.Series
        Must contain E_{band}_RP and E_{band}_RP_error or E_{band}_RP_err.
    model : PreconvolvedBOSZModel
        Open pre-convolved cache model.
    min_bands : int
        At least 3 colors are recommended. With 2 colors, the 5-parameter
        fit is strongly underconstrained.
    use_log_params : bool
        Reserved for future transforms. Kept False here for readability.
    use_covariance : bool
        If True, minimize DeltaE.T @ C^-1 @ DeltaE instead of diagonal chi2.
    include_shared_rp : bool
        If True, use RP_magerr to fill off-diagonal covariance terms because
        all colors share RP.
    pc_corr : float
        Optional assumed correlation coefficient between PC-relation residuals
        of different colors. Default is 0 because the PC covariance matrix is
        not currently saved.
    """
    del use_log_params

    used_bands, obs, err = collect_observed_colors(row, color_bands=color_bands)
    n_used = len(used_bands)

    out_base = {
        "Teff_fit": np.nan,
        "logg_fit": np.nan,
        "MH_fit": np.nan,
        "Rv_fit": np.nan,
        "Av_fit": np.nan,
        "chi2_min": np.nan,
        "chi2_red": np.nan,
        "nband_fit": n_used,
        "bands_fit": ",".join(used_bands),
        "fit_success": False,
        "fit_message": "",
        "covariance_used": False,
        "shared_rp_used": False,
        "rp_magerr_used": np.nan,
        "pc_corr_assumed": np.nan,
        "covariance_jitter": np.nan,
        "covariance_cholesky_ok": False,
        "jac_rank": np.nan,
        "param_err_reliable": False,
    }
    for par in ["Teff", "logg", "MH", "Rv", "Av"]:
        out_base[f"{par}_err"] = np.nan
    for band in color_bands:
        out_base[f"delta_sigma_{band}"] = np.nan

    if n_used < min_bands:
        out_base["fit_message"] = "too_few_bands"
        return pd.Series(out_base)

    if use_covariance:
        cov, cov_meta = build_color_covariance(
            row=row,
            used_bands=used_bands,
            color_err=err,
            include_shared_rp=include_shared_rp,
            pc_corr=pc_corr,
        )
        chol, jitter, chol_ok = cholesky_with_jitter(cov)
        whiten_sigma = np.sqrt(np.diag(cov))
        out_base.update(cov_meta)
        out_base["covariance_jitter"] = float(jitter)
        out_base["covariance_cholesky_ok"] = bool(chol_ok)
    else:
        cov = np.diag(err**2)
        chol = np.diag(err)
        whiten_sigma = err
        out_base["covariance_used"] = False
        out_base["covariance_cholesky_ok"] = True
        out_base["covariance_jitter"] = 0.0

    def residual(theta: np.ndarray) -> np.ndarray:
        teff, logg, mh, rv, av = theta
        try:
            pred_dict = model.color_excess(
                teff=teff,
                logg=logg,
                mh=mh,
                rv=rv,
                av=av,
                color_bands=used_bands,
            )
            pred = np.array([pred_dict[f"E_{b}_RP"] for b in used_bands], dtype=float)
            if not np.all(np.isfinite(pred)):
                return np.full(n_used, 1e6)
            return np.linalg.solve(chol, obs - pred)
        except Exception:
            return np.full(n_used, 1e6)

    egrp = row.get("E_G_RP", np.nan)
    av0 = initial_av_from_egrp(egrp)

    # Multi-start helps avoid shallow local minima in Teff/Rv/Av.
    starts = [
        [5500.0, 2.0, 0.0, 3.1, av0],
        [5000.0, 1.5, 0.0, 3.1, av0],
        [6000.0, 2.0, 0.0, 3.1, av0],
        [6500.0, 2.5, 0.0, 3.1, av0],
        [5500.0, 2.0, -0.5, 3.1, av0],
        [5500.0, 2.0, 0.25, 3.1, av0],
        [5500.0, 2.0, 0.0, 2.6, av0],
        [5500.0, 2.0, 0.0, 4.0, av0],
    ]

    lower = np.array(
        [
            float(np.nanmin(model.params[:, 0])),
            float(np.nanmin(model.params[:, 1])),
            float(np.nanmin(model.params[:, 2])),
            float(np.nanmin(model.rv_grid)),
            float(np.nanmin(model.av_grid)),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            float(np.nanmax(model.params[:, 0])),
            float(np.nanmax(model.params[:, 1])),
            float(np.nanmax(model.params[:, 2])),
            float(np.nanmax(model.rv_grid)),
            float(np.nanmax(model.av_grid)),
        ],
        dtype=float,
    )

    best = None
    best_chi2 = np.inf

    for x0 in starts:
        x0 = np.minimum(np.maximum(np.asarray(x0, dtype=float), lower), upper)
        res = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4,
            max_nfev=300,
        )
        chi2 = float(np.sum(res.fun**2))
        if np.isfinite(chi2) and chi2 < best_chi2:
            best = res
            best_chi2 = chi2

    if best is None:
        out_base["fit_message"] = "fit_failed"
        return pd.Series(out_base)

    teff, logg, mh, rv, av = best.x
    pred_dict = model.color_excess(teff, logg, mh, rv, av, color_bands=used_bands)
    pred = np.array([pred_dict[f"E_{b}_RP"] for b in used_bands], dtype=float)
    delta_sigma = (obs - pred) / whiten_sigma

    dof = max(n_used - 5, 1)
    jtj = best.jac.T @ best.jac
    jac_rank = int(np.linalg.matrix_rank(best.jac))
    param_err = np.full(5, np.nan, dtype=float)
    param_err_reliable = False
    try:
        cov_param = np.linalg.pinv(jtj) * (best_chi2 / dof)
        diag = np.diag(cov_param)
        if np.all(np.isfinite(diag)):
            param_err = np.sqrt(np.clip(diag, 0.0, np.inf))
            param_err_reliable = bool((n_used > 5) and (jac_rank >= 5) and best.success)
    except Exception:
        pass

    out_base.update(
        {
            "Teff_fit": float(teff),
            "logg_fit": float(logg),
            "MH_fit": float(mh),
            "Rv_fit": float(rv),
            "Av_fit": float(av),
            "chi2_min": best_chi2,
            "chi2_red": best_chi2 / dof,
            "fit_success": bool(best.success),
            "fit_message": str(best.message),
            "Teff_err": float(param_err[0]),
            "logg_err": float(param_err[1]),
            "MH_err": float(param_err[2]),
            "Rv_err": float(param_err[3]),
            "Av_err": float(param_err[4]),
            "jac_rank": jac_rank,
            "param_err_reliable": param_err_reliable,
        }
    )

    for band, ds in zip(used_bands, delta_sigma):
        out_base[f"delta_sigma_{band}"] = float(ds)

    return pd.Series(out_base)


def fit_dataframe(
    df: pd.DataFrame,
    cache_file: Path = CACHE_FILE,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for fitting a whole dataframe."""
    model = PreconvolvedBOSZModel(cache_file=cache_file)
    try:
        tqdm.pandas()
        fit_res = df.progress_apply(lambda row: fit_star_continuous(row, model), axis=1)
    finally:
        model.close()

    out = pd.concat([df.reset_index(drop=True), fit_res.reset_index(drop=True)], axis=1)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


if __name__ == "__main__":
    # Build cache only. Fitting is usually easier from a notebook.
    build_preconvolved_cache()
