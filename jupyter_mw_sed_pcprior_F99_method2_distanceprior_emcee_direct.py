#!/usr/bin/env python3
"""Direct-emcee MW Cepheid SED fit with X_RP PC prior and F99 extinction.

Jupyter usage
-------------

    %run -i "jupyter_mw_sed_pcprior_F99_method2_distanceprior_emcee_direct.py"

    selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_F99_method2_distanceprior_emcee_direct(
        n_stars=None,
        quality_mode="primary",
        nwalkers=32,
        nsteps=3000,
        burnin=500,
        thin=10,
        random_seed=42,
        save_samples="diagnostic",
        make_corner=False,
        progress=False,
        n_workers=48,
    )

The original PC-prior direct-emcee fitter is reused unchanged. This wrapper
only swaps the pre-convolved BOSZ cache to one built with Fitzpatrick (1999)
and writes outputs to an independent directory.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import interpolate

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


BASE = Path(__file__).resolve().parent
ASSET_BASE_CANDIDATES = [
    BASE,

]

G23_CACHE_FILE = BASE / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_preconvolved_ND.h5"
F99_CACHE_FILE = BASE / "bosz_r500_msmp_a0_c0_v4_Rv2p0_6p0_F99_preconvolved_ND.h5"
F99_OUT_DIR = BASE / "sed_pcprior_method2_distanceprior_emcee_direct_F99"


def _asset_base() -> Path:
    for candidate in ASSET_BASE_CANDIDATES:
        if (
            (candidate / "filters/GAIA_GAIA3.Gbp.dat").exists()
            and (candidate / "filters/WISE_WISE.W1.dat").exists()
            and (candidate / "r500/m+0.00/bosz2024_wave_r500.txt").exists()
        ):
            return candidate
    checked = ", ".join(str(p) for p in ASSET_BASE_CANDIDATES)
    raise FileNotFoundError(f"Cannot find filters and r500 wave file. Checked: {checked}")


def _fm90_exvebv(x: np.ndarray, C1: float, C2: float, C3: float, C4: float, xo: float, gamma: float) -> np.ndarray:
    """FM90 UV curve: E(lambda-V)/E(B-V)."""
    x = np.asarray(x, dtype=float)
    exvebv = C1 + C2 * x
    x2 = x * x
    exvebv += C3 * (x2 / ((x2 - xo**2) ** 2 + x2 * gamma**2))
    fuv = x >= 5.9
    if np.any(fuv):
        y = x[fuv] - 5.9
        exvebv[fuv] += C4 * (0.5392 * y**2 + 0.05644 * y**3)
    return exvebv


def _curve_f99_method(
    x: np.ndarray,
    Rv: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    xo: float,
    gamma: float,
    optnir_axav_x: np.ndarray,
    optnir_axav_y: np.ndarray,
) -> np.ndarray:
    """dust_extinction.shapes._curve_F99_method, compact local equivalent."""
    x = np.asarray(x, dtype=float)
    axav = np.zeros(x.shape, dtype=float)

    x_cutval_uv = 10000.0 / 2700.0
    x_splineval_uv = 10000.0 / np.array([2700.0, 2600.0], dtype=float)

    ind_uv = np.where(x >= x_cutval_uv)[0]
    if len(ind_uv) > 0:
        xuv = np.concatenate([x_splineval_uv, x[ind_uv]])
    else:
        xuv = x_splineval_uv

    axav_fm90 = _fm90_exvebv(xuv, C1, C2, C3, C4, xo, gamma) / Rv + 1.0
    y_splineval_uv = axav_fm90[:2]
    if len(ind_uv) > 0:
        axav[ind_uv] = axav_fm90[2:]

    ind_opir = np.where(x < x_cutval_uv)[0]
    if len(ind_opir) > 0:
        spline_x = np.concatenate([[0.0], optnir_axav_x, x_splineval_uv])
        spline_y = np.concatenate([[0.0], optnir_axav_y, y_splineval_uv])
        axav[ind_opir] = interpolate.CubicSpline(spline_x, spline_y, bc_type="natural")(x[ind_opir])

    return np.clip(axav, 0.0, None)


class F99:
    """Fitzpatrick (1999) Milky Way R(V)-dependent extinction law.

    This is a compact implementation matching dust_extinction.parameter_averages.F99.
    The official F99 x_range is [0.3, 10.0] 1/micron.  For W1, whose filter tail
    can go slightly below x=0.3, we allow the same F99 optical/NIR spline to
    evaluate down to x=0 using its built-in zero-extinction anchor.
    """

    Rv_range = (2.0, 6.0)
    x_range = (0.0, 10.0)

    def __init__(self, Rv=3.1):
        self.Rv = float(Rv)
        if not (self.Rv_range[0] <= self.Rv <= self.Rv_range[1]):
            raise ValueError(f"F99 Rv={self.Rv} outside [{self.Rv_range[0]}, {self.Rv_range[1]}]")

        self.C3 = 3.23
        self.C4 = 0.41
        self.xo = 4.596
        self.gamma = 0.99
        self.C2 = -0.824 + 4.717 / self.Rv
        self.C1 = 2.030 - 3.007 * self.C2

        optnir_x = 10000.0 / np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0], dtype=float)
        opt_y = np.array(
            [
                -0.426 + 1.0044 * self.Rv,
                -0.050 + 1.0016 * self.Rv,
                0.701 + 1.0016 * self.Rv,
                1.208 + 1.0032 * self.Rv - 0.00033 * self.Rv**2,
            ],
            dtype=float,
        )
        nir_y = np.array([0.265, 0.829], dtype=float) * self.Rv / 3.1

        self.optnir_axav_x = optnir_x
        self.optnir_axav_y = np.concatenate([nir_y, opt_y]) / self.Rv

    def __call__(self, inv_wave):
        if hasattr(inv_wave, "to_value"):
            import astropy.units as u

            x = inv_wave.to_value(1 / u.micron)
        else:
            x = np.asarray(inv_wave, dtype=float)

        x = np.clip(np.asarray(x, dtype=float), self.x_range[0], self.x_range[1])
        return _curve_f99_method(
            x,
            self.Rv,
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.xo,
            self.gamma,
            self.optnir_axav_x,
            self.optnir_axav_y,
        )


def _import_or_reload(module_name):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _patch_filter_paths(bosz):
    asset_base = _asset_base()
    bosz.FILTER_PATHS = {
        "BP": (asset_base / "filters/GAIA_GAIA3.Gbp.dat", True),
        "G": (asset_base / "filters/GAIA_GAIA3.G.dat", True),
        "RP": (asset_base / "filters/GAIA_GAIA3.Grp.dat", True),
        "J": (asset_base / "filters/2MASS_2MASS.J.dat", True),
        "H": (asset_base / "filters/2MASS_2MASS.H.dat", True),
        "K": (asset_base / "filters/2MASS_2MASS.Ks.dat", True),
        "W1": (asset_base / "filters/WISE_WISE.W1.dat", False),
    }


def build_f99_preconvolved_cache(overwrite_cache=False):
    """Build or reuse the independent F99 pre-convolved BOSZ cache."""
    sys.path.insert(0, str(BASE))
    bosz = _import_or_reload("bosz_preconvolved_interpolated_fit")
    _patch_filter_paths(bosz)
    asset_base = _asset_base()

    g23_cache = Path(getattr(_import_or_reload("jupyter_10star_sed_w1plz_emcee_fit"), "CACHE_FILE", G23_CACHE_FILE))
    if not g23_cache.exists():
        g23_cache = G23_CACHE_FILE
    if not g23_cache.exists():
        raise FileNotFoundError(f"Cannot find reference G23 cache for Av grid: {g23_cache}")

    with h5py.File(g23_cache, "r") as h5:
        av_grid = h5["av_grid"][:]

    rv_grid = np.arange(2.0, 6.0 + 1.0e-8, 0.02, dtype=np.float32)

    if F99_CACHE_FILE.exists() and not overwrite_cache:
        print(f"F99 cache already exists: {F99_CACHE_FILE}")
    else:
        bosz.build_preconvolved_cache(
            cache_file=F99_CACHE_FILE,
            spec_base_dir=asset_base / "r500",
            wave_file=asset_base / "r500/m+0.00/bosz2024_wave_r500.txt",
            rv_grid=rv_grid,
            av_grid=av_grid,
            ExtLaw=F99,
            overwrite=overwrite_cache,
        )

    with h5py.File(F99_CACHE_FILE, "r") as h5:
        print("cache:", F99_CACHE_FILE)
        print("ext_law:", h5.attrs.get("ext_law"))
        print("N shape:", h5["N"].shape)
        print("Rv:", h5["rv_grid"][:].min(), h5["rv_grid"][:].max())
        print("Av:", h5["av_grid"][:].min(), h5["av_grid"][:].max())
        print("asset_base:", asset_base)

    return F99_CACHE_FILE


def _configure_pcprior_runner(overwrite_cache=False):
    cache_file = build_f99_preconvolved_cache(overwrite_cache=overwrite_cache)

    sys.path.insert(0, str(BASE))
    pc = _import_or_reload("jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct")
    F99_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pc.mapbase.sedbase.CACHE_FILE = cache_file
    pc.OUT_DIR = F99_OUT_DIR
    pc.SELECTED_OUT = F99_OUT_DIR / "MW_SED_PCprior_F99_method2_distanceprior_emcee_direct_selected.csv"
    pc.SUMMARY_OUT = F99_OUT_DIR / "MW_SED_PCprior_F99_method2_distanceprior_emcee_direct_summary.csv"
    pc.FAIL_OUT = F99_OUT_DIR / "MW_SED_PCprior_F99_method2_distanceprior_emcee_direct_failures.csv"
    pc.RUNINFO_OUT = F99_OUT_DIR / "MW_SED_PCprior_F99_method2_distanceprior_emcee_direct_runinfo.json"
    return pc, cache_file


def run_mw_sed_pcprior_F99_method2_distanceprior_emcee_direct(
    n_stars=None,
    quality_mode="primary",
    nwalkers=32,
    nsteps=3000,
    burnin=500,
    thin=10,
    random_seed=42,
    save_samples="diagnostic",
    make_corner=False,
    progress=False,
    n_workers=48,
    overwrite_cache=False,
):
    """Run the original PC-prior direct-emcee fitter with an F99 cache."""
    pc, cache_file = _configure_pcprior_runner(overwrite_cache=overwrite_cache)
    selected, emcee_summary, emcee_samples = pc.run_mw_sed_pcprior_method2_distanceprior_emcee_direct(
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

    if pc.RUNINFO_OUT.exists():
        with open(pc.RUNINFO_OUT, "r", encoding="utf-8") as f:
            runinfo = json.load(f)
        runinfo.update(
            {
                "extinction_law": "F99",
                "cache_file": str(cache_file),
                "reference_cache_file": str(G23_CACHE_FILE),
                "output_dir": str(F99_OUT_DIR),
                "rv_grid_min": 2.0,
                "rv_grid_max": 6.0,
                "rv_grid_step": 0.02,
                "pc_prior": "X_RP default from jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct.py",
                "note": "F99 official range is x=[0.3,10] 1/micron; W1 is evaluated by the F99 optical/NIR spline with its zero-extinction anchor.",
            }
        )
        with open(pc.RUNINFO_OUT, "w", encoding="utf-8") as f:
            json.dump(runinfo, f, ensure_ascii=False, indent=2)

    return selected, emcee_summary, emcee_samples


print("Loaded F99 PC-prior direct-emcee wrapper.")
print("Call: selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_F99_method2_distanceprior_emcee_direct(...)")
