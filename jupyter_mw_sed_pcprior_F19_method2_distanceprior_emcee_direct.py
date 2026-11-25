#!/usr/bin/env python3
"""Direct-emcee MW Cepheid SED fit with X_RP PC prior and F19 extinction.

Jupyter usage
-------------

    %run -i "jupyter_mw_sed_pcprior_F19_method2_distanceprior_emcee_direct.py"

    selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_F19_method2_distanceprior_emcee_direct(
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
only swaps the pre-convolved BOSZ cache to one built with Fitzpatrick et al.
2019 (F19), and writes outputs to an independent directory.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import splrep, splev

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


BASE = Path(__file__).resolve().parent
ASSET_BASE_CANDIDATES = [
    BASE,

]

G23_CACHE_FILE = BASE / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_preconvolved_ND.h5"
F19_CACHE_FILE = BASE / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_F19_preconvolved_ND.h5"
F19_OUT_DIR = BASE / "sed_pcprior_method2_distanceprior_emcee_direct_F19"
F19_TABLE_FILE = BASE / "F19_tabulated.dat"


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


class F19:
    """Fitzpatrick et al. 2019 R(V)-dependent extinction law.

    This is a compact implementation of the dust_extinction F19 model using
    the bundled F19_tabulated.dat file:

        k_Rv(x) = k_3.02(x) + deltak(x) * (Rv - 3.10) * 0.990
        A(x) / A(V) = k_Rv(x) / Rv + 1

    The F19 table includes an x=0 anchor, which lets the spline evaluate W1's
    long-wavelength tail while keeping the same 7-band SED setup.
    """

    Rv_range = (2.0, 6.0)
    x_range = (0.0, 8.7)

    def __init__(self, Rv=3.1):
        self.Rv = float(Rv)
        if not (self.Rv_range[0] <= self.Rv <= self.Rv_range[1]):
            raise ValueError(f"F19 Rv={self.Rv} outside [{self.Rv_range[0]}, {self.Rv_range[1]}]")
        if not F19_TABLE_FILE.exists():
            raise FileNotFoundError(f"Missing F19 table: {F19_TABLE_FILE}")

        dat = np.genfromtxt(F19_TABLE_FILE, names=True)
        x = np.asarray(dat["x"], dtype=float)
        k_302 = np.asarray(dat["k_302"], dtype=float)
        deltak = np.asarray(dat["deltak"], dtype=float)
        k_rv = k_302 + deltak * (self.Rv - 3.10) * 0.990
        self._spline = splrep(x, k_rv)

    def __call__(self, inv_wave):
        if hasattr(inv_wave, "to_value"):
            import astropy.units as u

            x = inv_wave.to_value(1 / u.micron)
        else:
            x = np.asarray(inv_wave, dtype=float)

        x = np.asarray(x, dtype=float)
        x_eval = np.clip(x, self.x_range[0], self.x_range[1])
        k_rv = splev(x_eval, self._spline, der=0)
        return np.asarray(k_rv, dtype=float) / self.Rv + 1.0


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


def build_f19_preconvolved_cache(overwrite_cache=False):
    """Build or reuse the independent F19 pre-convolved BOSZ cache."""
    sys.path.insert(0, str(BASE))
    bosz = _import_or_reload("bosz_preconvolved_interpolated_fit")
    _patch_filter_paths(bosz)
    asset_base = _asset_base()

    g23_cache = Path(getattr(_import_or_reload("jupyter_10star_sed_w1plz_emcee_fit"), "CACHE_FILE", G23_CACHE_FILE))
    if not g23_cache.exists():
        g23_cache = G23_CACHE_FILE
    if not g23_cache.exists():
        raise FileNotFoundError(f"Cannot find reference G23 cache for grids: {g23_cache}")

    with h5py.File(g23_cache, "r") as h5:
        rv_grid = h5["rv_grid"][:]
        av_grid = h5["av_grid"][:]

    if F19_CACHE_FILE.exists() and not overwrite_cache:
        print(f"F19 cache already exists: {F19_CACHE_FILE}")
    else:
        bosz.build_preconvolved_cache(
            cache_file=F19_CACHE_FILE,
            spec_base_dir=asset_base / "r500",
            wave_file=asset_base / "r500/m+0.00/bosz2024_wave_r500.txt",
            rv_grid=rv_grid,
            av_grid=av_grid,
            ExtLaw=F19,
            overwrite=overwrite_cache,
        )

    with h5py.File(F19_CACHE_FILE, "r") as h5:
        print("cache:", F19_CACHE_FILE)
        print("ext_law:", h5.attrs.get("ext_law"))
        print("N shape:", h5["N"].shape)
        print("Rv:", h5["rv_grid"][:].min(), h5["rv_grid"][:].max())
        print("Av:", h5["av_grid"][:].min(), h5["av_grid"][:].max())
        print("asset_base:", asset_base)

    return F19_CACHE_FILE


def _configure_pcprior_runner(overwrite_cache=False):
    cache_file = build_f19_preconvolved_cache(overwrite_cache=overwrite_cache)

    sys.path.insert(0, str(BASE))
    pc = _import_or_reload("jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct")
    F19_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pc.mapbase.sedbase.CACHE_FILE = cache_file
    pc.OUT_DIR = F19_OUT_DIR
    pc.SELECTED_OUT = F19_OUT_DIR / "MW_SED_PCprior_F19_method2_distanceprior_emcee_direct_selected.csv"
    pc.SUMMARY_OUT = F19_OUT_DIR / "MW_SED_PCprior_F19_method2_distanceprior_emcee_direct_summary.csv"
    pc.FAIL_OUT = F19_OUT_DIR / "MW_SED_PCprior_F19_method2_distanceprior_emcee_direct_failures.csv"
    pc.RUNINFO_OUT = F19_OUT_DIR / "MW_SED_PCprior_F19_method2_distanceprior_emcee_direct_runinfo.json"
    return pc, cache_file


def run_mw_sed_pcprior_F19_method2_distanceprior_emcee_direct(
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
    """Run the original PC-prior direct-emcee fitter with an F19 cache."""
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
                "extinction_law": "F19",
                "cache_file": str(cache_file),
                "reference_cache_file": str(G23_CACHE_FILE),
                "output_dir": str(F19_OUT_DIR),
                "pc_prior": "X_RP default from jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct.py",
                "note": "F19 tabulated spline; W1 tail is evaluated with the bundled F19 table down to x=0.",
            }
        )
        with open(pc.RUNINFO_OUT, "w", encoding="utf-8") as f:
            json.dump(runinfo, f, ensure_ascii=False, indent=2)

    return selected, emcee_summary, emcee_samples


print("Loaded F19 PC-prior direct-emcee wrapper.")
print("Call: selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_F19_method2_distanceprior_emcee_direct(...)")
