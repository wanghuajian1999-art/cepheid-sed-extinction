#!/usr/bin/env python3
"""Direct-emcee MW Cepheid SED fit with X_RP PC prior and WC2019 extinction.

Jupyter usage
-------------

    %run -i "jupyter_mw_sed_pcprior_WC2019_method2_distanceprior_emcee_direct.py"

    selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_WC2019_method2_distanceprior_emcee_direct(
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

This wrapper keeps the fitting method from
``jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct.py`` unchanged,
but points the pre-convolved BOSZ cache to a separate WC2019 cache. The
original G23 cache and output directory are not overwritten.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


BASE = Path(__file__).resolve().parent
ASSET_BASE_CANDIDATES = [
    BASE,

]

G23_CACHE_FILE = BASE / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_preconvolved_ND.h5"
WC2019_CACHE_FILE = BASE / "bosz_r500_msmp_a0_c0_v4_Rv2p3_5p6_WC2019_preconvolved_ND.h5"
WC2019_OUT_DIR = BASE / "sed_pcprior_method2_distanceprior_emcee_direct_WC2019"


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


def Rvcodeg(Rv, num=1000, case1=1):
    """WC2019-style extinction curve helper from the user's notebook.

    Returns
    -------
    wave : ndarray
        Wavelength in micron.
    aw_av : ndarray
        A(lambda) / A(V) evaluated on ``wave``.
    """
    step = (np.log10(30) - np.log10(0.1)) / (num - 1)
    ccc = np.log10(0.1) + np.arange(num) * step
    wave = 10**ccc

    X = np.arange(1.1, 3.3, 0.0001)
    X1 = np.arange(0.33, 1.00, 0.0001)
    X2 = np.arange(3.3, 8.0, 0.0001)
    X3 = np.arange(0.02, 0.3, 0.0001)
    X4 = np.arange(8.0, 10.0, 0.0001)
    X5 = np.arange(10.0, 15.0, 0.0001)

    cX = np.arange(0.63, 2.2, 0.0001)
    cX1 = np.arange(0.12, 0.63, 0.0001)

    Y = X - 1.82

    if case1 == 2:
        A = (
            1
            + 0.17699 * Y
            - 0.50447 * Y**2
            - 0.02427 * Y**3
            + 0.72085 * Y**4
            + 0.01979 * Y**5
            - 0.77530 * Y**6
            + 0.32999 * Y**7
        )
        B = (
            1.41338 * Y
            + 2.28305 * Y**2
            + 1.07233 * Y**3
            - 5.38434 * Y**4
            - 0.62251 * Y**5
            + 5.30260 * Y**6
            - 2.09002 * Y**7
        )
    elif case1 == 3:
        A = 1
        B = 2.659 * (-1.857 + 1.040 / cX)
    else:
        A = (
            1
            + 0.7499 * Y
            - 0.1086 * Y**2
            - 0.08909 * Y**3
            + 0.02905 * Y**4
            + 0.01069 * Y**5
            + 0.001707 * Y**6
            - 0.001002 * Y**7
        )
        B = (
            1.41338 * Y
            + 2.28305 * Y**2
            + 1.07233 * Y**3
            - 5.38434 * Y**4
            - 0.62251 * Y**5
            + 5.30260 * Y**6
            - 2.09002 * Y**7
        ) * (1 - Rv / 3.1)

    result = A + B / Rv

    if case1 == 2:
        Y1 = X1**1.61
        A1 = 0.574 * Y1
        B1 = -0.527 * Y1
    elif case1 == 3:
        A1 = 1
        B1 = 2.659 * (-2.156 + 1.509 / cX1 - 0.198 / cX1**2 + 0.011 / cX1**3)
    else:
        Y1 = X1**2.07
        A1 = 0.3722 * Y1
        B1 = -0.5182 * Y1 * (1 - Rv / 3.1)

    result1 = A1 + B1 / Rv

    A2 = 1.752 - 0.316 * X2 - 0.104 / ((X2 - 4.67) ** 2 + 0.341)
    B2 = -3.090 + 1.825 * X2 + 1.206 / ((X2 - 4.62) ** 2 + 0.263)
    sel2 = X2 > 5.9
    A2[sel2] -= 0.04473 * (X2[sel2] - 5.9) ** 2 - 0.009779 * (X2[sel2] - 5.9) ** 3
    B2[sel2] += 0.2130 * (X2[sel2] - 5.9) ** 2 + 0.1207 * (X2[sel2] - 5.9) ** 3
    result2 = A2 + B2 / Rv

    if case1 == 2:
        Y3 = (X3 / 0.3) ** 2 * (0.3**1.61)
        A3 = 0.574 * Y3
        B3 = -0.527 * Y3
    elif case1 == 3:
        A3 = 0
        B3 = 0
    else:
        Y3 = (X3 / 0.3) ** 2 * (0.3**2.07)
        A3 = 0.3722 * Y3
        B3 = -0.5182 * Y3 * (1 - Rv / 3.1)

    result3i = A3 + B3 / Rv

    DIB = np.array([9.7e4, 2.5e4, 6.9e3])
    DIB1 = np.array([18.0e4, 5.0e4, 6.0e3])
    WAV = 1.0e4 / X3
    FAC = (0.574 - 0.527 / Rv) / (0.574 - 0.527 / 3.1)

    result3 = result3i + 0.63622 * FAC * (DIB[2] / Rv) * DIB[1] / (
        DIB[1] ** 2 + (WAV - DIB[0] ** 2 / WAV) ** 2
    ) + 0.63622 * FAC * (DIB1[2] / Rv) * DIB1[1] / (
        DIB1[1] ** 2 + (WAV - DIB1[0] ** 2 / WAV) ** 2
    )

    Y4 = X4 - 8
    A4 = -1.073 - 0.628 * Y4 + 0.137 * Y4**2 - 0.070 * Y4**3
    B4 = 13.670 + 4.257 * Y4 - 0.420 * Y4**2 + 0.374 * Y4**3
    result4 = A4 + B4 / Rv

    Y5 = X5 - 10
    A5 = -2.341 - 0.92 * Y5
    B5 = 23.496 + 7.065 * Y5
    result5 = A5 + B5 / Rv

    if case1 == 3:
        zongx = np.concatenate([cX1, cX])
        zongy = np.concatenate([result1, result])
        sort_idx = np.argsort(zongx)
        zongx1 = zongx[sort_idx]
        zongy1 = zongy[sort_idx]
        positive = zongy1 > 0
        negative = zongy1 < 0
        if np.any(negative):
            cutoff = np.min(zongx1[negative])
            mask = (ccc > np.log10(0.12)) & (ccc < np.log10(2.2)) & (ccc < np.log10(cutoff))
            wave = wave[mask]
            ccc = ccc[mask]
        log_awav = interp1d(
            np.log10(zongx1[positive]),
            np.log10(zongy1[positive]),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )(ccc)
    else:
        zongx = np.concatenate([1 / X, 1 / X1, 1 / X2, 1 / X3, 1 / X4, 1 / X5])
        zongy = np.concatenate([result, result1, result2, result3, result4, result5])
        sort_idx = np.argsort(zongx)
        zongx1 = zongx[sort_idx]
        zongy1 = np.clip(zongy[sort_idx], 1e-12, None)
        log_awav = interp1d(
            np.log10(zongx1),
            np.log10(zongy1),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )(ccc)

    return wave, 10**log_awav


class WC2019:
    """Callable extinction law object compatible with dust_extinction usage."""

    def __init__(self, Rv=3.1, num=1000, case1=1):
        self.Rv = float(Rv)
        wave, aw_av = Rvcodeg(self.Rv, num=num, case1=case1)
        x = 1.0 / np.asarray(wave, dtype=float)
        y = np.asarray(aw_av, dtype=float)
        order = np.argsort(x)
        self._interp = interp1d(
            x[order],
            y[order],
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

    def __call__(self, inv_wave):
        if hasattr(inv_wave, "to_value"):
            import astropy.units as u

            x = inv_wave.to_value(1 / u.micron)
        else:
            x = np.asarray(inv_wave, dtype=float)
        return np.clip(self._interp(x), 0.0, None)


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


def build_wc2019_preconvolved_cache(overwrite_cache=False):
    """Build or reuse the independent WC2019 pre-convolved BOSZ cache."""
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

    if WC2019_CACHE_FILE.exists() and not overwrite_cache:
        print(f"WC2019 cache already exists: {WC2019_CACHE_FILE}")
    else:
        bosz.build_preconvolved_cache(
            cache_file=WC2019_CACHE_FILE,
            spec_base_dir=asset_base / "r500",
            wave_file=asset_base / "r500/m+0.00/bosz2024_wave_r500.txt",
            rv_grid=rv_grid,
            av_grid=av_grid,
            ExtLaw=WC2019,
            overwrite=overwrite_cache,
        )

    with h5py.File(WC2019_CACHE_FILE, "r") as h5:
        print("cache:", WC2019_CACHE_FILE)
        print("ext_law:", h5.attrs.get("ext_law"))
        print("N shape:", h5["N"].shape)
        print("Rv:", h5["rv_grid"][:].min(), h5["rv_grid"][:].max())
        print("Av:", h5["av_grid"][:].min(), h5["av_grid"][:].max())
        print("asset_base:", asset_base)

    return WC2019_CACHE_FILE


def _configure_pcprior_runner(overwrite_cache=False):
    cache_file = build_wc2019_preconvolved_cache(overwrite_cache=overwrite_cache)

    sys.path.insert(0, str(BASE))
    pc = _import_or_reload("jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct")
    WC2019_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pc.mapbase.sedbase.CACHE_FILE = cache_file
    pc.OUT_DIR = WC2019_OUT_DIR
    pc.SELECTED_OUT = WC2019_OUT_DIR / "MW_SED_PCprior_WC2019_method2_distanceprior_emcee_direct_selected.csv"
    pc.SUMMARY_OUT = WC2019_OUT_DIR / "MW_SED_PCprior_WC2019_method2_distanceprior_emcee_direct_summary.csv"
    pc.FAIL_OUT = WC2019_OUT_DIR / "MW_SED_PCprior_WC2019_method2_distanceprior_emcee_direct_failures.csv"
    pc.RUNINFO_OUT = WC2019_OUT_DIR / "MW_SED_PCprior_WC2019_method2_distanceprior_emcee_direct_runinfo.json"
    return pc, cache_file


def run_mw_sed_pcprior_WC2019_method2_distanceprior_emcee_direct(
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
    """Run the original PC-prior direct-emcee fitter with a WC2019 cache."""
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
                "extinction_law": "WC2019",
                "cache_file": str(cache_file),
                "reference_cache_file": str(G23_CACHE_FILE),
                "output_dir": str(WC2019_OUT_DIR),
                "pc_prior": "X_RP default from jupyter_mw_sed_pcprior_method2_distanceprior_emcee_direct.py",
            }
        )
        with open(pc.RUNINFO_OUT, "w", encoding="utf-8") as f:
            json.dump(runinfo, f, ensure_ascii=False, indent=2)

    return selected, emcee_summary, emcee_samples


print("Loaded WC2019 PC-prior direct-emcee wrapper.")
print("Call: selected, emcee_summary, emcee_samples = run_mw_sed_pcprior_WC2019_method2_distanceprior_emcee_direct(...)")
