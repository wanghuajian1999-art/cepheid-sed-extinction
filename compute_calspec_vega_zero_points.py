#!/usr/bin/env python3
"""Compute Vega-system zero points from a CALSPEC alpha Lyrae spectrum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


FILTERS = {
    "BP": ("GAIA_GAIA3.Gbp.dat", "photon"),
    "G": ("GAIA_GAIA3.G.dat", "photon"),
    "RP": ("GAIA_GAIA3.Grp.dat", "photon"),
    "J": ("2MASS_2MASS.J.dat", "photon"),
    "H": ("2MASS_2MASS.H.dat", "photon"),
    "K": ("2MASS_2MASS.Ks.dat", "photon"),
    "W1": ("WISE_WISE.W1.dat", "energy"),
    "V": ("LCO_OGLE-IV.V.dat", "energy"),
    "I": ("LCO_OGLE-IV.I.dat", "photon"),
}

if hasattr(np, "trapezoid"):
    _TRAPEZOID = np.trapezoid
else:
    _TRAPEZOID = np.trapz


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_calspec(path: Path):
    with fits.open(path, memmap=True) as hdul:
        table_hdu = next(
            hdu for hdu in hdul if getattr(getattr(hdu, "data", None), "names", None)
        )
        names = {name.upper(): name for name in table_hdu.data.names}
        wave = np.asarray(table_hdu.data[names["WAVELENGTH"]], dtype=float)
        flux = np.asarray(table_hdu.data[names["FLUX"]], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0) & (flux >= 0)
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    return wave[order], flux[order]


def _read_filter(path: Path):
    data = np.loadtxt(path, comments="#", usecols=(0, 1), dtype=float)
    wave = np.asarray(data[:, 0], dtype=float)
    response = np.asarray(data[:, 1], dtype=float)
    good = np.isfinite(wave) & np.isfinite(response) & (wave > 0) & (response >= 0)
    wave = wave[good]
    response = response[good]
    order = np.argsort(wave)
    return wave[order], response[order]


def _atomic_csv(frame: pd.DataFrame, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)


def compute_zero_points(base_dir, vega_file, output, old_zp_file=None):
    base_dir = Path(base_dir).expanduser().resolve()
    vega_file = Path(vega_file).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    filter_dir = base_dir / "filters"
    wave_vega, flux_vega = _read_calspec(vega_file)
    rows = []

    for band, (filename, response_type) in FILTERS.items():
        filter_path = filter_dir / filename
        wave_filter, response = _read_filter(filter_path)
        if wave_filter[0] < wave_vega[0] or wave_filter[-1] > wave_vega[-1]:
            raise ValueError(
                f"CALSPEC wavelength coverage does not contain {band}: "
                f"filter={wave_filter[0]:.1f}-{wave_filter[-1]:.1f} Angstrom, "
                f"Vega={wave_vega[0]:.1f}-{wave_vega[-1]:.1f} Angstrom"
            )
        flux = np.interp(wave_filter, wave_vega, flux_vega)
        kernel = flux * response
        if response_type == "photon":
            kernel = kernel * wave_filter
        raw_integral = float(_TRAPEZOID(kernel, wave_filter))
        if not np.isfinite(raw_integral) or raw_integral <= 0:
            raise ValueError(f"non-positive CALSPEC Vega integral for {band}")
        rows.append(
            {
                "band": band,
                "filter_file": str(filter_path),
                "response_type": response_type,
                "vega_file": str(vega_file),
                "vega_sha256": _sha256(vega_file),
                "filter_wave_min_A": float(wave_filter[0]),
                "filter_wave_max_A": float(wave_filter[-1]),
                "vega_raw_integral": raw_integral,
                "ZP_X_cache_raw_integral": float(2.5 * np.log10(raw_integral)),
            }
        )

    result = pd.DataFrame(rows)
    if old_zp_file is not None:
        old_path = Path(old_zp_file).expanduser().resolve()
        old = pd.read_csv(old_path)[["band", "ZP_X_cache_raw_integral"]].rename(
            columns={"ZP_X_cache_raw_integral": "ZP_X_old"}
        )
        result = result.merge(old, on="band", how="left", validate="one_to_one")
        result["delta_ZP_new_minus_old"] = (
            result["ZP_X_cache_raw_integral"] - result["ZP_X_old"]
        )
    _atomic_csv(result, output)

    runinfo = {
        "output": str(output),
        "base_dir": str(base_dir),
        "vega_file": str(vega_file),
        "vega_sha256": _sha256(vega_file),
        "bands": list(FILTERS),
        "definition": "ZP_X_cache_raw_integral = 2.5 log10(integrated CALSPEC Vega kernel)",
        "photon_kernel": "F_lambda S(lambda) lambda",
        "energy_kernel": "F_lambda S(lambda)",
        "old_zp_file": None if old_zp_file is None else str(Path(old_zp_file).resolve()),
    }
    output.with_suffix(".runinfo.json").write_text(
        json.dumps(runinfo, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--vega-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-zp-file")
    args = parser.parse_args()
    result = compute_zero_points(
        args.base_dir, args.vega_file, args.output, args.old_zp_file
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
