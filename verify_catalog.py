"""Verify the released catalogue against the parent selection and fiducial fit."""
import argparse
import csv
from pathlib import Path

from astropy.table import Table
import numpy as np


def records(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(data):
    stem = data / "catalog" / "cepheids_2410_sed_coordinates_distances"
    catalogue = records(stem.with_suffix(".csv"))
    fits = Table.read(stem.with_suffix(".fits"))
    ids = [row["GaiaDR3"] for row in catalogue]
    flags = records(data / "selection.csv")
    selected = [row["GaiaDR3"] for row in flags if row["in_final_catalog"] == "1"]
    original = records(data / "results/fiducial/final_summary_converged.csv")
    by_id = {row["GaiaDR3"]: row for row in original}
    if len(flags) != 3640 or len(selected) != 2410 or len(ids) != len(set(ids)) or len(ids) != 2410:
        raise ValueError("Unexpected catalogue population")
    if set(ids) != set(selected) or set(ids) != set(by_id) or len(original) != len(ids):
        raise ValueError("Membership differs between the selection, fiducial fit and catalogue")
    if list(catalogue[0]) != fits.colnames:
        raise ValueError("CSV/FITS column names differ")
    for key in fits.colnames:
        values = [row[key] for row in catalogue]
        if fits[key].dtype.kind in "fiu":
            np.testing.assert_allclose(np.asarray(values, dtype=float), fits[key], rtol=1e-13, atol=0)
        else:
            np.testing.assert_array_equal(values, fits[key].astype(str))
    for param in ["Teff", "logg", "MH", "A550", "R550"]:
        for q in ["p16", "p50", "p84"]:
            key = f"{param}_{q}"
            np.testing.assert_allclose([float(row[key]) for row in catalogue],
                                       [float(by_id[i][key]) for i in ids], rtol=1e-12)
    print(f"Verified {len(ids)} unique Cepheids and {len(fits.colnames)} CSV/FITS columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Extracted tables-and-inputs directory")
    verify(parser.parse_args().data)
