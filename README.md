# Mapping the Milky Way extinction curve with classical Cepheids

Broadband SED inference for the Galactic Cepheid sample of Huajian Wang, Xiaodian Chen, Shu Wang and Richard I. Anderson. The fiducial model uses BOSZ spectra, the G23 extinction family and a multivariate period–colour (PC) prior. It samples effective temperature, surface gravity, metallicity, monochromatic extinction at 550 nm, the extinction-curve parameter R550, and the logarithm of the radius-to-distance ratio. The catalogue reports uncertainties through posterior quantiles.

## Reproduce one star

Use Linux with Python 3.12. The complete tested package set is recorded in `environment-linux.txt`. From the top level of this software distribution:

1. Download the companion dataset files `tables-and-inputs.zip` and `bosz_r500_G23_A550_R550_preconvolved_ND.h5`. Verify their SHA-256 values against the archive checksum manifest.
2. Unzip the tables archive in a separate directory. Copy the **contents** of its `inputs/` directory into the software directory, preserving subdirectories. Put the HDF5 file in the same directory as `fit_example.py`.
3. Run the following commands. The CALSPEC downloader checks the pinned Vega spectrum before saving it.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment-linux.txt
python fetch_calibration.py
OPENBLAS_NUM_THREADS=1 python fit_example.py --output aw-per
```

The example selects AW Per by Gaia DR3 source ID 174489098011145216. It compares a saved draw's log posterior and the model magnitudes at the archived marginal medians, then fits the star using 32 walkers and 10,000 steps. It checks each new marginal median against the original central 68% interval. The random chains can differ between runs. Use a new output directory for each run. Add `--short` for a 600-step functionality check.

## Fit the sample

```sh
python run_paper_fit_suite.py --mode full --branch fiducial_G23_pc1sigma --workers 4
python complete_fiducial_convergence.py --mode full --workers 4
```

The first command applies the production selection and runs the base and recovery chains. The second continues chains that need longer sampling. The parent photometry table contains 3,640 rows. Use `selection.csv` in the companion dataset to trace the 2,591 selected stars and the 2,410 entries in the final catalogue. Read Gaia DR3 identifiers as strings or 64-bit integers. The catalogue's spatial distances and the distance inputs used to construct a metallicity prior have distinct meanings; see `catalog/columns.csv`.

The suite also implements `G23_pc2sigma`, `G23_noPC`, `G23_fixedLogg2_MH0`, `WC2019_pc1sigma`, and `F19_pc1sigma`. Supply each requested extinction-law cache in the software directory. For WC2019 or F19, download the BOSZ library subset below and build that cache first.

```sh
python fetch_bosz.py
python bosz_a550_r550_preconvolved.py --law F19 --asset-base . --output-dir .
python bosz_a550_r550_preconvolved.py --law WC2019 --asset-base . --output-dir .
```

The native G23 grid can also be generated with `--law G23`. The companion dataset provides its computed version. A cache rebuild uses the instrument responses in `inputs/filters` and the 824 selected BOSZ model spectra in `bosz_spectra.txt`. The F19 table belongs to the F19 model; retain it beside the Python files when building that law. The code uses R550 defined by monochromatic extinction at 440 and 550 nm; the native G23 curve parameter is used internally during convolution.

## Provenance and licensing

The companion dataset includes input provenance, a column dictionary, branch summaries and machine-readable posterior associations. The MIT license applies to the authors' software. Cite the paper and the original data and model providers in analyses that use those products.

To verify the two catalogue formats, source IDs and fiducial posterior quantiles, run `python verify_catalog.py path/to/extracted/tables`. For upstream catalogue credits and the distinction between input and mapped distances, see the data README and column dictionary.
