"""Fit AW Per and verify the forward model against a stored posterior draw."""
import argparse
import json
from pathlib import Path

import emcee
import numpy as np
import run_paper_fit_suite as suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--short", action="store_true", help="Run the 600-step smoke configuration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    suite.configure_process_environment()
    ref = json.loads((suite.BASE_DIR / "example_reference.json").read_text())
    suite.validate_inputs(["G23"])
    runner = suite.load_runner()
    original = emcee.EnsembleSampler.run_mcmc
    verified = []

    def check_reference(sampler, *args, **kwargs):
        import jupyter_all_mw_sed_pcprior_method2_distanceprior as model_code
        row, model, zps, bands, *_ = sampler.log_prob_fn.args
        if list(bands) == ref["bands"]:
            lp = sampler.log_prob_fn(np.array(ref["draw"]))
            mags = model_code.model_mags(np.array(ref["median"]), model, zps, bands)
            expected = [ref["model_magnitudes"][b] for b in bands]
            np.testing.assert_allclose(lp, ref["draw_log_prob"], rtol=0, atol=1e-5)
            np.testing.assert_allclose(mags, expected, rtol=0, atol=1e-5)
            verified.append(True)
        return original(sampler, *args, **kwargs)

    emcee.EnsembleSampler.run_mcmc = check_reference
    try:
        _, summary, _ = suite.run_one_fit(runner, suite.BRANCHES[0], args.output,
                    suite.SMOKE_CHAIN if args.short else suite.BASE_CHAIN, 1, [ref["source_id"]])
    finally:
        emcee.EnsembleSampler.run_mcmc = original
    if not verified or len(summary) != 1 or not summary.iloc[0]["fit_success"]:
        raise RuntimeError("Example fit or forward-model validation failed")
    print("Verified archived log posterior and model magnitudes.")
    if not args.short:
        for i, param in enumerate(ref["parameters"]):
            med = float(summary.iloc[0][param + "_p50"])
            if not ref["p16"][i] <= med <= ref["p84"][i]:
                raise RuntimeError(f"{param} median {med} falls outside the archived central interval")
        print("All six refit medians lie inside their archived central 68% intervals.")


if __name__ == "__main__":
    main()
