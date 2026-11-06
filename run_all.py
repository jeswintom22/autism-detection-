"""Run the patient-safe pipeline: prepare + development training (never the test set).

    python run_all.py --config configs/hybrid.json --run-dir runs/hybrid/<run-id>

Final test scoring is separate:
    python -m asd_training evaluate-test --run-dir runs/hybrid/<run-id> --confirm-final
"""
import argparse
import sys

from asd_training.cli import DEFAULT_CONFIG, main as cli_main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--run-dir", required=True)
    args, extra = ap.parse_known_args(argv)
    if extra:
        ap.error(
            f"unsupported argument(s): {' '.join(extra)}. The legacy step1/step2/step3 options "
            "were removed; see docs/patient_safe_training.md"
        )
    return cli_main(["run", "--config", args.config, "--run-dir", args.run_dir])


if __name__ == "__main__":
    sys.exit(main())
