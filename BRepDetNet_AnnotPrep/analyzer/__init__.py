import argparse
from pathlib import Path
from .extract_brepnet_data_from_step import BRepExtractor
from OCC.Extend import TopologyUtils

import json


def get_args(args):
    """Parses the input arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--step_path",
        type=str,
        default="./step_examples/16550_e88d6986_0.stp",
        help=" Path to the step file to process.",
    )

    args = parser.parse_args(args=args)
    return args


def main(args):
    args = get_args(args)
    print(args)
    step_path = Path(args.step_path)
    assert step_path.exists(), "no step file found in input path"
    brep_extractor = BRepExtractor(step_path, "", "", scale_body=True)

    print(json.dumps(brep_extractor.body_properties(), sort_keys=False, indent=4))
