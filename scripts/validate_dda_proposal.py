"""Check a DDA suggestion against its context and exact player-profile snapshot."""

import argparse
import json
from pathlib import Path

from ai_platformer.adaptive.dda import DDAContext, validate_proposal
from ai_platformer.content.level_spec import parse_level_json


def load_document(path):
    with Path(path).open("rb") as stream:
        data = stream.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError("DDA input exceeds 4 MB limit; use a selected profile snapshot")
    return parse_level_json(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True,
                        help="standalone profile or summarize_play_sessions report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {
        path.resolve() for path in (args.context, args.proposal, args.profile)
    }:
        parser.error("output must be a new file distinct from inputs")
    try:
        context = DDAContext.model_validate(load_document(args.context))
        profile = load_document(args.profile)
        if isinstance(profile, dict) and profile.get("kind") == "play_metrics_and_profiles_v1":
            matches = [item for item in profile["profiles"]
                       if item["profile_id"] == context.profile.profile_id]
            if len(matches) != 1:
                raise ValueError("profile report must contain exactly one matching profile_id")
            profile = matches[0]
        report = validate_proposal(context, load_document(args.proposal), profile)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        report = {
            "schema_version": 1, "kind": "dda_contract_validation_v1", "contract_valid": False,
            "error": str(error), "application": "not_applied", "generation": "not_run",
            "static_validation": "not_run", "route_verification": "not_run", "ppo_evaluation": "not_run",
        }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as error:
        parser.exit(2, f"cannot write DDA report: {error}\n")
    print(f"contract_valid={report['contract_valid']}; application=not_applied; generation=not_run")
    if not report["contract_valid"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
