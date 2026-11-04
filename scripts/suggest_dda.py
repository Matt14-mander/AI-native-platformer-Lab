"""Produce an offline rules-v1 suggestion; never generate or apply a level."""

import argparse
import json
from pathlib import Path

from ai_platformer.adaptive.dda import DDAContext
from ai_platformer.adaptive.rules import suggest
from scripts.validate_dda_proposal import load_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True,
                        help="standalone profile or summarize_play_sessions report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.resolve() in {
            args.context.resolve(), args.profile.resolve()
        }:
            raise ValueError("output must be a new file distinct from inputs")
        context = DDAContext.model_validate(load_document(args.context))
        profile = load_document(args.profile)
        if isinstance(profile, dict) and profile.get("kind") == "play_metrics_and_profiles_v1":
            matches = [p for p in profile["profiles"] if p["profile_id"] == context.profile.profile_id]
            if len(matches) != 1:
                raise ValueError("report must contain exactly one matching profile")
            profile = matches[0]
        proposal = suggest(context, profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(proposal.model_dump(), stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        parser.error(str(error))
    print(f"action={proposal.action}; policy={proposal.policy_version}; application=not_applied; generation=not_run")


if __name__ == "__main__":
    main()
