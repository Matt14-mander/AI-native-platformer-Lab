"""Build the local C ABI bridge without modifying the TinyInfer checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="TinyInfer checkout")
    parser.add_argument("--build-dir", type=Path, default=Path("build/tinyinfer_bridge"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--native", action="store_true", help="enable host-specific CPU instructions"
    )
    args = parser.parse_args()
    source = args.source.resolve()
    build = args.build_dir.resolve()
    if not (source / "include/tinyinfer/tinyinfer.h").is_file() or args.jobs < 1:
        parser.error("source must be a TinyInfer checkout; jobs must be positive")
    bridge = Path(__file__).resolve().parents[1] / "native/tinyinfer_bridge"
    try:
        subprocess.run(
            [
                "cmake",
                "-S",
                str(bridge),
                "-B",
                str(build),
                f"-DTINYINFER_SOURCE_DIR={source}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DTINYINFER_ENABLE_NATIVE_ARCH={'ON' if args.native else 'OFF'}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--config",
                "Release",
                "--parallel",
                str(args.jobs),
            ],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        source_dirty = bool(
            subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
        libraries = sorted(
            path
            for path in build.rglob("*platformer_tinyinfer*")
            if path.suffix in {".dylib", ".so", ".dll"}
        )
        if len(libraries) != 1:
            raise ValueError(f"expected one bridge library, found {libraries}")
        report = {
            "library": str(libraries[0]),
            "library_sha256": hashlib.sha256(libraries[0].read_bytes()).hexdigest(),
            "tinyinfer_source": str(source),
            "tinyinfer_revision": revision,
            "tinyinfer_source_dirty": source_dirty,
            "configuration": "Release",
            "native_arch": args.native,
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        (build / "bridge_build.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"bridge build failed: {error}\n")


if __name__ == "__main__":
    main()
