#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
"""Compile the serial benchmark against an explicitly selected QUICK install."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--ddx-lib", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output already exists; select a new artifact path")
    prefix = args.prefix.resolve(strict=True)
    source = here / "quick_electrostatics_benchmark.f90"
    flavor, library = ("serial", "quick") if args.backend == "cpu" else ("cuda", "quick_cuda")
    command = [str(args.compiler.resolve(strict=True)), "-cpp", "-ffree-line-length-none"]
    if args.backend == "cuda":
        command += ["-DCUDA", "-DGPU"]
    command += ["-I" + str(root / "src/util"),
                "-I" + str(prefix / "include/amber-modules/quick" / flavor), str(source),
                "-L" + str(prefix / "lib"), "-Wl,-rpath," + str(prefix / "lib")]
    if args.ddx_lib:
        command += ["-Wl,-rpath," + str(args.ddx_lib.resolve(strict=True)),
                    "-Wl,-rpath-link," + str(args.ddx_lib.resolve(strict=True))]
    command += ["-l" + library, "-o", str(args.output)]
    process = subprocess.run(command, text=True, capture_output=True)
    record = {"argv": command, "returncode": process.returncode,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "stdout": process.stdout, "stderr": process.stderr}
    if process.returncode == 0:
        record["binary_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".compile.json").write_text(json.dumps(record, indent=2) + "\n")
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    print(json.dumps({"compiled": str(args.output), "backend": args.backend}))


if __name__ == "__main__":
    main()
