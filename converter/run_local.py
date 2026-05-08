#!/usr/bin/env python3
"""Run the converter locally (single process, iterative executor).

By default reads split_fractions from the config and writes separate
train / val / test H5 files.  Pass --out to override and write a single file.

Usage
-----
    python converter/run_local.py --config converter/configs/hza_signal.yaml
    python converter/run_local.py --config converter/configs/hza_signal.yaml \\
                                  --out data/all.h5 --max-events 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import uproot
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

from converter.processors.jet_dumper import process_events
from converter.processors.writer import H5Writer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="Write single file (disables train/val/test split)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for split shuffle")
    return p.parse_args()


def _assign_splits(n: int, fracs: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """Return an integer array of length n with values 0=train, 1=val, 2=test."""
    idx = rng.permutation(n)
    splits = np.empty(n, dtype=np.int8)
    cut1 = int(round(fracs["train"] * n))
    cut2 = cut1 + int(round(fracs["val"] * n))
    splits[idx[:cut1]]  = 0  # train
    splits[idx[cut1:cut2]] = 1  # val
    splits[idx[cut2:]]  = 2  # test
    return splits


def main():
    args = parse_args()
    cfg  = yaml.safe_load(Path(args.config).read_text())

    single_out = args.out  # None → use split mode
    chunk_size = cfg.get("chunk_size", 10_000)
    max_events = args.max_events
    tree_name  = cfg.get("tree", "Events")
    rng        = np.random.default_rng(args.seed)

    # ── collect ALL arrays first, then split ─────────────────────────────────
    # This is memory-efficient for pheno files (order of MB); for large
    # production samples use run_condor.py and merge afterwards.
    all_jets   = []
    all_tracks = []
    all_labels = []
    n_processed = 0

    for file_path in cfg["files"]:
        print(f"\nProcessing: {file_path}")
        try:
            f    = uproot.open(file_path)
            tree = f[tree_name]
        except FileNotFoundError:
            print(f"  WARNING: file not found, skipping")
            continue

        n_entries = tree.num_entries
        if max_events is not None:
            n_entries = min(n_entries, max_events - n_processed)
        if n_entries <= 0:
            break

        for start in range(0, n_entries, chunk_size):
            stop = min(start + chunk_size, n_entries)
            chunk = NanoEventsFactory.from_root(
                {file_path: tree_name},
                entry_start=start,
                entry_stop=stop,
                schemaclass=NanoAODSchema,
            ).events()
            chunk = ak.Array(chunk.compute())
            arrays = process_events(chunk)

            if len(arrays["jets"]) == 0:
                continue

            all_jets.append(arrays["jets"])
            all_tracks.append(arrays["tracks"])
            all_labels.append(arrays["labels"])
            n_a = int(arrays["labels"]["a_jet"].sum())
            print(f"  events {start}–{stop}: {len(arrays['jets'])} jets  ({n_a} a-jets)")

            n_processed += stop - start
            if max_events is not None and n_processed >= max_events:
                break

        if max_events is not None and n_processed >= max_events:
            break

    if not all_jets:
        print("No jets found — check your config files.")
        return

    jets   = np.concatenate(all_jets)
    tracks = np.concatenate(all_tracks)
    labels = np.concatenate(all_labels)
    total_jets = len(jets)
    total_a    = int(labels["a_jet"].sum())
    print(f"\nTotal jets: {total_jets}  a-jets: {total_a}  ({100*total_a/max(total_jets,1):.1f}%)")

    # ── write ─────────────────────────────────────────────────────────────────
    if single_out:
        with H5Writer(single_out) as w:
            w.write_chunk(jets, tracks, labels)
            w.finalize()
        print(f"Written to: {single_out}")
    else:
        fracs = cfg.get("split_fractions", {"train": 0.70, "val": 0.15, "test": 0.15})
        out   = cfg["output"]
        split_ids = _assign_splits(total_jets, fracs, rng)

        for split_idx, (split_name, path) in enumerate(out.items()):
            mask = split_ids == split_idx
            n_split = mask.sum()
            with H5Writer(path) as w:
                w.write_chunk(jets[mask], tracks[mask], labels[mask])
                w.finalize()
            n_a_split = int(labels[mask]["a_jet"].sum())
            print(f"  {split_name:5s}: {n_split:6d} jets  ({n_a_split} a-jets)  → {path}")


if __name__ == "__main__":
    main()
