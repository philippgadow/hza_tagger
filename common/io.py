"""H5 schema helpers shared by converter and analysis."""

from __future__ import annotations

import numpy as np

# ── Expected H5 dataset names ─────────────────────────────────────────────────

JETS_DATASET = "jets"
TRACKS_DATASET = "tracks"
LABELS_DATASET = "labels"

# dtype of the jets structured array stored in H5
# Note: a_jet label is included here so SALT's SaltDataset can find it
# when processing labels for the "jets" input.
JET_DTYPE = np.dtype([
    ("pt",    np.float32),
    ("eta",   np.float32),
    ("phi",   np.float32),
    ("mass",  np.float32),
    ("a_jet", np.int32),    # truth label: 1=a-jet, 0=background
])

# dtype of the tracks structured array stored in H5
# shape: (n_jets, N_TRACKS)
TRACK_DTYPE = np.dtype([
    ("pt",          np.float32),
    ("eta_rel",     np.float32),
    ("phi_rel",     np.float32),
    ("mass",        np.float32),
    ("charge",      np.int8),
    ("pdgId",       np.int32),
    ("dxy",         np.float32),
    ("dz",          np.float32),
    ("dxySig",      np.float32),
    ("dzSig",       np.float32),
    ("trkQuality",  np.int8),
    ("puppiWeight", np.float32),
    ("valid",       np.bool_),   # False for padding slots
])

LABEL_DTYPE = np.dtype([
    ("a_jet", np.int32),
])
