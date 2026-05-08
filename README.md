# hza_tagger

Jet tagger for H→Z(ll)+a(had) decays in CMS, targeting AK4 PUPPI jets from the hadronic decay of the pseudoscalar **a** (PDG 36).

```
hza_tagger/
├── common/          shared label defs, truth matching, IO schema, variable lists
├── converter/       btvNanoAOD ROOT → H5 (coffea, columnar)
├── tagger/          SALT submodule + training configs and scripts
└── analysis/        ROC curves, score distributions, working-point studies
```

## Quick start

### 1. Environment

```bash
mamba env create -f environment.yml
conda activate hza_tagger
pip install -e .        # installs common/, converter/, analysis/ as a package
```

### 2. SALT submodule

```bash
bash tagger/scripts/setup_salt.sh
```

### 3. Check your ROOT file's branch names

```bash
python converter/inspect_branches.py /path/to/hzanano_output_1.root
```

Compare with `common/variables.py` and adjust branch names there if needed.

### 4. Edit the converter config

```bash
nano converter/configs/hza_signal.yaml   # set file paths, cuts, chunk size
```

### 5. Run converter (local, quick test)

```bash
python converter/run_local.py --config converter/configs/hza_signal.yaml
```

This reads `split_fractions` from the config (default 70 / 15 / 15 %) and writes
three files: `data/train.h5`, `data/val.h5`, `data/test.h5`.

Pass `--out data/all.h5` to skip the split and write a single file (useful for quick tests).
Pass `--max-events N` to cap the number of events read.

### 6. Scale out on DESY NAF / HTCondor

```bash
python converter/run_condor.py \
    --config converter/configs/hza_signal.yaml \
    --outdir data/chunks/ \
    --merge
```

### 7. Preprocess + train

```bash
bash tagger/scripts/preprocess.sh   # computes normalisation dict
bash tagger/scripts/train.sh        # launches SALT training
```

On DESY NAF GPU nodes, add `--trainer.accelerator gpu --trainer.devices 1` to `train.sh`.

**Comet.ml logging** is enabled automatically when a `COMET_API_KEY` is present.

<details>
<summary>Setting up a Comet account (first time)</summary>

1. Go to **[comet.com](https://www.comet.com)** and sign up for a free account.
2. After logging in, open **[comet.com/api/my/settings](https://www.comet.com/api/my/settings)** and copy your **API key**.
3. Create `.env` in the project root (it is git-ignored):
   ```bash
   cp .env.example .env
   # then open .env and paste your key:
   #   COMET_API_KEY=<your_key>
   ```

</details>

`train.sh` sources `.env` on every run and passes the key to `CometLogger`. Without a key it falls back to offline mode (logs saved under `logs/`).

### 8. Evaluate

The evaluation script auto-discovers the test H5 file, the most recent checkpoint, and the training config from the standard project layout:

```bash
bash analysis/scripts/evaluate.sh
```

It runs two steps in sequence and writes plots to `analysis/plots/`:

1. **Score** — `eval_to_h5.py` loads the best checkpoint and appends a `scores` dataset (shape `(N, 2)`) to a copy of the test H5.
2. **Plot** — `plots.py` produces ROC curves, score distributions, and efficiency vs pT/η.

**Override any path** via argument or environment variable:

```bash
# Explicit test file
bash analysis/scripts/evaluate.sh data/my_test.h5

# Explicit test file + checkpoint
bash analysis/scripts/evaluate.sh data/my_test.h5 logs/my_run/checkpoints/best.ckpt

# Environment variable overrides
TEST_FILE=data/my_test.h5 \
CKPT=logs/my_run/checkpoints/best.ckpt \
PLOT_DIR=analysis/plots/my_run \
bash analysis/scripts/evaluate.sh
```

The auto-discovery priority is:

| Variable | Search order |
|----------|-------------|
| `TEST_FILE` | `data/test.h5` → `data/test_out.h5` → first `data/*.h5` |
| `CKPT` | lowest `val_loss` across all `logs/*/ckpts/epoch=*-val_loss=*.ckpt`; falls back to `best.ckpt` |
| `TRAIN_CFG` | `tagger/configs/hza_train.yaml` |
| `SCORES_FILE` | same dir as test file, `<name>_scores.h5` |
| `PLOT_DIR` | `analysis/plots/` |

## Tests

```bash
pytest -v
```

## Key design decisions

| Choice | Rationale |
|--------|-----------|
| Binary label (a-jet vs other) | Simplest discriminant; background jets taken from same signal sample |
| dR matching to a + daughters | Robust to multiple a's; requires all hadronic daughters inside the jet cone → clean merged-topology label |
| AK4 PUPPI jets | Standard CMS Run3 jet collection |
| PFCands as tracks | Rich per-constituent info in btvNanoAOD; IP variables zero-padded when absent (pheno files) |
| SALT via git submodule | No code fork; thin config layer only; easy to track upstream changes |
| Coffea columnar converter | Scales from laptop (iterative) to HTCondor (dask-jobqueue) without code changes |
