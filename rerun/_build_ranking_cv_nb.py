"""Generate the self-contained Colab notebook for Part A (12-model ranking) and
Part B (5-fold CV, A-law + B-law). Embeds the bug-fix engine plus the ranking_cv
extension so Colab runs exactly what was validated locally."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Base engine: hmd_leakage_rerun.py without the CLI main() or the __file__ default path.
base = (HERE / "hmd_leakage_rerun.py").read_text().split("\ndef main(")[0]
base = re.sub(r"\nDEFAULT_DATA_DIR = Path\(__file__\).*\n", "\n", base)
base = base.rstrip() + "\n\nprint('engine loaded | device =', DEVICE)\n"

# Extension: ranking_cv.py without the sibling import block (names are already global
# in the notebook from the engine cell above).
ext = (HERE / "ranking_cv.py").read_text()
ext = re.sub(r"\nfrom hmd_leakage_rerun import \([^)]*\)\n", "\n", ext)
ext = ext.rstrip() + "\n\nprint('extension loaded |', len(RANKING_CONFIGS), 'ranking configs,', len(CV_CONFIGS), 'CV configs')\n"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = []

cells.append(md(
    "# BR2 model ranking and 5-fold CV on the leakage-free (B-law) HMD\n"
    "\n"
    "Re-checks two earlier conclusions after the HMD measurement fix:\n"
    "\n"
    "- **Part A**: rank all 12 models (GRU / LSTM / TCN / MLP x small / medium / large) on a\n"
    "  single train/val split, B-law HMD, over 5 shared seeds. Tests whether GRU-small's lead\n"
    "  is real or within seed noise (descriptive evidence, no significance test).\n"
    "- **Part B**: 5-fold time-series CV for the 6 GRU/TCN models, under both A-law (frozen,\n"
    "  reproduces the previous CV) and B-law (rolling) HMD, same seed and same fold split.\n"
    "\n"
    "Only the HMD construction ever changes; models, training, and evaluation are taken from\n"
    "the original pipeline. CV uses standard TimeSeriesSplit with no embargo or purge, so the\n"
    "CV mean is still mildly optimistic (a separate rigor item, not addressed here).\n"
    "\n"
    "Run the cells in order. The only manual input is the data folder path in Step 2.\n"
))

cells.append(md(
    "## Step 1. Enable the GPU\n"
    "\n"
    "Menu: **Runtime -> Change runtime type -> Hardware accelerator -> GPU -> Save**.\n"
    "This sweep trains roughly 120 small models, so a GPU matters here.\n"
))
cells.append(code(
    "import torch\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('GPU:', torch.cuda.get_device_name(0))\n"
))

cells.append(md(
    "## Step 2. Connect the data\n"
    "\n"
    "The three split CSVs must be in Google Drive. Mount Drive, then set `DATA_DIR` to the\n"
    "folder that holds them.\n"
))
cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
))
cells.append(code(
    "from pathlib import Path\n"
    "\n"
    "# Set this to the Drive folder holding the three CSVs.\n"
    "DATA_DIR = Path('/content/drive/MyDrive/BR2_data')\n"
    "\n"
    "needed = ['Split_Train_Data.csv', 'Split_Validation_Data.csv', 'Split_Test_Data.csv']\n"
    "missing = [f for f in needed if not (DATA_DIR / f).exists()]\n"
    "assert not missing, f'Missing in {DATA_DIR}: {missing}. Fix DATA_DIR or re-upload.'\n"
    "print('Found all 3 data files in', DATA_DIR)\n"
))

cells.append(md("## Step 3. Load the engine and the ranking/CV extension\n\nRun both cells once."))
cells.append(code(base))
cells.append(code(ext))

cells.append(md(
    "## Step 4. Part A - 12-model ranking (5 shared seeds)\n"
    "\n"
    "Trains 12 models x 5 seeds. Prints each model's val IC mean +/- std and three pieces of\n"
    "descriptive evidence on whether GRU-small's lead is robust.\n"
))
cells.append(code(
    "import numpy as np, pandas as pd, json\n"
    "\n"
    "RANK_SEEDS = [42, 0, 1, 2, 3]\n"
    "df_train, df_val, df_test = load_data(DATA_DIR)\n"
    "\n"
    "rank_rows = run_ranking(df_train, df_val, df_test, RANK_SEEDS)\n"
    "rank_df = pd.DataFrame(rank_rows)\n"
    "\n"
    "out_dir = DATA_DIR / 'br2_outputs'\n"
    "out_dir.mkdir(parents=True, exist_ok=True)\n"
    "rank_df.to_csv(out_dir / 'ranking_rows.csv', index=False)\n"
    "\n"
    "agg = (rank_df.groupby(['model', 'family'])['val_IC']\n"
    "       .agg(['mean', 'std']).reset_index().sort_values('mean', ascending=False))\n"
    "params = rank_df.groupby('model')['n_params'].first()\n"
    "\n"
    "print('\\n' + '=' * 64)\n"
    "print('PART A - ranking by mean val IC (return space), 5 seeds')\n"
    "print('=' * 64)\n"
    "print('{:12s} {:>6s} {:>9s} {:>8s} {:>9s}'.format('model', 'fam', 'IC mean', 'IC std', 'params'))\n"
    "print('-' * 64)\n"
    "for _, r in agg.iterrows():\n"
    "    print('{:12s} {:>6s} {:9.4f} {:8.4f} {:9,}'.format(\n"
    "        r['model'], r['family'], r['mean'], r['std'], int(params[r['model']])))\n"
    "\n"
    "# Descriptive evidence (no significance test).\n"
    "winners = rank_df.loc[rank_df.groupby('seed')['val_IC'].idxmax()]\n"
    "gs_wins = int((winners['model'] == 'GRU-small').sum())\n"
    "best_name = agg.iloc[0]['model']; second_name = agg.iloc[1]['model']\n"
    "gs_mean = float(agg.loc[agg['model'] == 'GRU-small', 'mean'].iloc[0])\n"
    "gs_std = float(agg.loc[agg['model'] == 'GRU-small', 'std'].iloc[0])\n"
    "best_mean = float(agg.iloc[0]['mean']); second_mean = float(agg.iloc[1]['mean'])\n"
    "print('\\n--- descriptive evidence on \"GRU-small is best\" ---')\n"
    "print(f'(1) per-seed winner: GRU-small ranked #1 in {gs_wins} of {len(RANK_SEEDS)} seeds')\n"
    "print(f'    seed winners: {list(winners.sort_values(\"seed\")[[\"seed\",\"model\"]].itertuples(index=False, name=None))}')\n"
    "print(f'(2) top model by mean is {best_name} ({best_mean:.4f}); 2nd is {second_name} ({second_mean:.4f}); '\n"
    "      f'gap {best_mean - second_mean:+.4f}')\n"
    "print(f'(3) GRU-small mean {gs_mean:.4f}, seed std {gs_std:.4f}: the lead is '\n"
    "      f'{\"larger than\" if (best_mean - second_mean) > gs_std else \"smaller than / within\"} its seed std')\n"
))

cells.append(md(
    "## Step 5. Part B - 5-fold CV (A-law and B-law)\n"
    "\n"
    "6 models (GRU x3 + TCN x3), 5 folds, single seed. A-law reproduces the previous team's\n"
    "frozen-HMD CV; B-law is the clean rolling version. Same seed and fold split, so the only\n"
    "difference is the HMD construction.\n"
))
cells.append(code(
    "CV_SEED = 42\n"
    "pool, dates, feats = build_cv_pool(df_train, df_val)\n"
    "print('CV pool', pool.shape, '| days', len(dates), '| features', len(feats))\n"
    "\n"
    "cv_out = {}\n"
    "for name, builder in [('alaw', fold_rebuild_hmd_alaw), ('blaw', fold_rebuild_hmd_blaw)]:\n"
    "    print(f'\\n===== CV {name} =====')\n"
    "    cv_out[name] = run_cv(pool, dates, feats, CV_CONFIGS, builder, seed=CV_SEED, n_splits=5)\n"
    "\n"
    "def cv_summary(res):\n"
    "    out = {}\n"
    "    for m, folds in res.items():\n"
    "        ics = np.array([f['IC'] for f in folds], float)\n"
    "        out[m] = (float(np.nanmean(ics)), float(np.nanstd(ics)))\n"
    "    return out\n"
    "\n"
    "sa, sb = cv_summary(cv_out['alaw']), cv_summary(cv_out['blaw'])\n"
    "print('\\n' + '=' * 70)\n"
    "print('PART B - 5-fold CV mean +/- std IC  (standard TSCV, no embargo/purge)')\n"
    "print('=' * 70)\n"
    "print('{:12s} | {:>18s} | {:>18s}'.format('model', 'A-law (frozen)', 'B-law (rolling)'))\n"
    "print('-' * 70)\n"
    "for cfg in CV_CONFIGS:\n"
    "    m = cfg['name']\n"
    "    print('{:12s} | {:7.4f} +/- {:6.4f} | {:7.4f} +/- {:6.4f}'.format(\n"
    "        m, sa[m][0], sa[m][1], sb[m][0], sb[m][1]))\n"
    "\n"
    "print('\\n--- per-fold IC for GRU-small ---')\n"
    "for name in ('alaw', 'blaw'):\n"
    "    ics = [round(f['IC'], 4) for f in cv_out[name]['GRU-small']]\n"
    "    print(f'  {name}: {ics}')\n"
    "\n"
    "rows = []\n"
    "for name in ('alaw', 'blaw'):\n"
    "    for m, folds in cv_out[name].items():\n"
    "        for f in folds:\n"
    "            rows.append({'hmd': name, 'model': m, 'fold': f['fold'], 'IC': f['IC'], 'R2': f['R2']})\n"
    "pd.DataFrame(rows).to_csv(out_dir / 'cv_rows.csv', index=False)\n"
    "with open(out_dir / 'cv_summary.json', 'w') as fh:\n"
    "    json.dump({'alaw': sa, 'blaw': sb}, fh, indent=2)\n"
    "print('\\nsaved ->', out_dir)\n"
))

cells.append(md(
    "## Reading the results\n"
    "\n"
    "Part A answers whether GRU-small is the best model on the clean pipeline. Judge it from\n"
    "the three descriptive lines: how often it wins across seeds, and whether its lead over the\n"
    "second model exceeds the seed-to-seed std. A lead smaller than the std means the models\n"
    "are not separable at this sample size.\n"
    "\n"
    "Part B answers the CV-mean question. A-law should land near the previously reported CV\n"
    "mean (about 0.19 for GRU-small) and serves as a consistency anchor. Compare CV mean against\n"
    "the single-split val IC from Part A: a large drop reflects how optimistic a single split is.\n"
    "Under a single seed the A-law vs B-law CV-mean difference is confounded with seed noise; if\n"
    "it is smaller than the per-fold std, treat the two as indistinguishable.\n"
))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "BR2_ranking_cv_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "| cells", len(cells), "| base", len(base), "ext", len(ext))
