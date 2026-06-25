"""Generate a self-contained Colab notebook from hmd_leakage_rerun.py.

The notebook embeds the exact pipeline functions (so Colab runs what was validated
locally), wraps them with Drive-mount + GPU cells, and a run cell that sweeps
buggy / alaw / blaw over 5 seeds and prints a comparison table.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "hmd_leakage_rerun.py").read_text()

# Keep everything from the top through the function definitions, drop the CLI main().
engine = SRC.split("\ndef main(")[0]
# Remove the local-only default data dir (uses __file__, not valid in a notebook).
engine = re.sub(r"\nDEFAULT_DATA_DIR = Path\(__file__\).*\n", "\n", engine)
engine = engine.rstrip() + "\n\nprint('engine loaded | device =', DEVICE)\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


cells = []

cells.append(md(
    "# BR2 HMD leakage fix: 5-seed sweep (Colab GPU)\n"
    "\n"
    "Re-runs the two-stage volume pipeline with three HMD constructions, 5 seeds each,\n"
    "and reports a comparison table. The HMD build is the only thing that changes across\n"
    "variants; the feature set, scaler, model, and training are identical.\n"
    "\n"
    "- **buggy**: original asymmetric build (train rolling window, val/test frozen full-train mean).\n"
    "- **alaw**: frozen full-train mean on all splits.\n"
    "- **blaw**: causal rolling window on all splits (consistent, leakage-free).\n"
    "\n"
    "Run the cells in order. The only manual input is the data folder path in Step 2.\n"
))

cells.append(md(
    "## Step 1. Enable the GPU\n"
    "\n"
    "Menu: **Runtime -> Change runtime type -> Hardware accelerator -> GPU -> Save**.\n"
    "The cell below should then print a GPU name. A `cpu` result still runs, only slower.\n"
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
    "The three split CSVs (`Split_Train_Data.csv`, `Split_Validation_Data.csv`,\n"
    "`Split_Test_Data.csv`) must be in Google Drive.\n"
    "\n"
    "1. In Google Drive, create a folder named **`BR2_data`** and upload the three CSVs into it.\n"
    "2. Run the mount cell and authorize access when prompted.\n"
    "3. Set `DATA_DIR` to that folder (default assumes `BR2_data` at the Drive root).\n"
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

cells.append(md(
    "## Step 3. Load the pipeline\n"
    "\n"
    "Defines all pipeline functions. Run once.\n"
))
cells.append(code(engine))

cells.append(md(
    "## Step 4. Run the 5-seed sweep\n"
    "\n"
    "Trains 3 variants x 5 seeds = 15 GRU-small models (a few minutes on a GPU).\n"
    "Prints a comparison table and writes the raw rows and summary to the Drive folder.\n"
))
cells.append(code(
    "import pandas as pd, numpy as np, json\n"
    "\n"
    "SEEDS = [42, 0, 1, 2, 3]\n"
    "VARIANTS = ['buggy', 'alaw', 'blaw']\n"
    "\n"
    "df_train, df_val, df_test = load_data(DATA_DIR)\n"
    "\n"
    "all_rows = []\n"
    "for v in VARIANTS:\n"
    "    all_rows.extend(run_variant(v, df_train, df_val, df_test, SEEDS))\n"
    "res = pd.DataFrame(all_rows)\n"
    "\n"
    "out_dir = DATA_DIR / 'br2_outputs'\n"
    "out_dir.mkdir(parents=True, exist_ok=True)\n"
    "res.to_csv(out_dir / 'results_5seed.csv', index=False)\n"
    "\n"
    "def ms(g, col):\n"
    "    return g[col].mean(), g[col].std(ddof=0)\n"
    "\n"
    "print('\\n' + '=' * 84)\n"
    "print('COMPARISON  (mean +/- std over', len(SEEDS), 'seeds)')\n"
    "print('=' * 84)\n"
    "hdr = ('{:7s} | {:>16s} | {:>16s} | {:>11s} | {:>18s}'\n"
    "       .format('variant', 'val IC (return)', 'test IC (return)',\n"
    "               'HMDonly IC', 'IC delta (level)'))\n"
    "print(hdr)\n"
    "print('-' * 84)\n"
    "summary = {}\n"
    "for v in VARIANTS:\n"
    "    g = res[res.variant == v]\n"
    "    vic = ms(g, 'val_IC_return'); tic = ms(g, 'test_IC_return')\n"
    "    hmd = g['HMDonly_IC_level'].iloc[0]; dl = ms(g, 'IC_delta_level')\n"
    "    print('{:7s} | {:7.4f} +/- {:5.4f} | {:7.4f} +/- {:5.4f} | {:11.4f} | {:+7.4f} +/- {:5.4f}'\n"
    "          .format(v, vic[0], vic[1], tic[0], tic[1], hmd, dl[0], dl[1]))\n"
    "    summary[v] = {\n"
    "        'val_IC_return_mean': vic[0], 'val_IC_return_std': vic[1],\n"
    "        'test_IC_return_mean': tic[0], 'test_IC_return_std': tic[1],\n"
    "        'HMDonly_IC_level': float(hmd),\n"
    "        'HMDGRU_IC_level_mean': ms(g, 'HMDGRU_IC_level')[0],\n"
    "        'IC_delta_level_mean': dl[0], 'IC_delta_level_std': dl[1],\n"
    "    }\n"
    "with open(out_dir / 'summary_5seed.json', 'w') as f:\n"
    "    json.dump(summary, f, indent=2)\n"
    "print('\\nsaved ->', out_dir)\n"
))

cells.append(md(
    "## Reading the table\n"
    "\n"
    "- **val / test IC (return)**: how well the GRU predicts the Stage-2 residual (the hard\n"
    "  part). Higher is better. `+/-` is the spread across seeds; a large spread means the\n"
    "  point estimate is unstable.\n"
    "- **HMDonly IC**: the HMD baseline alone, no neural net.\n"
    "- **IC delta (level)**: the GRU's marginal contribution over HMD in level space. Under\n"
    "  the clean `blaw` build, a delta that stays clearly positive and larger than its `+/-`\n"
    "  indicates the two-stage design still adds value after the fix.\n"
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

out = HERE / "BR2_hmd_leakage_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
print("cells:", len(cells), "| engine chars:", len(engine))
