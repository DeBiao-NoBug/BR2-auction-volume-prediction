"""
Extension: 12-model ranking (Part A) and 5-fold time-series CV (Part B) on the
clean (B-law) HMD pipeline. Re-checks the previous team's "GRU-small is best"
(#19 / #27) and "CV mean << single-split val" (#15 / #12) conclusions after the
HMD measurement fix.

Builds on hmd_leakage_rerun.py (the bug-fix engine). Only the HMD construction
ever changes; model classes, training, evaluation, and the CV driver are lifted
verbatim from Final Code.ipynb (cells 23, 24, 26/28/30/32, 70/71/72).

Design notes (agreed with reviewer):
  Part A: all 12 models share the SAME 5 seeds; before each model we reset the RNG
          to that seed (paired comparison, difference is architecture only). The
          "is GRU-small best" call uses descriptive paired evidence, no t-test.
  Part B: plain TimeSeriesSplit, no embargo/purge (issue #10 stays open; CV mean is
          therefore still mildly optimistic). A-law and B-law CV use the SAME seed
          and the SAME fold split, so the only difference is the HMD construction.
          A-law reproduces the previous team's frozen-HMD CV (~0.1885) as an anchor.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hmd_leakage_rerun import (
    EPS, SEQ_LEN, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, DEVICE,
    load_data, build_hmd_blaw, build_intraday_pattern, generate_stage1_predictions,
    add_interactions_and_features, prepare_nn_data, GRUModel, train_model,
    evaluate_model, set_all_seeds,
)

CV_EPOCHS = 15
N_SPLITS = 5


# ----------------------------------------------------------------------------
# Model classes (Final Code.ipynb cell 23). GRUModel comes from the base module.
# .squeeze() left unchanged on purpose (review #24).
# ----------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.resid = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x)))
        out = out[:, :, :x.size(2)]
        out = self.drop(self.relu(self.conv2(out)))
        out = out[:, :, :x.size(2)]
        return self.relu(out + self.resid(x))


class TCNModel(nn.Module):
    def __init__(self, input_size, n_channels=64, n_blocks=3, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_blocks):
            in_ch = input_size if i == 0 else n_channels
            layers.append(TCNBlock(in_ch, n_channels, kernel_size, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x):
        out = self.network(x.transpose(1, 2))
        return self.fc(out[:, :, -1]).squeeze()


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = input_size
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------------
# Model configs (cells 26 / 28 / 30 / 32)
# ----------------------------------------------------------------------------
RANKING_CONFIGS = [
    {"name": "GRU-small",  "family": "GRU",  "hidden_size": 32,  "num_layers": 1, "dropout": 0.1},
    {"name": "GRU-medium", "family": "GRU",  "hidden_size": 64,  "num_layers": 2, "dropout": 0.2},
    {"name": "GRU-large",  "family": "GRU",  "hidden_size": 128, "num_layers": 2, "dropout": 0.2},
    {"name": "LSTM-small",  "family": "LSTM", "hidden_size": 32,  "num_layers": 1, "dropout": 0.1},
    {"name": "LSTM-medium", "family": "LSTM", "hidden_size": 64,  "num_layers": 2, "dropout": 0.2},
    {"name": "LSTM-large",  "family": "LSTM", "hidden_size": 128, "num_layers": 2, "dropout": 0.2},
    {"name": "TCN-small",  "family": "TCN", "n_channels": 32, "n_blocks": 2, "kernel_size": 3, "dropout": 0.1},
    {"name": "TCN-medium", "family": "TCN", "n_channels": 64, "n_blocks": 3, "kernel_size": 3, "dropout": 0.2},
    {"name": "TCN-large",  "family": "TCN", "n_channels": 96, "n_blocks": 3, "kernel_size": 5, "dropout": 0.2},
    {"name": "MLP-small",  "family": "MLP", "hidden_dims": [64, 32],   "dropout": 0.1},
    {"name": "MLP-medium", "family": "MLP", "hidden_dims": [128, 64],  "dropout": 0.2},
    {"name": "MLP-large",  "family": "MLP", "hidden_dims": [256, 128], "dropout": 0.2},
]

# CV uses the same 6 the previous team used (cell 71): GRU x3 + TCN x3.
CV_CONFIGS = [c for c in RANKING_CONFIGS if c["family"] in ("GRU", "TCN")]


def build_model(cfg, input_size):
    fam = cfg["family"]
    if fam == "GRU":
        return GRUModel(input_size, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
    if fam == "LSTM":
        return LSTMModel(input_size, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
    if fam == "TCN":
        return TCNModel(input_size, cfg["n_channels"], cfg["n_blocks"], cfg["kernel_size"], cfg["dropout"]).to(DEVICE)
    if fam == "MLP":
        return MLPModel(input_size, cfg["hidden_dims"], cfg["dropout"]).to(DEVICE)
    raise ValueError(fam)


# ----------------------------------------------------------------------------
# Part A: 12-model ranking on a single train/val split, B-law HMD, shared seeds.
# ----------------------------------------------------------------------------
def run_ranking(df_train, df_val, df_test, seeds, configs=RANKING_CONFIGS, epochs=EPOCHS):
    # B-law HMD reused directly from the bug-fix task (concat train+val+test rolling).
    dtr, dva, dte = build_hmd_blaw(df_train.copy(), df_val.copy(), df_test.copy())
    dtr = generate_stage1_predictions(dtr)
    dva = generate_stage1_predictions(dva)
    dte = generate_stage1_predictions(dte)
    nn_features = add_interactions_and_features(dtr, dva, dte)
    data = prepare_nn_data(dtr, dva, dte, nn_features, "return_diff", seq_len=SEQ_LEN)

    X_tr = torch.tensor(data["X_tr"]).to(DEVICE)
    y_tr = torch.tensor(data["y_tr"]).to(DEVICE)
    X_va = torch.tensor(data["X_va"]).to(DEVICE)
    y_va = torch.tensor(data["y_va"]).to(DEVICE)
    idx_va, df_va_scaled, scaler_y = data["idx_va"], data["df_va"], data["scaler_y"]
    n_features = len(nn_features)

    rows = []
    for seed in seeds:
        print(f"\n===== ranking | seed {seed} =====")
        for cfg in configs:
            # Paired: every model at this seed starts from the identical RNG state.
            set_all_seeds(seed)
            gen = torch.Generator()
            gen.manual_seed(seed)
            tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                                   shuffle=True, generator=gen)
            va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)

            model = build_model(cfg, n_features)
            n_params = sum(p.numel() for p in model.parameters())
            t0 = time.time()
            train_model(model, tr_loader, va_loader, epochs=epochs, lr=LR,
                        weight_decay=WEIGHT_DECAY, patience=PATIENCE)
            vm, _, _ = evaluate_model(model, va_loader, df_va_scaled, idx_va, scaler_y, "return_diff")
            rows.append({"model": cfg["name"], "family": cfg["family"], "seed": seed,
                         "n_params": n_params, "val_IC": vm["IC"], "val_R2": vm["R2"]})
            print(f"  {cfg['name']:11s} val IC {vm['IC']:.4f}  ({n_params:,} params, {time.time()-t0:.0f}s)")
    return rows


# ----------------------------------------------------------------------------
# Part B: 5-fold time-series CV. Two HMD builders, same seed, same fold split.
# ----------------------------------------------------------------------------
def fold_rebuild_hmd_alaw(df_tr_fold, df_va_fold):
    """A-law per fold: frozen train-only mean applied to train and val. Cell 70 verbatim."""
    tr = df_tr_fold.copy()
    va = df_va_fold.copy()
    for d in (tr, va):
        if "volume" not in d.columns:
            d["volume"] = d["imbalance_size"] + d["matched_size"]
        d["target_diff"] = d["target"] - d["volume"]

    st_diff = tr.groupby(["stock_id", "seconds_in_bucket"])["target_diff"].mean().to_dict()
    t_diff = tr.groupby("seconds_in_bucket")["target_diff"].mean().to_dict()
    st_vol = tr.groupby(["stock_id", "seconds_in_bucket"])["volume"].mean().to_dict()
    t_vol = tr.groupby("seconds_in_bucket")["volume"].mean().to_dict()

    def _apply(d):
        keys = list(zip(d["stock_id"].values, d["seconds_in_bucket"].values))
        d["stock_time_mean_diff"] = [st_diff.get(k, np.nan) for k in keys]
        d["time_mean_diff"] = d["seconds_in_bucket"].map(t_diff)
        d["stock_time_mean_diff"] = d["stock_time_mean_diff"].fillna(d["time_mean_diff"])
        d["stock_time_mean_vol"] = [st_vol.get(k, np.nan) for k in keys]
        d["time_mean_vol"] = d["seconds_in_bucket"].map(t_vol)
        d["stock_time_mean_vol"] = d["stock_time_mean_vol"].fillna(d["time_mean_vol"])
        d["vol_vs_intraday"] = d["volume"] / (d["stock_time_mean_vol"] + EPS)
        d["stage1_pred_diff"] = d["stock_time_mean_diff"]
        d["stage1_pred_vol"] = d["volume"] + d["stage1_pred_diff"]
        d["return_diff"] = d["target_diff"] - d["stage1_pred_diff"]
        return d

    return _apply(tr), _apply(va)


def fold_rebuild_hmd_blaw(df_tr_fold, df_va_fold):
    """
    B-law per fold: causal 10-day rolling (+shift(1)) on the fold's own timeline.
    Concat fold-train + fold-val (fold-val days are strictly later), run the same
    train-branch construction as the main pipeline, then split back. shift(1) keeps
    every fold-val row strictly past; the window rolling across the train/val boundary
    into the most recent fold-train days is the intended causal behavior.
    """
    parts = []
    for name, df in (("train", df_tr_fold), ("val", df_va_fold)):
        d = df.copy()
        d["split"] = name
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)
    combined, _ = build_intraday_pattern(combined, train_stats=None)
    combined = generate_stage1_predictions(combined)
    tr = combined[combined["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    va = combined[combined["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    return tr, va


def fold_prepare_nn(df_tr, df_va, feature_cols, target_col, seq_len=20):
    """Per-fold sequence builder (cell 70 verbatim). Scaler fit on this fold's train only."""
    cols_needed = feature_cols + [target_col, "stock_id_original", "date_id_original"]
    df_tr = df_tr.dropna(subset=cols_needed).copy()
    df_va = df_va.dropna(subset=cols_needed).copy()

    scaler_X = StandardScaler().fit(df_tr[feature_cols])
    scaler_y = StandardScaler().fit(df_tr[[target_col]])
    for d in [df_tr, df_va]:
        d[feature_cols] = scaler_X.transform(d[feature_cols])
        d[target_col] = scaler_y.transform(d[[target_col]])

    def make_sequences(df):
        X_list, y_list, idx_list = [], [], []
        df = df.sort_values(["stock_id_original", "date_id_original", "seconds_in_bucket"])
        for (sid, did), g in df.groupby(["stock_id_original", "date_id_original"]):
            feats = g[feature_cols].values
            targ = g[target_col].values
            idx = g.index.values
            if len(g) <= seq_len:
                continue
            for i in range(len(g) - seq_len):
                X_list.append(feats[i:i + seq_len])
                y_list.append(targ[i + seq_len])
                idx_list.append(idx[i + seq_len])
        return (np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.float32),
                np.array(idx_list, dtype=np.int64))

    X_tr, y_tr, _ = make_sequences(df_tr)
    X_va, y_va, idx_va = make_sequences(df_va)

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr).to(DEVICE), torch.tensor(y_tr).to(DEVICE)),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)),
                           batch_size=BATCH_SIZE)
    return tr_loader, va_loader, scaler_y, df_va, idx_va


def build_cv_pool(df_train, df_val):
    """CV pool = train + val (cell 69, HOLDOUT_TEST=True). Test kept as final holdout."""
    pool = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    # Row-local interaction features (no leakage) so folds have NN_FEATURES available.
    nn_features = add_interactions_and_features(pool, pool, pool)
    pool = pool.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).reset_index(drop=True)
    hmd_cols = ["stock_time_mean_diff", "time_mean_diff", "stock_time_mean_vol", "time_mean_vol",
                "vol_vs_intraday", "stage1_pred_diff", "stage1_pred_vol", "return_diff", "target_diff"]
    pool = pool.drop(columns=[c for c in hmd_cols if c in pool.columns], errors="ignore")
    unique_dates = sorted(pool["date_id"].unique())
    return pool, unique_dates, nn_features


def run_one_fold(df_tr_fold, df_va_fold, cfg, hmd_builder, nn_features, seed,
                 seq_len=SEQ_LEN, epochs=CV_EPOCHS):
    set_all_seeds(seed)
    df_tr_h, df_va_h = hmd_builder(df_tr_fold, df_va_fold)
    tr_loader, va_loader, scaler_y, df_va_scaled, idx_va = fold_prepare_nn(
        df_tr_h, df_va_h, nn_features, "return_diff", seq_len)
    model = build_model(cfg, len(nn_features))
    train_model(model, tr_loader, va_loader, epochs=epochs, lr=LR,
                weight_decay=WEIGHT_DECAY, patience=PATIENCE)
    vm, _, _ = evaluate_model(model, va_loader, df_va_scaled, idx_va, scaler_y, "return_diff")
    return vm


def run_cv(df_cv_pool, unique_dates, nn_features, configs, hmd_builder, seed,
           n_splits=N_SPLITS, epochs=CV_EPOCHS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    date_array = np.array(unique_dates)
    results = {cfg["name"]: [] for cfg in configs}
    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(date_array), start=1):
        tr_dates = set(date_array[tr_idx])
        va_dates = set(date_array[va_idx])
        df_tr_fold = df_cv_pool[df_cv_pool["date_id"].isin(tr_dates)].copy()
        df_va_fold = df_cv_pool[df_cv_pool["date_id"].isin(va_dates)].copy()
        print(f"  fold {fold_idx}/{n_splits} | train days {len(tr_dates)} val days {len(va_dates)}")
        for cfg in configs:
            vm = run_one_fold(df_tr_fold, df_va_fold, cfg, hmd_builder, nn_features, seed,
                              seq_len=SEQ_LEN, epochs=epochs)
            results[cfg["name"]].append({"fold": fold_idx, "IC": vm["IC"], "R2": vm["R2"]})
            print(f"    {cfg['name']:11s} val IC {vm['IC']:.4f}")
    return results
