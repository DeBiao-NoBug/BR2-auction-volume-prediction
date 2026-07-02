"""
HMD leakage fix rerun (issues #2 and #3).

Goal: re-run the two-stage main pipeline with the HMD feature rebuilt so that the
historical-mean estimator is constructed the SAME causal way on all three splits,
then check whether the headline conclusion (Stage 2 / GRU-small beats HMD-only by
+0.0148 IC in level space) survives.

Two HMD constructions are compared, everything else held identical:

  - "buggy": the original `build_intraday_pattern` (Final Code.ipynb cell 13).
             TRAIN uses a 10-day trailing rolling mean with shift(1); VAL/TEST use
             a single frozen full-train mean. Same column, two different qualities.

  - "blaw":  B-law fix. The exact cell-13 train-branch logic is applied to
             pd.concat([train, val, test]) sorted by date, so every split gets the
             same causal 10-day trailing rolling estimator (shift(1) keeps it
             strictly past). This is the production-realistic, fully symmetric build.

The only variable that changes between the two runs is the HMD build. Feature list,
scaler, sequence construction, model, training, and evaluation are lifted verbatim
from the notebook (cells 21-24, 26, 37, 39). The known `.squeeze()` edge case in the
RNN forward (review #24) is deliberately left unchanged to avoid confounding.

Run locally (CPU smoke test, single seed):
    .venv/bin/python rerun/hmd_leakage_rerun.py --seeds 42

Run the full multi-seed sweep (e.g. on Colab GPU):
    python rerun/hmd_leakage_rerun.py --seeds 42 0 1 2 3 --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------------
# Config (Final Code.ipynb cell 2)
# ----------------------------------------------------------------------------
EPS = 1e-6
N_STOCKS = 20
SEQ_LEN = 20
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
MAX_GRAD_NORM = 1.0

DEVICE = torch.device(
    os.environ.get("BR2_DEVICE")
    or ("cuda" if torch.cuda.is_available() else "cpu")
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "Previous Team Backup" / "Data"


# ----------------------------------------------------------------------------
# Data loading + base preprocessing (cells 5, 8, 10)
# ----------------------------------------------------------------------------
def load_data(data_dir: Path):
    df_train = pd.read_csv(data_dir / "Split_Train_Data.csv")
    df_val = pd.read_csv(data_dir / "Split_Validation_Data.csv")
    df_test = pd.read_csv(data_dir / "Split_Test_Data.csv")

    # Preserve original IDs for grouping (cell 5)
    for df in (df_train, df_val, df_test):
        df["stock_id_original"] = df["stock_id"].copy()
        df["date_id_original"] = df["date_id"].copy()

    # Compute volume and sort chronologically (cell 8)
    for df in (df_train, df_val, df_test):
        df["volume"] = df["imbalance_size"] + df["matched_size"]
        df.sort_values(["stock_id", "date_id", "seconds_in_bucket"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Step 1 assert: the splits are a clean, disjoint temporal partition. This is
    # the precondition for B-law's cross-boundary rolling window to be causal.
    tr_lo, tr_hi = df_train["date_id"].min(), df_train["date_id"].max()
    va_lo, va_hi = df_val["date_id"].min(), df_val["date_id"].max()
    te_lo, te_hi = df_test["date_id"].min(), df_test["date_id"].max()
    assert tr_hi < va_lo, f"train/val date overlap: {tr_hi} >= {va_lo}"
    assert va_hi < te_lo, f"val/test date overlap: {va_hi} >= {te_lo}"
    print(f"date_id ranges  train {tr_lo}-{tr_hi}  val {va_lo}-{va_hi}  test {te_lo}-{te_hi}")

    # Subset to the N most frequent stocks in train (cell 10)
    top_stocks = df_train["stock_id"].value_counts().head(N_STOCKS).index.tolist()
    df_train = df_train[df_train["stock_id"].isin(top_stocks)].reset_index(drop=True)
    df_val = df_val[df_val["stock_id"].isin(top_stocks)].reset_index(drop=True)
    df_test = df_test[df_test["stock_id"].isin(top_stocks)].reset_index(drop=True)
    print(f"Reduced to {N_STOCKS} stocks | train {df_train.shape} val {df_val.shape} test {df_test.shape}")

    return df_train, df_val, df_test


# ----------------------------------------------------------------------------
# HMD construction (cell 13, verbatim) -- used directly for the "buggy" build and
# reused (train branch) on the concatenated frame for the "blaw" build.
# ----------------------------------------------------------------------------
def build_intraday_pattern(df, train_stats=None):
    """Final Code.ipynb cell 13, unchanged."""
    is_train = train_stats is None
    if is_train:
        train_stats = {}

    df = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).copy()

    df["target_diff"] = df["target"] - df["volume"]
    df["target_log"] = np.log((df["target"] + EPS) / (df["volume"] + EPS))

    pattern_cols = ["stock_time_mean_diff", "time_mean_diff",
                    "stock_time_mean_log", "time_mean_log",
                    "stock_time_mean_vol", "time_mean_vol",
                    "vol_vs_intraday"]
    df = df.drop(columns=[c for c in pattern_cols if c in df.columns], errors="ignore")

    if is_train:
        daily = df.groupby(["stock_id", "seconds_in_bucket", "date_id"])["target_diff"].mean().reset_index()
        daily.columns = ["stock_id", "seconds_in_bucket", "date_id", "_d"]
        daily = daily.sort_values(["stock_id", "seconds_in_bucket", "date_id"])
        daily["stock_time_mean_diff"] = (
            daily.groupby(["stock_id", "seconds_in_bucket"])["_d"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        df = df.merge(daily[["stock_id", "seconds_in_bucket", "date_id", "stock_time_mean_diff"]],
                      on=["stock_id", "seconds_in_bucket", "date_id"], how="left")

        daily_t = df.groupby(["seconds_in_bucket", "date_id"])["target_diff"].mean().reset_index()
        daily_t.columns = ["seconds_in_bucket", "date_id", "_d"]
        daily_t = daily_t.sort_values(["seconds_in_bucket", "date_id"])
        daily_t["time_mean_diff"] = (
            daily_t.groupby("seconds_in_bucket")["_d"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        df = df.merge(daily_t[["seconds_in_bucket", "date_id", "time_mean_diff"]],
                      on=["seconds_in_bucket", "date_id"], how="left")
        df["stock_time_mean_diff"] = df["stock_time_mean_diff"].fillna(df["time_mean_diff"])

        daily_l = df.groupby(["stock_id", "seconds_in_bucket", "date_id"])["target_log"].mean().reset_index()
        daily_l.columns = ["stock_id", "seconds_in_bucket", "date_id", "_d"]
        daily_l = daily_l.sort_values(["stock_id", "seconds_in_bucket", "date_id"])
        daily_l["stock_time_mean_log"] = (
            daily_l.groupby(["stock_id", "seconds_in_bucket"])["_d"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        df = df.merge(daily_l[["stock_id", "seconds_in_bucket", "date_id", "stock_time_mean_log"]],
                      on=["stock_id", "seconds_in_bucket", "date_id"], how="left")

        daily_tl = df.groupby(["seconds_in_bucket", "date_id"])["target_log"].mean().reset_index()
        daily_tl.columns = ["seconds_in_bucket", "date_id", "_d"]
        daily_tl = daily_tl.sort_values(["seconds_in_bucket", "date_id"])
        daily_tl["time_mean_log"] = (
            daily_tl.groupby("seconds_in_bucket")["_d"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        df = df.merge(daily_tl[["seconds_in_bucket", "date_id", "time_mean_log"]],
                      on=["seconds_in_bucket", "date_id"], how="left")
        df["stock_time_mean_log"] = df["stock_time_mean_log"].fillna(df["time_mean_log"])

        daily_v = df.groupby(["stock_id", "seconds_in_bucket", "date_id"])["volume"].mean().reset_index()
        daily_v.columns = ["stock_id", "seconds_in_bucket", "date_id", "_d"]
        daily_v = daily_v.sort_values(["stock_id", "seconds_in_bucket", "date_id"])
        daily_v["stock_time_mean_vol"] = (
            daily_v.groupby(["stock_id", "seconds_in_bucket"])["_d"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        df = df.merge(daily_v[["stock_id", "seconds_in_bucket", "date_id", "stock_time_mean_vol"]],
                      on=["stock_id", "seconds_in_bucket", "date_id"], how="left")

        daily_tv = df.groupby(["seconds_in_bucket", "date_id"])["volume"].mean().reset_index()
        daily_tv.columns = ["seconds_in_bucket", "date_id", "_d"]
        daily_tv = daily_tv.sort_values(["seconds_in_bucket", "date_id"])
        daily_tv["time_mean_vol"] = (
            daily_tv.groupby("seconds_in_bucket")["_d"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        df = df.merge(daily_tv[["seconds_in_bucket", "date_id", "time_mean_vol"]],
                      on=["seconds_in_bucket", "date_id"], how="left")
        df["stock_time_mean_vol"] = df["stock_time_mean_vol"].fillna(df["time_mean_vol"])

        df["vol_vs_intraday"] = df["volume"] / (df["stock_time_mean_vol"] + EPS)

        train_stats["stock_time_mean_diff"] = df.groupby(["stock_id", "seconds_in_bucket"])["target_diff"].mean().to_dict()
        train_stats["time_mean_diff"] = df.groupby("seconds_in_bucket")["target_diff"].mean().to_dict()
        train_stats["stock_time_mean_log"] = df.groupby(["stock_id", "seconds_in_bucket"])["target_log"].mean().to_dict()
        train_stats["time_mean_log"] = df.groupby("seconds_in_bucket")["target_log"].mean().to_dict()
        train_stats["stock_time_mean_vol"] = df.groupby(["stock_id", "seconds_in_bucket"])["volume"].mean().to_dict()
        train_stats["time_mean_vol"] = df.groupby("seconds_in_bucket")["volume"].mean().to_dict()
    else:
        df["stock_time_mean_diff"] = df.set_index(["stock_id", "seconds_in_bucket"]).index.map(
            train_stats["stock_time_mean_diff"]).values
        df["time_mean_diff"] = df["seconds_in_bucket"].map(train_stats["time_mean_diff"]).fillna(0)
        df["stock_time_mean_diff"] = df["stock_time_mean_diff"].fillna(df["time_mean_diff"])

        df["stock_time_mean_log"] = df.set_index(["stock_id", "seconds_in_bucket"]).index.map(
            train_stats["stock_time_mean_log"]).values
        df["time_mean_log"] = df["seconds_in_bucket"].map(train_stats["time_mean_log"]).fillna(0)
        df["stock_time_mean_log"] = df["stock_time_mean_log"].fillna(df["time_mean_log"])

        df["stock_time_mean_vol"] = df.set_index(["stock_id", "seconds_in_bucket"]).index.map(
            train_stats["stock_time_mean_vol"]).values
        df["time_mean_vol"] = df["seconds_in_bucket"].map(train_stats["time_mean_vol"]).fillna(0)
        df["stock_time_mean_vol"] = df["stock_time_mean_vol"].fillna(df["time_mean_vol"])

        df["vol_vs_intraday"] = df["volume"] / (df["stock_time_mean_vol"] + EPS)

    return df, train_stats


def build_hmd_buggy(df_train, df_val, df_test):
    """Original asymmetric build: train rolling, val/test frozen full-train mean."""
    df_train, train_stats = build_intraday_pattern(df_train, train_stats=None)
    df_val, _ = build_intraday_pattern(df_val, train_stats=train_stats)
    df_test, _ = build_intraday_pattern(df_test, train_stats=train_stats)
    return df_train, df_val, df_test


def build_hmd_alaw(df_train, df_val, df_test):
    """
    A-law symmetric build (Final Code.ipynb cell 54 `rebuild_hmd_symmetric`).
    ONE frozen full-train mean per (stock, second) applied to ALL three splits,
    including train. Note: for val/test this is identical to the buggy build; the
    only thing A-law changes versus buggy is TRAIN (rolling -> frozen). This is the
    fix the previous team actually wired in (cell 81 reports its effect).
    """
    tr, va, te = df_train.copy(), df_val.copy(), df_test.copy()
    for d in (tr, va, te):
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
        return d

    return _apply(tr), _apply(va), _apply(te)


def build_hmd_blaw(df_train, df_val, df_test):
    """
    B-law symmetric build. Apply the cell-13 train-branch causal rolling estimator
    to the concatenated, date-ordered timeline so train/val/test share one identical
    construction. Because shift(1) keeps every day strictly past, a val/test day's
    10-day window simply rolls forward over the most recent prior days (which by then
    include late-train and earlier-val days). No future information is used.
    """
    parts = []
    for name, df in (("train", df_train), ("val", df_val), ("test", df_test)):
        d = df.copy()
        d["split"] = name
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)

    # Run the *train* branch (train_stats=None) over the full timeline.
    combined, _ = build_intraday_pattern(combined, train_stats=None)

    out = []
    for name in ("train", "val", "test"):
        d = combined[combined["split"] == name].drop(columns=["split"]).reset_index(drop=True)
        out.append(d)
    return out[0], out[1], out[2]


# ----------------------------------------------------------------------------
# Stage-1 residual construction (cell 15)
# ----------------------------------------------------------------------------
def generate_stage1_predictions(df):
    df = df.copy()
    df["stage1_pred_diff"] = df["stock_time_mean_diff"]
    df["stage1_pred_vol"] = df["volume"] + df["stock_time_mean_diff"]
    df["return_diff"] = df["target_diff"] - df["stage1_pred_diff"]
    return df


# ----------------------------------------------------------------------------
# Feature selection (cell 21)
# ----------------------------------------------------------------------------
def add_interactions_and_features(df_train, df_val, df_test):
    for df in (df_train, df_val, df_test):
        if "imbalance_ratio" not in df.columns:
            df["imbalance_ratio"] = df["imbalance_size"] / (df["matched_size"] + EPS)
        if "imbalance_pct" not in df.columns:
            df["imbalance_pct"] = df["imbalance_size"] / (df["volume"] + EPS)
        if "bid_ask_spread" not in df.columns and "ask_price" in df.columns and "bid_price" in df.columns:
            df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]

    nn_features = [
        "seconds_in_bucket",
        "imbalance_size", "imbalance_buy_sell_flag", "matched_size",
        "reference_price", "wap",
        "stock_time_mean_diff", "stock_time_mean_vol", "vol_vs_intraday",
        "imbalance_ratio", "imbalance_pct",
    ]
    if "bid_ask_spread" in df_train.columns:
        nn_features.append("bid_ask_spread")
    return nn_features


# ----------------------------------------------------------------------------
# Sequence construction + scaling (cell 22, returns tensors so loaders can be
# rebuilt per seed with an independent generator).
# ----------------------------------------------------------------------------
def prepare_nn_data(df_train, df_val, df_test, feature_cols, target_col, seq_len=20):
    cols_needed = feature_cols + [target_col, "stock_id_original", "date_id_original"]
    df_tr = df_train.dropna(subset=cols_needed).copy()
    df_va = df_val.dropna(subset=cols_needed).copy()
    df_te = df_test.dropna(subset=cols_needed).copy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(df_tr[feature_cols])
    scaler_y.fit(df_tr[[target_col]])

    for df in (df_tr, df_va, df_te):
        df[feature_cols] = scaler_X.transform(df[feature_cols])
        df[target_col] = scaler_y.transform(df[[target_col]])

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

    X_tr, y_tr, idx_tr = make_sequences(df_tr)
    X_va, y_va, idx_va = make_sequences(df_va)
    X_te, y_te, idx_te = make_sequences(df_te)
    print(f"  sequences | train {X_tr.shape} val {X_va.shape} test {X_te.shape}")

    return {
        "X_tr": X_tr, "y_tr": y_tr, "idx_tr": idx_tr,
        "X_va": X_va, "y_va": y_va, "idx_va": idx_va,
        "X_te": X_te, "y_te": y_te, "idx_te": idx_te,
        "df_tr": df_tr, "df_va": df_va, "df_te": df_te,
        "scaler_y": scaler_y,
    }


# ----------------------------------------------------------------------------
# Model (cell 23, GRU only) -- .squeeze() left unchanged on purpose (review #24)
# ----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()


# ----------------------------------------------------------------------------
# Training + evaluation (cell 24, verbatim logic)
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3,
                weight_decay=1e-4, max_grad_norm=1.0, patience=5):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        model.eval()
        t_loss = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                t_loss += criterion(model(xb), yb).item()
        t_loss /= len(train_loader)
        train_losses.append(t_loss)

        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                v_loss += criterion(model(xb), yb).item()
        v_loss /= len(val_loader)
        val_losses.append(v_loss)

        scheduler.step(v_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1:2d}/{epochs} | Train: {t_loss:.6f} | Val: {v_loss:.6f} | LR: {lr_now:.6f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_model(model, test_loader, df_te, idx_te, scaler_y, target_col):
    model.eval()
    preds_s, trues_s = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_s.append(model(xb).cpu().numpy())
            trues_s.append(yb.cpu().numpy())
    preds_s = np.concatenate(preds_s)
    trues_s = np.concatenate(trues_s)

    preds_real = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    trues_real = scaler_y.inverse_transform(trues_s.reshape(-1, 1)).flatten()

    r2 = r2_score(trues_real, preds_real)
    rmse = np.sqrt(mean_squared_error(trues_real, preds_real))
    mae = mean_absolute_error(trues_real, preds_real)
    smape = np.mean(2 * np.abs(trues_real - preds_real) / (np.abs(trues_real) + np.abs(preds_real) + EPS)) * 100

    stock_ids = df_te.loc[idx_te, "stock_id_original"].values
    date_ids = df_te.loc[idx_te, "date_id_original"].values
    ic_vals = []
    for sid in np.unique(stock_ids):
        sm = stock_ids == sid
        for did in np.unique(date_ids[sm]):
            dm = sm & (date_ids == did)
            if dm.sum() > 3:
                p_std = np.std(preds_real[dm])
                t_std = np.std(trues_real[dm])
                if p_std > EPS and t_std > EPS:
                    ic_vals.append(np.corrcoef(preds_real[dm], trues_real[dm])[0, 1])
    ic = np.nanmean(ic_vals) if ic_vals else 0.0

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "SMAPE": smape, "IC": ic}, preds_real, trues_real


# ----------------------------------------------------------------------------
# Level-space comparison (cell 37 / 39: full_metrics)
# ----------------------------------------------------------------------------
def full_metrics(actual, pred, te_df):
    valid = np.isfinite(pred) & np.isfinite(actual)
    a, p = actual[valid], pred[valid]
    r2 = r2_score(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mae = mean_absolute_error(a, p)
    smape = np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p) + EPS)) * 100
    sids = te_df["stock_id_original"].values[valid]
    dids = te_df["date_id_original"].values[valid]
    ics = []
    for sid in np.unique(sids):
        sm = sids == sid
        for did in np.unique(dids[sm]):
            dm = sm & (dids == did)
            if dm.sum() > 3 and np.std(a[dm]) > EPS and np.std(p[dm]) > EPS:
                ics.append(np.corrcoef(a[dm], p[dm])[0, 1])
    ic = np.nanmean(ics) if ics else 0.0
    return {"R2": r2, "RMSE": rmse, "SMAPE": smape, "MAE": mae, "IC": ic}


# ----------------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# One full variant run
# ----------------------------------------------------------------------------
def run_variant(variant, df_train, df_val, df_test, seeds):
    print("=" * 78)
    print(f"VARIANT: {variant}")
    print("=" * 78)

    if variant == "buggy":
        dtr, dva, dte = build_hmd_buggy(df_train.copy(), df_val.copy(), df_test.copy())
    elif variant == "alaw":
        dtr, dva, dte = build_hmd_alaw(df_train.copy(), df_val.copy(), df_test.copy())
    elif variant == "blaw":
        dtr, dva, dte = build_hmd_blaw(df_train.copy(), df_val.copy(), df_test.copy())
    else:
        raise ValueError(variant)

    dtr = generate_stage1_predictions(dtr)
    dva = generate_stage1_predictions(dva)
    dte = generate_stage1_predictions(dte)

    nn_features = add_interactions_and_features(dtr, dva, dte)
    data = prepare_nn_data(dtr, dva, dte, nn_features, "return_diff", seq_len=SEQ_LEN)

    # HMD-only and persistence baselines (seed independent), evaluated on the SAME
    # sequence-aligned rows the GRU predicts on (idx_te). This is the apples-to-apples
    # population that produced the buggy 0.8240 headline (cell 37).
    idx_te = data["idx_te"]
    te_orig = dte.loc[idx_te].copy()
    actual = te_orig["target"].values
    stage1_pred = te_orig["stage1_pred_vol"].values
    last_value_pred = te_orig["volume"].values

    m_hmd = full_metrics(actual, stage1_pred, te_orig)
    m_persist = full_metrics(actual, last_value_pred, te_orig)
    print(f"  HMD-only (level)     IC {m_hmd['IC']:.4f}  R2 {m_hmd['R2']:.4f}")
    print(f"  persistence (level)  IC {m_persist['IC']:.4f}  R2 {m_persist['R2']:.4f}")

    # Tensors -> device once (deterministic preprocessing).
    X_tr = torch.tensor(data["X_tr"]).to(DEVICE)
    y_tr = torch.tensor(data["y_tr"]).to(DEVICE)
    X_va = torch.tensor(data["X_va"]).to(DEVICE)
    y_va = torch.tensor(data["y_va"]).to(DEVICE)
    X_te = torch.tensor(data["X_te"]).to(DEVICE)
    y_te = torch.tensor(data["y_te"]).to(DEVICE)
    n_features = len(nn_features)

    seed_rows = []
    for seed in seeds:
        print(f"\n--- {variant} | seed {seed} ---")
        set_all_seeds(seed)

        gen = torch.Generator()
        gen.manual_seed(seed)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                                  shuffle=True, generator=gen)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE)

        # GRU-small (cell 26 config)
        model = GRUModel(input_size=n_features, hidden_size=32, num_layers=1, dropout=0.1).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        train_model(model, train_loader, val_loader,
                    epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                    max_grad_norm=MAX_GRAD_NORM, patience=PATIENCE)

        # Val metrics (return space) -- for model-selection lineage with the headline.
        val_metrics, _, _ = evaluate_model(model, val_loader, data["df_va"], data["idx_va"],
                                           data["scaler_y"], "return_diff")
        # Test metrics (return space)
        test_metrics, preds_return, _ = evaluate_model(model, test_loader, data["df_te"], idx_te,
                                                       data["scaler_y"], "return_diff")

        # Level space: HMD + GRU = stage1_pred_vol + predicted return residual
        stage1_plus_2 = stage1_pred + preds_return
        m_hmdgru = full_metrics(actual, stage1_plus_2, te_orig)

        row = {
            "variant": variant, "seed": seed, "n_params": n_params,
            "val_IC_return": val_metrics["IC"],
            "test_IC_return": test_metrics["IC"], "test_R2_return": test_metrics["R2"],
            "HMDonly_IC_level": m_hmd["IC"], "HMDonly_R2_level": m_hmd["R2"],
            "HMDGRU_IC_level": m_hmdgru["IC"], "HMDGRU_R2_level": m_hmdgru["R2"],
            "IC_delta_level": m_hmdgru["IC"] - m_hmd["IC"],
            "R2_delta_level": m_hmdgru["R2"] - m_hmd["R2"],
            "persistence_IC_level": m_persist["IC"], "persistence_R2_level": m_persist["R2"],
        }
        seed_rows.append(row)
        print(f"  val IC(return) {val_metrics['IC']:.4f} | test IC(return) {test_metrics['IC']:.4f} "
              f"R2 {test_metrics['R2']:.4f}")
        print(f"  level: HMD+GRU IC {m_hmdgru['IC']:.4f}  delta {row['IC_delta_level']:+.4f} | "
              f"R2 {m_hmdgru['R2']:.4f}  delta {row['R2_delta_level']:+.4f} | {time.time() - t0:.0f}s")

    return seed_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--variants", nargs="+", default=["buggy", "blaw"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "outputs")
    args = ap.parse_args()

    print(f"device={DEVICE}  data_dir={args.data_dir}")
    print(f"seeds={args.seeds}  variants={args.variants}")

    df_train, df_val, df_test = load_data(args.data_dir)

    all_rows = []
    for variant in args.variants:
        all_rows.extend(run_variant(variant, df_train, df_val, df_test, args.seeds))

    args.out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(all_rows)
    stamp = "_".join(str(s) for s in args.seeds)
    csv_path = args.out / f"results_seeds_{stamp}.csv"
    res.to_csv(csv_path, index=False)
    print(f"\nsaved raw rows -> {csv_path}")

    # Summary: mean +- std per variant for the headline level-space numbers.
    print("\n" + "=" * 78)
    print("SUMMARY (level space, mean +/- std over seeds)")
    print("=" * 78)
    summary = {}
    for variant, g in res.groupby("variant"):
        def ms(col):
            return float(g[col].mean()), float(g[col].std(ddof=0))
        summary[variant] = {
            "n_seeds": int(len(g)),
            "HMDonly_IC": g["HMDonly_IC_level"].iloc[0],
            "HMDGRU_IC_mean": ms("HMDGRU_IC_level")[0], "HMDGRU_IC_std": ms("HMDGRU_IC_level")[1],
            "IC_delta_mean": ms("IC_delta_level")[0], "IC_delta_std": ms("IC_delta_level")[1],
            "HMDonly_R2": g["HMDonly_R2_level"].iloc[0],
            "HMDGRU_R2_mean": ms("HMDGRU_R2_level")[0], "HMDGRU_R2_std": ms("HMDGRU_R2_level")[1],
            "R2_delta_mean": ms("R2_delta_level")[0], "R2_delta_std": ms("R2_delta_level")[1],
            "test_IC_return_mean": ms("test_IC_return")[0], "test_IC_return_std": ms("test_IC_return")[1],
            "persistence_IC": g["persistence_IC_level"].iloc[0],
        }
        s = summary[variant]
        print(f"\n[{variant}]  (n={s['n_seeds']})")
        print(f"  HMD-only IC (level):     {s['HMDonly_IC']:.4f}")
        print(f"  HMD+GRU  IC (level):     {s['HMDGRU_IC_mean']:.4f} +/- {s['HMDGRU_IC_std']:.4f}")
        print(f"  IC delta (level):        {s['IC_delta_mean']:+.4f} +/- {s['IC_delta_std']:.4f}")
        print(f"  HMD-only R2 (level):     {s['HMDonly_R2']:.4f}")
        print(f"  HMD+GRU  R2 (level):     {s['HMDGRU_R2_mean']:.4f} +/- {s['HMDGRU_R2_std']:.4f}")
        print(f"  GRU test IC (return):    {s['test_IC_return_mean']:.4f} +/- {s['test_IC_return_std']:.4f}")

    with open(args.out / f"summary_seeds_{stamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary -> {args.out / f'summary_seeds_{stamp}.json'}")


if __name__ == "__main__":
    main()
