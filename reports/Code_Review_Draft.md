# Code Review of Inherited Pipeline: Preliminary Findings

**Status**: Draft, pending verification. Each item has been identified through a code-level pass over `Final Code.ipynb`. None has been independently re-verified against a live notebook run yet. A finalized version will follow after weekend verification.

**Date**: 2026-06-12
**Source under review**: `BR_2/Final Code.ipynb` (the inherited notebook)
**Companion document**: `BR_2/Final Report.docx` (the inherited written report)

## Summary

A close reading of the inherited notebook surfaced 29 items that may affect the credibility, accuracy, or interpretation of the previously reported results. They span eight categories: data scope, feature engineering, loss function, evaluation metric, cross-validation, regime analysis, reporting and narrative, model selection, model implementation, and reproducibility.

Severity is assigned on the basis of impact to the headline conclusions of the inherited Final Report, not engineering difficulty.

- High (9 items): directly weaken the trustworthiness of headline numbers or main narrative claims.
- Medium (16 items): require correction or contextual qualification before downstream work proceeds.
- Low (4 items): minor inconsistencies, latent edge cases, or framing notes.

The most consequential finding is item 21, which suggests that the inherited team's regime analysis structurally tests the wrong rows. The systematic underestimation pattern they attributed to the "transition regime" actually peaks in the ramp-up window 30 to 60 seconds earlier. Other high-severity items concern the scope of evaluation (20 stocks rather than the full 200), the asymmetric HMD feature build that was patched late and never propagated to the main pipeline, and the use of level-space metrics whose values are dominated by trivial baselines.

## Findings Table

| # | Category | Description | Evidence | Severity | Possible Fix |
|---|---|---|---|---|---|
| 1 | Data scope | `N_STOCKS = 20` is hardcoded. All experiments use only the 20 most frequently appearing (most liquid) stocks. The Final Report describes the dataset as "approximately 200 NASDAQ-listed stocks." | Cell 2 `N_STOCKS = 20`; Cell 10 output "Reduced to 20 stocks" | High | Re-run on the full 200-stock universe; also report results split by liquidity tier. |
| 2 | Feature engineering | The HMD feature is built asymmetrically. TRAIN rows use an expanding window with shift(1), while VAL and TEST rows use the full-sample train mean. Same column name, different statistical quality across splits. | Cell 54 internal note acknowledges this as "the real problem behind 'test > val' and 'train loss > val loss.'" | High | Use the `rebuild_hmd_symmetric` function from Cell 54: freeze a single train-only statistic and apply it identically to all three splits. |
| 3 | Feature engineering | The HMD fix (dated 2026-04-14) only takes effect for the extended experiments. The main pipeline (Parts 5 and 6) still uses the buggy HMD, which is the source of all headline numbers. Under the fixed HMD, GRU-small Val IC drops from 0.2845 to 0.2137. | Cell 33 (buggy, 0.2845) vs Cell 81 (fixed, 0.2137); Cell 54 contains an explicit `[FIX 2026-04-14]` marker. | High | Re-run the main pipeline with the symmetric HMD and update all reported numbers. |
| 4 | Feature engineering | HMD uses a 10-day rolling mean. In a right-skewed window like transition, this is sensitive to outlier days. | Cell 13 `rolling(10, min_periods=1).mean()` | Medium | Try median or trimmed mean; consider a wider window with weighting. |
| 5 | Loss function | Symmetric losses (MSE, Huber) systematically underestimate in the right-skewed transition window. Biases are -1.2M (MSE) and -1.7M (Huber). | Cell 65 output. | Medium | Asymmetric Huber halves the bias in Cell 65 (-0.67M); extend it to the full pipeline. |
| 6 | Loss function | Asymmetric Huber was only tested on a transition-target subset using a simple MLP. The main training routine (Cell 24) still uses `nn.HuberLoss(delta=1.0)` by default. | Cell 65 vs Cell 24. | Medium | Add `AsymmetricHuberLoss` to `train_model` and retrain GRU-small to confirm bias reduction at scale. |
| 7 | Evaluation metric | The headline IC of 0.82 is computed in level space, where per-stock-day Pearson correlation is inflated by the strong intraday U-shape. The honest return-space IC is 0.19 to 0.25. | Cell 79 self-acknowledges this ("look artificially high"); Cell 39 (0.82) vs Cell 72 (0.19). | High | Use return-space IC as the headline metric and level-space as supplementary. |
| 8 | Evaluation metric | The headline R² of 0.9959 is in level space, where a trivial persistence baseline already explains roughly 99% of variance. Honest return-space R² is 0.13 to 0.18. | Cell 39 R² 0.9953 (level) vs Cell 15 R² 0.6116 (target_diff) vs Cell 56 R² 0.1586 (return). | High | Report return-space R² as headline and explicitly contrast it with the trivial persistence baseline. |
| 9 | Evaluation metric | The transition regime IC in the main evaluation is NaN. With only 3 to 4 buckets per stock-day, per-stock-day Pearson cannot be computed reliably. | Cell 50 output: `IC: nan` for both HMD only and HMD + GRU. | Medium | Use global Pearson (Cell 54's `transition_ic`) or aggregate by stock before computing IC. |
| 10 | Cross-validation | `TimeSeriesSplit` uses no embargo. With day-level residual autocorrelation, subtle leakage at fold boundaries is plausible. | Cell 71 uses standard `TimeSeriesSplit(n_splits=5)`. | Medium | Add a buffer of days at each fold boundary, or upgrade to CPCV. |
| 11 | Cross-validation | All CV splits are by date. No leave-one-stock-out or group k-fold. Generalization to unseen stocks is completely untested. | Cell 71: `tscv.split(date_array)`. | Medium | Add stock-level CV by holding out a random subset of stocks at a time. |
| 12 | Cross-validation | Fold 1 IC is 0.0784 (near noise) while Fold 5 IC is 0.2447. Results are highly sensitive to training data volume. | Cell 72 per-fold breakdown. | Medium | Cannot be fixed structurally, but the report should explicitly flag the dependency on at least 100+ training days. |
| 13 | Reporting / narrative | Cell 51 markdown claims that Stage 2 hurts in the transition regime. Cell 50 data shows Stage 2 actually improves transition RMSE by 6.5%, the largest relative improvement of any regime. The "regime-blend" intervention in Cell 52 reverses this gain. | Cell 51 markdown vs Cell 50/52 output (4,995,644 vs 5,340,529). | High | Correct the relevant section of the Final Report. Distinguish RMSE improvement (Stage 2 helps) from IC improvement (mixed). |
| 14 | Reporting / narrative | "Transition" is used in two incompatible senses. Cell 50 defines it by current bucket (280-310s). Cell 65 defines it by target bucket (60 seconds ahead lands in 280-310s). These cover different rows. | Cell 50 `if sec < 310` vs Cell 65 `is_transition_target`. | Medium | Standardize terminology: use `in-transition` versus `transition-target`. |
| 15 | Reporting / narrative | The CV mean IC (0.1885) is 34% lower than the headline single-run Val IC (0.2845). The report leads with the higher number. | Cell 33 (0.2845) vs Cell 72 (0.1885). | High | Promote CV mean ± std to the headline; single-run val/test as supplementary. |
| 16 | Reporting / narrative | The phrase "approximately 200 NASDAQ-listed stocks" is technically misleading because every reported experiment uses 20 stocks. | Final Report Section 1.1 vs Cell 2. | High | Rephrase as "20 most liquid stocks as a pilot subset." Explicitly list 200-stock validation as future work. |
| 17 | Regime analysis | The transition window contains only 3 or 4 buckets per stock-day. Stage 2 training data in transition is structurally sparse compared to ramp-up and plateau. | Cell 50 (10-second sampling); N=4,380 vs ramp-up 11,680 vs plateau 26,280. | Medium | Structural limit. Consider widening the window or switching to a volume-conditional regime definition. |
| 18 | Regime analysis | Regime classification is purely time-based; all stocks share the same 280/310 cutoffs. Variation across liquidity tiers is averaged out. | Cell 50 `regime_label(sec)`. | Low | Try stock-specific cutoffs or volume-deviation thresholds. |
| 19 | Model selection | The top six models in CV span an IC range of 0.170 to 0.189, well within the fold-to-fold std (~0.06). The claim that "GRU-small is optimal" is not statistically significant. | Cell 72. | Medium | Use paired t-tests on fold ICs. Do not rank models with sub-0.01 IC differences. |
| 20 | Model selection | Cell 81 ablation output is truncated. The "WITHOUT" condition's final result is not captured. | Cell 81 output. | Low | Re-run Cell 81 to capture the full output. |
| 21 | Regime analysis / reporting | Mean prediction error peaks at `seconds_in_bucket` 240 to 260 (around -3.5M underestimation), not in the transition regime itself (280 to 310). Because the target is 60 seconds ahead, the rows actually forecasting the transition jump are in the ramp-up window. Cell 50's current-bucket regime labels test the wrong rows. The systematic underestimation that the inherited team attributed to "transition" actually peaks before transition begins. | Cell 48 Mean_Error by bucket; Cell 50 `regime_label(sec)`; Cell 65 `is_transition_target` (which correctly identifies the right rows). | High | Redo all regime analyses using target-based regime labels (Cell 65 definition) rather than current-bucket labels. |
| 22 | Evaluation metric / reporting | Single-stage GRU has higher test IC (0.8407) than two-stage (0.8388). The Final Report recommends two-stage using RMSE as the deciding metric, despite model selection across Parts 5 and 6 being based on IC. This is a possible case of metric cherry-picking. | Cell 45 output; Cell 33 model selection by IC. | Medium | Commit to one primary metric throughout. Defend two-stage on interpretability and leakage robustness, not on universal metric dominance. |
| 23 | Evaluation metric | Spike RMSE (~14.1M) is roughly 3.4 times Normal RMSE (~4.2M). The model fails systematically on rare large events, which are exactly the trading-critical moments. | Cell 48 output. | Medium | Promote Spike and Tail RMSE to secondary headline metrics. Consider quantile or robust losses targeted at the tail. |
| 24 | Model implementation | `GRUModel`, `LSTMModel`, and `TCNModel` use bare `.squeeze()` in `forward`. When the batch size happens to be 1 (end-of-epoch remainder), this collapses a (1,) tensor to a 0-dim scalar and breaks the loss. `MLPModel` correctly uses `.squeeze(-1)`. | Cell 23. | Low | Change to `.squeeze(-1)` for consistency. |
| 25 | Evaluation metric | `full_metrics` in Cell 37 returns IC = 0.0 when computation fails, while Cell 50 returns NaN in the same situation. Inconsistent handling. The first silently disguises "cannot compute" as "IC is zero." | Cell 37 vs Cell 50. | Medium | Standardize on NaN. Distinguish NaN from real zero explicitly downstream. |
| 26 | Regime analysis / context | Predictions only cover `seconds_in_bucket` 200 to 480 (29 buckets) because `seq_len=20` requires 20 buckets of history. The 280-310 transition window represents only about 10% of predictions. | Cell 48 output range; Cell 2 `SEQ_LEN = 20`. | Low | Clarify the actual prediction coverage in the report. |
| 27 | Reproducibility | The same GRU-small configuration produces different Val IC at different points in the notebook: 0.2845 (Cell 26, buggy HMD), 0.2187 (Cell 58, fixed HMD), 0.2137 (Cell 81, fixed HMD). The HMD fix accounts for most but not all of the spread. Ranking decisions based on sub-0.01 IC gaps are within noise. | Cell 26, 58, 81 outputs. | Medium | Report mean ± std across multiple seeds. Do not trust rankings with gaps below ~0.01. |
| 28 | Evaluation metric | Single-stage GRU in target_diff space achieves IC 0.94 (Cell 44). The Final Report does not surface this number, presenting only the level-space IC of 0.84 and the return-space IC of 0.25. All three are legitimate in their respective spaces, but which one is highlighted shapes the reader's impression. | Cell 44 output. | Medium | Always state which space the IC is computed in. Report multiple spaces when their differences are material. |
| 29 | Evaluation metric | "Transition IC" has at least three incompatible definitions: (a) per-stock-day Pearson, level space (Cell 50) returning NaN; (b) global Pearson, level space (Cell 58 via `transition_ic`) returning 0.9975; (c) per-stock-day Pearson, return space, never computed. Cell 58's 0.9975 in particular reads as "transition is nearly solved" but only measures the seasonality fit. | Cell 50, Cell 54 `transition_ic`, Cell 58 output. | Medium | Standardize a single definition. The 0.9975 figure either gets removed or carries an explicit caveat. |

## Recommended Verification Order

1. The nine High-severity items first. These are the ones most likely to surface in any external review of the inherited work.
2. Item 21 deserves particular attention because it suggests the inherited regime narrative is structurally misaligned with the data.
3. Items 2 and 3 should be verified together, since they form a single bug-and-non-propagation pair.
4. Items 7 and 8 should be verified together, since they share the level-space inflation root cause.
5. Items 13, 15, and 16 are wording and framing issues that affect how the inherited report reads.
6. Medium-severity items follow next, prioritized by relevance to whichever research direction we choose.
7. Low-severity items can be deferred.

## Next Steps

- Verify each item against the live notebook over Saturday and Sunday.
- Produce a finalized version of this list.
- Circulate the finalized list to the TA and the Prof first for wording confirmation before broader distribution.
- Incorporate the confirmed items into the Executive Report. Target send date: Sunday afternoon.
