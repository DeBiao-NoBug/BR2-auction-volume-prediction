# Part B: 5-fold time-series CV on A-law vs B-law HMD

Re-checks the previous team's CV-based caveats on the corrected pipeline: that the
cross-validated mean IC is far below the single-split headline (review #15) and that the
early folds are near-noise with large fold-to-fold spread (review #12).

## Setup

- CV pool: train + validation (date_id 0-407, 408 trading days). Test held out.
- Scheme: standard expanding-window TimeSeriesSplit, 5 folds, split by trading day.
  Fold k trains on the first k segments and validates on segment k+1.
  Fold sizes: train 68/136/204/272/340 days, validation 68 days each.
- Two HMD constructions compared, same single seed (42) and same fold split, so the only
  difference is the HMD ruler:
  - A-law: frozen full-fold-train mean per fold (reproduces the previous team's CV).
  - B-law: causal 10-day rolling mean per fold (our clean construction).
- Metric: return-space validation IC. No embargo or purge (review #10 not addressed here),
  so the CV mean is still mildly optimistic.

## Summary: CV mean +/- std IC over 5 folds

| model | A-law (frozen) | B-law (rolling) |
|---|---|---|
| GRU-small | 0.1782 +/- 0.0667 | 0.2514 +/- 0.0399 |
| GRU-medium | 0.1769 +/- 0.0639 | 0.2409 +/- 0.0349 |
| GRU-large | 0.1689 +/- 0.0787 | 0.2410 +/- 0.0417 |
| TCN-small | 0.1911 +/- 0.0730 | 0.2511 +/- 0.0416 |
| TCN-medium | 0.1903 +/- 0.0690 | 0.2462 +/- 0.0337 |
| TCN-large | 0.1726 +/- 0.0559 | 0.2523 +/- 0.0416 |

## Full per-fold validation IC

A-law (frozen):

| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean | std |
|---|---|---|---|---|---|---|---|
| GRU-small | 0.0726 | 0.1306 | 0.2539 | 0.2093 | 0.2246 | 0.1782 | 0.0667 |
| GRU-medium | 0.0655 | 0.1513 | 0.2383 | 0.1961 | 0.2334 | 0.1769 | 0.0639 |
| GRU-large | 0.0588 | 0.0898 | 0.2453 | 0.2126 | 0.2382 | 0.1689 | 0.0787 |
| TCN-small | 0.0677 | 0.1539 | 0.2415 | 0.2194 | 0.2729 | 0.1911 | 0.0730 |
| TCN-medium | 0.0820 | 0.1347 | 0.2469 | 0.2397 | 0.2482 | 0.1903 | 0.0690 |
| TCN-large | 0.0972 | 0.1224 | 0.2232 | 0.1781 | 0.2421 | 0.1726 | 0.0559 |

B-law (rolling):

| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean | std |
|---|---|---|---|---|---|---|---|
| GRU-small | 0.2001 | 0.2102 | 0.2961 | 0.2599 | 0.2908 | 0.2514 | 0.0399 |
| GRU-medium | 0.1873 | 0.2140 | 0.2636 | 0.2578 | 0.2819 | 0.2409 | 0.0349 |
| GRU-large | 0.1886 | 0.1963 | 0.2796 | 0.2505 | 0.2898 | 0.2410 | 0.0417 |
| TCN-small | 0.2018 | 0.2011 | 0.2747 | 0.2770 | 0.3009 | 0.2511 | 0.0416 |
| TCN-medium | 0.1993 | 0.2121 | 0.2793 | 0.2646 | 0.2757 | 0.2462 | 0.0337 |
| TCN-large | 0.1978 | 0.2091 | 0.2703 | 0.2792 | 0.3050 | 0.2523 | 0.0416 |

## Reproduction anchor (A-law)

A-law GRU-small CV mean is 0.1782 +/- 0.0667, against the previous team's reported
0.1885 +/- 0.0677 (cell 72). The per-fold shape matches as well: fold 1 near noise
(0.0726 here, 0.0784 reported), rising to roughly 0.22 to 0.25 by the later folds. The
small offset is consistent with single-seed and library-version differences. This confirms
the CV framework is faithful, so the B-law numbers can be trusted.

## Findings

1. Under the clean B-law construction the CV mean rises by about 0.07 IC on every model
   (GRU-small 0.1782 to 0.2514), a systematic shift across all 6 models and all 5 folds,
   not a single lucky draw.

2. Review #15 (CV mean far below the single-split headline) is largely an artifact of the
   old ruler. Single-split GRU-small under B-law is about 0.295 (Part A, 5-seed mean); the
   B-law CV mean is 0.2514, only about 15% lower. The previous team's dramatic gap (single
   0.2845 to CV 0.1885, about 34%) was produced under the frozen HMD. On the clean pipeline
   the single split is still mildly optimistic, but the honest CV number is about 0.25, not 0.19.

3. Review #12 (early folds near-noise, large spread) is also largely an old-ruler artifact.
   Fold 1 GRU-small jumps from 0.0726 under A-law to 0.2001 under B-law, and the fold-to-fold
   std drops from 0.0667 to 0.0399. The mechanism: the frozen mean in fold 1 uses only that
   fold's short training history (68 days) and is stale by the time it is applied to the
   validation days, while the 10-day rolling mean only needs the most recent days and stays
   fresh even when the fold's total history is short. The rolling estimator is robust to
   data-starved early folds; the frozen one is not.

4. The single-split headline corresponds to the easiest CV fold. Fold 5 validates on days
   340-407, almost the same period as the original single-split validation (336-407), and uses
   the most training history. Its B-law IC (0.2908) is close to the single-split value (about
   0.295). The CV mean is lower mainly because it also averages in the earlier, data-poorer folds.

5. Model ranking in CV is again a statistical tie, consistent with Part A. The B-law CV means
   span 0.2409 to 0.2523 across the 6 models, a 0.011 range against fold stds of about 0.04.
   GRU-small (0.2514) is nominally second to TCN-large (0.2523), a gap of 0.0009. No model is
   separable from the others.

## Caveats

- Single seed for CV. The A-law vs B-law gap is large and systematic across every model and
  fold, so it is not plausibly seed noise, but the reported spread is fold-to-fold, not
  seed-to-seed.
- Standard TimeSeriesSplit with no embargo or purge (review #10). Adjacent folds touch at the
  boundary, so the CV mean is still mildly optimistic and should not be quoted as a final
  strict number.
- Per-epoch training curves from the run are not reproduced here; only result-level numbers are
  retained. The notebook also saves machine-readable `cv_rows.csv` and `cv_summary.json`.
