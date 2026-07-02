# Part A: 12-model ranking on the leakage-free (B-law) HMD pipeline

Re-checks the previous team's "GRU-small is the best model" conclusion (code-review
items #19 and #27) on the corrected pipeline.

## Setup

- Feature pipeline: B-law HMD (one consistent causal rolling construction on all splits).
- Models: 4 families (GRU, LSTM, TCN, MLP) x 3 sizes (small, medium, large), exact configs
  inherited from the original pipeline (cells 26/28/30/32). The grid is held fixed as a
  control; it was not re-tuned for this pipeline.
- Selection metric: return-space validation IC. Test set untouched (ranking is on validation only).
- Seeds: 42, 0, 1, 2, 3, shared across all 12 models (paired: same random condition per seed,
  only the architecture differs). Weights retrained from scratch each run.
- No significance test is used (per supervisor guidance); the conclusion rests on descriptive
  paired evidence.

## Ranking by mean validation IC (5 seeds)

| rank | model | family | IC mean | IC std | params |
|---|---|---|---|---|---|
| 1 | TCN-small | TCN | 0.3038 | 0.0100 | 10,945 |
| 2 | TCN-medium | TCN | 0.2971 | 0.0082 | 65,025 |
| 3 | MLP-medium | MLP | 0.2953 | 0.0073 | 9,985 |
| 4 | GRU-small | GRU | 0.2952 | 0.0061 | 4,449 |
| 5 | MLP-small | MLP | 0.2916 | 0.0081 | 2,945 |
| 6 | TCN-large | TCN | 0.2906 | 0.0107 | 238,081 |
| 7 | MLP-large | MLP | 0.2902 | 0.0067 | 36,353 |
| 8 | LSTM-small | LSTM | 0.2886 | 0.0091 | 5,921 |
| 9 | GRU-medium | GRU | 0.2829 | 0.0114 | 40,001 |
| 10 | LSTM-large | LSTM | 0.2818 | 0.0103 | 204,929 |
| 11 | LSTM-medium | LSTM | 0.2797 | 0.0044 | 53,313 |
| 12 | GRU-large | GRU | 0.2771 | 0.0008 | 153,729 |

## Full per-seed validation IC (return space)

| model | params | seed 42 | seed 0 | seed 1 | seed 2 | seed 3 | mean | std |
|---|---|---|---|---|---|---|---|---|
| TCN-small | 10,945 | 0.2893 | 0.3079 | 0.3168 | 0.3037 | 0.3012 | 0.3038 | 0.0100 |
| TCN-medium | 65,025 | 0.2883 | 0.3025 | 0.2880 | 0.3049 | 0.3017 | 0.2971 | 0.0082 |
| MLP-medium | 9,985 | 0.3018 | 0.2923 | 0.2863 | 0.2923 | 0.3039 | 0.2953 | 0.0073 |
| GRU-small | 4,449 | 0.2994 | 0.2907 | 0.2966 | 0.2871 | 0.3020 | 0.2952 | 0.0061 |
| MLP-small | 2,945 | 0.2851 | 0.3032 | 0.2827 | 0.2927 | 0.2945 | 0.2916 | 0.0081 |
| TCN-large | 238,081 | 0.2844 | 0.2908 | 0.2958 | 0.3051 | 0.2771 | 0.2906 | 0.0107 |
| MLP-large | 36,353 | 0.2913 | 0.2931 | 0.2838 | 0.2835 | 0.2994 | 0.2902 | 0.0067 |
| LSTM-small | 5,921 | 0.2888 | 0.2943 | 0.2734 | 0.2899 | 0.2968 | 0.2886 | 0.0091 |
| GRU-medium | 40,001 | 0.2711 | 0.2892 | 0.2799 | 0.2749 | 0.2992 | 0.2829 | 0.0114 |
| LSTM-large | 204,929 | 0.2858 | 0.2980 | 0.2778 | 0.2742 | 0.2732 | 0.2818 | 0.0103 |
| LSTM-medium | 53,313 | 0.2852 | 0.2823 | 0.2759 | 0.2803 | 0.2745 | 0.2797 | 0.0044 |
| GRU-large | 153,729 | 0.2773 | 0.2781 | 0.2768 | 0.2773 | 0.2758 | 0.2771 | 0.0008 |

## Descriptive evidence on "GRU-small is best"

- Per-seed winner (highest val IC in each seed): seed 42 MLP-medium, seed 0 TCN-small,
  seed 1 TCN-small, seed 2 TCN-large, seed 3 MLP-medium. GRU-small finished first in
  **0 of 5 seeds**.
- The first-place model is not stable: it rotates between TCN-small (2 seeds), MLP-medium
  (2 seeds), and TCN-large (1 seed).
- Top model by mean is TCN-small (0.3038); second is TCN-medium (0.2971); gap 0.0067. The
  per-model seed std across the top cluster is 0.006 to 0.011, comparable to or larger than
  the gaps between adjacent models.

## Conclusion

The previous team's specific claim "GRU-small is the best model" is not supported on the
corrected pipeline: GRU-small ranks 4th by mean and wins 0 of 5 seeds.

More importantly, no model is robustly best. The top cluster (roughly TCN-small, TCN-medium,
MLP-medium, GRU-small, and a few below) is separated by gaps smaller than the seed-to-seed
noise, and the per-seed winner changes from seed to seed. This confirms review items #19/#27:
at this sample size the model ranking is dominated by noise, so picking a single "best" model
is not meaningful. Two secondary observations point the same way: a plain MLP (no sequence
modeling) ties the sequence models, and the largest variants (TCN-large, LSTM-large,
GRU-large) sit in the middle or bottom, so added capacity does not help.

## Scope notes

- Ranking is on validation only; the test set is reserved for a single final evaluation.
- The hyperparameter grid was inherited from the original pipeline and not re-tuned for the
  B-law construction. This is intentional (it isolates the ranking effect by holding the model
  set fixed), so the result speaks to "is this model set separable," not "what is the globally
  optimal architecture." Given the top is a statistical tie, a fresh search would very likely
  also land in a tie.
- Per-epoch training curves from the run are not reproduced here; only the result-level numbers
  are retained. The machine-readable per-seed table is also saved by the notebook as
  `ranking_rows.csv`.
