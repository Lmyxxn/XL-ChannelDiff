# Evaluation Results

Saved OMP evaluations and 100-step evaluations of `checkpoints/paper_original/model_best.pt`, with 200 test channels per mask ratio.

| Result | Test data |
| --- | --- |
| `omp_random.csv` | `CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321_revised.mat` |
| `paper_weights_random.csv` | `CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321.mat` |

The curves use the saved measurements. The two methods were evaluated on different dataset versions; these plots are not a matched-test-set comparison.

From the repository root, regenerate the figures with:

```bash
python plot_results.py --series OMP=benchmark_results/omp_random.csv --series Proposed=benchmark_results/paper_weights_random.csv --output benchmark_results/omp_paper_weights_random --formats png
```

The random-sampling figure is provided in PNG format.
