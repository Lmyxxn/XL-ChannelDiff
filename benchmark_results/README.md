# Evaluation Results

Saved OMP evaluations and 100-step evaluations of `checkpoints/paper_original/model_best.pt`, with 200 test channels per mask ratio.

| Result | Test data |
| --- | --- |
| `omp_random.csv`, `omp_structured.csv` | `CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321_revised.mat` |
| `paper_weights_random.csv`, `paper_weights_structured.csv` | `CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321.mat` |

The curves use the saved measurements. The two methods were evaluated on different dataset versions; these plots are not a matched-test-set comparison.

From the repository root, regenerate the figures with:

```bash
python plot_results.py --series OMP=benchmark_results/omp_random.csv --series Proposed=benchmark_results/paper_weights_random.csv --output benchmark_results/omp_paper_weights_random
python plot_results.py --series OMP=benchmark_results/omp_structured.csv --series Proposed=benchmark_results/paper_weights_structured.csv --output benchmark_results/omp_paper_weights_structured
```

Figures are provided in PNG, PDF, and EPS formats.
