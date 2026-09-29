# FEDA: Homotopic and Homological Group Representations of Organic Molecules

Code and data for

> Qurat-ul-Ain and A. R. Nizami, "Homotopic and Homological Group Representations of Organic Molecules
> for Machine Learning Applications," *Proc. International Conference on
> Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2027)*, Rio de Janeiro,
> Brazil, 2–4 February 2027.

FEDA is a 261-feature molecular descriptor made of four blocks:

| Block | Object | Features | Function in `features.py` |
|---|---|---|---|
| FG | fundamental group (rank and minimum cycle basis) of the molecular graph | 28 | `fg_block` |
| ESPH | element-specific persistent homology, 6 atom selections × H0/H1/H2 × 8 statistics | 144 | `esph_block` |
| DM | pooled Euclidean distance matrix of the heavy atoms (+ gyration-tensor shape) | 47 | `adm_block` |
| AM | pooled bond-angle matrix (+ hybridisation fractions and torsion statistics) | 42 | `am_block` |

## Data

`BoilingData.xlsx` holds 5,432 organic compounds (SMILES, normal boiling point in °C) from

> Q. Zang *et al.*, "In silico prediction of physicochemical properties of environmental chemicals
> using molecular fingerprints and machine learning," *J. Chem. Inf. Model.* 57(1), 36–49, 2017.
> https://doi.org/10.1021/acs.jcim.6b00625

Please cite that paper if you use the data.

## Reproducing the results

```bash
pip install -r requirements.txt
bash run_all.sh
```

| Step | Script | Output |
|---|---|---|
| 1 | `conformers.py` | one ETKDGv3 + MMFF94 (≤200 steps) conformer per molecule, fixed seed `0xFEDA` |
| 2 | `features.py` | the four blocks; ESPH uses GUDHI Rips complexes up to 3-simplices, 12 Å cut-off, Z2 coefficients |
| 3 | `exp_main.py` (uses `common.py`) | five-fold CV for every configuration in Table II, plus four repeated partitions for FEDA and ESPH |
| 4 | `stats.py` | Table II (mean ± s.d.) and the corrected resampled t-test, FEDA vs ESPH |

`cv.py` is the original stage-3 script for the four blocks and FEDA alone.

Learners (not retuned per representation): XGBoost with 1200 trees, depth 8, learning rate 0.05 and
0.8 row/column subsampling; an MLP with layers 1024-512-256, Adam and early stopping, on inputs
standardised per training fold and clipped at ±5 s.d. Main partition: `KFold(5, shuffle=True,
random_state=20260902)`. Repeats use seeds 11, 22, 33 and 44.

## Results

| File | Contents |
|---|---|
| `results/results.json` | per-fold R² and RMSE for every configuration and partition |
| `results/per_fold_results.csv` | the same data as a flat table |
| `results/folds.csv` | fold assignment of every molecule for all five partitions |
| `results/results_review_version.json` | results as submitted for review (before the code was re-run) |

Main-partition summary (mean R² over five folds):

| Representation | d | XGBoost | ANN |
|---|---|---|---|
| FG | 28 | 0.6175 | 0.6385 |
| AM | 42 | 0.7701 | 0.7483 |
| DM | 47 | 0.8308 | 0.7854 |
| ESPH | 144 | 0.9224 | 0.9008 |
| **FEDA** | **261** | **0.9206** | **0.9192** |
| MACCS | 167 | 0.8164 | 0.8127 |

Over the 25 repeated folds, FEDA beats ESPH under the ANN in 22 of 25 folds (corrected t-test
p = 0.007). Under XGBoost the two are indistinguishable (p = 0.35).

## Figures

`figures/` holds the scripts for the paper's figures. Run them from inside that folder;
`palette.py` sets the shared style.

## License

Code: MIT (see `LICENSE`). Data: see the original publication cited above.
