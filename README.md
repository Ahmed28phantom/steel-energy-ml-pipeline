# Predicting Energy Use and Load State in a Smart Steel Plant

A machine-learning analysis of one year of 15-minute electricity readings from a steel manufacturer
in South Korea. The project predicts how much energy the plant uses, identifies which operating
state it is in, and looks for natural operating regimes in the data, using scikit-learn throughout.

The full analysis, with code, figures and discussion, is in
[`notebooks/steel_energy_analysis.ipynb`](notebooks/steel_energy_analysis.ipynb).

## Questions and results

| Question | Approach | Best model | Result (held-out test set) |
|---|---|---|---|
| How much electricity will the plant use in a 15-minute interval? | Regression on `Usage_kWh` | Random Forest | RMSE 0.92 kWh, R² 0.999 (linear models: RMSE 9.84 kWh, R² 0.914) |
| Which operating state is the plant in: light, medium or maximum load? | Classification on `Load_Type` | Random Forest | Macro-F1 0.922, accuracy 0.928, AUC 0.991 (linear models: macro-F1 0.68–0.71) |
| Are there natural operating regimes that can be found without labels? | PCA and K-Means | K-Means, K = 4 | Silhouette 0.357; four overlapping regimes |

## Key findings

- **Tree-based models beat linear ones on both supervised tasks,** so the relationships are
  strongly non-linear.
- **Energy use follows the machinery.** Lagging reactive power, the inductive load of running
  motors, accounts for about 72% of the Random Forest's importance. The time features add about 3%.
- **The load-type label follows the clock.** `Hour` and `Month` are the two strongest predictors.
  Removing all time features drops macro-F1 from 0.92 to 0.66. Electrical readings still carry
  real signal (random guessing gives 0.33), but much of the classifier's skill is recognising the
  plant's fixed tariff schedule.
- **The data has more structure than the three labels suggest.** K-Means finds four overlapping
  regimes, splitting light load into several low-intensity states and merging medium and maximum
  load into one production regime.

## Method in brief

- **Leakage control:** `CO2(tCO2)` is calculated directly from usage, so it is never used as a
  predictor.
- **Redundancy:** two redundant predictors are dropped (`NSM` duplicates `Hour`; leading reactive
  power duplicates leading power factor). That leaves six features for every model.
- **Evaluation:** an 80/20 train/test split, stratified for classification. Hyperparameters are
  tuned by 5-fold grid-search cross-validation on the training set only. The best model is chosen
  by cross-validated score, and the test set is used once.
- **Class imbalance:** macro-F1 is the primary classification metric, and class weights are used
  wherever a model supports them.
- **Models compared:**
  - Regression: Linear Regression, Lasso, Decision Tree, Random Forest
  - Classification: Logistic Regression, LDA, Decision Tree, Random Forest

## Repository layout

```
├── data/raw/Steel_industry_data.csv      # dataset (CC BY 4.0, see below)
├── notebooks/steel_energy_analysis.ipynb # the analysis
├── requirements.txt                      # exact library versions used
├── LICENSE                               # MIT (code)
└── README.md
```

## Running the notebook

Requires Python 3.12.

```bash
git clone https://github.com/Ahmed28phantom/steel-energy-ml-pipeline.git
cd steel-energy-ml-pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then either open `notebooks/steel_energy_analysis.ipynb` in VS Code or Jupyter (for Jupyter, also
run `pip install jupyterlab`), or execute it from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/steel_energy_analysis.ipynb
```

A full run takes about 4–5 minutes on a recent laptop, most of it spent on the grid searches.

## Reproducibility notes

- **Library versions matter.** All random seeds are fixed, so a run with the versions in
  `requirements.txt` reproduces the notebook's numbers exactly. Other scikit-learn versions
  build random forests slightly differently. With scikit-learn 1.7.1, for example, the regression
  Random Forest's test RMSE is 0.906 instead of 0.924. The conclusions don't change.
- **Midnight timestamps in the source data.** Each day's last reading (23:45–24:00) is stamped
  `00:00` on the same date instead of the next day. The notebook uses the data as published.

## Data and attribution

- **Dataset:** Sathishkumar V E, Shin C, Cho Y (2021). *Steel Industry Energy Consumption*
  [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C
  - Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  - 35,040 readings taken every 15 minutes in 2018 at DAEWOO Steel Co. Ltd, Gwangyang, South Korea.
  - The CSV is included unchanged in `data/raw/`.
- **Original study:** Sathishkumar V E, Shin C, Cho Y (2021). Efficient energy consumption
  prediction model for a data analytic-enabled industry building in a smart city.
  *Building Research & Information*, 49(1), 127–143. https://doi.org/10.1080/09613218.2020.1809983

## License

The code and notebook are released under the [MIT License](LICENSE). The dataset is not covered
by it: it remains under its original [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
license, with credit to its authors as above.
