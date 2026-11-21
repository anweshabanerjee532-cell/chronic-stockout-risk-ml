# Machine Learning for Chronic Stockout Risk Assessment

**Subtitle:** Implications for Quick-Commerce Inventory Planning  
**Academic project:** Advanced Machine Learning in Business Transformation (AMLBT), Goa Institute of Management  
**Authors:** Nishant Khandelwal, Anwesha Banerjee, Anisha Saha

## Project overview

This study uses a **simulated, cross-sectional dataset of 45,000 SKU summaries** to classify dataset-defined chronic stockouts (`stockout_frequency >= 0.20`). It compares five classifiers on 12 baseline variables and 16 policy-augmented variables, with one stratified 70/15/15 train/validation/test split and leakage-aware, training-only preprocessing. A subsequent **exploratory, post-hoc** analysis compares ML-based SKU review rankings with an interpretable safety-stock screening rule at fixed review capacities. Financial scenarios represent **hypothetical exposure and avoidance assumptions**, not realised savings or evidence of causal interventions.

## Main findings

- CatBoost with policy-augmented variables was chosen using **validation average precision (AP = 0.7897)**. On the test set, AP = **0.7692**, F1 = **0.6845**, and ROC-AUC = **0.8800**. LightGBM/policy had slightly higher test AP (**0.7710**), but was not selected using the validation rule.
- At a **10% review capacity (675/6,750 test records)**, CatBoost's ranking identified **603** chronic-stockout-labelled records and the interpretable rule identified **608**. The selected records represented **27.88%** and **32.42%**, respectively, of the test set's simulated shortage-cost exposure.
- These outcomes suggest **testing the simpler screening rule before investing in a more complex ML system**, not that either approach has been shown to prevent actual stockouts.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── inventory_optimization.csv             # Original input CSV, if redistribution is permitted
├── data_dictionary_inventory (2).csv
├── EDA_ETL_Pipeline.ipynb             # Phase 1: original-data audit / EDA
├── QCommerce_Stockout.ipynb      # Phases 2–4: model comparison and exploratory business analyses
├── experiment_setup.json                  # To upload: checksum, splits, model settings
├── outputs/                               # To upload: extracted experiment outputs
│   ├── tables/                            # Metrics, ranking comparisons, financial sensitivity
│   └── figures/                           # Six model and business graphics
└── paper/                                 # Optional: submitted IEEE paper PDF
```

The `experiment_setup.json` and `outputs/` directories should be added when the research-evidence archive is unpacked. **GitHub does not automatically extract ZIP files**; upload the individual experiment files for browsing and verification.

## Source data

**Source:** S. Nefedov, *Inventory and Stockout Optimization Dataset*, Kaggle  
https://www.kaggle.com/datasets/sergionefedov/inventory-and-stockout-optimization

The input must be saved as `inventory_optimization.csv` in the notebook working directory. Its original SHA-256 was recorded as:

`bef1c3cd6d53ce8c4397bb47d0dc4564941ee842ecdf984dba696860948347a6`

The original data comprise 45,000 records, 28 fields, 13,798 positive labels and 31,202 negative labels. If data redistribution is not allowed by the source licence, obtain the CSV from Kaggle directly instead of uploading a copy.

## Reproduction instructions

1. Install Python dependencies with `pip install -r requirements.txt`, or open both notebooks in Google Colab. If using Colab, its package versions may differ from those recorded below.
2. Download the **original** source CSV from Kaggle and put `inventory_optimization.csv` in the current notebook working directory (or upload it to the Colab Files panel).
3. Run **`EDA_ETL_Pipeline.ipynb`** top to bottom to verify schema, label definition and audit statistics. This notebook also generates a cleaned EDA file, but the model notebook intentionally uses the **original unscaled CSV**.
4. Run **`QCommerce_Stockout.ipynb`** top to bottom to reproduce the five-model comparison, plots, exploratory rule benchmark and financial sensitivity tables.
5. Compare the generated results with the archived files in `outputs/` and the experiment settings in `experiment_setup.json`.

**Recorded model-run environment:** Python 3.13.18; pandas 2.2.3; NumPy 2.4.6; scikit-learn 1.6.1; XGBoost 3.4.1; LightGBM 4.6.0; CatBoost 1.2.10; Matplotlib 3.10.0. Seaborn is additionally used by the EDA notebook.

## Methodological safeguards and limitations

- SKU identifier, target-defining frequency, realised shortages/demand, realised costs, and hindsight optimisation fields are **excluded from the classifier predictors**.
- Baseline features include `ignores_lead_time_variability`, which is itself a policy-design flag; the policy-augmented set adds **four further policy quantities**.
- Fixed model configurations were used; **no cross-validated hyperparameter optimisation** was conducted.
- Random record splitting of synthetic SKU summaries does **not** establish future-period forecasting or live quick-commerce performance.
- The conventional rule and review-capacity comparison were analysed **after inspecting the original model test results**. This is an exploratory benchmark, not a fresh confirmatory holdout evaluation.
- All proposed cost reductions are assumption-driven sensitivity scenarios, **not measured savings or ROI**.

## Paper and attribution

The accompanying IEEE-format paper reports the experiments, financial sensitivities, limitations and managerial recommendation. Generative AI assisted with code development, debugging, figures and drafting; the authors remain responsible for checking the scientific claims, code and submitted materials.
