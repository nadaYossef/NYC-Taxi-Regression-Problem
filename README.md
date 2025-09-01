# NYC Taxi Trip Duration – Regression Analysis

This project builds regression models to predict **taxi trip duration** in New York City.
Using the [NYC Taxi Trip Duration dataset](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration), we apply data preprocessing, feature engineering, and machine learning models to understand trip dynamics and improve prediction accuracy.

---

## Dataset

The dataset is available on Kaggle:

[NYC Taxi Trip Duration Dataset](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration)

Due to its large size, the dataset is stored using **Git Large File Storage (LFS)**.

* If you cloned this repository without Git LFS, run:

  ```bash
  git lfs install
  git lfs pull
  ```
* Alternatively, you can manually download the dataset from Kaggle and place it in:

  ```
  data/NYC.csv
  ```

---

## Requirements

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

*(Optional)* If you plan to test advanced models:

```bash
pip install xgboost lightgbm
```

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/nadaYossef/NYC-Taxi-Regression-Problem.git
   cd NYC-Taxi-Regression-Problem
   ```

2. Ensure the dataset is available in `data/NYC.csv`

   * Either via Git LFS, or by downloading from Kaggle and placing it manually.

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook NYC_Taxi_Regression_Problem.ipynb
   ```

4. Run the notebook cells to reproduce the full workflow.

---

## Methodology

The project follows a structured pipeline:

1. **Data Cleaning & Preprocessing**
   - Handled missing values and outliers.
   - Converted timestamps into `datetime` and extracted temporal features.
   - Applied log-transform to trip duration to reduce skewness.
   - Encoded categorical features and scaled where necessary.

2. **Exploratory Data Analysis (EDA)**
   - Distribution of trip durations (raw & log-transformed).
   - Spatial pickup/drop-off density mapping.
   - Temporal patterns (hourly, daily, monthly).

3. **Feature Engineering**
   - Computed Haversine distance from pickup/drop-off coordinates.
   - Derived datetime features (hour, weekday, month, weekend indicator).
   - Added passenger and vendor features where relevant.

4. **Modeling**
   - Baseline models: Linear Regression, Ridge, and Lasso.
   - Advanced models: Random Forest, XGBoost.
   - Evaluation metric: R² (main) and RMSE (secondary).

5. **Hyperparameter Optimization**
   To improve model performance, hyperparameter tuning was applied:

   - **Ridge (Polynomial Features)**: GridSearchCV over `alpha = [0.1, 1.0, 10.0, 100.0]`.
   - **Lasso (Polynomial Features)**: GridSearchCV over `alpha = [0.001, 0.01, 0.1, 1.0]`.
   - **Random Forest**: RandomizedSearchCV with 20 iterations, tuning depth, estimators, and feature splits.
   - **XGBoost**: RandomizedSearchCV with 20 iterations, tuning estimators, max depth, learning rate, subsample ratio, colsample_bytree, and regularization.

   Best parameter sets and optimized scores were logged and compared.

6. **Evaluation & Reporting**
   - R² and RMSE on the test set.
   - Predicted vs actual plots.
   - Feature importance analysis (tree-based models).

---

## ✅ Key Insights

* The target variable (trip duration) is highly skewed → log transformation improves model performance.
* Feature engineering (distance, datetime features) is crucial for predictive accuracy.
* Tree-based ensemble models outperform simple linear regression.

---
