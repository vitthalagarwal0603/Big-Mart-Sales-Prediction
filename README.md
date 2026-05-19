# Big Mart Sales Prediction using Machine Learning

This repository contains a comprehensive, hands-on Machine Learning project designed to predict item outlet sales for **Big Mart** locations. The project architecture follows a complete data science workflow, utilizing exploratory analysis, missing value imputation, categorical feature encoding, and predictive modeling with the **XGBoost Regressor framework**, implemented entirely in Python within Google Colab.

---

## Project Objective
The primary objective of this project is to construct a robust predictive model capable of forecasting sales performance across various items and distinct retail outlets. By parsing structural factors such as product attributes (MRP, visibility, weight) and store characteristics (establishment year, size, location tier), the model uncovers underlying consumption patterns. These insights enable Big Mart management to:
* Optimize warehouse inventory management across diverse item categories.
* Formulate strategic product placement layouts tailored to specific store demographics.
* Maximize revenue margins by identifying high-performing product-store pairs.

---

## Implementation & Technical Workflow

The structural code architecture implementation is split into modular execution pipelines:

### 1. Data Collection & Preprocessing
* **Source Workspace:** Executed using the Kaggle Big Mart Sales Dataset consisting of **8,523 distinct data points** spread across **12 structural attribute features**.
* **Missing Value Imputation:**
  * Handled the `Item_Weight` missing entries by substituting them with the global **Mean value** (`12.857`).
  * Addressed missing `Outlet_Size` categorical features by constructing a mapping layout to substitute missing values with the regional **Mode value** relative to their corresponding `Outlet_Type`.

### 2. Exploratory Data Analysis (EDA)
* Utilized **Seaborn** and **Matplotlib** to plot univariate and bivariate feature distributions (`sns.distplot`, `sns.countplot`).
* Exposed insights into continuous distributions (such as `Item_MRP` variances) and monitored categorical frequency weights across item types.

### 3. Feature Engineering & Data Transformations
* Adjusted structural discrepancies in categorical labels (cleaning up duplicate string tags like `Low Fat`, `LF`, and `low fat` into a unified category label).
* Implemented **Label Encoding** via `sklearn.preprocessing.LabelEncoder` to transform string-based categories (Store Size, Store Location Type, Product Types) into numeric arrays suitable for machine learning algorithms.
* Segregated data using a standard 80/20 division ratio (`train_test_split`) to safeguard modeling validity.

### 4. Predictive Modeling
* Leveraged the **XGBoost (Extreme Gradient Boosting) Regressor** framework (`XGBRegressor()`) as the primary prediction engine.
* Fit the model configuration architecture using training feature subsets (`regressor.fit(X_train, Y_train)`).

---

## 🧠 Technical Challenges & Problem Solving

During the development of this pipeline, two critical data quality issues were encountered and resolved:

1. **Structural Inconsistencies in Categorical Labels:** The dataset contained duplicate representations for identical classes (e.g., `Low Fat`, `LF`, and `low fat`). Leaving these unaddressed would have fragmented the feature space, leading to poor model performance. 
   * *Solution:* Implemented a data-cleansing map using Pandas to unify these values into a single, standardized string category before label encoding.
   
2. **Missing Value Propagation:** A significant number of records had missing values for `Outlet_Size`. Simply dropping these rows would have biased the dataset.
   * *Solution:* Instead of a generic global mode substitution, I analyzed the data structure and imputed missing `Outlet_Size` values by calculating the conditional mode relative to each specific `Outlet_Type`. This preserved structural relationships within store profiles.
  
---

## Evaluation & Performance Metrics

The predictive validity of the model layout was evaluated using the **Coefficient of Determination ($R^2$ Score)** to establish fit accuracy against targeted values:

* **Training Data Performance ($R^2$ Score):** `0.876` (Demonstrates strong capability in mapping complex patterns within the training features).
* **Test Data Performance ($R^2$ Score):** `0.501` (Indicates generalized performance metrics on brand new data splits).

---

## Key Libraries Used
* `xgboost` (XGBRegressor engine)
* `scikit-learn` (LabelEncoder, metrics, train_test_split modules)
* `pandas` & `numpy` (Advanced data manipulation and array computation matrices)
* `seaborn` & `matplotlib` (Statistical visualization suites)

---
*Note: This project architecture was built as an independent applied machine learning exercise utilizing public retail optimization datasets from the Kaggle analytics community.*
