# California Housing Price Prediction with KNN Regression

## Project Overview
Built a **K-Nearest Neighbors regression model** to predict California housing prices using the California Housing dataset. The workflow includes **data preprocessing, model pipeline creation, hyperparameter tuning, evaluation, and model persistence**.

---

## Dataset
- **Source:** `sklearn.datasets.fetch_california_housing`
- **Rows/Columns:** 20640 samples, 8 numerical features
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target:** Median house value

---

## Workflow
1. **Data Exploration:** Loaded dataset, checked first rows and feature info.
2. **Train-Test Split:** 80/20 split.
3. **Preprocessing:**  
   - Imputed missing values with median.  
   - Standardized all numerical features.  
   - Combined using `ColumnTransformer` and `Pipeline`.
4. **Modeling:** KNN regression integrated into pipeline.
5. **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation. Tuned `n_neighbors`, `weights`, `p`.
6. **Evaluation:** R², MSE, RMSE on test set.
7. **Model Persistence:** Saved trained pipeline with `pickle`.

---

## Skills Demonstrated / Concepts Applied
- **Predictive Modeling & Classical ML:** Built KNN regression model using scikit-learn.
- **Data Preprocessing:** Handled missing values with `SimpleImputer` and scaled numerical features with `StandardScaler`.
- **Feature Engineering:** Selected numerical features and combined preprocessing steps using `ColumnTransformer`.
- **Pipeline Creation:** Integrated preprocessing and model into a scikit-learn `Pipeline` for reproducibility.
- **Hyperparameter Tuning:** Optimized `n_neighbors`, `weights`, and `p` using `GridSearchCV`.
- **Model Evaluation:** Measured performance using R², Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- **Model Persistence:** Saved trained pipeline with `pickle` for future use.

---

## Outputs / Visualizations
- Preprocessing pipeline summary and hyperparameter tuning results
- Model evaluation metrics: R², MSE, RMSE
- Predictions on the test dataset
---
