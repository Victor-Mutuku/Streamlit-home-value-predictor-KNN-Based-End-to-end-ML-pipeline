import os
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

#Load Dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
print(X.head())

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Reprocessing:Imputation + Scaling for numerical features
numerical_features= X.columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#Combine preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features)
])

#Build pipeline: preprocessing + KNN
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor())
])
#This step preprocess the data hence creates a pipeline structure

#Define the hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance'],
    'knn__p':[1,2]
}

#Apply GridSearchCV with 5-fold cross-validation
grid_search= GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)

#Fit the model
grid_search.fit(X_train, y_train)
#This is the training step of the entire workflow

#Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

#The metrics
r2_score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse=mean_squared_error = mse**0.5

#Print results
print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV R2 Score:", grid_search.best_score_)
print("R2 Score:", r2_score)
print("MSE:", mse)
print("RMSE:", rmse)

save_path="Model/model.pkl"

os.makedirs("Model",exist_ok=True)
with open(save_path, "wb") as file:
    pickle.dump(best_model, file)
    
print("model saved successfully")    