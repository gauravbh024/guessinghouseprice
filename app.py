import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv('housing.csv')

    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop('income_cat', axis=1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)
    columns_to_save = [col for col in housing.columns if col != 'median_house_value']

    # Save only the column names (no data) to CSV
    pd.DataFrame(columns=columns_to_save).to_csv('sample.csv', index=False)

    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # OPTIMIZED MODEL - fewer trees and less depth to reduce memory
    model = RandomForestRegressor(
        n_estimators=50,      # Reduced from default 100
        max_depth=15,         # Limit tree depth
        min_samples_split=5,  # Prevent overfitting
        random_state=23,
        n_jobs=1              # Use single thread to save memory
    )
    model.fit(housing_prepared, housing_labels)

    # Save with compression to reduce file size
    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(pipeline, PIPELINE_FILE, compress=3)

    print('Model successfully trained with memory optimization.')
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('sample.csv')
    transformed_input = pipeline.transform(input_data)

    prediction_input = model.predict(transformed_input)
    input_data['median_house_value'] = prediction_input
    input_data.to_csv('output.csv', index=False)
    print('Process completed. Results saved in output.csv.')
