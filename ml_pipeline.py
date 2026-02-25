# ===============================
# STEP 1: IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ===============================
# STEP 2: LOAD EXCEL FILE
# ===============================
file_path = "agriculture_advanced_ml_ready_dataset.xlsx"

sheets = pd.read_excel(file_path, sheet_name=None)

fact = sheets["Fact_Production"]
crop = sheets["Dim_Crop"]
soil = sheets["Dim_Soil"]
irrigation = sheets["Dim_Irrigation"]
geo = sheets["Dim_Geography"]
time = sheets["Dim_Time"]

# ===============================
# STEP 3: MERGE TABLES (STAR SCHEMA)
# ===============================
df = fact.merge(crop, on="Crop_ID", how="left") \
         .merge(soil, on="Soil_ID", how="left") \
         .merge(irrigation, on="Irrigation_ID", how="left") \
         .merge(geo, on="Geo_ID", how="left") \
         .merge(time, on="Time_ID", how="left")

print(df.columns.tolist())


# ===============================
# STEP 4: SELECT REQUIRED COLUMNS
# ===============================
df = df[
    [
        "Rainfall_mm",
        "Avg_Temperature_C",
        "Base_Yield_q_per_acre",
        "Fertility_Index",
        "Efficiency_Index",
        "Crop_Name",
        "Soil_Type",
        "Irrigation_Type",
        "Season_y",
        "State",
        "Yield_q_per_acre"
    ]
]

df = df.rename(columns={"Season_y": "Season"})
print(df.columns.tolist())

# ===============================
# STEP 5: DROP MISSING VALUES
# ===============================
df = df.dropna()

# ===============================
# STEP 6: SPLIT FEATURES & TARGET
# ===============================
X = df.drop("Yield_q_per_acre", axis=1)
y = df["Yield_q_per_acre"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# ===============================
# STEP 7: PREPROCESSOR
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# ===============================
# STEP 8: MODEL PIPELINE
# ===============================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# ===============================
# STEP 9: TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 10: TRAIN MODEL
# ===============================
model.fit(X_train, y_train)

# ===============================
# STEP 11: EVALUATION
# ===============================
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ===============================
# STEP 12: SAVE MODEL
# ===============================
joblib.dump(model, "yield_model.pkl")

print("Model saved as yield_model.pkl")
