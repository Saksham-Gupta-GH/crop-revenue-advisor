# =====================================================
# TRAIN.PY — FINAL, ROBUST VERSION
# =====================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


# =========================
# 1. LOAD & CLEAN CROP DATA
# =========================
crop = pd.read_csv("data/crop_production.csv")
crop = crop[crop["Area"] > 0]

crop["Yield"] = crop["Production"] / crop["Area"]

crop = crop[
    ["State_Name", "Season", "Crop", "Yield"]
]

crop.columns = ["State", "Season", "Crop", "Yield"]

crop["State"] = crop["State"].str.upper().str.strip()
crop["Crop"] = crop["Crop"].str.strip().str.title()
crop["Season"] = crop["Season"].str.strip()

valid_seasons = ["Kharif", "Rabi", "Whole Year"]
crop = crop[crop["Season"].isin(valid_seasons)]
crop.dropna(inplace=True)


# =========================
# 2. LOAD & CLEAN RAINFALL
# =========================
rain = pd.read_csv("data/rainfall_district.csv")
rain = rain[["STATE_UT_NAME", "ANNUAL"]]
rain.columns = ["State", "Annual_Rainfall"]

rain["State"] = rain["State"].str.upper().str.strip()
rain_state = rain.groupby("State", as_index=False).mean()


# =========================
# 3. MERGE
# =========================
data = crop.merge(rain_state, on="State", how="inner")
data.dropna(inplace=True)


# =========================
# 4. ENCODE CATEGORICALS
# =========================
encoders = {}
for col in ["State", "Season", "Crop"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le


# =========================
# 5. TRAIN MODEL (TARGET = YIELD)
# =========================
X = data.drop("Yield", axis=1)
y = data["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))


# =========================
# 6. SAVE ARTIFACTS
# =========================
joblib.dump(model, "model/yield_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")

print("✅ Model trained and saved successfully.")
