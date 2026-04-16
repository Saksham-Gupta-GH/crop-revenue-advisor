import joblib
import pandas as pd

# =========================
# LOAD MODEL & ENCODERS
# =========================
model = joblib.load("model/yield_model.pkl")
encoders = joblib.load("model/encoders.pkl")

# =========================
# LOAD RAINFALL DATA
# =========================
rain = pd.read_csv("data/rainfall_district.csv")
rain = rain[["STATE_UT_NAME", "DISTRICT", "ANNUAL"]]
rain.columns = ["State", "District", "Annual_Rainfall"]

rain["State"] = rain["State"].str.upper().str.strip()
rain["District"] = rain["District"].str.upper().str.strip()

# fallback: district → state rainfall
rainfall_state = rain.groupby("State")["Annual_Rainfall"].mean().to_dict()

# =========================
# LOAD MARKET DATA
# =========================
market_raw = pd.read_csv(
    "data/Marketwise_Price_Arrival_09-02-2026_07-49-39_PM.csv",
    header=None
)

market = market_raw.iloc[3:, :6].copy()
market.columns = ["Group", "Crop", "MSP", "P1", "P2", "P3"]

for col in ["MSP", "P1", "P2", "P3"]:
    market[col] = pd.to_numeric(market[col], errors="coerce")

market["Market_Price"] = market[["P1", "P2", "P3"]].mean(axis=1)
market = market[["Crop", "Market_Price"]].dropna()
market["Crop"] = market["Crop"].str.strip().str.title()

crop_map = {
    "Paddy(Common)": "Rice",
    "Bajra(Pearl Millet/Cumbu)": "Bajra",
    "Jowar(Sorghum)": "Jowar",
    "Ragi(Finger Millet)": "Ragi",
    "Soyabean": "Soybean"
}
market["Crop"] = market["Crop"].replace(crop_map)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend_crop(state, district, season, land_area):
    state = state.upper().strip()
    district = district.upper().strip()
    season = season.strip()

    if state not in rainfall_state:
        raise ValueError("State not found in rainfall data")

    results = []

    for crop in market["Crop"].unique():

        if crop not in encoders["Crop"].classes_:
            continue

        X = pd.DataFrame([{
            "State": state,
            "Season": season,
            "Crop": crop,
            "Annual_Rainfall": rainfall_state[state]
        }])

        # Encode ONLY features used in training
        for col in ["State", "Season", "Crop"]:
            if X[col].iloc[0] not in encoders[col].classes_:
                break
            X[col] = encoders[col].transform(X[col])
        else:
            predicted_yield = model.predict(X)[0]
            price = market.loc[market["Crop"] == crop, "Market_Price"].values[0]
            total_profit = predicted_yield * price * land_area

            results.append({
                "Crop": crop,
                "Yield": round(predicted_yield, 2),
                "Total_Profit": round(total_profit, 2)
            })

    return sorted(results, key=lambda x: x["Total_Profit"], reverse=True)


# =========================
# USER INPUT (FOR NOW)
# =========================
if __name__ == "__main__":
    print("\n🌾 FARMING ADVISOR SYSTEM\n")

    state = input("Enter State: ")
    district = input("Enter District: ")
    season = input("Enter Season (Kharif / Rabi / Whole Year): ")
    land_area = float(input("Enter Land Area (hectares): "))

    recs = recommend_crop(state, district, season, land_area)

    if not recs:
        print("\n⚠️ No suitable crop found for given inputs.")
    else:
        print("\n✅ Top Crop Recommendations:\n")
        for i, r in enumerate(recs[:3], 1):
            print(
                f"{i}. {r['Crop']} | "
                f"Yield: {r['Yield']} | "
                f"Expected Profit: ₹{r['Total_Profit']}"
            )
