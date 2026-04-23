# Crop Revenue Advisor

A machine learning web application that predicts the most suitable crop and estimates expected revenue for a given location, season, and farm area across India. Built with Flask and deployed on Vercel.

Live: https://crop-revenue-advisor.vercel.app/

---

## Overview

Farmers and agricultural planners often lack data-driven tools to decide which crop to grow and what revenue to expect. This application addresses that by combining historical crop production data, rainfall records, and live market price data to produce per-hectare and total revenue estimates for any supported district in India.

The user selects a state, district, season, and farm area. The backend runs two sequential ML models: a crop classifier that predicts the optimal crop for that location and season, followed by a revenue regression model that estimates expected earnings.

---

## Features

- Interactive district selection via a clickable map of India
- Automatic fuzzy matching and alias resolution for renamed or reorganized districts
- Crop recommendation using a trained Random Forest classifier
- Revenue estimation using a trained Gradient Boosting regression model
- Fallback logic: if a district is not in the training data, the nearest supported district is used
- Clean landing page and a separate prediction interface
- Deployed as a serverless Flask app on Vercel

---

## Project Structure

```
.
├── app.py                    # Flask application, prediction logic, API routes
├── train.py                  # Script to retrain the yield regression model
├── requirements.txt          # Python dependencies
├── vercel.json               # Vercel deployment configuration
│
├── templates/
│   ├── landing.html          # Landing page
│   └── index.html            # Main prediction UI with map
│
├── public/
│   └── INDIA_DISTRICTS.geojson  # GeoJSON with district geometries and model mappings
│
├── data/
│   ├── crop_production.csv      # Historical crop production dataset
│   ├── rainfall_district.csv    # District-level annual rainfall data
│   └── Marketwise_Price_Arrival_09-02-2026_07-49-39_PM.csv  # Market price data
│
├── crop_model.pkl.gz         # Trained crop classifier (Random Forest, compressed)
├── revenue_model.pkl         # Trained revenue regressor (Gradient Boosting)
├── le_state.pkl              # Label encoder for states
├── le_dist.pkl               # Label encoder for districts
├── le_season.pkl             # Label encoder for seasons
├── le_crop.pkl               # Label encoder for crops
├── avg_yield.pkl             # Precomputed average yield per crop
├── rainfall_state.pkl        # Precomputed average rainfall per state
├── market_data.pkl           # Processed market price lookup table
├── model_columns.pkl         # Feature column names for the revenue model
└── encoders.pkl              # Combined encoders (used by train.py)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (Random Forest, Gradient Boosting) |
| Data processing | pandas, joblib |
| Frontend | HTML, CSS, JavaScript |
| Map | GeoJSON district boundaries |
| Deployment | Vercel (serverless) |

---

## Machine Learning Models

### Crop Classifier

- **Algorithm**: Random Forest Classifier
- **Input features**: State, District, Year, Season (all label-encoded)
- **Output**: Predicted crop label
- **File**: `crop_model.pkl.gz`

### Revenue Regressor

- **Algorithm**: Gradient Boosting Regressor
- **Input features**: State, District, Year, Season, Crop, Average Yield, Annual Rainfall, Market Price
- **Output**: Predicted revenue per hectare (INR)
- **File**: `revenue_model.pkl`

---

## API

### `GET /`
Returns the landing page.

### `GET /app`
Returns the main prediction interface with the map and form.

### `POST /predict`
Accepts a JSON body and returns crop and revenue predictions.

**Request body:**
```json
{
  "state": "KARNATAKA",
  "district": "MYSURU",
  "season": "KHARIF",
  "area": 2.5
}
```

**Response:**
```json
{
  "crop": "Rice",
  "revenue_per_hectare": 48500.0,
  "total_revenue": 121250.0,
  "prediction_state": "KARNATAKA",
  "prediction_district": "MYSORE",
  "note": "MYSURU was matched to the model district MYSORE.",
  "used_district_fallback": true
}
```

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/Saksham-Gupta-GH/crop-revenue-advisor.git
cd crop-revenue-advisor
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run the Flask development server**
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## Retraining the Model

To retrain the revenue model from scratch:

```bash
python train.py
```

This reads from `data/crop_production.csv` and `data/rainfall_district.csv`, trains a Gradient Boosting regressor, and saves the output to `model/yield_model.pkl`.

Note: The crop classifier (`crop_model.pkl.gz`) was trained separately and is not regenerated by `train.py`.

---

## Data Sources

- **Crop production**: Government of India crop production statistics (state, district, season, area, production)
- **Rainfall**: District-level annual average rainfall data
- **Market prices**: Marketwise arrival and price data (February 2026)

---

## Deployment

The app is live at https://crop-revenue-advisor.vercel.app/ and is configured for Vercel using a serverless Flask setup. The compressed model file (`crop_model.pkl.gz`) is used at runtime to stay within Vercel's bundle size limits. The uncompressed `crop_model.pkl` is excluded from version control via `.gitignore`.

---

## District Name Resolution

Indian district names have changed significantly over time due to bifurcations and renaming. The app handles this through a multi-layer resolution strategy:

1. Exact match against the training data label set
2. Hardcoded alias table covering renamed districts across 20+ states
3. Fuzzy string matching with a confidence threshold as a final fallback
4. GeoJSON-based nearest-district lookup for completely unsupported districts

---

## License

This project is for educational and demonstration purposes.
