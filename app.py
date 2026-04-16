import json
from pathlib import Path
import re
from collections import defaultdict
from difflib import SequenceMatcher

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


app = Flask(__name__, static_folder="public", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_GEOJSON = STATIC_DIR / "INDIA_DISTRICTS.geojson"
CROP_HISTORY_PATH = BASE_DIR / "data" / "crop_production.csv"
MARKET_HISTORY_PATH = BASE_DIR / "data" / "Marketwise_Price_Arrival_09-02-2026_07-49-39_PM.csv"


# Load the already-trained models and encoders.
CROP_MODEL_PATH = BASE_DIR / "crop_model.pkl.gz"
if not CROP_MODEL_PATH.exists():
    CROP_MODEL_PATH = BASE_DIR / "crop_model.pkl"

CROP_MODEL = joblib.load(CROP_MODEL_PATH)
REVENUE_MODEL = joblib.load(BASE_DIR / "revenue_model.pkl")
LE_STATE = joblib.load(BASE_DIR / "le_state.pkl")
LE_DIST = joblib.load(BASE_DIR / "le_dist.pkl")
LE_SEASON = joblib.load(BASE_DIR / "le_season.pkl")
LE_CROP = joblib.load(BASE_DIR / "le_crop.pkl")

# Helper data used by the original recommend_crop() logic.
AVG_YIELD_DF = joblib.load(BASE_DIR / "avg_yield.pkl")
RAINFALL_BY_STATE = joblib.load(BASE_DIR / "rainfall_state.pkl")
MARKET_DATA = joblib.load(BASE_DIR / "market_data.pkl")

PREDICTION_YEAR = 2020


def normalize_text(value):
    """Uppercase and clean text so user/map input matches encoder labels."""
    text = str(value or "").strip().upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def format_label(value):
    """Convert labels like 'GROUNDNUT' into 'Groundnut' for the UI."""
    return " ".join(part.capitalize() for part in str(value).split())


STATE_ALIASES = {
    "CHHATTISGARH": "CHHATTISGARH",
}

SEASON_ALIASES = {
    "ZAID": "SUMMER",
}

MODEL_LOCATIONS_DF = pd.read_csv(CROP_HISTORY_PATH, usecols=["State_Name", "District_Name"])
MODEL_LOCATIONS_DF["State_Name"] = MODEL_LOCATIONS_DF["State_Name"].astype(str).str.upper().str.strip()
MODEL_LOCATIONS_DF["District_Name"] = MODEL_LOCATIONS_DF["District_Name"].astype(str).str.upper().str.strip()
MODEL_LOCATIONS_DF = MODEL_LOCATIONS_DF[
    MODEL_LOCATIONS_DF["State_Name"].isin(LE_STATE.classes_)
    & MODEL_LOCATIONS_DF["District_Name"].isin(LE_DIST.classes_)
][["State_Name", "District_Name"]].drop_duplicates()
MODEL_LOCATIONS_DF.columns = ["State", "District"]

SUPPORTED_STATES = sorted(MODEL_LOCATIONS_DF["State"].unique().tolist())
DISTRICTS_BY_STATE = {
    state: sorted(
        MODEL_LOCATIONS_DF.loc[MODEL_LOCATIONS_DF["State"] == state, "District"].unique().tolist()
    )
    for state in SUPPORTED_STATES
}

STATE_LOOKUP = {normalize_text(label): label for label in SUPPORTED_STATES}
DISTRICT_LOOKUP_BY_STATE = {
    state: {normalize_text(district): district for district in districts}
    for state, districts in DISTRICTS_BY_STATE.items()
}
SEASON_LOOKUP = {normalize_text(label): label for label in LE_SEASON.classes_}

RAW_FRONTEND_DISTRICT_ALIASES = {
    "ANDHRA PRADESH": {
        "ANANTHAPURAMU": "ANANTAPUR",
        "VISAKHAPATNAM": "VISAKHAPATANAM",
        "YSR KADAPA": "KADAPA",
        "SRI POTTI SRIRAMULU NELLORE": "SPSR NELLORE",
        "ALLURI SITHARAMA RAJU": "VISAKHAPATANAM",
        "ANAKAPALLI": "VISAKHAPATANAM",
        "DR B R AMBEDKAR KONASEEMA": "EAST GODAVARI",
        "ELURU": "WEST GODAVARI",
        "KAKINADA": "EAST GODAVARI",
        "NANDYAL": "KURNOOL",
        "NTR": "KRISHNA",
        "PALNADU": "GUNTUR",
        "PARVATHIPURAM MANYAM": "VIZIANAGARAM",
        "SRI SATHYA SAI": "ANANTAPUR",
        "TIRUPATI": "CHITTOOR",
    },
    "ASSAM": {
        "DARANG": "DARRANG",
        "SIBSAGAR": "SIVASAGAR",
        "CHARAIDEO": "SIVASAGAR",
        "BISWANATH": "SONITPUR",
        "HOJAI": "NAGAON",
        "KAMRUP RURAL": "KAMRUP",
        "MAJULI": "JORHAT",
        "BAJALI": "BARPETA",
        "TAMULPUR": "BAKSA",
        "SOUTH SALMARA MANCACHAR": "DHUBRI",
    },
    "BIHAR": {
        "JAHANABAD": "JEHANABAD",
    },
    "GUJARAT": {
        "DANGS": "DANG",
        "DAHOD": "DOHAD",
        "ARVALLI": "SABAR KANTHA",
        "BOTAD": "BHAVNAGAR",
        "CHHOTAUDEPUR": "VADODARA",
        "DEVBHUMI DWARKA": "JAMNAGAR",
        "GIR SOMNATH": "JUNAGADH",
        "MORBI": "RAJKOT",
        "MAHISAGAR": "PANCH MAHALS",
    },
    "HARYANA": {
        "GURUGRAM": "GURGAON",
        "NUH": "MEWAT",
        "CHARKI DADRI": "BHIWANI",
    },
    "JAMMU AND KASHMIR": {
        "BARAMULA": "BARAMULLA",
        "BANDIPURA": "BANDIPORA",
        "RIASI": "REASI",
        "SHUPIYAN": "SHOPIAN",
        "PUNCH": "POONCH",
    },
    "KARNATAKA": {
        "BELAGAVI": "BELGAUM",
        "BALLARI": "BELLARY",
        "VIJAYAPURA": "BIJAPUR",
        "KALABURAGI": "GULBARGA",
        "MYSURU": "MYSORE",
        "SHIVAMOGGA": "SHIMOGA",
        "DAKSHINA KANNADA": "DAKSHIN KANNAD",
        "UTTARA KANNADA": "UTTAR KANNAD",
        "BENGALURU RURAL": "BANGALORE RURAL",
        "BENGALURU URBAN": "BENGALURU URBAN",
    },
    "MAHARASHTRA": {
        "RAIGARH": "RAIGAD",
        "DHARASHIV": "OSMANABAD",
        "CHHATRAPATI SAMBHAJINAGAR": "AURANGABAD",
    },
    "PUDUCHERRY": {
        "PUDUCHERRY": "PONDICHERRY",
    },
    "PUNJAB": {
        "FIROZPUR": "FIROZEPUR",
        "SRI MUKTSAR SAHIB": "MUKTSAR",
        "SAS NAGAR SAHIBZADA AJIT SINGH NAGAR": "S.A.S NAGAR",
        "SHAHID BHAGAT SINGH NAGAR": "NAWANSHAHR",
        "MALER KOTLA": "SANGRUR",
    },
    "RAJASTHAN": {
        "CHITTAURGARH": "CHITTORGARH",
        "DHAULPUR": "DHOLPUR",
        "JALOR": "JALORE",
        "JHUNJHUNUN": "JHUNJHUNU",
        "ANOOPGARH": "GANGANAGAR",
        "BALOTRA": "BARMER",
        "BEAWAR": "AJMER",
        "DEEG": "BHARATPUR",
        "DIDWANA KUCHAMAN": "NAGAUR",
        "DUDU": "JAIPUR",
        "GANGAPURCITY": "SAWAI MADHOPUR",
        "JAIPUR GRAMIN": "JAIPUR",
        "JODHPUR GRAMIN": "JODHPUR",
        "KEKRI": "AJMER",
        "KHAIRTHAL TIJARA": "ALWAR",
        "NEEM KA THANA": "SIKAR",
        "PHALODI": "JODHPUR",
        "SALUMBAR": "UDAIPUR",
        "SANCHORE": "JALORE",
        "SHAHPURA": "BHILWARA",
    },
    "SIKKIM": {
        "GANGTOK": "EAST DISTRICT",
        "PAKYONG": "EAST DISTRICT",
        "MANGAN": "NORTH DISTRICT",
        "NAMCHI": "SOUTH DISTRICT",
        "GYALSHING": "WEST DISTRICT",
        "SORENG": "WEST DISTRICT",
    },
    "TAMIL NADU": {
        "THOOTHUKUDI": "TUTICORIN",
        "VILUPPURAM": "VILLUPURAM",
        "CHENGALPATTU": "KANCHIPURAM",
        "KALLAKKURICHI": "VILLUPURAM",
        "MAYILADUTHURAI": "NAGAPATTINAM",
        "RANIPET": "VELLORE",
        "TENKASI": "TIRUNELVELI",
        "TIRUPATHUR": "VELLORE",
    },
    "TRIPURA": {
        "GOMTI": "GOMATI",
    },
    "UTTAR PRADESH": {
        "BARA BANKI": "BARABANKI",
        "BHADOHI": "SANT RAVIDAS NAGAR",
        "AYODHYA": "FAIZABAD",
        "KUSHINAGAR": "KUSHI NAGAR",
        "MAHRAJGANJ": "MAHARAJGANJ",
        "SANT KABIR NAGAR": "SANT KABEER NAGAR",
        "SHRAWASTI": "SHRAVASTI",
        "SIDDHARTHNAGAR": "SIDDHARTH NAGAR",
    },
    "WEST BENGAL": {
        "HUGLI": "HOOGHLY",
        "PURULIYA": "PURULIA",
        "DAKSHIN DINAJPUR": "DINAJPUR DAKSHIN",
        "UTTAR DINAJPUR": "DINAJPUR UTTAR",
        "PURBA MEDINIPUR": "MEDINIPUR EAST",
        "PASCHIM MEDINIPUR": "MEDINIPUR WEST",
        "PURBA BARDHAMAN": "BARDHAMAN",
        "PASCHIM BARDHAMAN": "BARDHAMAN",
    },
}

FRONTEND_DISTRICT_ALIASES = {
    state: {normalize_text(alias): target for alias, target in aliases.items()}
    for state, aliases in RAW_FRONTEND_DISTRICT_ALIASES.items()
}

GEO_DISTRICT_LOOKUP_BY_STATE = defaultdict(dict)
RAW_GEO_DISTRICT_LOOKUP_BY_STATE = defaultdict(dict)

# Build lookup tables that match the notebook's average-value logic.
AVG_YIELD_DF = AVG_YIELD_DF.copy()
AVG_YIELD_DF["Crop_Key"] = AVG_YIELD_DF["Crop"].apply(normalize_text)
AVG_YIELD_BY_CROP = AVG_YIELD_DF.groupby("Crop_Key")["Yield"].mean().to_dict()

RAINFALL_LOOKUP = {
    normalize_text(state): float(value)
    for state, value in RAINFALL_BY_STATE.items()
}

MARKET_PRICE_LOOKUP = {}
for crop_name, values in MARKET_DATA.items():
    crop_key = normalize_text(crop_name)
    market_price = values.get("Market_Price")
    if market_price is not None:
        MARKET_PRICE_LOOKUP[crop_key] = float(market_price)

if CROP_HISTORY_PATH.exists():
    crop_history_df = pd.read_csv(CROP_HISTORY_PATH)
    crop_history_df = crop_history_df[crop_history_df["Area"] > 0].copy()
    crop_history_df["Crop"] = crop_history_df["Crop"].astype(str).apply(normalize_text)
    crop_history_df["Yield"] = crop_history_df["Production"] / crop_history_df["Area"]
    HISTORICAL_YIELD_BY_CROP = crop_history_df.groupby("Crop")["Yield"].mean().to_dict()
else:
    HISTORICAL_YIELD_BY_CROP = {}

if MARKET_HISTORY_PATH.exists():
    market_history_df = pd.read_csv(MARKET_HISTORY_PATH, header=None)
    market_history_df = market_history_df.iloc[3:, :6].copy()
    market_history_df.columns = ["Group", "Crop", "MSP", "P1", "P2", "P3"]
    for column in ["P1", "P2", "P3"]:
        market_history_df[column] = pd.to_numeric(market_history_df[column], errors="coerce")
    market_history_df["Crop"] = market_history_df["Crop"].astype(str).apply(normalize_text)
    market_history_df["Market_Price"] = market_history_df[["P1", "P2", "P3"]].mean(axis=1)
    fallback_prices = (
        market_history_df[["Crop", "Market_Price"]]
        .dropna()
        .groupby("Crop")["Market_Price"]
        .mean()
        .to_dict()
    )
    for crop_key, market_price in fallback_prices.items():
        MARKET_PRICE_LOOKUP.setdefault(crop_key, float(market_price))


def resolve_label(raw_value, lookup, aliases=None):
    """Map a user/map value to the exact label stored by the encoder."""
    cleaned = normalize_text(raw_value)
    if aliases and cleaned in aliases:
        cleaned = normalize_text(aliases[cleaned])
    return lookup.get(cleaned)


def resolve_state_label(raw_state):
    return resolve_label(raw_state, STATE_LOOKUP, STATE_ALIASES)


def best_unique_match(cleaned_value, candidates, threshold=0.88, min_gap=0.05):
    if not candidates:
        return None

    scored = sorted(
        (
            (
                SequenceMatcher(None, cleaned_value, normalize_text(candidate)).ratio(),
                candidate,
            )
            for candidate in candidates
        ),
        reverse=True,
    )

    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0

    if best_score >= threshold and (best_score - second_score) >= min_gap:
        return best_candidate

    return None


def fuzzy_match_district(cleaned_district, state_label):
    candidates = DISTRICTS_BY_STATE.get(state_label, [])
    return best_unique_match(cleaned_district, candidates)


def resolve_district_label(raw_district, state_label, allow_fuzzy=False):
    if not state_label:
        return None

    cleaned = normalize_text(raw_district)
    state_district_lookup = DISTRICT_LOOKUP_BY_STATE.get(state_label, {})

    if cleaned in state_district_lookup:
        return state_district_lookup[cleaned]

    state_aliases = FRONTEND_DISTRICT_ALIASES.get(state_label, {})
    aliased = state_aliases.get(cleaned)
    if aliased in DISTRICTS_BY_STATE.get(state_label, []):
        return aliased

    if allow_fuzzy:
        return fuzzy_match_district(cleaned, state_label)

    return None


def extract_geometry_points(coordinates):
    points = []
    if isinstance(coordinates, (list, tuple)):
        if coordinates and isinstance(coordinates[0], (int, float)):
            if len(coordinates) >= 2:
                points.append((float(coordinates[0]), float(coordinates[1])))
        else:
            for item in coordinates:
                points.extend(extract_geometry_points(item))
    return points


def geometry_centroid(geometry):
    if not geometry:
        return None

    points = extract_geometry_points(geometry.get("coordinates"))
    if not points:
        return None

    avg_lon = sum(point[0] for point in points) / len(points)
    avg_lat = sum(point[1] for point in points) / len(points)
    return (avg_lon, avg_lat)


def distance_between(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def resolve_geo_district_entry(raw_district, raw_state, allow_fuzzy=False):
    raw_state_key = normalize_text(raw_state)
    if not raw_state_key:
        return None

    cleaned = normalize_text(raw_district)
    state_geo_lookup = RAW_GEO_DISTRICT_LOOKUP_BY_STATE.get(raw_state_key, {})

    if cleaned in state_geo_lookup:
        return state_geo_lookup[cleaned]

    if allow_fuzzy:
        matched_name = best_unique_match(
            cleaned,
            [entry["raw_district"] for entry in state_geo_lookup.values()],
            threshold=0.84,
            min_gap=0.03,
        )
        if matched_name:
            return state_geo_lookup.get(normalize_text(matched_name))

    return None


def load_frontend_geojson_metadata():
    """Populate district lookup tables from the prepared frontend GeoJSON."""
    if not STATIC_GEOJSON.exists():
        return

    geojson_data = json.loads(STATIC_GEOJSON.read_text())

    for feature in geojson_data.get("features", []):
        properties = feature.get("properties") or {}
        raw_state = properties.get("state")
        raw_district = properties.get("district")
        model_state = properties.get("model_state")
        model_district = properties.get("model_district")
        nearest_model_district = properties.get("nearest_model_district")
        prediction_state = properties.get("prediction_state")
        prediction_district = properties.get("prediction_district")

        if raw_state and raw_district:
            RAW_GEO_DISTRICT_LOOKUP_BY_STATE[normalize_text(raw_state)][normalize_text(raw_district)] = {
                "raw_state": raw_state,
                "raw_district": raw_district,
                "model_state": model_state,
                "model_district": model_district,
                "nearest_model_district": nearest_model_district,
                "prediction_state": prediction_state,
                "prediction_district": prediction_district,
            }

        if model_state and raw_district:
            GEO_DISTRICT_LOOKUP_BY_STATE[model_state][normalize_text(raw_district)] = {
                "raw_district": raw_district,
                "model_state": model_state,
                "model_district": model_district,
                "nearest_model_district": nearest_model_district,
                "prediction_state": prediction_state,
                "prediction_district": prediction_district,
            }


load_frontend_geojson_metadata()


def get_average_yield(crop_key):
    """Prefer the small precomputed table, then fall back to historical crop averages."""
    return AVG_YIELD_BY_CROP.get(crop_key, HISTORICAL_YIELD_BY_CROP.get(crop_key))


def resolve_prediction_district(raw_district, state_label):
    direct_district = resolve_district_label(raw_district, state_label, allow_fuzzy=False)
    if direct_district:
        if normalize_text(raw_district) != normalize_text(direct_district):
            return {
                "district_label": direct_district,
                "note": f"{raw_district} was matched to the model district {direct_district}.",
                "used_fallback": True,
            }
        return {
            "district_label": direct_district,
            "note": None,
            "used_fallback": False,
        }

    geo_entry = resolve_geo_district_entry(raw_district, state_label, allow_fuzzy=True)
    if geo_entry and geo_entry.get("prediction_district"):
        prediction_district = geo_entry["prediction_district"]

        if geo_entry.get("model_district"):
            note = (
                f"{geo_entry['raw_district']} was matched to the model district "
                f"{prediction_district}."
            )
        else:
            note = (
                f"{geo_entry['raw_district']} is not covered directly by the model. "
                f"Showing prediction using the nearest supported district in {state_label}: "
                f"{prediction_district}."
            )

        return {
            "district_label": prediction_district,
            "note": note,
            "used_fallback": True,
        }

    raise ValueError("Invalid or unsupported district for the selected state.")


def resolve_prediction_location(raw_state, raw_district):
    state_label = resolve_state_label(raw_state)

    if state_label:
        district_resolution = resolve_prediction_district(raw_district, state_label)
        district_resolution["state_label"] = state_label
        return district_resolution

    geo_entry = resolve_geo_district_entry(raw_district, raw_state, allow_fuzzy=True)
    if geo_entry and geo_entry.get("prediction_state") and geo_entry.get("prediction_district"):
        return {
            "state_label": geo_entry["prediction_state"],
            "district_label": geo_entry["prediction_district"],
            "note": (
                f"{raw_state} is not covered directly by the model. "
                f"Showing prediction using the nearest supported district: "
                f"{geo_entry['prediction_district']}, {geo_entry['prediction_state']}."
            ),
            "used_fallback": True,
        }

    raise ValueError("Invalid or unsupported state selected.")


def predict_crop_and_revenue(state, district, season, area):
    """Apply the same flow as recommend_crop(): classify crop, then predict revenue."""
    location_resolution = resolve_prediction_location(state, district)
    state_label = location_resolution["state_label"]
    district_resolution = location_resolution
    district_label = district_resolution["district_label"]

    season_label = resolve_label(season, SEASON_LOOKUP, SEASON_ALIASES)
    if not season_label:
        raise ValueError("Invalid season. Please choose KHARIF, RABI, or ZAID.")

    state_encoded = LE_STATE.transform([state_label])[0]
    district_encoded = LE_DIST.transform([district_label])[0]
    season_encoded = LE_SEASON.transform([season_label])[0]

    crop_input = pd.DataFrame(
        [[state_encoded, district_encoded, PREDICTION_YEAR, season_encoded]],
        columns=list(CROP_MODEL.feature_names_in_),
    )

    crop_prediction = int(CROP_MODEL.predict(crop_input)[0])
    crop_label = LE_CROP.inverse_transform([crop_prediction])[0]
    crop_key = normalize_text(crop_label)

    average_yield = get_average_yield(crop_key)
    rainfall = RAINFALL_LOOKUP.get(normalize_text(state_label))
    market_price = MARKET_PRICE_LOOKUP.get(crop_key)

    if average_yield is None:
        raise ValueError("Average yield data is missing for the predicted crop.")
    if rainfall is None:
        raise ValueError("Rainfall data is missing for the selected state.")
    if market_price is None:
        raise ValueError("Market price data is missing for the predicted crop.")

    revenue_input = pd.DataFrame(
        [[
            state_encoded,
            district_encoded,
            PREDICTION_YEAR,
            season_encoded,
            crop_prediction,
            average_yield,
            rainfall,
            market_price,
        ]],
        columns=list(REVENUE_MODEL.feature_names_in_),
    )

    revenue_per_hectare = float(REVENUE_MODEL.predict(revenue_input)[0])
    total_revenue = revenue_per_hectare * area

    return {
        "crop": format_label(crop_label),
        "revenue_per_hectare": round(revenue_per_hectare, 2),
        "total_revenue": round(total_revenue, 2),
        "prediction_state": state_label,
        "prediction_district": district_label,
        "note": district_resolution["note"],
        "used_district_fallback": district_resolution["used_fallback"],
    }


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/app", methods=["GET"])
def index():
    frontend_config = {
        "states": SUPPORTED_STATES,
        "districts_by_state": DISTRICTS_BY_STATE,
    }
    return render_template("index.html", frontend_config=frontend_config)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    required_fields = ["state", "district", "season", "area"]
    missing_fields = [field for field in required_fields if payload.get(field) in (None, "")]
    if missing_fields:
        return jsonify(
            {"error": f"Missing input: {', '.join(missing_fields)}"}
        ), 400

    try:
        area = float(payload["area"])
    except (TypeError, ValueError):
        return jsonify({"error": "Area must be a valid number."}), 400

    if area <= 0:
        return jsonify({"error": "Area must be greater than 0."}), 400

    try:
        result = predict_crop_and_revenue(
            state=payload["state"],
            district=payload["district"],
            season=payload["season"],
            area=area,
        )
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        return jsonify({"error": "Something went wrong while making the prediction."}), 500


if __name__ == "__main__":
    app.run(debug=True)
