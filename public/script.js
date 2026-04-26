const mapStatus = document.getElementById("map-status");
const selectedLocation = document.getElementById("selected-location");
const stateInput = document.getElementById("state");
const districtInput = document.getElementById("district");
const districtOptions = document.getElementById("district-options");
const districtHint = document.getElementById("district-hint");
const seasonInput = document.getElementById("season");
const areaInput = document.getElementById("area");
const predictButton = document.getElementById("predict-btn");
const resultMessage = document.getElementById("result-message");
const resultsGrid = document.getElementById("results-grid");
const cropResult = document.getElementById("crop-result");
const revenuePerHectareResult = document.getElementById("revenue-hectare-result");
const totalRevenueResult = document.getElementById("total-revenue-result");
const frontendConfig = JSON.parse(document.getElementById("frontend-config").textContent);
const states = frontendConfig.states || [];
const districtsByState = frontendConfig.districts_by_state || {};

const currencyFormatter = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
});

const defaultStyle = {
    color: "#dadce0",
    weight: 1,
    fillColor: "#f8f9fa",
    fillOpacity: 1,
};

const unsupportedStyle = {
    color: "#f1f3f4",
    weight: 0.8,
    fillColor: "#ffffff",
    fillOpacity: 1,
};

const hoverStyle = {
    color: "#1e8e3e",
    weight: 1.5,
    fillColor: "#e6f4ea",
    fillOpacity: 1,
};

const selectedStyle = {
    color: "#137333",
    weight: 2,
    fillColor: "#1e8e3e",
    fillOpacity: 0.2,
};

let selectedLayer = null;
let selectedFeature = null;

const map = L.map("map", {
    preferCanvas: true,
    zoomSnap: 0.5,
    zoomControl: true,
}).setView([22.5, 79.5], 4.5);

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    maxZoom: 18,
}).addTo(map);


function setMessage(message, type = "info") {
    resultMessage.textContent = message;
    resultMessage.className = `message-box ${type}`;
}


function getDistrictsForState(state) {
    return districtsByState[state] || [];
}


function populateStateOptions() {
    stateInput.innerHTML = states
        .map((state) => `<option value="${state}">${state}</option>`)
        .join("");

    if (states.length > 0) {
        stateInput.value = states[0];
    }
}


function updateDistrictOptions(state) {
    const districts = getDistrictsForState(state);
    districtOptions.innerHTML = districts
        .map((district) => `<option value="${district}"></option>`)
        .join("");

    districtHint.textContent = districts.length
        ? `Valid model districts for ${state}: ${districts.length} available.`
        : "No model districts available for the selected state.";
}


function updateSelectedLocation() {
    if (!selectedFeature) {
        selectedLocation.textContent = "No district selected yet";
        return;
    }

    if (selectedFeature.modelDistrict) {
        selectedLocation.textContent =
            `Map: ${selectedFeature.rawDistrict}, ${selectedFeature.rawState} | Model district: ${selectedFeature.modelDistrict}`;
        return;
    }

    if (selectedFeature.predictionDistrict) {
        selectedLocation.textContent =
            `Map: ${selectedFeature.rawDistrict}, ${selectedFeature.rawState} | Fallback prediction: ${selectedFeature.predictionDistrict}, ${selectedFeature.predictionState}`;
        return;
    }

    selectedLocation.textContent =
        `Map: ${selectedFeature.rawDistrict}, ${selectedFeature.rawState} | No direct model district match`;
}


function showResults(data) {
    cropResult.textContent = data.crop;
    const baseRevenue = data.revenue_per_hectare;

    // ±20%
    const lower = baseRevenue * 0.8;
    const upper = baseRevenue * 1.2;

    revenuePerHectareResult.textContent =
        `${currencyFormatter.format(lower)} – ${currencyFormatter.format(upper)}`;

    // total revenue range
    const totalLower = lower * (data.total_revenue / baseRevenue);
    const totalUpper = upper * (data.total_revenue / baseRevenue);

totalRevenueResult.textContent =
    `${currencyFormatter.format(totalLower)} – ${currencyFormatter.format(totalUpper)}`;
    resultsGrid.classList.remove("hidden");
    setMessage(data.note || "Prediction completed successfully.", "info");
}


function resetResults() {
    resultsGrid.classList.add("hidden");
    cropResult.textContent = "-";
    revenuePerHectareResult.textContent = "-";
    totalRevenueResult.textContent = "-";
}


function getBaseStyle(feature) {
    return feature.properties?.can_predict ? defaultStyle : unsupportedStyle;
}


function onEachFeature(feature, layer) {
    const district = feature.properties?.district || "Unknown District";
    const state = feature.properties?.state || "Unknown State";
    const modelDistrict = feature.properties?.model_district || "";
    const modelState = feature.properties?.model_state || "";
    const predictionState = feature.properties?.prediction_state || "";
    const predictionDistrict = feature.properties?.prediction_district || "";
    const canPredict = Boolean(feature.properties?.can_predict);

    layer.bindPopup(`<strong>${district}</strong><br>${state}`);

    layer.on("mouseover", () => {
        if (layer !== selectedLayer) {
            layer.setStyle(canPredict ? hoverStyle : unsupportedStyle);
        }
        layer.bringToFront();
    });

    layer.on("mouseout", () => {
        if (layer !== selectedLayer) {
            layer.setStyle(getBaseStyle(feature));
        }
    });

    layer.on("click", () => {
        if (selectedLayer) {
            selectedLayer.setStyle(getBaseStyle(selectedLayer.feature));
        }

        selectedLayer = layer;
        selectedLayer.setStyle(selectedStyle);

        selectedFeature = {
            rawDistrict: district,
            rawState: state,
            modelDistrict,
            modelState,
            predictionState,
            predictionDistrict,
            canPredict,
        };

        if (predictionState) {
            stateInput.value = predictionState;
            updateDistrictOptions(predictionState);
        }

        if (modelDistrict) {
            districtInput.value = modelDistrict;
            setMessage(
                "Map selection matched to your model district. You can still edit the district manually if needed.",
                "info",
            );
        } else if (predictionDistrict) {
            districtInput.value = predictionDistrict;
            setMessage(
                `This place is not covered directly by the model. Using fallback prediction from ${predictionDistrict}, ${predictionState}.`,
                "info",
            );
        } else if (modelState) {
            districtInput.value = "";
            setMessage(
                "This map district could not be matched to a nearby supported district. Choose from the suggestions or type a valid model district manually.",
                "error",
            );
        } else {
            districtInput.value = "";
            setMessage(
                "This state is not supported by the saved model. Please choose a supported state and district manually.",
                "error",
            );
        }

        updateSelectedLocation();
        resetResults();
    });
}


async function loadMap() {
    try {
        const response = await fetch("/INDIA_DISTRICTS.geojson");
        if (!response.ok) {
            throw new Error("GeoJSON file could not be loaded.");
        }

        const geoJson = await response.json();

        const geoJsonLayer = L.geoJSON(geoJson, {
            style: getBaseStyle,
            onEachFeature,
        }).addTo(map);

        map.fitBounds(geoJsonLayer.getBounds(), {
            padding: [16, 16],
        });

        mapStatus.textContent = "Map ready";
    } catch (error) {
        mapStatus.textContent = "Map error";
        setMessage(error.message, "error");
    }
}


async function predictCrop() {
    const state = stateInput.value;
    const district = districtInput.value.trim();

    if (!state) {
        setMessage("Please select a valid state.", "error");
        return;
    }

    if (!district) {
        setMessage("Please select a district from the map or type a valid model district manually.", "error");
        return;
    }

    if (!areaInput.value) {
        setMessage("Please enter the farm area in hectares.", "error");
        return;
    }

    const area = Number(areaInput.value);
    if (Number.isNaN(area) || area <= 0) {
        setMessage("Area must be a number greater than 0.", "error");
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";
    setMessage("Running the ML models for your selected district...", "info");
    resetResults();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                state,
                district,
                season: seasonInput.value,
                area,
            }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        showResults(data);
    } catch (error) {
        setMessage(error.message, "error");
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Crop";
    }
}


stateInput.addEventListener("change", () => {
    updateDistrictOptions(stateInput.value);

    if (!getDistrictsForState(stateInput.value).includes(districtInput.value.trim())) {
        districtInput.value = "";
    }

    setMessage("State updated. Choose or type a valid district for that state.", "info");
    resetResults();
});

predictButton.addEventListener("click", predictCrop);
populateStateOptions();
updateDistrictOptions(stateInput.value);
loadMap();
