import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌾",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #f5f8f3;
}

.block-container {
    padding-top: 1.5rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 20px;
}

.cloud-box {
    padding: 14px;
    border-radius: 12px;
    background: #e8f5e9;
    border: 1px solid #b7dfb9;
    text-align: center;
    font-weight: 600;
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg,#e9f7e8,#ffffff);
    border: 1px solid #c9e6c5;
}

.big {
    font-size: 32px;
    font-weight: 800;
}

.small {
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CROP DATA
# N, P, K, Temperature, Humidity, pH, Rainfall
# =========================================================

crop_profiles = {
    "Rice": [90, 42, 43, 21, 82, 6.5, 205],
    "Maize": [78, 48, 42, 22.5, 65, 6.0, 85],
    "Chickpea": [40, 68, 80, 18.5, 16, 7.0, 65],
    "Kidney Beans": [20, 68, 20, 20, 22, 5.8, 105],
    "Pigeon Peas": [20, 65, 20, 27, 48, 6.0, 110],
    "Moth Beans": [25, 55, 20, 28.5, 55, 7.0, 45],
    "Mung Bean": [25, 55, 20, 28, 80, 6.5, 50],
    "Black Gram": [40, 68, 20, 27, 65, 7.0, 70],
    "Lentil": [20, 68, 20, 18, 65, 6.9, 45],
    "Pomegranate": [18, 18, 40, 21, 90, 6.5, 110],
    "Banana": [100, 82, 50, 27, 80, 6.0, 105],
    "Mango": [20, 28, 30, 31, 50, 5.8, 95],
    "Grapes": [25, 35, 30, 24, 80, 6.0, 70],
    "Watermelon": [100, 17, 50, 26, 85, 6.5, 50],
    "Muskmelon": [100, 18, 50, 27, 90, 6.5, 35],
    "Apple": [20, 135, 200, 22, 90, 6.0, 115],
    "Orange": [20, 20, 10, 22, 92, 6.5, 110],
    "Papaya": [50, 55, 50, 25, 90, 6.5, 150],
    "Coconut": [20, 20, 30, 27, 95, 5.8, 160],
    "Cotton": [120, 45, 20, 24, 75, 7.0, 80],
    "Jute": [80, 40, 40, 26, 85, 6.5, 175],
    "Coffee": [100, 20, 30, 23.5, 60, 6.0, 160]
}

crop_names = list(crop_profiles.keys())

# =========================================================
# CREATE AI MODELS
# =========================================================

@st.cache_resource
def create_models():

    rng = np.random.default_rng(42)

    X = []
    y_crop = []
    y_yield = []

    for crop, profile in crop_profiles.items():

        N, P, K, temp, humidity, ph, rainfall = profile

        for _ in range(120):

            n = np.clip(
                rng.normal(N, max(4, N * 0.12)),
                0, 140
            )

            p = np.clip(
                rng.normal(P, max(4, P * 0.12)),
                0, 145
            )

            k = np.clip(
                rng.normal(K, max(4, K * 0.12)),
                0, 210
            )

            t = np.clip(
                rng.normal(temp, 2.5),
                8, 45
            )

            h = np.clip(
                rng.normal(humidity, 8),
                10, 100
            )

            ph_value = np.clip(
                rng.normal(ph, 0.4),
                3.5, 9.5
            )

            rain = np.clip(
                rng.normal(
                    rainfall,
                    max(8, rainfall * 0.15)
                ),
                20, 300
            )

            X.append([
                n, p, k,
                t, h,
                ph_value,
                rain
            ])

            y_crop.append(crop)

            nutrient_score = (
                1 - abs(n - N) / (N + 30)
                + 1 - abs(p - P) / (P + 30)
                + 1 - abs(k - K) / (K + 30)
            ) / 3

            weather_score = (
                1 - abs(t - temp) / 25
                + 1 - abs(h - humidity) / 100
                + 1 - abs(rain - rainfall) /
                (rainfall + 100)
            ) / 3

            base_yield = 2.5 + (
                nutrient_score + weather_score
            ) * 2

            yield_value = max(
                0.5,
                base_yield + rng.normal(0, 0.25)
            )

            y_yield.append(yield_value)

    X = np.array(X)
    y_crop = np.array(y_crop)
    y_yield = np.array(y_yield)

    # -----------------------------
    # Classification
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_crop,
        test_size=0.20,
        random_state=42,
        stratify=y_crop
    )

    crop_model = RandomForestClassifier(
        n_estimators=180,
        random_state=42
    )

    crop_model.fit(
        X_train,
        y_train
    )

    prediction = crop_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    # -----------------------------
    # Regression
    # -----------------------------

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X,
        y_yield,
        test_size=0.20,
        random_state=42
    )

    yield_model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    yield_model.fit(
        Xr_train,
        yr_train
    )

    yield_prediction = yield_model.predict(
        Xr_test
    )

    mae = mean_absolute_error(
        yr_test,
        yield_prediction
    )

    return (
        crop_model,
        yield_model,
        accuracy,
        mae
    )


crop_model, yield_model, model_accuracy, model_mae = create_models()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🌾 Smart Agriculture AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Cloud-Based Intelligent Agriculture Decision Support System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="cloud-box">
    ☁️ CLOUD-BASED AI APPLICATION
    <br>
    Streamlit Community Cloud + GitHub + Artificial Intelligence
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌱 Farm Parameters")

st.sidebar.info(
    "Enter soil and environmental conditions. "
    "The AI system will analyse the farm."
)

N = st.sidebar.slider(
    "Nitrogen (N)",
    0, 140, 80
)

P = st.sidebar.slider(
    "Phosphorus (P)",
    0, 145, 45
)

K = st.sidebar.slider(
    "Potassium (K)",
    0, 210, 40
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    8.0, 45.0, 25.0
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    10.0, 100.0, 70.0
)

ph = st.sidebar.slider(
    "Soil pH",
    3.5, 9.5, 6.5
)

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    20.0, 300.0, 100.0
)

st.sidebar.button(
    "🤖 ANALYSE FARM",
    use_container_width=True
)

# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "AI Model Accuracy",
        f"{model_accuracy * 100:.1f}%"
    )

with c2:
    st.metric(
        "Crops Analysed",
        len(crop_names)
    )

with c3:
    st.metric(
        "Soil pH",
        f"{ph:.1f}"
    )

with c4:
    st.metric(
        "Rainfall",
        f"{rainfall:.0f} mm"
    )

st.divider()

# =========================================================
# AI INPUT
# =========================================================

input_data = np.array([[
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
]])

# =========================================================
# CROP RECOMMENDATION
# =========================================================

probabilities = crop_model.predict_proba(
    input_data
)[0]

classes = crop_model.classes_

sorted_indices = np.argsort(
    probabilities
)[::-1]

top_crops = [
    (
        classes[i],
        probabilities[i]
    )
    for i in sorted_indices[:5]
]

recommended_crop = top_crops[0][0]

confidence = top_crops[0][1] * 100

# =========================================================
# YIELD
# =========================================================

predicted_yield = yield_model.predict(
    input_data
)[0]

predicted_yield = max(
    0.5,
    round(predicted_yield, 2)
)

# =========================================================
# SUITABILITY
# =========================================================

profile = np.array(
    crop_profiles[recommended_crop],
    dtype=float
)

current = np.array([
    N, P, K,
    temperature,
    humidity,
    ph,
    rainfall
])

ranges = np.array([
    100, 100, 120,
    20, 100, 5, 250
])

difference = np.abs(
    current - profile
)

component_scores = 100 * (
    1 - difference / ranges
)

component_scores = np.clip(
    component_scores,
    0, 100
)

suitability = int(
    np.mean(component_scores)
)

final_score = int(
    confidence * 0.65 +
    suitability * 0.35
)

final_score = min(
    99,
    max(50, final_score)
)

# =========================================================
# RISK
# =========================================================

risk_points = 0
risk_messages = []

if ph < 5.5 or ph > 8.0:
    risk_points += 1
    risk_messages.append(
        "Soil pH is outside the preferred range."
    )

if humidity > 90:
    risk_points += 1
    risk_messages.append(
        "Very high humidity may increase disease risk."
    )

if rainfall < 40:
    risk_points += 1
    risk_messages.append(
        "Low rainfall indicates possible water stress."
    )

if rainfall > 250:
    risk_points += 1
    risk_messages.append(
        "Very high rainfall may create waterlogging risk."
    )

if temperature > 35:
    risk_points += 1
    risk_messages.append(
        "High temperature may cause heat stress."
    )

if temperature < 12:
    risk_points += 1
    risk_messages.append(
        "Low temperature may slow crop growth."
    )

if risk_points == 0:
    risk_level = "LOW"
elif risk_points <= 2:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# =========================================================
# IRRIGATION
# =========================================================

if rainfall < 50:

    irrigation = "HIGH"

    irrigation_text = (
        "Frequent irrigation may be required."
    )

elif rainfall < 120:

    irrigation = "MEDIUM"

    irrigation_text = (
        "Moderate irrigation is recommended."
    )

else:

    irrigation = "LOW"

    irrigation_text = (
        "Rainfall may satisfy a major part "
        "of water demand."
    )

# =========================================================
# AI RESULT
# =========================================================

st.subheader(
    "🤖 AI Decision Support"
)

r1, r2 = st.columns([1.4, 1])

with r1:

    st.markdown(
        f"""
        <div class="result">

        <div class="small">
        AI RECOMMENDED CROP
        </div>

        <div class="big">
        🌱 {recommended_crop}
        </div>

        <br>

        <b>AI Confidence:</b>
        {confidence:.1f}%

        <br>

        <b>Suitability Score:</b>
        {final_score}/100

        <br>

        <b>Expected Yield:</b>
        {predicted_yield} tons/hectare

        <br>

        <b>Water Requirement:</b>
        {irrigation}

        </div>
        """,
        unsafe_allow_html=True
    )

with r2:

    st.markdown("### ⚠️ Farm Risk")

    if risk_level == "LOW":

        st.success(
            "LOW RISK — Conditions are favourable."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "MEDIUM RISK — Monitor farm conditions."
        )

    else:

        st.error(
            "HIGH RISK — Immediate monitoring recommended."
        )

    st.info(
        irrigation_text
    )

# =========================================================
# TOP 5 CROPS
# =========================================================

st.subheader(
    "🌾 Top AI Crop Recommendations"
)

chart_crops = [
    x[0]
    for x in top_crops
]

chart_scores = [
    x[1] * 100
    for x in top_crops
]

fig = go.Figure(
    go.Bar(
        x=chart_scores,
        y=chart_crops,
        orientation="h",
        text=[
            f"{x:.1f}%"
            for x in chart_scores
        ],
        textposition="auto"
    )
)

fig.update_layout(
    title="AI Crop Probability",
    xaxis_title="AI Confidence (%)",
    yaxis_title="Crop",
    height=400
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# ALL CROPS
# =========================================================

st.subheader(
    "🌾 All 22 Crops Analysed"
)

st.write(
    ", ".join(crop_names)
)

# =========================================================
# ANALYTICS
# =========================================================

st.subheader(
    "🧪 Soil & Environmental Analytics"
)

a1, a2 = st.columns(2)

with a1:

    soil_values = [
        N,
        P,
        K
    ]

    fig_soil = go.Figure()

    fig_soil.add_trace(
        go.Bar(
            x=[
                "Nitrogen",
                "Phosphorus",
                "Potassium"
            ],
            y=soil_values,
            text=soil_values,
            textposition="auto"
        )
    )

    fig_soil.update_layout(
        title="Soil Nutrient Profile",
        yaxis_title="Value"
    )

    st.plotly_chart(
        fig_soil,
        use_container_width=True
    )

with a2:

    radar_categories = [
        "N",
        "P",
        "K",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall"
    ]

    normalized = [
        min(N / 140 * 100, 100),
        min(P / 145 * 100, 100),
        min(K / 210 * 100, 100),
        min(temperature / 45 * 100, 100),
        humidity,
        ph / 9.5 * 100,
        rainfall / 300 * 100
    ]

    radar = go.Figure()

    radar.add_trace(
        go.Scatterpolar(
            r=normalized,
            theta=radar_categories,
            fill="toself",
            name="Farm Profile"
        )
    )

    radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Farm Environmental Profile"
    )

    st.plotly_chart(
        radar,
        use_container_width=True
    )

# =========================================================
# WHY CROP
# =========================================================

st.subheader(
    "🧠 Why did AI recommend this crop?"
)

reason_list = []

target_profile = crop_profiles[
    recommended_crop
]

if abs(N - target_profile[0]) < 25:
    reason_list.append(
        "Nitrogen level is suitable."
    )

if abs(P - target_profile[1]) < 25:
    reason_list.append(
        "Phosphorus level is suitable."
    )

if abs(K - target_profile[2]) < 30:
    reason_list.append(
        "Potassium level is suitable."
    )

if abs(
    temperature - target_profile[3]
) < 5:

    reason_list.append(
        "Temperature is close to the crop's preferred range."
    )

if abs(
    humidity - target_profile[4]
) < 15:

    reason_list.append(
        "Humidity is favourable."
    )

if abs(
    ph - target_profile[5]
) < 1:

    reason_list.append(
        "Soil pH is compatible."
    )

if abs(
    rainfall - target_profile[6]
) < 50:

    reason_list.append(
        "Rainfall conditions are suitable."
    )

if not reason_list:

    reason_list.append(
        "Random Forest selected this crop based "
        "on the combined soil and environmental pattern."
    )

for reason in reason_list:

    st.write(
        "✅",
        reason
    )

# =========================================================
# RISK MONITORING
# =========================================================

st.subheader(
    "⚠️ AI Risk Monitoring"
)

if risk_messages:

    for message in risk_messages:

        st.warning(message)

else:

    st.success(
        "No major environmental risk detected "
        "from the supplied parameters."
    )

# =========================================================
# CLOUD ARCHITECTURE
# =========================================================

st.subheader(
    "☁️ Cloud Computing Architecture"
)

st.info(
    """
    👨‍🌾 User
       ↓
    🌐 Internet
       ↓
    ☁️ Streamlit Community Cloud
       ↓
    🖥️ Smart Agriculture AI Application
       ↓
    🤖 Random Forest AI Models
       ↓
    📊 AI Dashboard & Analytics
    """
)

# =========================================================
# TECHNICAL DETAILS
# =========================================================

with st.expander(
    "🔬 AI & Cloud Technical Details"
):

    st.write(
        """
        **Artificial Intelligence:**
        Random Forest Classification and Regression

        **Classification:**
        Crop recommendation

        **Regression:**
        Expected yield prediction

        **Input Parameters:**
        Nitrogen, Phosphorus, Potassium,
        Temperature, Humidity, Soil pH and Rainfall

        **Application Framework:**
        Streamlit

        **Visualization:**
        Plotly

        **Machine Learning Library:**
        Scikit-learn

        **Numerical Processing:**
        NumPy

        **Cloud Platform:**
        Streamlit Community Cloud

        **Cloud Source Repository:**
        GitHub

        **Cloud Architecture:**
        User → Internet → Streamlit Cloud →
        AI Application → AI Models → Dashboard
        """
    )

    st.metric(
        "Yield Model MAE",
        f"{model_mae:.2f} tons/hectare"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <center>
    🌾 <b>Smart Agriculture AI</b><br>
    Cloud-Based Intelligent Farming Decision Support System
    <br><br>
    Developed by <b>Krupa Modhale</b>
    </center>
    """,
    unsafe_allow_html=True
)