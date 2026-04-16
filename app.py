import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

HAS_XGB = True
HAS_SHAP = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

try:
    import shap
except Exception:
    HAS_SHAP = False

st.set_page_config(page_title="NYC AQI Interactive Demo", layout="wide")
st.title("🌫️ NYC AQI Interactive Demo")
st.caption("AQI in this app is PM2.5-based daily air quality index (city-level, from EPA).")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    df["Traffic_Index"] = df["Traffic_Vol"] / (df["Wind_Speed"] + 0.1)
    df["Temp_Change"] = df["Temp_Avg"].diff().fillna(0)
    df["Target_NextDay_AQI"] = df["AQI"].shift(-1)

    df = df.dropna().copy()
    df = df[(df["AQI"] <= 150) & (df["Target_NextDay_AQI"] <= 150) & (df["AQI"] >= 0) & (df["Target_NextDay_AQI"] >= 0)]
    return df.reset_index(drop=True)


@st.cache_resource
def train_models(df: pd.DataFrame):
    feature_cols = [
        "Traffic_Vol", "Temp_Avg", "Precipitation", "Wind_Speed",
        "Pressure", "AQI", "Traffic_Index", "Temp_Change",
    ]

    X = df[feature_cols]
    y = df["Target_NextDay_AQI"]
    d = df["Date"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    d_test = d.iloc[split:]

    models = {}

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    models["Linear Regression"] = lin

    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, objective="reg:squarederror",
        )
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb

    compare = pd.DataFrame({
        "Date": d_test.values,
        "AQI_Today": X_test["AQI"].values,
        "Actual": y_test.values,
    })

    metrics = {}
    for name, m in models.items():
        pred = m.predict(X_test)
        compare[name] = pred

        true_dir = np.sign(y_test.values - X_test["AQI"].values)
        pred_dir = np.sign(pred - X_test["AQI"].values)

        metrics[name] = {
            "Directional_Accuracy": float((true_dir == pred_dir).mean()),
            "R2": float(r2_score(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": float(mean_absolute_error(y_test, pred)),
        }

    return models, metrics, feature_cols, X_test.reset_index(drop=True), y_test.reset_index(drop=True), compare.reset_index(drop=True)


def direction_label(delta: float):
    if delta > 0.5:
        return "⬆️ Up"
    if delta < -0.5:
        return "⬇️ Down"
    return "➡️ Flat"


def local_similar_direction_acc(X_test, y_test, pred_arr, query_row, k=40):
    mu = X_test.mean()
    sd = X_test.std().replace(0, 1)

    z_test = (X_test - mu) / sd
    z_q = (query_row - mu) / sd
    dist = ((z_test - z_q) ** 2).sum(axis=1).pow(0.5)

    idx = np.argsort(dist.values)[: min(k, len(dist))]
    Xn = X_test.iloc[idx]
    yn = y_test.iloc[idx].values
    pn = pred_arr[idx]

    true_dir = np.sign(yn - Xn["AQI"].values)
    pred_dir = np.sign(pn - Xn["AQI"].values)
    return float((true_dir == pred_dir).mean())


def linear_contrib(model, row, feature_cols):
    coef = pd.Series(model.coef_, index=feature_cols)
    vals = pd.Series(row[feature_cols].values, index=feature_cols)
    return (coef * vals).sort_values(key=lambda s: np.abs(s), ascending=False)


def xgb_shap_contrib(model, input_df, feature_cols):
    explainer = shap.Explainer(model)
    sv = explainer(input_df[feature_cols])
    return pd.Series(sv.values[0], index=feature_cols).sort_values(key=lambda s: np.abs(s), ascending=False)


def level_text(val, series):
    q33, q66 = series.quantile(0.33), series.quantile(0.66)
    if val <= q33:
        return "Low"
    if val <= q66:
        return "Medium"
    return "High"


# ----- load -----
DATA_PATH = "final_model_data.csv"
df = load_data(DATA_PATH)
models, metrics, feature_cols, X_test, y_test, compare_df = train_models(df)

if HAS_XGB:
    active_model_name = "XGBoost"
    st.info("Linear baseline can be shown in the comparison chart.")
else:
    active_model_name = "Linear Regression"
    st.warning("XGBoost not available in this environment. Falling back to Linear Regression.")

active_model = models[active_model_name]
active_pred_test = compare_df[active_model_name].values

# ----- mode switch -----
mode = st.radio("Mode", ["What-if Scenario", "Historical Challenge"], horizontal=True)

# ----- sidebar inputs -----
st.sidebar.header("🎛️ Inputs")
med = df[feature_cols].median()
q01 = df[feature_cols].quantile(0.01)
q99 = df[feature_cols].quantile(0.99)

traffic = st.sidebar.slider("Traffic Volume", float(q01.Traffic_Vol), float(q99.Traffic_Vol), float(np.clip(med.Traffic_Vol, q01.Traffic_Vol, q99.Traffic_Vol)), step=100.0)
temp = st.sidebar.slider("Average Temperature (°C)", float(q01.Temp_Avg), float(q99.Temp_Avg), float(np.clip(med.Temp_Avg, q01.Temp_Avg, q99.Temp_Avg)), step=0.1)
prec = st.sidebar.slider("Precipitation (mm)", float(q01.Precipitation), float(q99.Precipitation), float(np.clip(med.Precipitation, q01.Precipitation, q99.Precipitation)), step=0.1)
wind = st.sidebar.slider("Wind Speed (m/s)", float(q01.Wind_Speed), float(q99.Wind_Speed), float(np.clip(med.Wind_Speed, q01.Wind_Speed, q99.Wind_Speed)), step=0.1)
press = st.sidebar.slider("Pressure (hPa)", float(q01.Pressure), float(q99.Pressure), float(np.clip(med.Pressure, q01.Pressure, q99.Pressure)), step=0.1)
aqi_today = st.sidebar.slider("Today's AQI", float(max(0, q01.AQI)), 150.0, float(np.clip(med.AQI, max(0, q01.AQI), 150.0)), step=0.1)
st.sidebar.caption("AQI is capped at 150 because AQI > 150 is rare/extreme in this NYC study setup.")
temp_change = st.sidebar.slider("Temperature Change From Yesterday", float(q01.Temp_Change), float(q99.Temp_Change), float(np.clip(med.Temp_Change, q01.Temp_Change, q99.Temp_Change)), step=0.1)

traffic_index = traffic / (wind + 0.1)
input_df = pd.DataFrame([{
    "Traffic_Vol": traffic,
    "Temp_Avg": temp,
    "Precipitation": prec,
    "Wind_Speed": wind,
    "Pressure": press,
    "AQI": aqi_today,
    "Traffic_Index": traffic_index,
    "Temp_Change": temp_change,
}])

# ----- main content -----
if mode == "What-if Scenario":
    pred_next = float(active_model.predict(input_df[feature_cols])[0])
    delta = pred_next - aqi_today

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Next-day AQI", f"{pred_next:.1f}", f"{delta:+.1f} vs today")
    c2.metric("Predicted Direction", direction_label(delta))
    c3.metric("Directional Accuracy (test)", f"{metrics[active_model_name]['Directional_Accuracy']*100:.1f}%")

    with st.expander("Technical metrics (optional)"):
        st.write({
            "Model": active_model_name,
            "R2": round(metrics[active_model_name]["R2"], 3),
            "RMSE": round(metrics[active_model_name]["RMSE"], 3),
            "MAE": round(metrics[active_model_name]["MAE"], 3),
        })

    st.subheader("🔍 What impacts the prediction?")
    if active_model_name == "XGBoost" and HAS_SHAP:
        try:
            s = xgb_shap_contrib(active_model, input_df, feature_cols)
            top = s.head(8).sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
            fig.update_layout(title="SHAP value impact (local)", xaxis_title="SHAP value")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("SHAP value > 0 pushes predicted AQI upward; SHAP value < 0 pushes it downward.")
        except Exception as e:
            st.warning(f"SHAP failed in this env: {e}")
    else:
        contrib = linear_contrib(active_model, input_df.iloc[0], feature_cols)
        top = contrib.head(8).sort_values()
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
        fig.update_layout(title="Linear contribution proxy (coef × input)", xaxis_title="Impact score")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("🎯 Challenge the model")
    st.caption("Pick a historical date from test data, make your guess, then reveal model result and true outcome.")

    options = compare_df["Date"].dt.strftime("%Y-%m-%d").tolist()
    idx = st.selectbox("Historical date", list(range(len(options))), format_func=lambda i: options[i])

    row_cmp = compare_df.iloc[idx]
    row_x = X_test.iloc[idx]

    # Context card so audience can make an informed guess
    st.markdown("**Scenario card (selected date conditions)**")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Traffic", f"{row_x['Traffic_Vol']:.0f}", level_text(row_x['Traffic_Vol'], df['Traffic_Vol']))
    k2.metric("Wind", f"{row_x['Wind_Speed']:.1f}", level_text(row_x['Wind_Speed'], df['Wind_Speed']))
    k3.metric("Precip", f"{row_x['Precipitation']:.1f}", level_text(row_x['Precipitation'], df['Precipitation']))
    k4.metric("Pressure", f"{row_x['Pressure']:.1f}", level_text(row_x['Pressure'], df['Pressure']))
    k5.metric("AQI today", f"{row_cmp['AQI_Today']:.1f}")

    st.caption("Use the scenario card above to make your direction guess before revealing the result.")

    user_guess = st.radio("Your guess for next-day direction", ["Up", "Down"], horizontal=True)
    if st.button("Reveal"):
        model_delta = row_cmp[active_model_name] - row_cmp["AQI_Today"]
        true_delta = row_cmp["Actual"] - row_cmp["AQI_Today"]
        model_dir = "Up" if model_delta > 0 else "Down"
        true_dir = "Up" if true_delta > 0 else "Down"

        local_da = local_similar_direction_acc(X_test, y_test, active_pred_test, row_x, k=40)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model direction", model_dir)
        c2.metric("True direction", true_dir)
        c3.metric("Similar-scenario directional accuracy", f"{local_da*100:.1f}%")

        hit = (user_guess == true_dir)
        st.success("✅ Your guess hit the true outcome!" if hit else "❌ Your guess missed this date.")

        st.write({
            "Date": options[idx],
            "AQI_today": round(float(row_cmp["AQI_Today"]), 2),
            "Pred_next": round(float(row_cmp[active_model_name]), 2),
            "True_next": round(float(row_cmp["Actual"]), 2),
        })

# ----- comparison chart -----
st.subheader("📈 Model comparison over time")
show_linear = st.checkbox("Show linear baseline", value=False)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=compare_df["Date"], y=compare_df["Actual"], mode="lines", name="Actual", line=dict(color="gray", width=2)))

# Active model line
fig2.add_trace(go.Scatter(
    x=compare_df["Date"], y=compare_df[active_model_name], mode="lines", name=active_model_name,
    line=dict(color="firebrick" if active_model_name == "XGBoost" else "royalblue", width=3)
))

if show_linear and "Linear Regression" in compare_df.columns and active_model_name != "Linear Regression":
    fig2.add_trace(go.Scatter(
        x=compare_df["Date"], y=compare_df["Linear Regression"], mode="lines", name="Linear Regression",
        line=dict(color="royalblue", width=1.8, dash="dash"), opacity=0.7
    ))

fig2.update_layout(xaxis_title="Date", yaxis_title="Next-day AQI", height=420)
st.plotly_chart(fig2, use_container_width=True)
