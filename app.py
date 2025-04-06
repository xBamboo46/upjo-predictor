

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import shutil
shutil.rmtree("/home/adminuser/.cache/matplotlib", ignore_errors=True)

import math
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import trapezoid
from utils.shap_plot import plot_shap_waterfall


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'  # ✅ 强制用默认字体，避免找不到
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Pediatric UPJO Prediction Platform", layout="wide")

# Apply English font site-wide
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Noto Sans', 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Pediatric UPJO Surgery Prediction Platform")
# st.sidebar.markdown("🛠️ **Debug Tools**")
# debug_mode = st.sidebar.checkbox("Enable SHAP Debug Mode", value=False)

@st.cache_resource
def load_model():
    return joblib.load("model/svm_model.pkl")

model = load_model()

def extract_features(apd_values):
    time_points = np.array([0, 5, 15, 30, 45, 60])
    apd_values = np.array(apd_values)
    spline = make_interp_spline(time_points, apd_values, k=3)
    dense_time = np.linspace(0, 60, 300)
    smooth_curve = spline(dense_time)
    area = trapezoid(smooth_curve, dense_time)
    apd_0 = apd_values[0]
    apd_45 = apd_values[4]
    apd_max = np.max(apd_values)
    decline_ratio = (apd_max - apd_45) / (apd_max - apd_0) * 100 if (apd_max - apd_0) != 0 else 0
    return [area, decline_ratio], (dense_time, smooth_curve, apd_values)

# Input Section
st.header("1. Enter Patient Information")
col1, col2, col3 = st.columns(3)
name = col1.text_input("Name")
age = col2.number_input("Age (years)", 0, 18)
gender_input = col3.radio("Gender", ["Male", "Female"])
gender = {"Male": "男", "Female": "女"}[gender_input]

side_input = st.radio("Affected Side", ["Left", "Right"])
side = {"Left": "左", "Right": "右"}[side_input]

st.header("2. Enter Affected Kidney APD (cm)")
time_labels = ["0", "5", "15", "30", "45", "60"]
affected = [st.number_input(f"Affected {t}min", key=f"a{t}", step=0.1, format="%.1f") for t in time_labels]

st.header("3. Enter Unaffected Kidney APD (cm)")
unaffected = [st.number_input(f"Unaffected {t}min", key=f"u{t}", step=0.1, format="%.1f") for t in time_labels]

st.header("4. Enter Recoil Speed")
col1, col2 = st.columns(2)
recoil_speed_a = col1.selectbox("Affected Kidney Recoil Speed", [1, 2, 3], key="rsa")
recoil_speed_u = col2.selectbox("Unaffected Kidney Recoil Speed", [1, 2, 3], key="rsu")

if st.button("Run Prediction"):
    feature_names = ["回缩速度", "曲线面积", "45min下降百分比"]
    features_a, (t_dense_a, smooth_a, raw_a) = extract_features(affected)
    features_a = [recoil_speed_a] + features_a
    X_a = pd.DataFrame([features_a], columns=feature_names)
    y_pred_a = model.predict(X_a)[0]
    y_prob_a = model.predict_proba(X_a)[0, 1]

    features_u, (t_dense_u, smooth_u, raw_u) = extract_features(unaffected)
    features_u = [recoil_speed_u] + features_u
    X_u = pd.DataFrame([features_u], columns=feature_names)
    y_pred_u = model.predict(X_u)[0]
    y_prob_u = model.predict_proba(X_u)[0, 1]

    st.header("Model Input Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Affected Kidney Features")
        st.markdown(f"- Recoil Speed: **{features_a[0]}**")
        st.markdown(f"- Curve Area: **{features_a[1]:.2f}**")
        st.markdown(f"- 45min Decline: **{features_a[2]:.2f}%**")

    with col2:
        st.markdown("#### Unaffected Kidney Features")
        st.markdown(f"- Recoil Speed: **{features_u[0]}**")
        st.markdown(f"- Curve Area: **{features_u[1]:.2f}**")
        st.markdown(f"- 45min Decline: **{features_u[2]:.2f}%**")

    st.header("5. Prediction Results")
    col1, col2 = st.columns(2)
    col1.success(f"Affected Kidney: {'Surgery Needed' if y_pred_a==1 else 'No Surgery Needed'}, Probability: {y_prob_a:.2f}")
    col2.success(f"Unaffected Kidney: {'Surgery Needed' if y_pred_u==1 else 'No Surgery Needed'}, Probability: {y_prob_u:.2f}")

    st.header("6. Bilateral Kidney Drainage Curves")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    time_points = np.array([0, 5, 15, 30, 45, 60])
    affected_spline = make_interp_spline(time_points, raw_a, k=3)
    unaffected_spline = make_interp_spline(time_points, raw_u, k=3)
    affected_dense = np.linspace(0, 60, 300)
    unaffected_dense = np.linspace(0, 60, 300)
    affected_smooth = affected_spline(affected_dense)
    unaffected_smooth = unaffected_spline(unaffected_dense)
    ax.plot(affected_dense, affected_smooth, label='Affected Kidney', color='#800080')
    ax.plot(unaffected_dense, unaffected_smooth, label='Unaffected Kidney', color='#FF8C00')
    ax.scatter(time_points, raw_a, color='#800080', s=40)
    ax.scatter(time_points, raw_u, color='#FF8C00', s=40)
    ax.set_xlabel("Time (minutes)", fontsize=10)
    ax.set_ylabel("APD (cm)", fontsize=10)
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("7. Model Interpretation (SHAP Waterfall Plot)")
    st.subheader("Affected Kidney Waterfall Plot")
    fig_wfa = plot_shap_waterfall(model, X_a, feature_names=feature_names, debug=False)

    if fig_wfa:
        try:
            st.pyplot(fig_wfa, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render plot: {str(e)}")
    else:
        st.warning("Failed to generate SHAP plot. Please check input data.")

    st.subheader("Unaffected Kidney Waterfall Plot")
    fig_wfu = plot_shap_waterfall(model, X_u, feature_names=feature_names, debug=False)

    if fig_wfu:
        try:
            st.pyplot(fig_wfu, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render plot: {str(e)}")
    else:
        st.warning("Failed to generate SHAP plot. Please check input data.")








