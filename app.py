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
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Pediatric UPJO Prediction Platform", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Noto Sans', 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("Pediatric UPJO Surgery Prediction Platform")

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

# Input
st.header("1. Enter Patient Information")
col1, col2, col3 = st.columns(3)
name = col1.text_input("Name")
age = col2.number_input("Age (years)", 0, 18)
gender_input = col3.radio("Gender", ["Male", "Female"])
gender = {"Male": "男", "Female": "女"}[gender_input]

st.header("2. Enter Left Kidney APD (cm)")
time_labels = ["0", "5", "15", "30", "45", "60"]
left_kidney = [st.number_input(f"Left {t}min", key=f"l{t}", step=0.1, format="%.1f") for t in time_labels]

st.header("3. Enter Right Kidney APD (cm)")
right_kidney = [st.number_input(f"Right {t}min", key=f"r{t}", step=0.1, format="%.1f") for t in time_labels]

st.header("4. Enter Recoil Speed")
col1, col2 = st.columns(2)
recoil_speed_l = col1.selectbox("Left Kidney Recoil Speed", [1, 2, 3], key="rsl")
recoil_speed_r = col2.selectbox("Right Kidney Recoil Speed", [1, 2, 3], key="rsr")

if st.button("Run Prediction"):
    feature_names = ["回缩速度", "曲线面积", "45min下降百分比"]

    features_l, (t_dense_l, smooth_l, raw_l) = extract_features(left_kidney)
    features_l = [recoil_speed_l] + features_l
    X_l = pd.DataFrame([features_l], columns=feature_names)
    y_pred_l = model.predict(X_l)[0]
    y_prob_l = model.predict_proba(X_l)[0, 1]

    features_r, (t_dense_r, smooth_r, raw_r) = extract_features(right_kidney)
    features_r = [recoil_speed_r] + features_r
    X_r = pd.DataFrame([features_r], columns=feature_names)
    y_pred_r = model.predict(X_r)[0]
    y_prob_r = model.predict_proba(X_r)[0, 1]

    st.header("Model Input Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Left Kidney Features")
        st.markdown(f"- Recoil Speed: **{features_l[0]}**")
        st.markdown(f"- Curve Area: **{features_l[1]:.2f}**")
        st.markdown(f"- 45min Decline: **{features_l[2]:.2f}%**")

    with col2:
        st.markdown("#### Right Kidney Features")
        st.markdown(f"- Recoil Speed: **{features_r[0]}**")
        st.markdown(f"- Curve Area: **{features_r[1]:.2f}**")
        st.markdown(f"- 45min Decline: **{features_r[2]:.2f}%**")

    st.header("5. Prediction Results")
    col1, col2 = st.columns(2)
    col1.success(f"Left Kidney: {'Surgery Needed' if y_pred_l==1 else 'No Surgery Needed'}, Probability: {y_prob_l:.2f}")
    col2.success(f"Right Kidney: {'Surgery Needed' if y_pred_r==1 else 'No Surgery Needed'}, Probability: {y_prob_r:.2f}")

    st.header("6. Bilateral Kidney Drainage Curves")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    time_points = np.array([0, 5, 15, 30, 45, 60])
    left_spline = make_interp_spline(time_points, raw_l, k=3)
    right_spline = make_interp_spline(time_points, raw_r, k=3)
    left_dense = np.linspace(0, 60, 300)
    right_dense = np.linspace(0, 60, 300)
    left_smooth = left_spline(left_dense)
    right_smooth = right_spline(right_dense)
    ax.plot(left_dense, left_smooth, label='Left Kidney', color='#800080')
    ax.plot(right_dense, right_smooth, label='Right Kidney', color='#FF8C00')
    ax.scatter(time_points, raw_l, color='#800080', s=40)
    ax.scatter(time_points, raw_r, color='#FF8C00', s=40)
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
    st.subheader("Left Kidney Waterfall Plot")
    fig_l = plot_shap_waterfall(model, X_l, feature_names=feature_names, debug=False)

    if fig_l:
        try:
            st.pyplot(fig_l, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render plot: {str(e)}")
    else:
        st.warning("Failed to generate SHAP plot. Please check input data.")

    st.subheader("Right Kidney Waterfall Plot")
    fig_r = plot_shap_waterfall(model, X_r, feature_names=feature_names, debug=False)

    if fig_r:
        try:
            st.pyplot(fig_r, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render plot: {str(e)}")
    else:
        st.warning("Failed to generate SHAP plot. Please check input data.")







