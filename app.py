from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # ⚠️ 临时关闭限制，避免报错
import shutil
shutil.rmtree("/home/adminuser/.cache/matplotlib", ignore_errors=True)

import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置全局字体配置
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 简体中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


import math
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import trapezoid
from utils.shap_plot import plot_shap_waterfall

# 页面配置
st.set_page_config(page_title="小儿UPJO预测平台", layout="wide")
st.title("小儿UPJO手术需求预测平台")
# 开关：是否启用调试模式
st.sidebar.markdown("🛠️ **调试工具**")
debug_mode = st.sidebar.checkbox("开启 SHAP 调试模式", value=False)
# 载入模型
@st.cache_resource
def load_model():
    return joblib.load("model/svm_model.pkl")

model = load_model()

# 特征提取函数
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

# 页面输入
st.header("1. 输入患者信息")
col1, col2, col3 = st.columns(3)
name = col1.text_input("姓名")
age = col2.number_input("年龄（岁）", 0, 18)
gender = col3.radio("性别", ["男", "女"])
side = st.radio("患侧", ["左", "右"])

st.header("2. 输入患肾APD（单位：cm）")
time_labels = ["0", "5", "15", "30", "45", "60"]
affected = [st.number_input(f"患{t}min", key=f"a{t}", step=0.1, format="%.1f") for t in time_labels]

st.header("3. 输入健肾APD（单位：cm）")
unaffected = [st.number_input(f"健{t}min", key=f"u{t}", step=0.1, format="%.1f") for t in time_labels]

st.header("4. 输入回缩速度")
col1, col2 = st.columns(2)
recoil_speed_a = col1.selectbox("患肾回缩速度", [1, 2, 3], key="rsa")
recoil_speed_u = col2.selectbox("健肾回缩速度", [1, 2, 3], key="rsu")



# 模型预测
if st.button("开始预测"):
    feature_names = ["回缩速度", "曲线面积", "45min下降百分比"]
    # 提取患肾特征
    features_a, (t_dense_a, smooth_a, raw_a) = extract_features(affected)
    features_a = [recoil_speed_a] + features_a
    X_a = pd.DataFrame([features_a], columns=feature_names)
    y_pred_a = model.predict(X_a)[0]
    y_prob_a = model.predict_proba(X_a)[0, 1]
    # 提取健肾特征
    features_u, (t_dense_u, smooth_u, raw_u) = extract_features(unaffected)
    features_u = [recoil_speed_u] + features_u
    X_u = pd.DataFrame([features_u], columns=feature_names)
    y_pred_u = model.predict(X_u)[0]
    y_prob_u = model.predict_proba(X_u)[0, 1]

    st.header(" 模型输入特征")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 患肾特征")
        st.markdown(f"- 回缩速度：**{features_a[0]}**")
        st.markdown(f"- 曲线面积：**{features_a[1]:.2f}**")
        st.markdown(f"- 45min下降百分比：**{features_a[2]:.2f}**")

    with col2:
        st.markdown("#### 健肾特征")
        st.markdown(f"- 回缩速度：**{features_u[0]}**")
        st.markdown(f"- 曲线面积：**{features_u[1]:.2f}**")
        st.markdown(f"- 45min下降百分比：**{features_u[2]:.2f}**")

    st.header("5. 预测结果")
    col1, col2 = st.columns(2)
    col1.success(f"患肾：{'需要手术' if y_pred_a==1 else '无需手术'}，概率：{y_prob_a:.2f}")
    col2.success(f"健肾：{'需要手术' if y_pred_u==1 else '无需手术'}，概率：{y_prob_u:.2f}")

    st.header("6. 双肾排泄曲线")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    time_points = np.array([0, 5, 15, 30, 45, 60])
    affected_spline = make_interp_spline(time_points, raw_a, k=3)
    unaffected_spline = make_interp_spline(time_points, raw_u, k=3)
    affected_dense = np.linspace(0, 60, 300)
    unaffected_dense = np.linspace(0, 60, 300)
    affected_smooth = affected_spline(affected_dense)
    unaffected_smooth = unaffected_spline(unaffected_dense)
    ax.plot(affected_dense, affected_smooth, label='患肾', color='#800080')
    ax.plot(unaffected_dense, unaffected_smooth, label='健肾', color='#FF8C00')
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

    
    st.header("8. 模型解释 (SHAP Waterfall Plot)")

    st.subheader("患肾 Waterfall 图")
    fig_wfa = plot_shap_waterfall(model, X_a, feature_names=feature_names, debug=debug_mode)
    # 修改模型解释部分的调用逻辑
    if fig_wfa:
        try:
            st.pyplot(fig_wfa, clear_figure=True, use_container_width=True)  # 使用容器宽度自适应
        except Exception as e:
            st.error(f"图像渲染失败: {str(e)}")
    else:
        st.warning("SHAP解释图生成失败，请检查输入数据格式。")

    st.subheader("健肾 Waterfall 图")
    fig_wfu = plot_shap_waterfall(model, X_u, feature_names=feature_names, debug=debug_mode)
    # 修改模型解释部分的调用逻辑
    if fig_wfu:
        try:
            st.pyplot(fig_wfu, clear_figure=True, use_container_width=True)  # 使用容器宽度自适应
        except Exception as e:
            st.error(f"图像渲染失败: {str(e)}")
    else:
        st.warning("SHAP解释图生成失败，请检查输入数据格式。")







