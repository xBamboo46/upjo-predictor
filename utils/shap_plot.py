from utils.custom_shap_waterfall import plot_custom_waterfall

from matplotlib import rcParams
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

rcParams['font.family'] = 'Noto Sans CJK SC'
rcParams['axes.unicode_minus'] = False

def plot_shap_waterfall(pipeline_model, X_input, feature_names=None, debug=False):
    try:
        # ✅ 使用传入的 pipeline_model
        scaler = pipeline_model.named_steps['scaler']
        model = pipeline_model.named_steps['svm']

        # ✅ 加载背景数据
        background = pd.read_csv("model/shap_background.csv")
        background_scaled = scaler.transform(background)

        # ✅ 标准化输入数据
        X_input_df = pd.DataFrame(X_input, columns=feature_names)
        X_scaled = scaler.transform(X_input_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        # ✅ 创建 explainer
        explainer = shap.LinearExplainer(model, background_scaled, feature_names=feature_names)
        shap_values = explainer.shap_values(X_scaled_df)

    # ✅ ✅ ✅ 调试输出 SHAP 贡献值 + 特征值
        if debug:
            print("🔍 SHAP值详细贡献如下：")
            for name, shap_val, raw_val in zip(feature_names, shap_values[0], X_input_df.values[0]):
                print(f"  特征 {name:<10} | 输入值 = {raw_val:<8.3f} | SHAP值 = {shap_val:<+8.6f}")

        raw_values = []
        for name, val in zip(feature_names, X_input_df.values[0]):
            if name == "回缩速度":
                raw_values.append(f"{int(val)}")  # 不保留小数
            elif name in ["曲线面积", "45min下降百分比"]:
                raw_values.append(f"{val:.2f}")  # 保留两位
            else:
                raw_values.append(f"{val:.3f}")  # 默认
        # ✅ 构建 Explanation
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=raw_values, # ✅ 原始未标准化的特征值
            feature_names=feature_names
        )

# ✅ 修改 SHAP 显示格式为三位小数（箭头中的值）
        shap.plots._utils.format_value = lambda x: f"{x:+.3f}"
        # ✅ 绘图
        plt.figure(figsize=(10, 6))
        fig = plot_custom_waterfall(explanation, max_display=5, show=False)



        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig

    except Exception as e:
        print(f"[SHAP Waterfall Error] {e}")
        print(traceback.format_exc())  # ✅ 打印完整错误堆栈
        plt.close()
        return None



