from utils.custom_shap_waterfall import plot_custom_waterfall
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import traceback

# ✅ 显示英文特征名用（不影响模型）
feature_display_names = {
    "回缩速度": "Recoil Speed",
    "曲线面积": "Curve Area",
    "45min下降百分比": "45min Decline (%)"
}

def plot_shap_waterfall(pipeline_model, X_input, feature_names=None, debug=False):
    try:
        scaler = pipeline_model.named_steps['scaler']
        model = pipeline_model.named_steps['svm']

        # ✅ 加载背景数据
        background = pd.read_csv("model/shap_background.csv")
        background_scaled = scaler.transform(background)

        # ✅ 标准化输入数据
        X_input_df = pd.DataFrame(X_input, columns=feature_names)
        X_scaled = scaler.transform(X_input_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        # ✅ 创建 Explainer
        explainer = shap.LinearExplainer(model, background_scaled, feature_names=feature_names)
        shap_values = explainer.shap_values(X_scaled_df)

        # ✅ 调试打印信息（英文输出）
        if debug:
            print("🔍 SHAP Value Contributions:")
            for name, shap_val, raw_val in zip(feature_names, shap_values[0], X_input_df.values[0]):
                print(f"  Feature: {name:<10} | Input = {raw_val:<8.3f} | SHAP = {shap_val:<+8.6f}")

        # ✅ 格式化原始特征值用于展示（不影响模型）
        raw_values = []
        for name, val in zip(feature_names, X_input_df.values[0]):
            if name == "回缩速度":
                raw_values.append(f"{int(val)}")
            elif name in ["曲线面积", "45min下降百分比"]:
                raw_values.append(f"{val:.2f}")
            else:
                raw_values.append(f"{val:.3f}")

        # ✅ 英文展示名（只用于 SHAP 图显示）
        display_names = [feature_display_names.get(name, name) for name in feature_names]

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=raw_values,  # ✅ 原始未标准化的值
            feature_names=display_names  # ✅ 显示用英文名
        )

        # ✅ 格式化箭头中的数值
        shap.plots._utils.format_value = lambda x: f"{x:+.3f}"

        # ✅ 绘图并返回 Figure
        plt.figure(figsize=(10, 6))
        fig = plot_custom_waterfall(explanation, max_display=5, show=False)
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig

    except Exception as e:
        print(f"[SHAP Waterfall Error] {e}")
        print(traceback.format_exc())
        plt.close()
        return None




