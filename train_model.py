import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os

# 加载数据
data = pd.read_excel("F:/研究生生涯/数据/交流用.xlsx")

# 特征和标签
X = data[['回缩速度', '曲线面积', '45min下降百分比']]
y = data['梗阻状态']

# 建立模型管道
scaler = StandardScaler()
model = SVC(C=0.1, kernel='linear', probability=True)

X_scaled = scaler.fit_transform(X)  # ✅ 标准化特征
model.fit(X_scaled, y)

# 创建 Pipeline
final_model = Pipeline([
    ('scaler', scaler),
    ('svm', model)
])

# ✅ 创建 model 文件夹（如果不存在）
os.makedirs("model", exist_ok=True)

# ✅ 保存模型
joblib.dump(final_model, 'model/svm_model.pkl')

# ✅ 保存背景数据（取前100个样本作为 SHAP 背景）
background = pd.DataFrame(X_scaled[:100], columns=X.columns)
background.to_csv('model/shap_background.csv', index=False)

print("✅ 模型和 SHAP 背景样本已保存！")
