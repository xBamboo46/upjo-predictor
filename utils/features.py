from scipy.interpolate import make_interp_spline
from scipy.integrate import trapezoid
import numpy as np

def extract_features(apd_values):
    """
    从6个时间点的APD数据中提取两个特征：
    - 曲线面积（通过三次样条插值+积分）
    - 45min下降百分比
    还返回绘图需要的时间点和曲线
    """
    time_points = np.array([0, 5, 15, 30, 45, 60])
    apd_values = np.array(apd_values)

    # 三次样条插值
    spline = make_interp_spline(time_points, apd_values, k=3)
    dense_time = np.linspace(0, 60, 300)
    smooth_curve = spline(dense_time)

    # ✅ 特征①：曲线面积（平滑曲线下的积分）
    area = trapezoid(smooth_curve, dense_time)

    # ✅ 特征②：45min下降百分比
    apd_0 = apd_values[0]
    apd_45 = apd_values[4]
    apd_max = np.max(apd_values)
    if (apd_max - apd_0) != 0:
        decline_ratio = (apd_max - apd_45) / (apd_max - apd_0)
    else:
        decline_ratio = 0  # 或者设为 np.nan / 特殊值

    return [area, decline_ratio], (dense_time, smooth_curve, apd_values)
