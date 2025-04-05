import matplotlib
import matplotlib.font_manager as fm

# 刷新字体缓存
print("🌀 正在刷新 matplotlib 字体缓存...")
fm.fontManager.ttflist = []  # 清空字体缓存列表
fm._rebuild()  # 重建字体缓存
print("✅ 刷新完成")

# 查看是否能找到目标字体
available_fonts = [f.name for f in fm.fontManager.ttflist if "Noto" in f.name]
print("📚 可用的 Noto 字体: ", available_fonts)
