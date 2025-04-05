import matplotlib
import matplotlib.font_manager as fm
import os
import shutil

# 刷新字体缓存
print("🌀 正在刷新 matplotlib 字体缓存...")

# 删除 matplotlib 缓存文件，这样下次加载时会重新创建
cache_dir = os.path.join(matplotlib.get_data_path(), 'fontlist-v310.json')
if os.path.exists(cache_dir):
    os.remove(cache_dir)
    print(f"已删除缓存文件：{cache_dir}")
else:
    print("未找到缓存文件。")

# 强制重新加载字体列表
fm._rebuild()

print("✅ 刷新完成")

# 查看是否能找到目标字体
available_fonts = [f.name for f in fm.fontManager.ttflist if "Noto" in f.name]
print("📚 可用的 Noto 字体: ", available_fonts)
