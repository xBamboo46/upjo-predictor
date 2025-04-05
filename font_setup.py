# font_setup.py
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path

def init_fonts():
    # 设置缓存目录（解决部署环境权限问题）
    cache_dir = Path(mpl.get_cachedir())
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # 添加系统字体路径（适用于Linux环境）
    font_dirs = ['/usr/share/fonts/truetype/noto/']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    
    # 动态添加字体到Matplotlib
    for font_file in font_files:
        if 'NotoSansCJK' in font_file:
            fm.fontManager.addfont(font_file)
    
    # 强制刷新字体列表
    fm._load_fontmanager(try_read_cache=False)
    
    # 配置全局字体
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']