import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result")


def plt_set():
    """设置绘图相关参数（ggplot 风格 + 中文支持），并设置颜色方案为 Dark2 """

    # 绘图风格
    plt.style.use('ggplot')
    # 颜色方案
    sns.set_palette("Set2")

    # 字体设置
    plt.rcParams['font.family'] = ['serif', 'sans-serif']
    plt.rcParams['font.serif'] = ['Times New Roman',]
    plt.rcParams['font.sans-serif'] = ['SimSun',]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['font.size'] = 10
    
    # 出图参数
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['figure.autolayout'] = False


def save_fig(save_name='None'):
    """保存当前绘图到指定文件"""
    if save_name == 'None':
        save_name = f'默认名称_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.svg'
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, save_name)
    # 统一固定所有导出参数
    plt.savefig(
        out_path,
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
        pad_inches=0.02
    )
    plt.close()
    print(f"图片保存：{out_path}")
