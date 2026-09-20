import seaborn as sns
import matplotlib.pyplot as plt
from _basic_set import plt_set

plt_set() # 调用你之前的绘图配置
# 绘制Set2色卡
sns.palplot(sns.color_palette("Set2"))
plt.title("Set2 Color Palette")
plt.show()

# 切换使用
sns.set_palette("Set2")