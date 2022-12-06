import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

x_10 = [10, 9, 8, 7, 6, 5]

# 已修
LassoCVOneCNN =    [0.98187,0.98092,0.98187,0.98187,0.9771,0.97233]
ElasticNetCVOneCNN =[0.97901,0.97901,0.97996,0.98092,0.97901,0.9771]
RandomForestOneCNN =[0.97519,0.98092,0.97615,0.97328,0.97233,0.96756]




LassoCVTAOneCNN = [ 0.9876,0.9876,0.98664,0.98569,0.98664,0.98187]

ElasticNetCVTAOneCNN = [0.98378,0.98569,0.98664,0.98664,0.98664,0.98378]

RandomForestTAOneCNN = [ 0.98473,0.98187,0.98092,0.98187,0.97901,0.97901]

plt.rcParams["figure.figsize"] = [9, 6]
fig, ax = plt.subplots(1, 1)

##E4392E #3979F2
# ----------
ax.plot(x_10, LassoCVOneCNN, color='#3979F2', linewidth=2, linestyle='--', marker='+', ms=7, label='LassoCV 1D-CNN')
ax.plot(x_10, ElasticNetCVOneCNN, color='#3979F2', linewidth=2, linestyle='--', marker='o', ms=6,
        label='ElasticNetCV 1D-CNN')
ax.plot(x_10, RandomForestOneCNN, color='#3979F2', linewidth=2, linestyle='--', marker='s', ms=6, label='RandomForest 1D-CNN')
# ----------
ax.plot(x_10, LassoCVTAOneCNN, color='#E4392E', linewidth=2, linestyle='-', marker='+', ms=7, label='LassoCV 1D-CNN')
ax.plot(x_10, ElasticNetCVTAOneCNN, color='#E4392E', linewidth=2, linestyle='-', marker='o', ms=6,
        label='ElasticNetCV 1D-CNN')
ax.plot(x_10, RandomForestTAOneCNN, color='#E4392E', linewidth=2, linestyle='-', marker='s', ms=6, label='RandomForest 1D-CNN')

font1 = {'size': 8.4}
ax.legend(loc="lower right",prop=font1)
ax.set_xlabel('Number of Features',fontsize=15)
ax.set_ylabel('Classification Accuracy',fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.show()