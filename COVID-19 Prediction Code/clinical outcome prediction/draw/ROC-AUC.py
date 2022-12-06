import matplotlib.pyplot as plt

import numpy as np
from numpy import interp
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score,roc_auc_score



KNN_True = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\ElasticNetCV KNN(9)正确标签.npy")
KNN_Pro = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\ElasticNetCV KNN(9)概率输出值.npy")

SVM_True = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\LassoCV-ElasticNetCV SVM(8)正确标签.npy")
SVM_Pro = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\LassoCV-ElasticNetCV SVM(8)概率输出值.npy")

TAOneDCNN_True = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\LassoCV TA 1D-CNN(9)正确标签.npy")
TAOneDCNN_Pro = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\LassoCV TA 1D-CNN(9)概率输出值.npy")


TALSTM_True = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\ElasticNetCV TA-LSTM(8)正确标签.npy")
TALSTM_Pro = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\ElasticNetCV TA-LSTM(8)概率输出值.npy")

TwoDCNN_True = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\LassoCV 2D-CNN(9)正确标签.npy")
TwoDCNN_Pro = np.load(r"D:\pythonProject\COVID-19-Prediction\clinical outcome prediction\draw\PR-ROC data\LassoCV 2D-CNN(9)概率输出值.npy")



KNN_fpr, KNN_tpr, KNN_threshold = roc_curve(KNN_True,KNN_Pro)
KNN_AUC = auc(KNN_fpr, KNN_tpr)

SVM_fpr, SVM_tpr, SVM_threshold = roc_curve(SVM_True,SVM_Pro)
SVM_AUC = auc(SVM_fpr, SVM_tpr )

TAOneCNN_fpr, TAOneCNN_tpr, TAOneCNN_threshold = roc_curve(TAOneDCNN_True,TAOneDCNN_Pro)
TAOneCNN_AUC = auc(TAOneCNN_fpr, TAOneCNN_tpr)

TALSTM_fpr, TALSTM_tpr, TALSTM_threshold = roc_curve(TALSTM_True, TALSTM_Pro)
TALSTM_AUC = auc(TALSTM_fpr, TALSTM_tpr)


"""
2D-CNN 比较奇葩
"""

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes =2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(TwoDCNN_True[:, i],TwoDCNN_Pro[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    # macro（方法一）
    # First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.plot(fpr["macro"], tpr["macro"],
             label='LassoCV 2D-CNN (AUC: {0:0.5f})'
                   ''.format(roc_auc["macro"]),
             color='#f8ac8c', linestyle='-',lw=2)

# micro（方法二）
#fpr["micro"], tpr["micro"], _ = roc_curve(TwoDCNN_True.ravel(), TwoDCNN_Pro.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])#
#
#plt.plot(fpr["macro"], tpr["macro"],
#         label='LassoCV 2D-CNN (area = {0:0.5f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle='-', linewidth=2)





"""
#2878b5
#9ac9db
#f8ac8c
#c82423
#ff8884
"""




plt.plot(KNN_fpr, KNN_tpr, color='#2878b5',
         lw=2, label='ElasticNetCV KNN (AUC:%0.5f)' %KNN_AUC, linestyle='-')
plt.plot(SVM_fpr, SVM_tpr, color='#9ac9db',
         lw=2, label='LassoCV-ElasticNetCV SVM (AUC:%0.5f)' %SVM_AUC, linestyle='-')
plt.plot(TAOneCNN_fpr, TAOneCNN_tpr, color='#c82423',
         lw=2, label='LassoCV TA 1D-CNN (AUC:%0.5f)'%TAOneCNN_AUC, linestyle='-')
plt.plot(TALSTM_fpr, TALSTM_tpr, color='#ff8884',
         lw=2, label='ElasticNetCV TA LSTM (AUC:%0.5f)'%TALSTM_AUC, linestyle='-')


plt.plot([0, 1], [0, 1],  lw=1.5, linestyle='-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="best")
plt.show()