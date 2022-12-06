import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay



plt.rcParams["figure.figsize"] = [8, 6]

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


KNN_precision, KNN_recall, KNN_threshold =  precision_recall_curve(KNN_True, KNN_Pro,pos_label=1)
KNN_AUC = average_precision_score(KNN_True, KNN_Pro)

SVM_precision, SVM_recall, SVM_threshold =  precision_recall_curve(SVM_True, SVM_Pro,pos_label=1)
SVM_AUC = average_precision_score(SVM_True, SVM_Pro , pos_label=1)

TAOneDCNN_precision, TAOneDCNN_recall, TAOneDCNN_threshold =  precision_recall_curve(TAOneDCNN_True, TAOneDCNN_Pro,pos_label=1)
TAOneDCNN_AUC = average_precision_score(TAOneDCNN_True, TAOneDCNN_Pro, pos_label=1)

TALSTM_precision, TALSTM_recall, TALSTM_threshold =  precision_recall_curve(TALSTM_True, TALSTM_Pro,pos_label=1)
TALSTM_AUC = average_precision_score(TALSTM_True, TALSTM_Pro,  pos_label=1)



def plot_pr_multi_label(n_classes, Y_test, y_score):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # 绘制每个类的PR曲线和 iso-f1 曲线
    # setup plot details
    _, ax = plt.subplots(figsize=(7, 8))
    display = PrecisionRecallDisplay(recall=recall[1], precision=precision[1])
    display.plot(ax=ax, name=f"LassoCV 2D-CNN",lw=2,linestyle='-', color='#f8ac8c')



plot_pr_multi_label(2,TwoDCNN_True,TwoDCNN_Pro)



"""
#2878b5
#9ac9db
#f8ac8c
#c82423
#ff8884
"""

plt.plot(KNN_recall,KNN_precision, color='#2878b5',
         lw=2, label='ElasticNetCV KNN', linestyle='-')
plt.plot(SVM_recall,SVM_precision, color='#9ac9db',
        lw=2, label='LassoCV-ElasticNetCV SVM', linestyle='-')
plt.plot(TAOneDCNN_recall,TAOneDCNN_precision, color='#c82423',
         lw=2, label='LassoCV TA 1D-CNN', linestyle='-')
plt.plot(TALSTM_recall,TALSTM_precision, color='#ff8884',
         lw=2, label='ElasticNetCV TA LSTM', linestyle='-')


plt.plot([0, 1], [0, 1], lw=1.5, linestyle='--')
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('PR Curves',fontsize=15)
plt.legend(loc="best")
plt.show()
