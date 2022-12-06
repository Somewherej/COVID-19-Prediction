import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from itertools import chain
from sklearn.metrics import classification_report,confusion_matrix
from utils import sampling,division_X,division_Y,featureSelectionMethod
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

#随机种子
seed = 5
#每位患者的记录数量
dataNumber = 14
#每条记录的特征数
featureNumber = 5
#使用的特征选择方法
Method = 3


""":特征选择后的特征(Y:症状程度)
LassoCV 
['血_球蛋白','高血压(0=无，1=有)','血_淋巴细胞(%)','年龄','血_尿酸','血_白/球比值',
'血_乳酸脱氢酶','糖尿病(0=无，1=有)','血_白蛋白','性别']           
ElasticNetCV
['血_球蛋白','高血压(0=无，1=有)','血_淋巴细胞(%)','年龄','血_尿酸','血_白/球比值',
'血_乳酸脱氢酶','糖尿病(0=无，1=有)','血_白蛋白','性别']   
RandomForest:
['血_RBC分布宽度SD', '血_白细胞计数', '血_尿酸', '年龄', '血_中性粒细胞(#)',
'血_白蛋白','血_D-D二聚体定量','血_乳酸脱氢酶', '血_中性粒细胞(%)', '血_淋巴细胞(%)']      
"""

df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed outcome and severity dataset (with Time).xlsx')
data = df[['病人ID','严重程度（最终）', '血_RBC分布宽度SD', '血_白细胞计数', '血_尿酸', '年龄', '血_中性粒细胞(#)',
'血_白蛋白','血_D-D二聚体定量','血_乳酸脱氢酶', '血_中性粒细胞(%)', '血_淋巴细胞(%)']      ]

data_X = data[['病人ID',
'血_白蛋白','血_D-D二聚体定量','血_乳酸脱氢酶', '血_中性粒细胞(%)', '血_淋巴细胞(%)']     ]
data_Y = data[['病人ID','严重程度（最终）']]
#listname 只是为了打印列名
listName= data_X



""" data_X: [[['A0001',2,3],['A0001',4,5],['A0001',7,8]],[['A0002',10,11],['A0002',13,14]]]
    X:      [[[2, 3], [4, 5], [7, 8]],    [[10, 11], [13, 14]]]
                  患者A0001                   患者A0002
             X是三维的   [患者数量,每个患者的样本数,每份样本的特征数]                  
"""
X = division_X(data_X)

for i in X:
    # i 是一个患者样本集
    for j in i:
        # 对 i 这个患者样本集中的每个样本 j
        # j.pop(0) 去掉样本ID
        j.pop(0)





temp = division_Y(data_Y)
""" data_Y:   [['A0001', 0], ['A0001', 0],['A0001', 0]
               ['A0002', 0], ['A0002', 0],['A0002', 0]
               ........, ['n', 0]]
               
    temp:     [['A0001', 0], ['A0002', 0], ........, ['n', 0]]
    n对n问题  但是同个患者的标签都是一样的  转化成n对1问题
"""
#把标签取出来放进Y里面
Y = []
for i in range(0,len(temp)):
    Y.append(temp[i][-1])


#由于每个患者的样本数量不同,对每个患者的样本采样sampling,来限定每个患者的样本数量
for i in range(0,len(X)):
    X[i] = sampling(X[i],dataNumber,featureNumber)   #sampling(样本,限定每个患者的样本数量,特征数量)




X = np.array(X)
Y = np.array(Y)

# (1048, dataNumber * featureNumber), SVM和KNN输入需要二维
"""  
   input:  [[[1,2], [4,5],[7,8]],       
         [[9,10],[11,12],[12,13]]]
      dataNumber = 3
      featureNumber = 2
      
   output: [[ 1  2  4  5  7  8]
            [ 9 10 11 12 12 13]]
"""
X = np.reshape(X, (len(X), dataNumber * featureNumber))


#knn的五折分数集合
knn_scoreSet = []
#knn预测的类别
knn_Y_predict = []
#svm的五折分数集合
svm_scoreSet = []
#svm预测的类别
svm_Y_predict = []
#实际类别
Y_true = []
#knn,svm预测的类别概率   绘制ROC PR曲线会用到
knn_Y_predict_probability = []
svm_Y_predict_probability = []


knn = KNeighborsClassifier()
svm = svm.SVC(probability=True)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    xTrain, xTest = X[train_index], X[test_index]
    yTrain, yTest = Y[train_index], Y[test_index]

    #训练
    svm.fit(xTrain, yTrain)
    knn.fit(xTrain, yTrain)

    # sklearn.predict()  模型预测输入样本所属的类别
    knn_part_predict = knn.predict(xTest)        # <class 'numpy.ndarray'>
    knn_part_predict = knn_part_predict.astype(np.int).tolist()  # float->int   ndarray->list
    knn_Y_predict.append(knn_part_predict)

    svm_part_predict = svm.predict(xTest)  # <class 'numpy.ndarray'>
    svm_part_predict = svm_part_predict.astype(np.int).tolist()  # float->int   ndarray->list
    svm_Y_predict.append(svm_part_predict)


    yTest = yTest.tolist()  # ndarray->list
    Y_true.append(yTest)

    # 测试数据类别概率值
    knn_part_predict_probability = knn.predict_proba(xTest)[:, -1]
    knn_part_predict_probability = knn_part_predict_probability.tolist()  # ndarray->list
    knn_Y_predict_probability.extend(knn_part_predict_probability)

    svm_part_predict_probability = svm.predict_proba(xTest)[:, -1]
    svm_part_predict_probability = svm_part_predict_probability.tolist()  # ndarray->list
    svm_Y_predict_probability.extend(svm_part_predict_probability)


knn_Y_predict = list(chain.from_iterable(knn_Y_predict))  # [[1],[2],...,[5]] -> [1,2,4,5]
svm_Y_predict = list(chain.from_iterable(svm_Y_predict))  # [[1],[2],...,[5]] -> [1,2,4,5]
Y_true = list(chain.from_iterable(Y_true))                # [[1],[2],...,[5]] -> [1,2,4,5]

print('disease severity prediction (remove ICU and Day):')
featureSelectionMethod(Method,featureNumber,listName)
print("KNN")
print(classification_report(Y_true, knn_Y_predict, digits=5))
print("SVM")
print(classification_report(Y_true, svm_Y_predict, digits=5))



confusion_matrix_knn = confusion_matrix(Y_true, knn_Y_predict)
print('KNN:', confusion_matrix_knn)
confusion_matrix_svm = confusion_matrix(Y_true, svm_Y_predict)
print('SVM:', confusion_matrix_svm)





