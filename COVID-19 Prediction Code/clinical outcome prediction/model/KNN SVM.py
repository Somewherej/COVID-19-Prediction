import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix
from utils import sampling,division_X,division_Y,featureSelectionMethod
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

#随机种子
seed = 5
#每位患者的记录数量
dataNumber = 14
#每条记录的特征数量
featureNumber = 7
#使用的特征选择方法
Method = 3


""":特征选择后的特征(Y:临床结局)
LassoCV 
['血_RBC分布宽度CV','血_血红蛋白','血_单核细胞(%)','血_国际标准化比值','血_白蛋白','血_超敏C反应蛋白',
'血_中性粒细胞(#)','血_血小板计数','血_尿素','血_乳酸脱氢酶']       
ElasticNetCV
['血_血红蛋白','血_白细胞计数','血_单核细胞(%)','血_国际标准化比值','血_中性粒细胞(#)',
'血_白蛋白','血_超敏C反应蛋白','血_血小板计数','血_尿素','血_乳酸脱氢酶'] 
RandomForest:
['血_钠','血_血小板计数','血_超敏C反应蛋白','血_淋巴细胞(#)','血_单核细胞(%)',
'血_中性粒细胞(#)','血_尿素','血_乳酸脱氢酶','血_淋巴细胞(%)','血_中性粒细胞(%)']  
"""


df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed outcome and severity dataset (with Time).xlsx')
data = df[['病人ID','临床结局 ','血_钠','血_血小板计数','血_超敏C反应蛋白','血_淋巴细胞(#)','血_单核细胞(%)',
'血_中性粒细胞(#)','血_尿素','血_乳酸脱氢酶','血_淋巴细胞(%)','血_中性粒细胞(%)']  ]
data_X = data[['病人ID', '血_淋巴细胞(#)','血_单核细胞(%)',
'血_中性粒细胞(#)','血_尿素','血_乳酸脱氢酶','血_淋巴细胞(%)','血_中性粒细胞(%)'] ]
data_Y = data[['病人ID','临床结局 ']]
#listname 只是为了打印列名
listName= data_X





X = division_X(data_X)
"""  
   [ [['A0001',1,1,1],
      ['A0001',2,2,2],
      ['A0001',3,3,3]], 
     
     [['A0002',1,1,1],
      ['A0002',2,2,2],
      ['A0002',3,3,3]], 
              ...  
    ]      
"""
for i in X:
    for j in i:
        j.pop(0)
#去除每位患者的每条数据的ID
#到这步,X是三维的   [患者,每个患者的样本数,每份样本的特征]
#但是每个患者的样本数是不同的, 所以我们还要对这个样本进行一次采样sampling



temp = division_Y(data_Y)
"""  [['A0001', 0], 
      ['A0002', 0], 
      ['A0003', 0], 
        ...
      ]   
"""
Y = []
for i in range(0,len(temp)):
    Y.append(temp[i][-1])
#依次取出每个患者的标签,放进Y列表中(跟上面去ID的方式有点不同)


#由于每个患者的样本数量不同,对每个患者的样本采样sampling,来限定每个患者的样本数量
for i in range(0,len(X)):
    X[i] = sampling(X[i],dataNumber,featureNumber)   #sampling(样本,限定每个患者的样本数量,特征数量)




X = np.array(X)
Y = np.array(Y)

# (1048, dataNumber * featureNumber), SVM和KNN输入需要二维
# 等于说我们把三维的数据压成了二维
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
svm = svm.SVC(class_weight='balanced',probability=True)




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
    #knn_Y_predict 保存每折knn预测的样本类别
    knn_Y_predict.extend(knn_part_predict)
    #svm_Y_predict 保存每折svm预测的样本类别
    svm_part_predict = svm.predict(xTest)  # <class 'numpy.ndarray'>
    svm_part_predict = svm_part_predict.astype(np.int).tolist()  # float->int   ndarray->list
    svm_Y_predict.extend(svm_part_predict)



    #Y_true        保存真实样本类别
    yTest = yTest.tolist()  # ndarray->list
    Y_true.extend(yTest)


    # 预测保留的测试数据以生成概率值
    knn_part_predict_probability = knn.predict_proba(xTest)[:, -1]
    knn_part_predict_probability = knn_part_predict_probability.tolist()   # ndarray->list
    knn_Y_predict_probability.extend(knn_part_predict_probability)

    svm_part_predict_probability = svm.predict_proba(xTest)[:,-1]
    svm_part_predict_probability =  svm_part_predict_probability.tolist()  # ndarray->list
    svm_Y_predict_probability.extend(svm_part_predict_probability)





featureSelectionMethod(Method,featureNumber,listName)
print("delete ICU and Dau")
print("KNN")
print(classification_report(Y_true, knn_Y_predict, digits=5))
print("SVM")
print(classification_report(Y_true, svm_Y_predict, digits=5))
confusion_matrix_knn = confusion_matrix(Y_true,knn_Y_predict)
print('KNN:',confusion_matrix_knn)
confusion_matrix_svm = confusion_matrix(Y_true,svm_Y_predict)
print('SVM:',confusion_matrix_svm)








