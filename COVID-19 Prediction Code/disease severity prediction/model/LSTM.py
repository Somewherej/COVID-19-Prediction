import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

import numpy as np
import pandas as pd
from utils import sampling,division_X,division_Y,featureSelectionMethod
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import LSTM,Dense,BatchNormalization,Flatten,Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

#随机种子
seed = 5
#每位患者的记录数量
dataNumber = 14
#每条记录的特征数量
featureNumber = 9
#使用的特征选择方法
Method = 1
""":特征选择后的特征(Y:临床结局)
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
data = df[['病人ID','严重程度（最终）','血_球蛋白','高血压(0=无，1=有)','血_淋巴细胞(%)','年龄','血_尿酸','血_白/球比值',
'血_乳酸脱氢酶','糖尿病(0=无，1=有)','血_白蛋白','性别'] ]

data_X = data[['病人ID','高血压(0=无，1=有)', '血_淋巴细胞(%)','年龄','血_尿酸','血_白/球比值',
'血_乳酸脱氢酶','糖尿病(0=无，1=有)','血_白蛋白','性别']  ]
data_Y = data[['病人ID','严重程度（最终）']]
#listname 只是为了打印列名
listName= data_X




X = division_X(data_X)
"""    患者ID  特征1  特征2  特征3
    [[['A0001',2,3],['A0001',4,5],['A0001',7,8]],[['A0002',10,11],['A0002',13,14]]]
    [[[2, 3], [4, 5], [7, 8]], [[10, 11], [13, 14]]]
    去除ID
"""
for i in X:
    for j in i:
        j.pop(0)
#到这步,X是三维的   [患者数量,每个患者的样本数,每份样本的特征数]





temp = division_Y(data_Y)
"""   患者ID  标签
    [['A0001', 0], ['A0002', 0], ........, ['n', 0]]
     [0,0,.....0]
     去除ID
"""
Y = []
for i in range(0,len(temp)):
    Y.append(temp[i][-1])




#由于每个患者的样本数量不同,对每个患者的样本采样sampling,来限定每个患者的样本数量
for i in range(0,len(X)):
    X[i] = sampling(X[i],dataNumber,featureNumber)   #sampling(样本,限定每个患者的样本数量,特征数量)



X = np.array(X)
Y = np.array(Y)


#不均衡样本数量集,提取类型权重参数： class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)
class_weights = dict(enumerate(class_weights))
print('类型权重参数:',class_weights)


from utils import timeAttention_Two
def build_model():
    model = Sequential()
    model.add(LSTM(160, activation='relu', return_sequences=True,
                   input_shape=(np.array(X[train_index]).shape[1], np.array(X[train_index]).shape[2])))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


"""factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
   patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
"""
reduce_lr = ReduceLROnPlateau(factor=0.9,monitor='loss', patience=2, mode='auto')




#存放五折得分
scoreSet = []
#存放五折(每折的预测标签)
Y_predict = []
#存放五折(每折的实际标签)
Y_true = []
#存放五折(每折预测的概率,便于画AUC,PR图)
Y_predict_probability = []

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    LSTM_model = build_model()
    LSTM_model.fit(X[train_index],  Y[train_index], epochs=20,batch_size=128,
              validation_data=(X[test_index], Y[test_index]),
                   verbose=2,class_weight=class_weights,callbacks=[reduce_lr])
    #no time series models.evaluate  输入数据和标签，输出损失值和选定的指标值
    loss,accuracy =  LSTM_model.evaluate(X[test_index],  Y[test_index], verbose=2)

    #预测  返回的是类别的索引，即该样本所属的类别标签
    #[[0],[0],[0]......[0]]
    part_predict =  LSTM_model.predict_classes(X[test_index])
    #[0,0,0,....,0]
    part_predict = [i for item in part_predict for i in item]

    Y_predict.extend(part_predict)
    Y_true.extend(Y[test_index])
    print('loss:',loss,'accuracy:',accuracy*100)
    scoreSet.append(accuracy * 100)
    part_predict_probability = LSTM_model.predict(X[test_index]).ravel()
    Y_predict_probability.extend(part_predict_probability)



print("LSTM (disease severity):")
featureSelectionMethod(Method,featureNumber,listName)
print('准确率集合 ',scoreSet)
print('准确率平均值%.3f '%np.mean(scoreSet))
print(classification_report(Y_true,Y_predict,digits=5))
confusion_matrix = confusion_matrix(Y_true,Y_predict)
print('LSTM:',confusion_matrix)



