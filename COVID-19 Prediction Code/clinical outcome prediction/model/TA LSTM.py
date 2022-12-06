import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.compat.v1.Session(config=config))
import time
import numpy as np
import pandas as pd
from utils import sampling,division_X,division_Y,featureSelectionMethod
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from utils import timeAttention_Two
from keras.models import Sequential
from keras.layers import LSTM,Dense,BatchNormalization,Flatten,Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report,confusion_matrix

#随机种子
seed = 15
#每位患者的记录数量
dataNumber = 14
#每条记录的特征数量
featureNumber = 8
#使用的特征选择方法
Method = 2
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
data = df[['病人ID','临床结局 ','血_血红蛋白','血_白细胞计数','血_单核细胞(%)','血_国际标准化比值','血_中性粒细胞(#)',
'血_白蛋白','血_超敏C反应蛋白','血_血小板计数','血_尿素','血_乳酸脱氢酶']]
data_X = data[['病人ID', '血_单核细胞(%)','血_国际标准化比值','血_中性粒细胞(#)',
'血_白蛋白','血_超敏C反应蛋白','血_血小板计数','血_尿素','血_乳酸脱氢酶']]
data_Y = data[['病人ID','临床结局 ']]
#listname 只是为了打印列名
listName= data_X




X = division_X(data_X)
""" [[['A0001',2,3],['A0001',4,5],['A0001',7,8]],[['A0002',10,11],['A0002',13,14]]]
    [[[2, 3], [4, 5], [7, 8]], [[10, 11], [13, 14]]]
    去除ID
"""
for i in X:
    for j in i:
        j.pop(0)
#到这步,X是三维的   [患者数量,每个患者的样本数,每份样本的特征数]





temp = division_Y(data_Y)
""" [['A0001', 0], ['A0002', 0], ........, ['n', 0]]
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



def build_model():
    model = Sequential()
    model.add(timeAttention_Two(input_shape=(np.array(X[train_index]).shape[1],
                                             np.array(X[train_index]).shape[2])))
    model.add(LSTM(160, activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(80))
    model.add(Flatten())
    model.add(Dropout(0.2))
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
training_time  = 0



kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    model = build_model()
    time_start = time.time()  # 记录开始时间

    model.fit(X[train_index],  Y[train_index], epochs=75,batch_size=128,
              validation_data=(X[test_index], Y[test_index]),
                   verbose=2,class_weight=class_weights,callbacks=[reduce_lr])

    time_end = time.time()  # 记录结束时间
    part_time = time_end - time_start
    training_time = training_time + part_time


    #model.evaluate  输入数据和标签，输出损失值和选定的指标值
    loss,accuracy =  model.evaluate(X[test_index],  Y[test_index], verbose=2)

    #预测  返回的是类别的索引，即该样本所属的类别标签
    #[[0],[0],[0]......[0]]
    part_predict =  model.predict_classes(X[test_index])
    #[0,0,0,....,0]
    part_predict = [i for item in part_predict for i in item]

    Y_predict.extend(part_predict)
    Y_true.extend(Y[test_index])

    # 预测保留的测试数据以生成概率值
    part_predict_probability = model.predict(X[test_index]).ravel()
    Y_predict_probability.extend(part_predict_probability)

    print('loss:',loss,'accuracy:',accuracy*100)
    scoreSet.append(accuracy * 100)






print("TA LSTM (clinical outcome):")
print("delete ICU and Day")
featureSelectionMethod(Method,featureNumber,listName)
print('准确率集合 ',scoreSet)
print('准确率平均值%.3f '%np.mean(scoreSet))
print(classification_report(Y_true,Y_predict,digits=5))
print('随机种子:',seed)
print('训练时间:',training_time)
confusion_matrix = confusion_matrix(Y_true,Y_predict)
print('TA LSTM:',confusion_matrix)


