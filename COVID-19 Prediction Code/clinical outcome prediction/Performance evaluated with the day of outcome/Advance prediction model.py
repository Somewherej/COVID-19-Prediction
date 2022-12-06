"""
   临床结局部分
   提前预测: 五折交叉验证
            对于每折的测试集中的样本,我们逐步删掉样本中的最后一条血液样本,再通过规范化公式规范化到14条,再添加回测试集中
            例如 患者A有10条样本, 我们规范化到14条,得到A1
                删掉患者A最后1条样本,再规范化到14条,得到A2
                删掉患者A最后1条样本,再规范化到14条,得到A3
                          ......
                删掉患者A最后1条样本,再规范化到14条,得到A9
                删掉患者A最后1条样本,再规范化到14条,得到A10

                A1,A2,A3,....,A9,A10可视为10份不同的测试样本
                (他们最后一条血液样本的检测时间是不同的  ==  提前预测的时间不同)

            我们对五折都进行这样的操作, 汇总所有的准确率和提前时间并进行绘图
"""
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
from utils import sampling,division_X,division_Y
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from utils import timeAttention_Two
from keras.models import Sequential
from keras.layers import LSTM,Dense,BatchNormalization,Flatten,Dropout
from keras.callbacks import ReduceLROnPlateau


#随机种子
seed = 5
#每位患者的记录数量
dataNumber = 14
#每条记录的特征数量
featureNumber = 8
#使用的特征选择方法
Method = 2


#LassoCV
# ['血_RBC分布宽度CV','血_血红蛋白','血_单核细胞(%)','血_国际标准化比值','血_白蛋白',
#  '血_超敏C反应蛋白','血_中性粒细胞(#)','血_血小板计数','血_尿素','血_乳酸脱氢酶']

df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed outcome and severity dataset (with Time) delete.xlsx')

data = df[['病人ID','临床结局 ','血_血红蛋白','血_白细胞计数','血_单核细胞(%)','血_国际标准化比值','血_中性粒细胞(#)',
'血_白蛋白','血_超敏C反应蛋白','血_血小板计数','血_尿素','血_乳酸脱氢酶','提前预测时间']]
data_X = data[['病人ID','血_单核细胞(%)','血_国际标准化比值','血_中性粒细胞(#)',
'血_白蛋白','血_超敏C反应蛋白','血_血小板计数','血_尿素','血_乳酸脱氢酶']]
data_Y = data[['病人ID','临床结局 ']]
data_gap = data[['病人ID','提前预测时间']]
#listname 只是为了打印列名
listName= data_X

print(data.size)


X = division_X(data_X)
""" [[['A0001',2,3],['A0001',4,5],['A0001',7,8]],[['A0002',10,11],['A0002',13,14]]]
    [[[2, 3], [4, 5], [7, 8]], [[10, 11], [13, 14]]]
    去除ID
"""
for i in X:
    for j in i:
        j.pop(0)
#到这步,X是三维的   [患者数量,每个患者的样本数,每份样本的特征数]


gap = division_X(data_gap)

for i in gap:
    for j in i:
        j.pop(0)




temp = division_Y(data_Y)
""" [['A0001', 0], ['A0002', 0], ........, ['n', 0]]
     [0,0,.....0]
     去除ID
"""
Y = []
for i in range(0,len(temp)):
    Y.append(temp[i][-1])






#不均衡样本数量集,提取类型权重参数： class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)
class_weights = dict(enumerate(class_weights))
print('类型权重参数:',class_weights)








def build_model():
    model = Sequential()
    model.add(timeAttention_Two(input_shape=(14,8)))
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


# factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
#   patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
 
reduce_lr = ReduceLROnPlateau(factor=0.9,monitor='loss', patience=2, mode='auto')




#存放五折得分
scoreSet = []
#存放五折(每折的预测标签)
Y_predict = []
#存放五折(每折的实际标签)
Y_true = []







X = np.array(X)
Y = np.array(Y)
gap = np.array(gap)

predict_gap = []

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    part_train_X = []
    part_test_X = []
    part_train_Y = []
    part_test_Y = []
    part_gapday = []

    for i in range(0, len(train_index.tolist())):
        part_train_X.append(X[train_index[i]])
    for i in range(0, len(test_index.tolist())):
        part_test_X.append(X[test_index[i]])
    for i in range(0, len(train_index.tolist())):
        part_train_Y.append(Y[train_index[i]])
    for i in range(0, len(test_index.tolist())):
        part_test_Y.append(Y[test_index[i]])
    for i in range(0, len(test_index.tolist())):
        part_gapday.append(gap[test_index[i]])




    for i in range(0, len(part_train_X)):
        part_train_X[i] = sampling(part_train_X[i], dataNumber, featureNumber)




    limit_number = 0
    while limit_number < len(part_test_X):
        if len(part_test_X[limit_number]) >= 2:
             part_test_X.append(part_test_X[limit_number][0:-1])
             part_gapday.append(part_gapday[limit_number][0:-1])
             part_test_Y.append(part_test_Y[limit_number])
        limit_number = limit_number + 1



    for i in range(0, len(part_test_X)):
        part_test_X[i] = sampling(part_test_X[i], dataNumber, featureNumber)
        part_gapday[i] = sampling(part_gapday[i], dataNumber, featureNumber)


    part_train_X = np.array(part_train_X)
    part_test_X = np.array(part_test_X)
    part_train_Y = np.array(part_train_Y)
    part_test_Y = np.array(part_test_Y)


    for i in part_gapday:
        print(i)
        predict_gap.append(i[len(i)-1])

    model = build_model()


    model.fit(part_train_X,  part_train_Y, epochs=75,batch_size=128,
              validation_data=(part_test_X,part_test_Y),
              verbose=2,class_weight=class_weights,callbacks=[reduce_lr])




    
    print(predict_gap)



    #model.evaluate  输入数据和标签，输出损失值和选定的指标值
    loss,accuracy =  model.evaluate(part_test_X,part_test_Y, verbose=2)

    #预测  返回的是类别的索引，即该样本所属的类别标签
    #[[0],[0],[0]......[0]]
    part_predict =  model.predict_classes(part_test_X)
    #[0,0,0,....,0]
    part_predict = [i for item in part_predict for i in item]

    Y_predict.extend(part_predict)
    Y_true.extend(part_test_Y)


    print('loss:',loss,'accuracy:',accuracy*100)
    scoreSet.append(accuracy * 100)






print("TA LSTM (clinical outcome):")
print("The forecasting performance evaluated with respect to the day of outcome.")
predict_gap = [int(x) for item in predict_gap for x in item]
print('predict_gap',predict_gap)
print('------------')
print('Y_true',Y_true)
print('------------')
print('Y_predict',Y_predict)