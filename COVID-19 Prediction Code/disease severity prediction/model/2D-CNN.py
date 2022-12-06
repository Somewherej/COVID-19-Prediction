import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))


import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from utils import sampling,division_X,division_Y,featureSelectionMethod
from sklearn.model_selection import KFold


#随机种子
seed = 5
#每位患者的记录数量
dataNumber = 14
#每条记录的特征数量
featureNumber = 5
#使用的特征选择方法
Method = 1

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
data = df[['病人ID','严重程度（最终）','血_球蛋白','高血压(0=无，1=有)','血_淋巴细胞(%)','年龄','血_尿酸','血_白/球比值',
'血_乳酸脱氢酶','糖尿病(0=无，1=有)','血_白蛋白','性别']  ]
data_X = data[['病人ID', '血_白/球比值',
'血_乳酸脱氢酶','糖尿病(0=无，1=有)','血_白蛋白','性别']  ]
data_Y = data[['病人ID','严重程度（最终）']]
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


X = X.reshape(X.shape[0], dataNumber, featureNumber, 1)
Y = np_utils.to_categorical(Y, 2)
def build_model():
    tf.compat.v1.reset_default_graph()
    network = input_data(shape=[None, np.array(X[train_index]).shape[1], np.array(X[train_index]).shape[2], 1])
    network = conv_2d(network, 4, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 12, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 48, 3, activation='relu')
    network = conv_2d(network, 48, 3, activation='relu')
    network = conv_2d(network, 36, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 576, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 576, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2 ,activation='softmax')
    network = regression(network, optimizer="adam",
                         loss = 'categorical_crossentropy',
                         learning_rate=0.001)
    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model





"""
scoreSet存放每折的accuracy, 最后取mean得泛化accuracy
Y_true KFold打乱了数据，因此我们要把每折的的Y_true存下来
Y_predict 同理
"""
scoreSet = []
Y_predict = []
Y_true = []
Y_predict_probability = []
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    AlexNet_mdoel = build_model()

    AlexNet_mdoel.fit(X[train_index], Y[train_index], n_epoch=100, batch_size=256,
                      snapshot_epoch=False, snapshot_step=None )
    part_predict = AlexNet_mdoel.predict(X[test_index])  # part_predict <class 'numpy.ndarray'>  输出[[A,B],...,[],[],[]]
    part_predict = part_predict.argmax(
        axis=1)  # 当axis=1,是在行中比较，选出最大的列索引    input:array[[0, 6, 2], [3, 4, 5]]  output:[1 2]
    part_predict = part_predict.tolist()  # Y_predict是list类型,为了统一，part_predict要从ndarray->list
    Y_predict.extend(part_predict)

    # Y[test_index]  <class 'numpy.ndarray'>   [[1. 0.]  ... [1. 0.] [1. 0.] [1. 0.]]
    part_true = Y[test_index].argmax(axis=1)
    part_true = part_true.tolist()  # Y_true是list类型,为了统一，part_true要从ndarray->list
    Y_true.extend(part_true)

    accuracy = AlexNet_mdoel.evaluate(X[test_index], Y[test_index])  # type(accuracy):list
    accuracy = float(accuracy[0])  # list 转float纯数字
    print('Classification accuracy:', accuracy * 100)
    scoreSet.append(accuracy)

    part_predict_pro = AlexNet_mdoel.predict(X[test_index])
    Y_predict_probability.extend(part_predict_pro)




from sklearn.metrics import classification_report, confusion_matrix
print('2D-CNN (disease severity):')
featureSelectionMethod(Method,featureNumber,listName)
print('准确率集合',scoreSet)
print('准确率平均值%.5f'%np.mean(scoreSet))
print(classification_report(Y_true,Y_predict,digits=5))
confusion_matrix = confusion_matrix(Y_true,Y_predict)
print('2D-CNN:',confusion_matrix)

from keras.utils.np_utils import to_categorical
Y_true = to_categorical(Y_true)

