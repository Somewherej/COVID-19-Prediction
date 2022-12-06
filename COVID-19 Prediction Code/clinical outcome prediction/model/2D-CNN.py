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
featureNumber = 9
#使用的特征选择方法
Method = 1
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

data = df[['病人ID','临床结局 ','血_RBC分布宽度CV','血_血红蛋白','血_单核细胞(%)','血_国际标准化比值','血_白蛋白','血_超敏C反应蛋白',
'血_中性粒细胞(#)','血_血小板计数','血_尿素','血_乳酸脱氢酶']]
data_X = data[['病人ID', '血_血红蛋白','血_单核细胞(%)','血_国际标准化比值','血_白蛋白','血_超敏C反应蛋白',
'血_中性粒细胞(#)','血_血小板计数','血_尿素','血_乳酸脱氢酶'] ]
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



class weighted_cross_entropy(object):
    def __call__(self, y_pred, y_true):
        """
        logits: a Tensor with shape [batch_size, image_width, image_height, channel], score from the unet conv10
        label: a Tensor with shape [batch_size, image_width, image_height], ground truth
        """
        #类型权重参数: {0: 0.5402061855670103, 1: 6.717948717948718}
        weight = [0.5402061855670103, 6.717948717948718]
        # label = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), y_pred.get_shape()[-1])
        prob = tf.nn.softmax(y_pred, dim=-1)
        loss = -tf.reduce_mean(y_true * tf.log(prob) * weight)
        return loss



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
    network = fully_connected(network, 576, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 576, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2 ,activation='softmax')
    network = regression(network, optimizer="adam",
                         loss = weighted_cross_entropy(),
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
    part_predict = part_predict.argmax(axis=1)  # 当axis=1,是在行中比较，选出最大的列索引    input:array[[0, 6, 2], [3, 4, 5]]  output:[1 2]
    part_predict = part_predict.tolist()  # Y_predict是list类型,为了统一，part_predict要从ndarray->list
    Y_predict.extend(part_predict)

    # Y[test_index]  <class 'numpy.ndarray'>   [[1. 0.]  ... [1. 0.] [1. 0.] [1. 0.]]
    part_true = Y[test_index].argmax(axis=1)
    part_true = part_true.tolist()  # Y_true是list类型,为了统一，part_true要从ndarray->list
    Y_true.extend(part_true)

    accuracy = AlexNet_mdoel.evaluate(X[test_index], Y[test_index])  # type(accuracy):list
    #tflearn 固定用法 score[0]
    accuracy = float(accuracy[0])  # list 转float纯数字
    print('Classification accuracy:', accuracy * 100)
    scoreSet.append(accuracy)

    part_predict_probability = AlexNet_mdoel.predict(X[test_index])
    Y_predict_probability.extend(part_predict_probability)




from sklearn.metrics import classification_report,confusion_matrix
print('2D-CNN (clinical outcome):')
print('delete ICU and Day')
featureSelectionMethod(Method,featureNumber,listName)
print('准确率集合',scoreSet)
print('准确率平均值%.4f'%np.mean(scoreSet))
print(classification_report(Y_true,Y_predict,digits=5))

confusion_matrix = confusion_matrix(Y_true,Y_predict)
print('2D-CNN:',confusion_matrix)

from keras.utils.np_utils import to_categorical
Y_true = to_categorical(Y_true)







