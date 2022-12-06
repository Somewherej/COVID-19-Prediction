import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
from utils import division_X
import numpy as np
seed = 5

df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed antibody dataset.xlsx')
data = df[['病人ID', '血_肌酐','血_超敏C反应蛋白','血_红细胞压积','血_尿素','血_钠','血_红细胞计数','血_白/球比值',
'血_平均RBC体积','血_嗜酸细胞(%)','血_淋巴细胞(#)','血_大血小板比率','血_谷丙转氨酶', '血_总蛋白',
'血_碳酸氢根','血_平均血红蛋白含量','血_碱性磷酸酶','血_总胆固醇','血_血小板压积',
'血_钾','血_尿酸','血_γ-谷氨酰转肽酶','血_氯','血_球蛋白',
'血_乳酸脱氢酶','血_eGFR(基于CKD-EPI方程)','血_嗜酸细胞(#)',
'血_血小板计数','血_RBC分布宽度SD','血_RBC分布宽度CV','年龄','S1_IgG']]

data_X = data[['病人ID','血_肌酐','血_超敏C反应蛋白','血_红细胞压积','血_尿素','血_钠','血_红细胞计数','血_白/球比值',
'血_平均RBC体积','血_嗜酸细胞(%)','血_淋巴细胞(#)','血_大血小板比率','血_谷丙转氨酶', '血_总蛋白',
'血_碳酸氢根','血_平均血红蛋白含量','血_碱性磷酸酶','血_总胆固醇','血_血小板压积',
'血_钾','血_尿酸','血_γ-谷氨酰转肽酶','血_氯','血_球蛋白',
'血_乳酸脱氢酶','血_eGFR(基于CKD-EPI方程)','血_嗜酸细胞(#)',
'血_血小板计数','血_RBC分布宽度SD','血_RBC分布宽度CV','年龄','S1_IgG']]
data_Y = data[['病人ID','S1_IgG']]
#listname 只是为了打印列名
listName= data_X


""":
CatBoost
['血_肌酐','血_超敏C反应蛋白','血_红细胞压积','血_尿素','血_钠','血_红细胞计数','血_白/球比值',                 
'血_平均RBC体积','血_嗜酸细胞(%)','血_淋巴细胞(#)','血_大血小板比率','血_谷丙转氨酶', '血_总蛋白',               
'血_碳酸氢根','血_平均血红蛋白含量','血_碱性磷酸酶','血_总胆固醇','血_血小板压积',                 
'血_钾','血_尿酸','血_γ-谷氨酰转肽酶','血_氯','血_球蛋白',                 
'血_乳酸脱氢酶','血_eGFR(基于CKD-EPI方程)','血_嗜酸细胞(#)',               
'血_血小板计数','血_RBC分布宽度SD','血_RBC分布宽度CV','年龄']                        

XGBoost
'血_嗜碱细胞(#)', '血_白细胞计数','血_血小板计数','血_红细胞计数','血_超敏C反应蛋白',             
'血_谷丙转氨酶','血_淋巴细胞(#)', '血_红细胞压积','血_γ-谷氨酰转肽酶',              
'血_总胆固醇','血_钙','血_尿酸','血_eGFR(基于CKD-EPI方程)',    
'性别','血_总蛋白','血_PLT分布宽度','血_大血小板比率',             
'血_尿素', '血_氯', '血_肌酐','血_嗜酸细胞(%)','血_碳酸氢根', '血_乳酸脱氢酶',
'血_RBC分布宽度SD','血_嗜酸细胞(#)', '糖尿病(0=无，1=有)','血_球蛋白',
'血_RBC分布宽度CV','血_血小板压积','年龄'] 

                     
LightGBM
['血_平均RBC体积','血_大血小板比率', '血_平均血红蛋白浓度','血_嗜酸细胞(%)','血_PLT分布宽度','血_钠','血_总胆红素',                  
'血_血小板计数','血_白/球比值','血_平均血红蛋白含量','血_钙','血_尿酸','血_碱性磷酸酶','血_总蛋白',
'血_球蛋白','血_血小板压积', '血_谷丙转氨酶','血_总胆固醇','血_肌酐','血_碳酸氢根','血_氯','血_尿素','血_嗜酸细胞(#)',               
'血_eGFR(基于CKD-EPI方程)', '血_淋巴细胞(#)','血_γ-谷氨酰转肽酶','血_乳酸脱氢酶','血_RBC分布宽度SD',
'血_RBC分布宽度CV','年龄']    
"""

X = division_X(data_X)
for i in X:
    for j in i:
        j.pop(0)
#到这步,X是三维的   [患者数量,每个患者的样本数,每份样本的特征数]

#删除患者血液样本数量为1的样本
new_X = []
for i in X:
    if len(i) != 1:
        new_X.append(i)
X = new_X
#去掉预测那次的S1_IgG 设置为0
new_Y = []
for i in X:
    #new_Y 预测每位患者最后一次的抗体水平
    new_Y.append(i[len(i)-1][30])
    #取完标签 原数据置0
    i[len(i) - 1][30] = 0
Y = new_Y






X = np.array(X)
Y = np.array(Y)

from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, LSTM, Bidirectional ,Merge
from keras.layers.core import*

X =sequence.pad_sequences(X,maxlen=20,value=0,padding='post')
import numpy as np

def build_model():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(X.shape[1],X.shape[2])))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='Adam')
    return model





Y_predict = []
Y_true = []
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    model = build_model()
    model.fit(X[train_index], Y[train_index], epochs=25, batch_size=16, verbose=2
    ,validation_data=(X[test_index], Y[test_index]))
    Y_predict.extend(model.predict(X[test_index]))
    Y_true.extend(Y[test_index])


import matplotlib.pyplot as plt
plt.figure(figsize=(50, 4))
from sklearn.metrics import mean_squared_error,mean_absolute_error\
    ,mean_absolute_percentage_error, r2_score




print('MSE {}'.format((mean_squared_error(Y_true ,Y_predict))))
#RMSE 可以调用 mean_squared_error 方法实现, 设置 squared=False 即可;
print('RMSE {}'.format((mean_squared_error(Y_true ,Y_predict, squared=False))))
print('RAE {}'.format(mean_absolute_error(Y_true ,Y_predict)))
print('RAPE {}'.format(mean_absolute_percentage_error(Y_true ,Y_predict)))
print('r2 {}'.format(r2_score(Y_true ,Y_predict)))


print('CatBoost LSTM')
plt.plot(Y_true, "blue")
plt.plot(Y_predict, "red")
plt.legend(['True','Predict'],loc='best')
plt.show()




