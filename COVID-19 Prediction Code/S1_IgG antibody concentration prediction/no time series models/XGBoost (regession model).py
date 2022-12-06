import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd

seed = 5
Method = 3
df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed antibody dataset.xlsx')


X = df[['血_嗜碱细胞(#)', '血_白细胞计数','血_血小板计数','血_红细胞计数','血_超敏C反应蛋白',
'血_谷丙转氨酶','血_淋巴细胞(#)', '血_红细胞压积','血_γ-谷氨酰转肽酶',
'血_总胆固醇','血_钙','血_尿酸','血_eGFR(基于CKD-EPI方程)',
'性别','血_总蛋白','血_PLT分布宽度','血_大血小板比率',
'血_尿素', '血_氯', '血_肌酐','血_嗜酸细胞(%)','血_碳酸氢根', '血_乳酸脱氢酶',
'血_RBC分布宽度SD','血_嗜酸细胞(#)', '糖尿病(0=无，1=有)','血_球蛋白',
'血_RBC分布宽度CV','血_血小板压积','年龄']  ]
Y = df['S1_IgG']


"""
CatBoost
['血_肌酐','血_超敏C反应蛋白','血_红细胞压积','血_尿素','血_钠','血_红细胞计数','血_白/球比值',                 
'血_平均RBC体积','血_嗜酸细胞(%)','血_淋巴细胞(#)','血_大血小板比率','血_谷丙转氨酶', '血_总蛋白',               
'血_碳酸氢根','血_平均血红蛋白含量','血_碱性磷酸酶','血_总胆固醇','血_血小板压积',                 
'血_钾','血_尿酸','血_γ-谷氨酰转肽酶','血_氯','血_球蛋白',                 
'血_乳酸脱氢酶','血_eGFR(基于CKD-EPI方程)','血_嗜酸细胞(#)',               
'血_血小板计数','血_RBC分布宽度SD','血_RBC分布宽度CV',  '年龄']                        

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








import numpy as np
X = np.array(X)
Y = np.array(Y)
Y_predict = []
Y_true = []
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    model = XGBRegressor(max_depth=6,
                         n_estimators=10000,
                         objective='reg:squarederror',
                         booster='gbtree',
                         eval_metric='mae',
                         learning_rate=0.01,
                         gamma=0.1,
                         tree_method='exact',
                         min_child_weight=1,
                         subsample=0.71,
                         colsample_bytree=1,
                         reg_alpha=0,
                         reg_lambda=10,
                         random_state=0)
    model.fit(X[train_index], Y[train_index], eval_set=[(X[test_index], Y[test_index])], verbose=True)

    part_predict = model.predict(X[test_index])
    Y_predict.append(part_predict)
    Y_true.append(Y[test_index])


from itertools import chain
Y_predict  = list(chain(*Y_predict))
Y_true = list(chain(*Y_true))


import matplotlib.pyplot as plt
plt.figure(figsize=(50, 4))
from sklearn.metrics import mean_squared_error,mean_absolute_error\
    ,mean_absolute_percentage_error, r2_score



print("特征选择方法")
if Method == 1:
    print("CatBoost")
if Method == 2:
    print("XGBoost")
if Method == 3:
    print("LightGBM")
print('XGBoost分类器')

print('MSE {}'.format((mean_squared_error(Y_true ,Y_predict))))
#RMSE 可以调用 mean_squared_error 方法实现, 设置 squared=False 即可;
print('RMSE {}'.format((mean_squared_error(Y_true ,Y_predict, squared=False))))
print('RAE {}'.format(mean_absolute_error(Y_true ,Y_predict)))
print('RAPE {}'.format(mean_absolute_percentage_error(Y_true ,Y_predict)))
print('r2 {}'.format(r2_score(Y_true ,Y_predict)))


plt.title('S1_IgG antibody concentration prediction based on XGBoost')
plt.plot(Y_true, "blue")
plt.plot(Y_predict, "red")
plt.legend(['True','Predict'],loc='best')
plt.show()
