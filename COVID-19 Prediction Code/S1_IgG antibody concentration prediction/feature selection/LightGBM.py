

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
np.set_printoptions(threshold=np.inf)



seed = 5
df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed antibody dataset.xlsx')
X = df.drop(['S1_IgG','病人ID'],axis=1)
Y = df['S1_IgG']

from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import KFold
importance = np.zeros(52)
cv = 10
kf = KFold(n_splits=cv, shuffle=False, random_state=None)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = LGBMRegressor(n_estimators=30000, learning_rate=0.001, boosting_type='gbdt', objective='regression',
                          metric='mae', eval_metric='mae', od_wait=500,
                          num_leaves=60, colsample_bytree=0.7, feature_fraction=0.7, min_data=100,
                          min_hessian=1)
    model.fit(X_train, Y_train, eval_set=(X_test, Y_test), early_stopping_rounds=500,
              verbose=True)
    importance = importance + model.feature_importances_


from sklearn import preprocessing
MinMaxSc= preprocessing.MinMaxScaler()

#np.true_divide 返回除法的浮点数结果而不作截断
output_importance = pd.DataFrame(np.true_divide(importance,cv))
output_importance = MinMaxSc.fit_transform(output_importance)


from itertools import chain
output_importance  = list(chain(*output_importance))

output_importance = pd.Series(output_importance, index=X.columns)
output_importance = output_importance.sort_values().tail(30)
print(output_importance)



#作图
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
output_importance.plot(kind="barh")
plt.title("Selections in the LightGBM Model")
plt.show()