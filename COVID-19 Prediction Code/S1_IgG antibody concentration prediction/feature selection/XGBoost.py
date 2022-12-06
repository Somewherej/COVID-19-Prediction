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

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
importance = np.zeros(52)
cv = 10
kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = XGBRegressor(max_depth=10, n_estimators=10000, objective='reg:squarederror',
                         booster='gbtree', eval_metric='mae', learning_rate=0.001, gamma=0.1,
                         tree_method='exact', min_child_weight=1, subsample=0.71,
                         colsample_bytree=1, reg_alpha=0,
                         reg_lambda=10, random_state=0)
    model.fit(X_train,Y_train, eval_set=[(X_test,Y_test)], early_stopping_rounds = 500,
                   verbose=True)
    importance = importance + model.feature_importances_


from sklearn import preprocessing
MinMaxSc= preprocessing.MinMaxScaler()

#np.true_divide 返回除法的浮点数结果而不作截断
#output_importance是十次特征重要性的平均值
output_importance = pd.DataFrame(np.true_divide(importance,cv))
#因为三种算法 CatBoost XGBoost LightGBM得出的特征重要性量纲差异大  归一化下
output_importance = MinMaxSc.fit_transform(output_importance)


from itertools import chain
output_importance  = list(chain(*output_importance))

output_importance = pd.Series(output_importance, index=X.columns)
output_importance = output_importance.sort_values().tail(30)
print('去除ICU and Day')
print(output_importance)

#作图
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
output_importance.plot(kind="barh")
plt.title("Selections in the XGBoost Model")
plt.show()
