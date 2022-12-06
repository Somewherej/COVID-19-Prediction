import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold



seed = 5
df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed outcome and severity dataset (with Time).xlsx')
df = df.drop(['提前预测时间','检测日期','出院/死亡时间'],axis=1)
#预测临床结局,不应该知道症状程度
df = df.drop(['严重程度（最终）'],axis=1)
X = df.drop(['临床结局 ','病人ID'],axis=1)
Y = df['临床结局 ']

cv = 10
kf = KFold(n_splits=cv, shuffle=True, random_state=seed)


#已知 len(model.feature_importances_) = 57, and type(model.feature_importances_) ==<class 'numpy.ndarray'>
#importance 将每折的feature importances累加
#importance使用np.true_divide   除以cv  得到   out_importance
importance = np.zeros(57)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = RandomForestClassifier(n_estimators=10000, random_state=5, n_jobs=-1)
    model.fit(X_train, Y_train.astype('int'))
    importance = importance + model.feature_importances_


#np.true_divide 返回除法的浮点数结果而不作截断
output_importance = pd.Series(np.true_divide(importance,cv), index=X.columns)
output_importance = output_importance.sort_values().tail(10)
print('去掉ICU和Day的数据集')
print('RandomForest')
print(output_importance)

#作图
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
output_importance.plot(kind="barh")
plt.title("Selections in the RandomForest Model")
plt.show()
