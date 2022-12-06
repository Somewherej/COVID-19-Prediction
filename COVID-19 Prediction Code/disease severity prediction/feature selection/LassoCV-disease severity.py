import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
from sklearn.linear_model import LassoCV



seed = 5
df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed outcome and severity dataset (with Time).xlsx')
df = df.drop(['提前预测时间','检测日期','出院/死亡时间'],axis=1)


#预测症状程度,不应该知道临床结局
#在特征选择阶段,我们没有利用患者样本的时序性, 而是将每条样本视为一个独立的个体
df = df.drop(['临床结局 '],axis=1)
X = df.drop(['严重程度（最终）','病人ID'],axis=1)
Y = df['严重程度（最终）']


alphas = np.logspace(-3,1,100)
model =  LassoCV(alphas = alphas, cv = 10, max_iter = 300000).fit(X,Y) #cv, cross-validation
feature = pd.Series(model.coef_, index=X.columns)
print("LassoCV picked " + str(sum(feature != 0)) + " variables and eliminated the other " +  str(sum(feature == 0)) + " variables")

#画出特征变量的重要程度并打印，这里面选出前10个重要的特征,由于相关性有正有负,取abs()
#sort_values()升序排列,
#DataFrame.tail(n)据位置返回对象的最后n行
feature = abs(feature)
feature_important = pd.concat([feature.sort_values().tail(10)])
print('去掉ICU和Day的数据集')
print('LassoCV')
print(feature_important)




#画图部分
matplotlib.rcParams['figure.figsize'] = (11.0, 5.0)
feature_important.plot(kind="barh")
plt.title("Selection in the LassoCV Model")
plt.show()