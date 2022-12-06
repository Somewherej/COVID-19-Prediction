import pandas as pd
from utils import del_rows,del_columns
from sklearn.preprocessing import StandardScaler

"""
生成数据集B
只考虑S1_IgG
"""

#设定随机种子为5
seed = 5
df = pd.read_excel(r'D:\临床指标简化整理    实验报告.xlsx')
# 临床结局  严重程度（最终） N_IgG  是否进入ICU  发病天数         和             S1_IgG 已知强相关
# 故去掉
df = df.drop(['发病日期','入院时间','出院/死亡时间','检测日期','临床结局 ','严重程度（最终）','N_IgG','是否进入ICU','发病天数'], axis=1)
# 字符型进行映射数值型
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})

#存在部分'S1_IgG'缺失的行, 去掉
df = df.dropna(axis=0, how='any', thresh=None, subset=['S1_IgG'], inplace=False)


#缺失值
print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)




#对缺失值进行补0
df = df.fillna(0)

#先把病人ID和分类变量去除,再进行标准化
data = df.drop(['病人ID','性别','高血压(0=无，1=有)','糖尿病(0=无，1=有)'], axis=1)

#去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
sc = StandardScaler()
features = sc.fit_transform(data)
#标准化之后会丢失列名,再补回去     https://blog.csdn.net/duxinyuhi/article/details/88792738
data_transform = pd.DataFrame(features,columns=data.columns)

#因为我们之前将分类变量去除了,所以我们要将分类型变量重新添加回来
"""
df = pd.DataFrame(np.arange(12).reshape(4,3),columns=['a','b','c'])
print(df)
df.insert(0,'insert',[2,22,222,2222])
print(df)
   a   b   c
0  0   1   2
1  3   4   5
2  6   7   8
3  9  10  11
   insert  a   b   c
0       2  0   1   2
1      22  3   4   5
2     222  6   7   8
3    2222  9  10  11
"""
data_transform.insert(0,'糖尿病(0=无，1=有)', df['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0,'高血压(0=无，1=有)', df['高血压(0=无，1=有)'].tolist())
data_transform.insert(0,'性别', df['性别'].tolist())
data_transform.insert(0,'病人ID', df['病人ID'].tolist())

print(data_transform)


#保存
data_transform.to_excel('Pre-processed antibody dataset.xlsx')
