import pandas as pd
from utils import del_rows,del_columns
from sklearn.preprocessing import StandardScaler

"""
生成数据集A
"""
#设定随机种子为5
seed = 5
df = pd.read_excel(r'D:\临床指标简化整理    实验报告.xlsx')
# ICU经历,发病天数  和   症状程度,临床结局  逻辑强相关      故去掉
df = df.drop(['发病日期','入院时间','是否进入ICU','发病天数'], axis=1)
#字符型变量转离散型变量
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})
df['临床结局 ']=df['临床结局 '].astype(str).map({'出院':0,'死亡':1})
df['严重程度（最终）']=df['严重程度（最终）'].astype(str).map({'无症状感染者':0,'轻型':0,'重型':1,'危重型':1})
""" 
存在部分临床结局,严重程度缺失的行, 去掉
  axis	0为行 1为列，default 0，数据删除维度
  how	{‘any’, ‘all’}, default ‘any’，any：删除带有nan的行；all：删除全为nan的行
  thresh	int，保留至少 int 个非nan行
  subset	list，在特定列缺失值处理
  inplace	bool，是否修改源文件
"""
df = df.dropna(axis=0, how='any', thresh=None, subset=['临床结局 ', '严重程度（最终）'], inplace=False)

print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)
#统计缺失值是到这步统计的

#对缺失值进行补0
df = df.fillna(0)


#先把分类变量去除再进行标准化
data = df.drop(['病人ID','临床结局 ','性别','严重程度（最终）','出院/死亡时间','检测日期',
                '高血压(0=无，1=有)','糖尿病(0=无，1=有)'], axis=1)



#去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
sc = StandardScaler()
features = sc.fit_transform(data)
#标准化之后会丢失列名,再补回去
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
data_transform.insert(0,'严重程度（最终）', df['严重程度（最终）'].tolist())
data_transform.insert(0,'性别', df['性别'].tolist())
data_transform.insert(0,'临床结局 ', df['临床结局 '].tolist())
data_transform.insert(0,'出院/死亡时间', df['出院/死亡时间'].tolist())
data_transform.insert(0,'检测日期', df['检测日期'].tolist())
data_transform.insert(0,'病人ID', df['病人ID'].tolist())


data_transform.to_excel('Pre-processed outcome and severity dataset (with Time).xlsx')

