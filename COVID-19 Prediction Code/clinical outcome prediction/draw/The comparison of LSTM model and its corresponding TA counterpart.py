
import matplotlib.pyplot as plt
x_10 = [10,9,8,7,6,5 ]



LassoCVLSTM =    [0.98569	,0.98664,	0.98664,	0.98282	,0.97615	,0.97615]
ElasticNetCVLSTM =  [0.98378	,0.98569,	0.98569	,0.98282	,0.98282	,0.98092]
RandomForestLSTM = [0.9771	,0.97901,	0.97615	,0.97424	,0.97328	,0.96947]


 

LassoCVTALSTM =  [0.9876,	0.98855,	0.9876,	0.98664,	0.98664,	0.98378]
ElasticNetCVTALSTM = [0.9876,	0.9876,	0.98855,	0.9876,	0.98569,	0.98378]
RandomForestTALSTM =  [0.98473,	0.98092,	0.98092,	0.97615,	0.97424,	0.97328]



plt.rcParams["figure.figsize"] = [9, 6]
fig, ax = plt.subplots(1, 1)

##E4392E #3979F2
#----------
ax.plot(x_10,LassoCVLSTM,color='#3979F2',linewidth = 2,linestyle='--',marker='+',ms=7,label='LassoCV LSTM')
ax.plot(x_10,ElasticNetCVLSTM,color='#3979F2',linewidth = 2,linestyle='--',marker='o',ms=6,label='ElasticNetCV LSTM')
ax.plot(x_10,RandomForestLSTM,color='#3979F2',linewidth = 1.5,linestyle='--',marker='s',ms=6,label='RandomForest LSTM')
#----------
ax.plot(x_10,LassoCVTALSTM,color='#E4392E',linewidth = 2,linestyle='-',marker='+',ms=7,label='LassoCV TA LSTM')
ax.plot(x_10,ElasticNetCVTALSTM,color='#E4392E',linewidth = 2,linestyle='-',marker='o',ms=6,label='ElasticNetCV TA LSTM')
ax.plot(x_10,RandomForestTALSTM,color='#E4392E',linewidth = 2,linestyle='-',marker='s',ms=6,label='RandomForest TA LSTM')

font1 = {'size': 8.4}
ax.legend(loc="lower right",prop=font1)
ax.set_xlabel('Number of Features',fontsize=15)
ax.set_ylabel('Classification Accuracy',fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.show()

