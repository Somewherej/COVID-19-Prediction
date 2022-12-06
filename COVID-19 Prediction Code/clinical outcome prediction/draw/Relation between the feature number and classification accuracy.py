import matplotlib.pyplot as plt
x_10 = [10,9,8,7,6,5 ]





LassoCVKNN =[0.9771,0.97805,0.97615	,0.97519,	0.9771	,0.97042]
ElasticNetCVKNN =[0.9771,0.97901,0.97615,	0.97519,	0.9771,	0.9771]
RandomForestKNN = [0.9771,0.97615,0.97328,	0.97615,	0.97519,	0.97424]



LassoCVSVM =[0.98282	,0.98378	,0.98473	,0.98378,	0.98187	,0.97519]
ElasticNetCVSVM =[0.98282,	0.98473,	0.98473,	0.98378,	0.98187,	0.98092 ]
RandomForestSVM = [0.98187,	0.98187,	0.97805,	0.97328,	0.97328,	0.97042]



LassoCVTwoDCNN =[0.97996,	0.98187,	0.97901,	0.97996,	0.97519,	0.9666]
ElasticNetCVTwoDCNN =[0.97233,	0.97996,	0.97805,	0.9771,	0.97424,	0.97137]
RandomForestTwoDCNN = [0.97233,	0.97137,	0.97233,	0.9666,	0.96851	,0.97328]






LassoCVTAOneDCNN =[0.9876,	0.9876,	0.98664	,0.98569,	0.98664	,0.98187]
ElasticNetCVTAOneDCNN =[0.98378,	0.98569,	0.98664,	0.98664,	0.98664,	0.98378]
RandomForestTAOneDCNN = [0.98473,	0.98187	,0.98092,	0.98187	,0.97901	,0.97901]



LassoCVTALSTM =[0.9876,	0.98855,0.9876,	0.98664,	0.98664,	0.98378]
ElasticNetCVTALSTM =[0.9876,0.9876,	0.98855,	0.9876,	0.98569,	0.98378]
RandomForestTALSTM =[0.98473,	0.98092,	0.98092,	0.97615,	0.97424,	0.97328]









plt.rcParams["figure.figsize"] = [8, 6]
fig, ax = plt.subplots(1, 1)


ax.plot(x_10,LassoCVTALSTM,color='#f37649',linewidth = 2,linestyle='-',marker='+',ms=6,label='LassoCV TA LSTM')
ax.plot(x_10,ElasticNetCVTALSTM,color='#f37649',linewidth = 2,linestyle='-',marker='o',ms=5,label='ElasticNetCV TA LSTM')
ax.plot(x_10,RandomForestTALSTM,color='#f37649',linewidth = 2,linestyle='-',marker='s',ms=5,label='RandomForest TA LSTM')
#----------
ax.plot(x_10,LassoCVTAOneDCNN,color='#5fc6c9',linewidth = 2,linestyle='-',marker='+',ms=6,label='LassoCV TA 1D-CNN')
ax.plot(x_10,ElasticNetCVTAOneDCNN,color='#5fc6c9',linewidth = 2,linestyle='-',marker='o',ms=5,label='ElasticNetCV TA 1D-CNN')
ax.plot(x_10,RandomForestTAOneDCNN,color='#5fc6c9',linewidth = 2,linestyle='-',marker='s',ms=5,label='RandomForest TA 1D-CNN')
#----------
ax.plot(x_10,LassoCVTwoDCNN,color='#fac00f',linewidth = 2,linestyle='-',marker='+',ms=6,label='LassoCV 2D-CNN')
ax.plot(x_10,ElasticNetCVTwoDCNN,color='#fac00f',linewidth = 2,linestyle='-',marker='o',ms=5,label='ElasticNetCV 2D-CNN')
ax.plot(x_10,RandomForestTwoDCNN,color='#fac00f',linewidth = 2,linestyle='-',marker='s',ms=5,label='RandomForest 2D-CNN')
#----------
ax.plot(x_10,LassoCVKNN,color='#45596d',linewidth = 2,linestyle='-',marker='+',ms=6,label='LassoCV KNN')
ax.plot(x_10,ElasticNetCVKNN,color='#45596d',linewidth = 2,linestyle='-',marker='o',ms=5,label='ElasticNetCV KNN')
ax.plot(x_10,RandomForestKNN,color='#45596d',linewidth = 2,linestyle='-',marker='s',ms=5,label='RandomForest KNN')
#----------
ax.plot(x_10,LassoCVSVM,color='#015699',linewidth = 2,linestyle='-',marker='+',ms=6,label='LassoCV SVM')
ax.plot(x_10,ElasticNetCVSVM,color='#015699',linewidth = 2,linestyle='-',marker='o',ms=5,label='ElasticNetCV SVM')
ax.plot(x_10,RandomForestSVM,color='#015699',linewidth = 2,linestyle='-',marker='s',ms=5,label='RandomForest SVM')

font1 = {'size': 8.4}
ax.legend(loc="lower right",prop=font1)
ax.set_xlabel('Number of Features',fontsize=15)
ax.set_ylabel('Classification Accuracy',fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.ylim(0.95,1)
plt.show()
