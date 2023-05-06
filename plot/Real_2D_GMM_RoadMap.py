import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import time
import numpy as np
import pandas as pd
import sklearn.preprocessing  as  preprocessing
# from sklearn.neighbors import kde
import matplotlib.pyplot as plt
from sklearn import mixture

def roadData():
    #显示停等点。

    #data1=pd.read_csv('.\\travelInfoAll2016\\travelInfoAll201607.csv')
    #plt.plot(data1.StopLon,data1.StopLat,'.',markersize=1,color='blue')
    #data2=pd.read_csv('.\\travelInfoAll2016\\travelInfoAll201606.csv')
    #plt.plot(data2.StopLon,data2.StopLat,'.',markersize=1,color='blue')

    ## 读OSM的json文件，生成路网
    with open('luohu3.json','r',encoding='utf-8') as mh:
        jsondata1 = pd.read_json(mh)
    #print(jsondata1)
    #jsondata1 = pd.read_json(jsonURL)
    # 有用的路网类型,排除了步行道、自行车道之类
    roadnetworkType = ['primary_link', 'secondary_link', 'tertiary_link', 'primary', 'secondary', 'trunk_link', 'service','tertiary', 'trunk', 'residential', 'unclassified']
    #roadnetworkType = ['primary', 'secondary','primary_link', 'secondary_link','tertiary_link','tertiary','trunk_link','trunk']
    #roadnetworkType = ['primary', 'secondary','tertiary','trunk','trunk_link','secondary_link','tertiary_link','primary_link','residential']
    #roadnetworkType = ['primary', 'secondary','tertiary','trunk','trunk_link','secondary_link','tertiary_link','primary_link','residential']
    # 筛选出需要的路网类型 # 路网，0阶段
    roadList = []
    for item in jsondata1.features:
        if item is not None:
            if  item['geometry'] is not None  and (len(item['geometry']) > 0):  # 保证有数据
                if item['geometry']['type'] == 'LineString':  # 保证是直线类型
                    if item['properties'] is not None and('highway' in item['properties']):  # 是highway类型
                        if item['properties']['highway'] in roadnetworkType:  # 在预设规定的类型中
                             if len(item['geometry']['coordinates'])>0:
                                   roadList.append(item['geometry']['coordinates'])
    print('roadList: the number of road:', len(roadList))
    return roadList

# from sklearn.mixture import BayesianGaussianMixture
fig = plt.figure()
np.set_printoptions(threshold=np.inf)  #超过阈值就缩写
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#处理数据   73*137
df = pd.read_csv("clean201803-201809.csv",header=None)
# df = pd.read_csv("data/201807.txt", sep="\t")
data=np.array(df)
dataSet = data.tolist()       #将数组或者矩阵转换成列表
month_gridnum = []
Month,predict_day = 9,8
startH = 12              #11
endH = 13              #14
alldensity = []
for day in range(predict_day,predict_day+1):
    day_gridnum = []
    position = 0
    for hour in range(startH,endH):
        X1 = []
        listx = []
        listy = []
        for i in range(0, len(dataSet), 1):
            Starttime = time.strptime(dataSet[i][1], "%Y-%m-%d %H:%M:%S.%f")  # 开始等待
            Endtime = time.strptime(dataSet[i][2], "%Y-%m-%d %H:%M:%S.%f")  # 结束等待
            if (Endtime[1] == Month and Endtime[2] == day):  # 0914的数据
                all_Starttime = Starttime[3] * 3600 + Starttime[4] * 60 + Starttime[5]  # 时分转换成秒
                all_Endtime = Endtime[3] * 3600 + Endtime[4] * 60 + Endtime[5]  # 时分转换成秒
                listx.append(dataSet[i][3])  # 经度添加
                listy.append(dataSet[i][4])  # 纬度
                templist = []
                templist.append(dataSet[i][3])  # 经度添加
                templist.append(dataSet[i][4])  # 纬度
                if (all_Starttime >= hour * 3600 and all_Starttime <= (hour + 1) * 3600) or (
                        all_Endtime >= hour * 3600 and all_Starttime <= hour * 3600):
                    X1.append(templist)
        #坐标归一化
        X=np.mat(X1)  #创建经纬度，时间矩阵                  ##########自定义
        listx = np.array(listx)  # 列表转化为数组
        listy = np.array(listy)
        xmin, xmax, ymin, ymax = min(listx), max(listx), min(listy), max(listy)
        # 坐标归一化
        min_max_scaler = preprocessing.MinMaxScaler()  # 按比例缩放，归一化
        X_longitude = min_max_scaler.fit_transform(X[:, 0])  # 经度
        X_latitude = min_max_scaler.fit_transform(X[:, 1])  # 纬度
        # X13 = min_max_scaler.fit_transform(X[:, 2])    #时间
        XX = np.hstack((X_longitude, X_latitude))  # 将X11,X12,X13连接并平铺
        print('XX.shape:', XX.shape)
        # print(XX)
        # 选择合适的GMM   diag，spherical，tied，full
        gmm = mixture.BayesianGaussianMixture(n_components= 9,  covariance_type='full',
                                              weight_concentration_prior_type='dirichlet_distribution',
                                              weight_concentration_prior = 1e+2,mean_precision_prior=1e-4,max_iter=100,
                                              warm_start=True
                                              )
        gmm.fit(XX)
        dx, dy = (max(listy) - min(listy)) / float(100), (max(listx) - min(listx)) / float(100)  # x,y坐标步长
        print("dx,dy",dx,dy)
        y2, x2 = np.mgrid[slice(min(listy), max(listy) + dy, dy), slice(min(listx), max(listx) + dx, dx)]  # mgrid坐标范围y,x
        yt = np.mgrid[slice(min(listy), max(listy) + dy, dy)].reshape(-1,1)
        xt = np.mgrid[slice(min(listx), max(listx) + dx, dx)].reshape(-1,1)
        print(max(listx),min(listx),max(listy), min(listy))
        yt = min_max_scaler.fit_transform(yt)  # y按相同比例缩放
        xt = min_max_scaler.fit_transform(xt)  # x按相同比例缩放

        z = np.zeros([yt.shape[0], xt.shape[0]])  # 构造z,作为密度矩阵
        print("z.shape", z.shape)
        for i in range(0, yt.shape[0] - 1, 1):
            for j in range(0, xt.shape[0] - 1, 1):
                input_point = np.hstack((xt[j], yt[i])).reshape(1,-1)  # [开始：结尾：步长]
                z[i][j] = np.exp(gmm.score_samples(input_point)) # np.exp()：返回e的幂次方
        z = z[:-1, :-1]  # 删除最后一行和最后一列
        z = z / z.max()
        # z = min_max_scaler.fit_transform(z)
        levels = np.linspace(z.min(), z.max(), 30)
        # levels = np.linspace(0, 4, 30)
        ax = plt.subplot(111 + position)
        cmap = plt.get_cmap('RdYlBu_r')  # 设置色阶
        # cmap = plt.get_cmap('RdYlGn_r')
        # ax.set_title('GMM201805%d:%d:00~%d:00' % (predict_day, hour, hour + 1))
        # ax.set_title('GMM201805%d:%d:00~%d:00,%dpoints' % (predict_day,hour,hour+1 , X.shape[0]))
        cs = ax.contourf(x2[:-1, :-1] + dx / 2., y2[:-1, :-1] + dy / 2., z, levels, cmap = cm.coolwarm)    #等高线的绘制
        z = z[:73, :137]
        alldensity.append(z)
        print(z.shape)
        # np.savetxt('dataout/GMM_20180531_11-12.txt', z)
        cbar = fig.colorbar(cs)#显示色度条
        plt.xlim([114.06, 114.17])
        plt.ylim([22.53, 22.61])
        #画路网
        roadList = roadData()
        for i in range(len(roadList)):
            road = roadList[i]
            road = np.array(road)
            plt.plot(road[:, 0], road[:, 1], '-', color='gray', linewidth=0.5)
        plt.tick_params(labelsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("latitude")
        plt.show()
        fig.savefig('road_color-10-11.png', dpi=600, format='png')
