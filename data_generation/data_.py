"""

    将traj数据处理成两个采样点数据：起点，终点

"""
# -*- coding: utf-8 -*-
import numpy as np

import pandas as pd
from static_graph import *
import random
import os
class DataTransit:
    '''
    [This class is used to proceed the raw data ]

    csvfile:the raw trip data （清洗起始点不是目标城市以及偏离平均值太大的数据）
    如
    '''
    def __init__(self,csvfile):
        self.csvfile = csvfile

    def load(self):
        # is used to pre-proceed the raw data
        df = pd.read_csv(self.csvfile,usecols=['ObjectID', 'StartTime','StartLon','StartLat','StartPos','StopTime',
                                               'StopLon','StopLat','StopPos','TravelMileage','TravelPeriod','TravelOil','avgspeed'],dtype= {'TravelOil':np.float32,'avgspeed':np.float32})

        df = df.drop_duplicates()

        #df = df.drop(df[(df.TravelMileage < 100) & (df.TravelPeriod < 600)].index)#删除漂移点

        # df = df.sort_values(by = 'StopTime')
        # df.reset_index(drop=True, inplace=True)
        #df['TravelOil'] = df['TravelOil'].map(lambda x: round(x * 0.001, 1))
        return df

    def traj2grid(self):
        # is used to tansfer the traj data to s-t point that we need to generate the static feature
        df = self.load()

        df_start = df[['ObjectID', 'StartTime','StartLon','StartLat']]
        df_start.rename(columns={'StartTime': 'Time', 'StartLon': 'Lon','StartLat':'Lat'}, inplace=True)

        df_end = df[['ObjectID','StopTime','StopLon','StopLat']]
        df_end.rename(columns={'StopTime': 'Time', 'StopLon': 'Lon','StopLat':'Lat'}, inplace=True)

        point_data = pd.concat([df_start, df_end], ignore_index=True)

        grid_distance = 7500
        is_delete = False
        if is_delete:
            tracks_data, grid_list, num_set = grid_process(point_data, grid_distance,is_delete)
            #筛选数据
            df = df[~df.index.isin(num_set)]
            df.reset_index(drop=True, inplace=True)
            df.to_csv('Sz_mesh.csv', index=None, encoding='utf_8_sig')
        else:
            tracks_data, grid_list = grid_process(point_data, grid_distance, is_delete)


        grid_nums = len(grid_list)

        #删除个数少于10的数据
        grid_data = tracks_data.loc[:,['Lon','Lat','grid_ID']]

        if os.path.exists('grid.csv'):
            print("grid.csv exists!")
        else:
            grid_data.to_csv('grid.csv', index=None, encoding='utf_8_sig')
            f = open("graph_grid_ids.txt", "w")
            for line in grid_list:
                f.write(str(line) + '\n')
            f.close()

        return tracks_data,grid_nums

    def mesh_grid(self):
        df = self.load()

        lenth = len(df)

        dt_start = df.loc[:, ['ObjectID','StartLon', 'StartLat','StartTime']]
        dt_start = dt_start.rename(columns={"StartLat": "Lat", "StartLon": "Lon"})

        dt_end = df.loc[:, ['ObjectID','StopLon', 'StopLat','StopTime','TravelOil','avgspeed','TravelMileage'
                               ,'TravelPeriod']]
        dt_end = dt_end.rename(columns={"StopLon": "Lon", "StopLat": "Lat"})

        gd = pd.read_csv('grid.csv')

        gd_start = gd[:lenth]

        gd_end = gd[lenth:]
        gd_end.reset_index(drop=True, inplace=True)

        dis_start = pd.concat([dt_start, gd_start], axis=1)
        dis_end = pd.concat([dt_end, gd_end], axis=1)

        dis_start = dis_start.drop(dis_start.columns[[1, 2]], axis=1)
        dis_end = dis_end.drop(dis_end.columns[[1, 2]], axis=1)

        # dis_start = dis_start.groupby(level=0, axis=1).last()
        # dis_end = dis_end.groupby(level=0, axis=1).last()

        # dis_start = dt_start.merge(gd_start,how='inner',on = ['Lon', 'Lat'])
        # dis_end = dt_end.merge(gd_end, how='inner', on=['Lon', 'Lat'])


        # dis_start.to_csv('dis_start.csv', index=None, encoding='utf_8_sig')
        # dis_end.to_csv('dis_end.csv', index=None, encoding='utf_8_sig')
        # dis_start = pd.merge(dt_start, gd, left_on=['StartLon', 'StartLat'], right_on=['Lon', 'Lat'])
        # dis_end = pd.merge(dt_end, gd, left_on=['StopLon', 'StopLat'], right_on=['Lon', 'Lat'])


        dis = pd.DataFrame()
        dis.loc[:, 'ObjectID'] = dis_end.loc[:, 'ObjectID']
        dis.loc[:, 'from'] = dis_start.loc[:, 'grid_ID']
        dis.loc[:, 'to'] = dis_end.loc[:, 'grid_ID']


        dis.loc[:, 'StopLon'] = dt_end.loc[:, 'Lon']
        dis.loc[:, 'StopLat'] = dt_end.loc[:, 'Lat']

        dis.loc[:, 'StartTime'] = dis_start.loc[:, 'StartTime']
        dis.loc[:, 'StopTime'] = dis_end.loc[:, 'StopTime']
        dis.loc[:, 'Oil'] = dis_end.loc[:, 'TravelOil']
        dis.loc[:, 'Speed'] = dis_end.loc[:, 'avgspeed']
        dis.loc[:, 'Period'] = dis_end.loc[:, 'TravelPeriod']
        dis.loc[:, 'Mile'] = dis_end.loc[:, 'TravelMileage']


        dis.StartTime = pd.to_datetime(dis.StartTime, format='%Y-%m-%d')
        dis.StopTime = pd.to_datetime(dis.StopTime, format='%Y-%m-%d')

        dis['StopTime_hour'] = dis['StopTime'].dt.hour
        dis['StopTime_minute'] = dis['StopTime'].dt.minute

        #30min 采集一次
        #dis['StopTime_emb'] = 2 * dis['StopTime_hour'] + dis['StopTime_minute'].apply(lambda x: 1 if x >= 30 else 0)
        #60min采集一次
        dis['StopTime_emb'] = dis['StopTime_hour']
        dis.sort_values(by=['StopTime'])
        dis = dis.dropna(axis=0, how='any')
        dis.reset_index(drop=True,inplace=True)

        #dis.sort_values(by='from', inplace=True)
        #再去ftd.csv里筛查
        if os.path.exists('from_to_data.csv'):
            print("Params csvfile exists!")
        else:
            dis.to_csv('from_to_data.csv', index=None, encoding='utf_8_sig')

        return dis

    def external_data(self):
        #把数据整理成 num_node * time 的形式
        grid_nums = self.extract_grid()

        data_df = self.graph_distance()
        time_df = data_df.loc[:, ['to', 'stoptime_hour','stoptime_minute']]
        time_df['stoptime'] = 2 * time_df['stoptime_hour'] + time_df['stoptime_minute'].apply(lambda x : 1 if x >= 30 else 0)
        weather_df = data_df.loc[:, ['to', 'tianqi']]

        weather_df.rename(columns={'tianqi': 'weather_emb',},inplace=True)
        def classify(x):
            if '多云' in x:
                return 1
            elif '晴' in x:
                return 2
            elif '雨' in x:
                return 4
            else:
                return 3
        weather_df['weather_emb'] = weather_df['weather_emb'].apply(lambda x :classify(x))
        time_df = time_df.loc[:, ['to', 'time_emb']]
        oil_df = data_df.loc[:, ['to', 'oil']]
        speed_df = data_df.loc[:, ['to', 'speed']]

        df3 = pd.concat([oil_df, speed_df,time_df,weather_df], join="inner", axis=1)
        df3.drop(columns=['to'], inplace=True)

        df3.oil = df3.oil.apply(lambda x : round(x,1))
        df3.speed = df3.speed.apply(lambda x: round(x, 1))

        dict = {}

        for i in range(grid_nums):
            dict[i] = []

        for row in oil_df.values:
            dict[row[0]].append(row[1])

        data = pd.DataFrame.from_dict(dict, orient='index')
        pd.set_option('display.max_rows', None)

        data.T.to_csv('data.csv', index=None)

    def data_written(self):

        #_, grid_nums = self.traj2grid()
        grid_nums = 70
        dis = pd.read_csv('from_to_data.csv')
        # 标注停车状态

        data_list = []
        time_list = []
        flow_list = []
        for index, row in dis.iterrows():
            if not time_list:
                oil = [0] * grid_nums
                oil[row['to']] += row['Oil']
                flow = [0] * grid_nums
                flow[row['to']] += 1
                time_list.append(row['StopTime_emb'])
                data_list.append(oil)
                flow_list.append(flow)
            elif row['StopTime_emb'] in time_list:
                data_list[-1][row['to']] += row['Oil']
                flow_list[-1][row['to']] += 1
            else:
                if time_list[0] < row['StopTime_emb']:
                    disc = row['StopTime_emb'] - time_list[0]
                else:
                    disc = 24 - time_list[0] + row['StopTime_emb']
                while disc > 1:
                    data_list.append(data_list[-disc])
                    flow_list.append(flow_list[-disc])
                    disc -= 1
                oil = [0] * grid_nums
                oil[row['to']] += row['Oil']
                flow = [0] * grid_nums
                flow[row['to']] += 1
                time_list.pop(0)
                time_list.append(row['StopTime_emb'])
                data_list.append(oil)
                flow_list.append(flow)

        data_arr = np.array(data_list)
        oil_data = np.expand_dims(data_arr, -1)
        flow_arr = np.array(flow_list)
        flow_data = np.expand_dims(flow_arr, -1)

        data = np.concatenate((oil_data, flow_data), axis=2)

        np.savez_compressed(
            'original_data',
            data=data
        )

if __name__ == "__main__":
    Data = DataTransit('Sz.csv')
    #Data.load()
    #Data.traj2grid()
    #Data.mesh_grid()
    Data.data_written()
















