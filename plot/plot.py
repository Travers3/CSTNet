import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#grid_size
# x_axis_data = [1000,2500,5000,7500]
# y_mae_sz = [2.532,2.2165,3.8301,4.79]
# y_mae_cs = [1.9837,1.8509,2.3181,2.8239]
#
# y_rmse_sz = [4.7764,3.6762,5.2463,5.7789]
# y_rmse_cs = [3.8374,3.2621,4.1176,5.2178]
#
# x_index=['1000','2500','5000','7500']  # x 轴显示的刻度
#
# # plt.figure(figsize=(5,5))
# ax = plt.gca()
#
# plt.xlabel(u'grid_size',fontsize=14)
# plt.plot(x_axis_data,y_mae_sz,color="tomato",linewidth=2,linestyle='-',label='MAE', marker='s')
# plt.plot(x_axis_data,y_rmse_sz,color="cadetblue",linewidth=1,linestyle='-',label='RMSE', marker='v')
#
# _ = plt.xticks(x_axis_data,x_index)
# # ax.set_ylim(1.5,6)
#
# plt.legend(loc=2)
# plt.savefig('Sz.png',dpi=600)
# plt.show()

#horizon
#mae
# horizon_mae = pd.read_csv('para/horizon.csv',usecols=[1,2,3,4,5,6])
# horizon_mae = horizon_mae.rename(columns=lambda x:x.replace('_mae',''))
# print(horizon_mae)
# lstm_mae = horizon_mae.iloc[12:24,0]
# gru_mae = horizon_mae.iloc[12:24,1]
# ogcrnn_mae = horizon_mae.iloc[12:24,2]
#
# Gated_STGCN_mae = horizon_mae.iloc[12:24,3]
# GWNET_mae = horizon_mae.iloc[12:24,4]
# CSTNet_mae = horizon_mae.iloc[12:24,5]
#
# name = horizon_mae.columns.tolist()
#
# model = []
# x_axis = [i for i in range(1,13)]
# color = ['coral','green','teal',"slategray",'brown','peru']
#
# plt.figure()
# plt.xlabel(u'Horizon',fontsize=14)
# ax = plt.gca()
# for i in range(6):
#     model.append(horizon_mae.iloc[12:24,i])
#     plt.plot(x_axis,model[i],color=color[i],linestyle='-',label=name[i],marker='o')
# ax.set_ylabel('MAE')
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
# ax.legend(loc='upper left', fontsize='x-small', ncol=3)
# # ax.legend(loc='center left',bbox_to_anchor=(0.08, 1.14), ncol=3)
#
# plt.savefig('horizon_cs_mae.png',dpi=600)
# plt.show()

#rmse
# horizon_rmse = pd.read_csv('horizon.csv',usecols=[7,8,9,10,11,12])
# horizon_rmse = horizon_rmse.rename(columns=lambda x:x.replace('_rmse',''))
# print(horizon_rmse)
#
#
# name = horizon_rmse.columns.tolist()
#
# model = []
# x_axis = [i for i in range(1,13)]
# color = ['coral','green','teal',"slategray",'brown','peru']
#
# plt.figure()
# plt.xlabel(u'Horizon_Cs',fontsize=14)
# ax = plt.gca()
# for i in range(6):
#     model.append(horizon_rmse.iloc[12:24,i])
#     plt.plot(x_axis,model[i],color=color[i],linestyle='-',label=name[i],marker='o')
#
# ax.set_ylabel('RMSE')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
# ax.legend(loc='center left', bbox_to_anchor=(0.08, 1.14), ncol=3)
#
# plt.savefig('horizon_cs_rmse.png',dpi=600)
# plt.show()

#time_interval

# sz_mae_60 = pd.read_csv('time_interval.csv',usecols=[1,2,3]).rename(columns=lambda x:x.replace('_mae',''))
# sz_mae_30 = pd.read_csv('time_interval.csv',usecols=[4,5,6]).rename(columns=lambda x:x.replace('_mae_30',''))
#
# sz_rmse_60 = pd.read_csv('time_interval.csv',usecols=[7,8,9]).rename(columns=lambda x:x.replace('_mae',''))
# sz_rmse_30 = pd.read_csv('time_interval.csv',usecols=[10,11,12]).rename(columns=lambda x:x.replace('_mae_30',''))
#
# name = sz_mae_30.columns.tolist()
# name_30 = []
# name_60 = []
# for n in name:
#     name_30.append(str(n)+'_30min')
#     name_60.append(str(n) + '_60min')
#
# model_60 = []
# model_30 = []
#
# x_axis = [i for i in range(1,13)]
# color_60 = ['coral',"slategray",'teal']
# color_30 = ['green','peru','brown']
#
# plt.figure()
# plt.xlabel(u'Horizon',fontsize=14)
# ax = plt.gca()
# for i in range(3):
#     model_60.append(sz_mae_60.iloc[:12,i])
#     model_30.append(sz_mae_30.iloc[:12,i])
#     plt.plot(x_axis,model_60[i],color=color_60[i],linestyle='-',label=name_60[i],marker='o')
#     plt.plot(x_axis, model_30[i], color=color_30[i], linestyle='-', label=name_30[i], marker='v')
#
# ax.set_ylabel('MAE')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
# ax.legend(loc='center left', bbox_to_anchor=(-0.085, 1.14), ncol=3)
#
# plt.savefig('sz_mae.png',dpi=600)
# plt.show()

#node embedding
# x_axis_data = [i for i in range(20,80,10)]
# y_mae_sz = [3.0301,2.728,2.43, 2.5203,2.5799, 2.8211]
# y_mae_cs = [2.3653, 2.342,2.1834,2.0037,1.8509,2.2181]
#
# y_rmse_sz = [4.187, 4.074,3.6762,3.7821,3.8801,4.09245]
# y_rmse_cs = [3.7423, 3.6302,3.6072,3.4374,3.0621,3.4110]
#
# x_index=[i for i in range(20,80,10)]  # x 轴显示的刻度
#
# ax = plt.gca()
#
# plt.xlabel(u'Dimension of node embedding',fontsize=14)
# plt.plot(x_axis_data,y_mae_cs,color="tomato",linewidth=2,linestyle='-',label='MAE', marker='s')
# plt.plot(x_axis_data,y_rmse_cs,color="cadetblue",linewidth=1,linestyle='-',label='RMSE', marker='v')
#
# _ = plt.xticks(x_axis_data,x_index)
# # ax.set_ylim(2,5.5)
#
# plt.legend(loc=2)
# plt.savefig('node_embedding_cs.png',dpi=600)
# plt.show()

#heatmap

# data = pd.read_csv('heatmap_7am_gru.csv')
#
#
# fig, ax = plt.subplots()
#
# x = range(0,10)
# y = range(0,10)
# plt.xticks(x)
# plt.yticks(y)
# #设置标注前后左右的距离
# plt.imshow(data, cmap='coolwarm')
#
# cb = plt.colorbar()
# cb.set_label('Carbon Emission')
#
# plt.show()
# fig.savefig('10am.png',dpi=600,format='png')

# from mpl_toolkits.mplot3d import Axes3D    #这里只是用Axes3D函数，所以只导入了Axes3D
#
# data = pd.read_csv('heatmap_7am.csv')
# print(data.shape)
# fig=plt.figure()
#
# ax=Axes3D(fig,auto_add_to_figure=False)
# ax.view_init(elev=90, azim=0)
# fig.add_axes(ax)
#
# x=np.arange(0,10)     # 生成数据
# y=np.arange(0,10)
# X,Y=np.meshgrid(x,y)   #生成x,y轴数据
#
#                  #生成z值
# sc = ax.plot_surface(X, Y, data,rstride=1,cstride=1,cmap=plt.get_cmap('coolwarm'))
#
# plt.colorbar(sc,shrink=0.5)
# plt.savefig("2d-7am-graph.png",dpi=600,format='png')
# plt.show()

#ablation study
# coding:utf8


# plt.rcParams['font.sas-serig']=['simfang']
# plt.rcParams['axes.unicode_minus']=False

# n_groups = 2
#
# MAE_A = (1.75, 2.37)
#
# MAE_C = (1.81, 2.34)
#
# MAE_D = (1.63, 2.13)
#
# MAE_T = (1.68, 2.30)
#
# MAE = (1.58, 2.02)
#
# rmse_A = (3.06, 3.81)
#
# rmse_C = (2.97, 3.80)
#
# rmse_D = (2.93, 3.71)
#
# rmse_T = (2.93, 3.75)
#
# rmse = (2.88, 3.67)
# fig, ax = plt.subplots()
#
# index = np.arange(n_groups)
# bar_width = 0.125
#
# opacity = 0.8
# error_config = {'ecolor': '0.3'}
#
# rects1 = ax.bar(index, rmse_A, bar_width,
#                 alpha=opacity, color='#3174A1',
#                 error_kw=error_config,
#                 label='CSTNet-A')
#
# rects2 = ax.bar(index + bar_width, rmse_C, bar_width,
#                 alpha=opacity, color='#E1812B',
#                 error_kw=error_config,
#                 label='CSTNet-C')
#
# rects3 = ax.bar(index + bar_width + bar_width, rmse_D, bar_width,
#                 alpha=opacity, color='#3A923B',
#                 error_kw=error_config,
#                 label='CSTNet-D')
# rects4 = ax.bar(index + bar_width + bar_width + bar_width, rmse_T, bar_width,
#                 alpha=opacity, color='#BEA63E',
#                 error_kw=error_config,
#                 label='CSTNet-T')
# rects4 = ax.bar(index + bar_width + bar_width + bar_width+ bar_width, rmse, bar_width,
#                 alpha=opacity, color='#BF3D3D',
#                 error_kw=error_config,
#                 label='CSTNet')
# # plt.xlabel(u'Mae',fontsize=14)
# ax.set_xticks(index + 2 * bar_width)
# ax.set_xticklabels(('Changsha', 'Shenzhen'))
# ax.legend()
# ax.legend(loc='upper left', fontsize='small', ncol=2)
#
# fig.tight_layout()
# plt.savefig("ablation-rmse.png",dpi=600,format='png')
# plt.show()

#region heat
#time_line
# ground_truth = [31, 18, 11, 7, 5, 4, 57, 73, 108, 136, 100, 92, 103, 110, 101, 96, 102, 108, 155, 156, 133, 102, 109, 63,
#                  39, 22, 13, 8, 2, 3, 1, 60, 62, 107, 155, 117, 104, 130, 107, 92, 114, 106, 138, 136, 133, 122, 105, 60,
#                  37, 16, 15, 4, 4, 2, 5, 53, 82, 126, 160, 121, 110, 123, 114, 121, 135, 116, 153, 148, 123, 116, 121, 80,
#                  21, 22, 14, 10, 7, 4, 5, 46, 70, 91, 128, 103, 107, 114, 113, 103, 107, 101, 130, 119, 118, 97, 106, 64,
#                  34, 32, 23, 8, 12, 7, 6, 60, 61, 88, 127, 127, 114, 102, 115, 102, 114, 96, 128, 119, 121, 104, 101, 70,
#                  34, 16, 15, 4, 4, 12, 5, 53, 82, 126, 160, 111, 120, 129, 124, 135, 162, 153, 148, 123, 116, 121, 111, 70,
#                  28, 20, 17, 9, 5, 4, 2, 33, 72, 116, 140, 111, 120, 129, 124, 111, 117, 105, 123, 129, 132, 106, 101, 90]
#
# hier = [37, 15, 11, 6, 6, 4, 53, 79, 106, 130, 104, 92, 106, 111, 94, 106, 100, 114, 150, 152, 128, 99, 109, 57,
#              30, 28, 17, 8, 10, 7, 4, 59, 65, 104, 155, 113, 102, 129, 108, 98, 119, 100, 140, 144, 136, 123, 107, 66,
#              40, 18, 18, 7, 3, 4, 7, 56, 81, 129, 163, 122, 111, 120, 115, 123, 134, 114, 153, 147, 127, 106, 126, 82,
#              24, 22, 13, 8, 7, 4, 8, 46, 70, 91, 125, 103, 107, 114, 117, 113, 107, 101, 131, 118, 109, 111, 104, 64,
#              31, 32, 24, 10, 12, 7, 6, 60, 61, 78, 115, 126, 113, 106, 117, 112, 115, 93, 126, 116, 120, 106, 97, 70,
#             32, 22, 13, 5, 2, 8, 4, 57, 88, 116, 160, 123, 113, 131, 128, 137, 153, 161, 143, 130, 112, 123, 112, 68,
#              30, 19, 20, 10, 8, 4, 1, 37, 80, 119, 150, 121, 110, 129, 111, 95, 110, 117, 133, 142, 132, 116, 103, 88]
#
# hier_f = [28, 12, 13, 8, 12, 7, 6, 30, 51, 68, 85, 127, 124, 122, 105, 122, 124, 106, 138, 144, 111, 94, 88, 50,
#             11, 22, 17, 18,  37, 34, 45, 66, 50, 91, 100, 103, 107, 104, 123, 123, 127, 111, 131, 129, 118, 107, 86, 58,
#             30, 19, 20, 10, 8, 4, 1, 37, 80, 119, 130, 111, 110, 119, 111, 95, 110, 117, 133, 142, 132, 116, 103, 88,
#              24, 22, 13, 8, 12, 7, 6, 60, 91, 148, 125, 117, 114, 122, 115, 102, 114, 96, 158, 149, 131, 124, 111, 70,
#              30, 28, 17, 8, 10, 7, 4, 59, 65, 104, 155, 113, 102, 129, 108, 98, 119, 100, 140, 144, 136, 123, 107, 66,
#             32, 22, 13, 5, 2, 8, 4, 57, 88, 106, 137, 121, 110, 120, 114, 129, 139, 130, 135, 133, 117, 123, 111, 70,
#              37, 15, 11, 6, 6, 4, 53, 79, 106, 130, 104, 92, 106, 111, 94, 106, 100, 114, 150, 152, 128, 99, 109, 57]
# import matplotlib.pyplot as plt
# from datetime import datetime
#
# x = range(168)
#
# plt.ylabel('Heat Values')  # y轴标题
# plt.plot(x, ground_truth, label='Ground Truth', color = '#BC0826')
# plt.plot(x, hier, label = 'HierSTNet', color = '#3D6D9E')
# plt.plot(x, hier_f, label = 'HierSTNet-f', color = 'slateblue')
# plt.plot(x, ground_truth, label='Ground Truth', color = '#BC0826')
#
# plt.xticks([])
# plt.legend(loc='upper left', fontsize='small')  # 设置折线名称
# plt.savefig("gwn.png",dpi=600,format='png')
# plt.show()  # 显示折线图

#comparison
# ground = [31, 18, 11, 7, 5, 4, 57, 73, 108, 136, 100, 92, 103, 110, 101, 96, 102, 108, 155, 156, 133, 102, 109, 63,
#           37, 16, 15, 4, 4, 2, 5, 53, 82, 126, 160, 121, 110, 123, 114, 121, 135, 116, 153, 148, 123, 116, 121, 80]
# hier = [37, 15, 11, 6, 6, 4, 53, 79, 106, 130, 104, 92, 106, 111, 94, 106, 100, 114, 150, 152, 128, 99, 109, 57,
#         40, 18, 18, 7, 3, 4, 7, 56, 81, 129, 163, 122, 111, 120, 115, 123, 134, 114, 153, 147, 127, 106, 126, 82]
# ast = [39, 22, 13, 8, 2, 3, 1, 60, 62, 107, 155, 117, 104, 130, 107, 92, 114, 106, 138, 136, 133, 122, 105, 60,
#        37, 15, 11, 6, 6, 4, 53, 79, 106, 130, 104, 92, 106, 111, 94, 106, 100, 114, 150, 152, 128, 99, 109, 57]
# gwn = [37, 16, 15, 4, 4, 2, 5, 53, 101, 139, 109, 91, 101, 114, 101, 93, 102, 116, 153, 148, 123, 116, 121, 80,
#        30, 19, 20, 10, 8, 4, 1, 37, 80, 119, 150, 121, 110, 129, 111, 95, 110, 117, 133, 142, 132, 116, 103, 88]
# mdl = [30, 28, 17, 8, 10, 7, 4, 59, 65, 104, 155, 113, 102, 129, 108, 98, 119, 100, 120, 124, 136, 123, 107, 66,
#        30, 28, 17, 8, 10, 7, 4, 59, 65, 104, 155, 113, 102, 129, 108, 98, 119, 100, 140, 144, 136, 123, 107, 66]
#
# x = range(48)
# # color_60 = ['coral',"slategray",'teal']'#BC0826' 'mediumslateblue' 'cornflowerblue'
# # color_30 = ['green','peru','brown']'#6B6B6B' 'indianred'
# plt.ylabel('Heat Values')  # y轴标题
# plt.plot(x, ground, label='Ground Truth', color = '#6B6B6B', marker='s',markersize=4)
# plt.plot(x, gwn, label = 'GWNET', color = 'lightsalmon', marker='s',markersize=4)
# plt.xticks(np.arange(0,50,12))
# plt.legend(loc='upper left', fontsize='small')  # 设置折线名称
# plt.savefig("gwn.png",dpi=600,format='png')
# plt.show()  # 显示折线图

#multi-step
n_groups = 4

rmse_Hier = (8.06, 8.81, 9.34, 13.86)

rmse_GW = (8.47, 9.26, 10.24, 14.36)

rmse_MD = (8.63, 9.43, 10.64, 14.62)

rmse_Step = (8.89, 9.83, 11.74, 16.32)

rmse_AS = (9.01, 10.02, 11.84, 16.41)

rmse_Conv = (9.78, 10.63, 12.04, 16.93)
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.125

opacity = 0.8
error_config = {'ecolor': '0.3'}
plt.ylabel('RMSE')
rects1 = ax.bar(index, rmse_Hier, bar_width,
                alpha=opacity, color='#3174A1',
                error_kw=error_config,
                label='HierSTNet')

rects2 = ax.bar(index + bar_width, rmse_GW, bar_width,
                alpha=opacity, color='#E1812B',
                error_kw=error_config,
                label='GWNET')

rects3 = ax.bar(index + bar_width + bar_width, rmse_MD, bar_width,
                alpha=opacity, color='#3A923B',
                error_kw=error_config,
                label='MDL')
rects4 = ax.bar(index + bar_width + bar_width + bar_width, rmse_Step, bar_width,
                alpha=opacity, color='#BEA63E',
                error_kw=error_config,
                label='StepDeep')
rects4 = ax.bar(index + bar_width + bar_width + bar_width+ bar_width, rmse_AS, bar_width,
                alpha=opacity, color='#BF3D3D',
                error_kw=error_config,
                label='ASTGCN')

rects5 = ax.bar(index + bar_width + bar_width + bar_width + bar_width + bar_width, rmse_Conv, bar_width,
                alpha=opacity, color='#8865A7',
                error_kw=error_config,
                label='ConvLSTM')
# plt.xlabel(u'Mae',fontsize=14)
ax.set_xticks(index + 2 * bar_width)
ax.set_xticklabels(('1', '3', '6','12'))
ax.legend()
ax.legend(loc='upper left', fontsize='small', ncol=2)

fig.tight_layout()
plt.savefig("multistep-rmse.png",dpi=600,format='png')
plt.show()