import numpy as np
import torch
# content = np.load('./garage/Oil/Sz/60-5000-150/OCST_net_tcn/OCST_net_tcn_prediction_results.npz')
# ls = content.files
#
# spatial_at = content['spatial_at']
# for item in ls:
#     print(item)
#     print(content[item].shape)
#
# spatial_at = content['spatial_at']
# content_pth = torch.load('./garage/Oil/Sz/60-5000-150/gwnet/gwnet_exp1_best_3.64.pth')
# print(content_pth)



npz = np.load('D:\\Space\\Spaceself\\实验\\OCST_net\\data\\Oil\\Sz\\60-2500-319\\original_data.npz')
print(npz['data'].shape)