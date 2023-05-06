import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device,supports, days=24,
                 dims=40, order=2):

        self.model = OCST_net(device, num_nodes, dropout, supports=supports, supports_a = supports,
                              in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid,
                              skip_channels=nhid * 8, end_channels=nhid * 16, days= days, order=order, dims=dims)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input, (1, 0, 0, 0))
        output,_,_ = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)#将标准化后的数据转换为原始数据
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)#梯度裁剪
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()#得到一个元素张量里面的元素值
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(),mae, mape, rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        #input = nn.functional.pad(input, (1, 0, 0, 0))
        output,_,_ = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(),mae, mape, rmse

class trainer1():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = gwnet(device, num_nodes, dropout, supports=supports,
                           in_dim=in_dim, out_dim=seq_length,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer2():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = Gated_STGCN(device, num_nodes, dropout, supports=supports,
                                 in_dim=in_dim, out_dim=seq_length,
                                 residual_channels=nhid, dilation_channels=nhid,
                                 skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer3():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = OGCRNN(device, num_nodes, dropout, supports=supports,
                            in_dim=in_dim, out_dim=seq_length,
                            residual_channels=nhid, dilation_channels=nhid,
                            skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer4():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = OTSGGCN(device, num_nodes, dropout, supports=supports,
                             in_dim=in_dim, out_dim=seq_length,
                             residual_channels=nhid, dilation_channels=nhid,
                             skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer5():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = LSTM(device, num_nodes, dropout, supports=supports,
                          in_dim=in_dim, out_dim=seq_length,
                          residual_channels=nhid, dilation_channels=nhid,
                          skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,1,num_nodes,12]

        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer6():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = GRU(device, num_nodes, dropout, supports=supports,
                         in_dim=in_dim, out_dim=seq_length,
                         residual_channels=nhid, dilation_channels=nhid,
                         skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer7():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = ASTGCN_Recent(device, num_nodes, dropout, supports=supports,
                                   in_dim=in_dim, out_dim=seq_length,
                                   residual_channels=nhid, dilation_channels=nhid,
                                   skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer8():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = H_GCN_wh(device, num_nodes, dropout, supports=supports,
                              in_dim=in_dim, out_dim=seq_length,
                              residual_channels=nhid, dilation_channels=nhid,
                              skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer9():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports,
                 days=24,
                 dims=40, order=2):
        self.model = OCST_net_tcn(device, num_nodes, dropout, supports=supports, supports_a=supports,
                              in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid,
                              end_channels=nhid * 16)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10,
                                                              threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, _, _ = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)  # 将标准化后的数据转换为原始数据
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # 梯度裁剪
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()  # 得到一个元素张量里面的元素值
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, _, _ = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

