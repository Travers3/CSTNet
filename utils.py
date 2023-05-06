import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

#ocst_

#dynamic gcn
class nconv_dyn(nn.Module):
    def __init__(self):
        super(nconv_dyn, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()
class linear_dyn(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear_dyn, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv_dyn()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear_dyn(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# SA and TA
class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, device, c_in, num_nodes, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(c_in, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(c_in).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_nodes, num_nodes).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).to(device))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized
class Temporal_Attention_layer(nn.Module):
    def __init__(self, device, c_in, num_nodes, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(c_in, num_nodes).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(c_in).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_nodes, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized

#ST-block
class linear_time(nn.Module):
    def __init__(self, c_in, c_out, Kt):
        super(linear_time, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)
class multi_gcn_time(nn.Module):
    def __init__(self, c_in, c_out, Kt, dropout, support_len=3, order=2):
        super(multi_gcn_time, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear_time(c_in, c_out, Kt)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        # print(c2.shape)
        f2 = self.conv2(c2).squeeze()  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs

class ST_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2):
        super(ST_block, self).__init__()
        self.time_conv = Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_time(c_out, 2 * c_out, Kt, dropout, support_len, order)

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        self.TAT = TATT_1(c_out, num_nodes, tem_size)
        self.c_out = c_out
        # self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, x, support):
        residual = self.conv1(x)

        x = self.time_conv(x)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        x = self.multigcn(x, support)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)

        T_coef = self.TAT(x)
        T_coef = T_coef.transpose(-1, -2)
        x = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        out = self.bn(x + residual[:, :, :, -x.size(3):])
        return out

#gwnet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
class multi_gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(multi_gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

###Gated-STGCN(IJCAI)
class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(cheby_conv, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out
class ST_BLOCK_4(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_4,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out//2,c_out,K,1)
        self.conv2=Conv2d(c_out, c_out*2, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        #self.conv_2=Conv2d(c_out//2, c_out, kernel_size=(1, 1),
          #                stride=(1,1), bias=True)

    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)
        filter1,gate1=torch.split(x1,[self.c_out//2,self.c_out//2],1)
        x1=(filter1)*torch.sigmoid(gate1)
        x2=self.gcn(x1,supports)
        x2=torch.relu(x2)
        #x_input2=self.conv_2(x2)
        x3=self.conv2(x2)
        filter2,gate2=torch.split(x3,[self.c_out,self.c_out],1)
        x=(filter2+x_input1)*torch.sigmoid(gate2)
        return x

###OTSGGCN(ITSM)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(gcn_conv_hop, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv1d(c_in_new, c_out, kernel_size=1,
                            stride=1, bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode = x.shape

        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out
class ST_BLOCK_5(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_5, self).__init__()
        self.gcn_conv = gcn_conv_hop(c_out + c_in, c_out * 4, K, 1)
        self.c_out = c_out
        self.tem_size = tem_size

    def forward(self, x, supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        out = []

        for k in range(self.tem_size):
            input1 = x[:, :, :, k]
            tem1 = torch.cat((input1, h), 1)
            fea1 = self.gcn_conv(tem1, supports)
            i, j, f, o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c = c * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(j)
            new_h = torch.tanh(new_c) * (torch.sigmoid(o))
            c = new_c
            h = new_h
            out.append(new_h)
        x = torch.stack(out, -1)
        return x

## OTSGGCN
class ST_BLOCK_6(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_6, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.gcn = cheby_conv(c_out, 2 * c_out, K, 1)

        self.c_out = c_out
        self.conv_1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)

    def forward(self, x, supports):
        x_input1 = self.conv_1(x)
        x1 = self.conv1(x)
        x2 = self.gcn(x1, supports)
        filter, gate = torch.split(x2, [self.c_out, self.c_out], 1)
        x = (filter + x_input1) * torch.sigmoid(gate)
        return x

###ASTGCN_block

class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze(1)  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)  # b,c,l

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        return coefs

class SATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(SATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)

        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  # b,n,l

        c2 = seq.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        return coefs

class cheby_conv_ds(nn.Module):
    def __init__(self, c_in, c_out, K):
        super(cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj, ds):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj

        L = ds * adj
        I = ds * torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            L3 = ds * L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out

    ###ASTGCN_block

class ST_BLOCK_7(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_7, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_in, num_nodes, tem_size)
        self.SATT = SATT(c_in, num_nodes, tem_size)
        self.dynamic_gcn = cheby_conv_ds(c_in, c_out, K)
        self.K = K

        self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        T_coef = self.TATT(x)
        T_coef = T_coef.transpose(-1, -2)
        x_TAt = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        S_coef = self.SATT(x)  # B x N x N

        spatial_gcn = self.dynamic_gcn(x_TAt, supports, S_coef)
        spatial_gcn = torch.relu(spatial_gcn)
        time_conv_output = self.time_conv(spatial_gcn)
        out = self.bn(torch.relu(time_conv_output + x_input))

        return out, S_coef, T_coef

