import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d
from torch.autograd import Variable

from utils import ST_block
from utils import multi_gcn,gcn  # gwnet


from utils import ST_BLOCK_4  # Gated-STGCN
from utils import ST_BLOCK_5  # GRCN
from utils import ST_BLOCK_6  # OTSGGCN
from utils import ST_BLOCK_7 #astgcn

class OCST_net(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, supports_a=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, days=24, order=2,dims=10):

        super(OCST_net, self).__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        self.supports = supports
        self.supports_a = supports_a

        self.supports_len = 0
        self.supports_len_a = 0

        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_a += len(supports_a)

        if supports is None:
            self.supports = []
            self.supports_a = []

        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_a = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h_a, a=0, b=0.0001)
        self.supports_len += 1
        self.supports_len_a += 1

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(dims, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec1_a = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a = nn.Parameter(torch.randn(dims, num_nodes).to(device), requires_grad=True).to(device)

        self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)

        self.block1 = ST_block(dilation_channels, dilation_channels, num_nodes, length - 6, 3, dropout, num_nodes,
                               self.supports_len)
        self.block2 = ST_block(dilation_channels, dilation_channels, num_nodes, length - 9, 2, dropout, num_nodes,
                               self.supports_len)

        self.skip_conv1 = Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.gconv_a2p = gcn(dilation_channels, residual_channels, dropout, support_len=1, order=order)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, inputs, ind):

        xo = inputs

        x = self.start_conv(xo[:, [0]])

        x_a = self.start_conv_a(xo[:, [1]])

        skip = 0

        if self.supports is not None:
            # nodes
            A = F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d = 1 / (torch.sum(A, -1))
            D = torch.diag_embed(d)
            A = torch.matmul(D, A)

            new_supports = self.supports + [A]

            A_a = F.relu(torch.mm(self.nodevec1_a, self.nodevec2_a))
            d_c = 1 / (torch.sum(A_a, -1))
            D_c = torch.diag_embed(d_c)
            A_a = torch.matmul(D_c, A_a)

            new_supports_a = self.supports_a + [A_a]

        adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)
        new_supports_a2p = [adp_a2p]

        # network
        #1.
        x = self.block1(x, new_supports)
        x_a = self.block1(x_a, new_supports_a)

        x_a2p = self.gconv_a2p(x_a, new_supports_a2p)
        x = x_a2p + x

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        #2.
        x = self.block2(x, new_supports)
        x_a = self.block2(x_a, new_supports_a)

        x_a2p = self.gconv_a2p(x_a, new_supports_a2p)
        x = x_a2p + x

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
        # x = x + residual[:, :, :, -x.size(3):]
        # x = self.normal(x)
        # x_a = self.normal_a(x_a)

        # output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, A, A

class OCST_net_tcn(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, supports_a=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 end_channels=512, days=24, kernel_size=2, blocks=4, layers=2,dims=10):
        super(OCST_net_tcn, self).__init__()

        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        self.bn_a = nn.ModuleList()

        self.gconv_a = nn.ModuleList()

        self.gconv_a2p = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))


        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        self.supports = supports
        self.supports_a = supports_a

        receptive_field = 1

        self.supports_len = 1
        self.supports_len_a = 1

        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_a += len(supports_a)

        if supports is None:
            self.supports = []
            self.supports_a = []

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(dims, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec1_a = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec2_a = nn.Parameter(torch.randn(dims, num_nodes).to(device), requires_grad=True).to(device)

        self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.bn_a.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                self.gconv_a.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

                self.gconv_a2p.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, inputs, ind):

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs

        x = self.start_conv(xo[:, [0]])

        x_a = self.start_conv_a(xo[:, [1]])  # torch.Size([32, 32, 307, 13])
        skip = 0

        if self.supports is not None:
            # nodes
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

            adp_a = F.softmax(F.relu(torch.mm(self.nodevec1_a, self.nodevec2_a)), dim=1)
            new_supports_a = self.supports + [adp_a]

        adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)
        new_supports_a2p = [adp_a2p]

        for i in range(self.blocks * self.layers):
            # tcn for primary part
            residual = x

            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # tcn for auxiliary part
            residual_a = x_a
            filter_a = self.filter_convs_a[i](residual_a)
            filter_a = torch.tanh(filter_a)

            gate_a = self.gate_convs_a[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)  # b f n t
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, new_supports)  # torch.Size([32, 32, 307, 12]) [torch.Size([32, 307, 307])]
            x_a = self.gconv_a[i](x_a, new_supports_a)
            # multi-faceted fusion module
            x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
            # SUM（）
            x = x_a2p + x

            # residual and normalization
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)
            x_a = self.bn_a[i](x_a)

            # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, adp, adp_a2p

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        input = self.bn_1(input[:,[0]])
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, adp, adp

class Gated_STGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(Gated_STGCN, self).__init__()
        tem_size = length
        self.block1 = ST_BLOCK_4(in_dim, dilation_channels, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_4(dilation_channels, dilation_channels, num_nodes, tem_size, K, Kt)
        self.block3 = ST_BLOCK_4(dilation_channels, dilation_channels, num_nodes, tem_size, K, Kt)

        self.conv1 = Conv2d(dilation_channels, 12, kernel_size=(1, tem_size), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.supports = supports
        self.bn = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        x = self.bn(input[:,[0]])
        adj = self.supports[0]

        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = self.block3(x, adj)
        x = self.conv1(x)  # b,12,n,1
        return x, adj, adj

class OGCRNN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(OGCRNN, self).__init__()

        self.block1 = ST_BLOCK_5(in_dim, dilation_channels, num_nodes, length, K, Kt)

        self.tem_size = length

        self.conv1 = Conv2d(dilation_channels, out_dim, kernel_size=(1, length),
                            stride=(1, 1), bias=True)
        self.supports = supports
        self.bn = BatchNorm2d(in_dim, affine=False)
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(self, input):
        x = self.bn(input[:,[0]])
        A = self.h + self.supports[0]
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A = F.dropout(A, 0.5)
        x = self.block1(x, A)

        x = self.conv1(x)
        return x, A, A

    # OTSGGCN

class OTSGGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(OTSGGCN, self).__init__()
        tem_size = length
        self.num_nodes = num_nodes
        self.block1 = ST_BLOCK_6(in_dim, dilation_channels, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_6(dilation_channels, dilation_channels, num_nodes, tem_size, K, Kt)

        self.conv1 = Conv2d(dilation_channels, 12, kernel_size=(1, tem_size), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.supports = supports
        self.bn = BatchNorm2d(in_dim, affine=False)
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(self, input):
        x = self.bn(input[:,[0]])
        A = self.h + self.supports[0]
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A1 = torch.eye(self.num_nodes).cuda() - A
        A1 = F.dropout(A1, 0.5)
        x = self.block1(x, A1)
        x = self.block2(x, A1)

        x = self.conv1(x)  # b,12,n,1
        return x, A1, A1

    # gwnet

class GRU(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(GRU, self).__init__()
        self.gru = nn.GRU(in_dim, dilation_channels, batch_first=True)  # b*n,l,c
        self.c_out = dilation_channels
        tem_size = length
        self.tem_size = tem_size

        self.conv1 = Conv2d(dilation_channels, 12, kernel_size=(1, tem_size),
                            stride=(1, 1), bias=True)

    def forward(self, input):
        x = input[:,[0]]
        shape = x.shape
        h = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda()
        hidden = h

        x = x.permute(0, 2, 3, 1).contiguous().view(shape[0] * shape[2], shape[3], shape[1])
        x, hidden = self.gru(x, hidden)
        x = x.view(shape[0], shape[2], shape[3], self.c_out).permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)  # b,c,n,l
        return x, hidden[0], hidden[0]

class LSTM(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_dim, dilation_channels, batch_first=True)  # b*n,l,c
        self.c_out = dilation_channels
        tem_size = length
        self.tem_size = tem_size

        self.conv1 = Conv2d(dilation_channels, 12, kernel_size=(1, tem_size), padding=(0, 0),
                            stride=(1, 1), bias=True)

    def forward(self, input):
        x = input[:,[0]]
        shape = x.shape
        h = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda()
        c = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda()
        hidden = (h, c)

        x = x.permute(0, 2, 3, 1).contiguous().view(shape[0] * shape[2], shape[3], shape[1])
        x, hidden = self.lstm(x, hidden)
        x = x.view(shape[0], shape[2], shape[3], self.c_out).permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)  # b,c,n,l
        return x, hidden[0], hidden[0]

class ASTGCN_Recent(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(ASTGCN_Recent, self).__init__()
        self.block1 = ST_BLOCK_7(in_dim, dilation_channels, num_nodes, length, K, Kt)
        self.block2 = ST_BLOCK_7(dilation_channels, dilation_channels, num_nodes, length, K, Kt)
        self.final_conv = Conv2d(length, 12, kernel_size=(1, dilation_channels), padding=(0, 0),
                                 stride=(1, 1), bias=True)
        self.supports = supports
        self.bn = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        x = self.bn(input[:,[0]])
        adj = self.supports[0]
        x, _, _ = self.block1(x, adj)
        x, d_adj, t_adj = self.block2(x, adj)
        x = x.permute(0, 3, 2, 1)
        x = self.final_conv(x)  # b,12,n,1
        return x, d_adj, t_adj

class H_GCN_wh(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(H_GCN_wh, self).__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        self.supports_len = 0

        if supports is not None:
            self.supports_len += len(supports)

        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

        self.supports_len += 1

        Kt1 = 2
        self.block1 = ST_block(dilation_channels, dilation_channels, num_nodes, length - 6, 3, dropout, num_nodes,
                              self.supports_len)
        self.block2 = ST_block(dilation_channels, dilation_channels, num_nodes, length - 9, 2, dropout, num_nodes,
                              self.supports_len)

        self.skip_conv1 = Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.bn = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        x = self.bn(input[:,[0]])
        shape = x.shape

        if self.supports is not None:
            # nodes
            # A=A+self.supports[0]
            A = F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d = 1 / (torch.sum(A, -1))
            D = torch.diag_embed(d)
            A = torch.matmul(D, A)

            new_supports = self.supports + [A]

        skip = 0
        x = self.start_conv(x)

        # 1
        x = self.block1(x, new_supports)

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # 2
        x = self.block2(x, new_supports)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, x, A
