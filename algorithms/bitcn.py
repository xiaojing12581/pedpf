""""""
"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Parameter
import numpy as np
#%% Monotonic Quantile TCN 

# This implementation of causal conv is faster than using normal conv1d module
# 这种因果conv的实现比使用普通的conv1d模块更快
class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, mode='backward', groups=1):
        super(CustomConv1d, self).__init__()
        k = np.sqrt(1 / (in_channels * kernel_size))
        weight_data = -k + 2 * k * torch.rand((out_channels, in_channels // groups, kernel_size))#初始化
        bias_data = -k + 2 * k * torch.rand((out_channels))#初始化
        self.weight = Parameter(weight_data, requires_grad=True)#转化为可以训练优化的值
        self.bias = Parameter(bias_data, requires_grad=True)  
        self.dilation = dilation#默认1，不进行膨胀卷积（有空隙的过滤）
        """
        假设卷积操作的输入通道数是in_channels,输出通道数是out_channles，分组数是groups，
        分组卷积就是把原本的整体卷积操作分成groups个小组来分别处理，其中每个分组的输入通道数是in_channles / groups,
        输出通道数是out_channles / groups，最后将所有分组的输出通道数concat，得到最终的输出通道数out_channles，
        所以在做分组卷积的时候，in_channels和out_channels需要被groups整除
        """
        self.groups = groups
        if mode == 'backward':#反向传播
            self.padding_left = padding
            self.padding_right= 0
        elif mode == 'forward':#前向传播
            self.padding_left = 0
            self.padding_right= padding            

    def forward(self, x):
        xp = F.pad(x, (self.padding_left, self.padding_right))#对x进行扩充
        return F.conv1d(xp, self.weight, self.bias, dilation=self.dilation, groups=self.groups)

class tcn_cell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, mode, groups, dropout):
        super(tcn_cell, self).__init__()
        #weight_norm将权重归一化应用于给定模块中的参数
        self.conv1 = weight_norm(CustomConv1d(in_channels, out_channels, kernel_size, padding, dilation, mode, groups))
        self.conv2 = weight_norm(CustomConv1d(out_channels, in_channels * 2, 1))
        self.drop = nn.Dropout(dropout)#防止过拟合
        
    def forward(self, x):
        h_prev, out_prev = x
        h = self.drop(F.gelu(self.conv1(h_prev)))
        h_next, out_next = self.conv2(h).chunk(2, 1)#把张量在1维度上拆分成2部分
        return (h_prev + h_next, out_prev + out_next)

class bitcn(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, Nl, kernel_size):
        super(bitcn, self).__init__()
        # Embedding layer for time series ID时序ID的嵌入层
        # nn.ModuleList是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器
        # nn.Embedding创建一个词嵌入模型：共d_emb[i, 0]个词，为每个词创建一个d_emb[i, 1]维的向量来表示该词
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        self.upscale_lag = nn.Linear(d_lag + d_emb_tot + d_cov, d_hidden)
        self.upscale_cov = nn.Linear(d_emb_tot + d_cov, d_hidden)
        self.drop_lag = nn.Dropout(dropout)
        self.drop_cov = nn.Dropout(dropout)
        # tcn
        layers_fwd = nn.ModuleList([tcn_cell(
                    d_hidden, d_hidden * 4, 
                    kernel_size, padding=(kernel_size-1)*2**i, 
                    dilation=2**i, mode='forward', 
                    groups=d_hidden, 
                    dropout=dropout) for i in range(Nl)])  
        layers_bwd = nn.ModuleList([tcn_cell(
                    d_hidden, d_hidden * 4, 
                    kernel_size, padding=(kernel_size-1)*2**i, 
                    dilation=2**i, mode='backward', 
                    groups=1, 
                    dropout=dropout) for i in range(Nl)])
        self.net_fwd = nn.Sequential(*layers_fwd)#参数可变*
        self.net_bwd = nn.Sequential(*layers_bwd)
        # Output layer
        self.loc_scale = nn.Linear(d_hidden * 2, d_output * 2)
        self.epsilon = 1e-6
        
    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):       
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])
            x_emb.append(out)
        x_emb = torch.cat(x_emb, -1)#在-1维度上拼接x_emb
        # Concatenate inputs连接输入
        dim_seq = x_lag.shape[0]#读取第0维长度
        h_cov = torch.cat((x_cov, x_emb), dim=-1)
        h_lag = torch.cat((x_lag, h_cov[:dim_seq]), dim=-1)
        h_lag = self.drop_lag(self.upscale_lag(h_lag)).permute(1,2,0)
        h_cov = self.drop_cov(self.upscale_cov(h_cov)).permute(1,2,0)
        # Apply bitcn
        _, out_cov = self.net_fwd((h_cov, 0))
        _, out_lag = self.net_bwd((h_lag, 0))
        # Output layers - location & scale of the distribution输出图层-分布的位置和比例
        out = torch.cat((out_cov[:, :, :dim_seq], out_lag), dim = 1)
        output = out[:, :, -d_outputseqlen:].permute(2, 0, 1)
        loc, scale = F.softplus(self.loc_scale(output)).chunk(2, -1)
        return loc, scale + self.epsilon
