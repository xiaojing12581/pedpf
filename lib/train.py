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
import time
import numpy as np
import torch.utils.data as torchdata
from torch.distributions import StudentT
from lib.utils import calc_metrics
#%% Training loop
def loop(model, data, optimizer, batch_size, id_samples, train, metrics, scaling):
    """ Loop to calculate output of one epoch循环计算一个时期的输出"""
    # Run model in train mode if train, otherwise in evaluation mode如果是训练，则在训练模式下运行模型，否则在评估模式下运行
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device#使用和参数相同的设备
    data_subset = torchdata.Subset(data, id_samples)#获取指定一个索引序列对应的子数据集
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)#划分数据集
    # Quantile forecasting分位数预测
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    # Initiate dimensions and book-keeping variables初始化维度和簿记变量
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window, dim_lag, dim_emb, dim_cov = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window, data.d_lag, data.d_emb, data.d_cov
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')
    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0
    # Student's t distribution settingsStudent's t分布设置
    v = 3
    factor = (v / (v - 2))
    n_samples_dist = 1000
    # Datamax
    data_max = 1.0 if data.name == 'uci_traffic' else 1e12
    data_min = 0.0
    # Loop
    start = time.time()#计算代码运行时间，默认为秒
    for i, (X, Y) in enumerate(data_generator):#将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # Batch
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        # Permute to [seqlen x batch x feature] and transfer to device置换到[seqlen x batch x feature]并传送到设备
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2)
        # Fill bookkeeping variables填充簿记变量
        y_tot[:, i*batch_size:j] = Y.detach().numpy()#张量分离并转换为NumPy数组
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        # Create lags and covariate tensors创建滞后和协变量张量
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0)#对X的第0维度求平均
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY
        else:
            scaleY = torch.tensor([1.0])#直接根据数据创建Tensor：tensor([1.0])
        # Create three inputs: (i) time series index, (ii) covariates, (iii) lags创建三个输入:(一)时间序列索引，(二)协变量，(三)滞后
        X_idx = X[:, :, 0:dim_emb].long()#转为
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]
        # Send to device
        X_lag, X_cov, X_idx, Y = X_lag.to(device), X_cov.to(device), X_idx.to(device), Y.to(device)
        scaleY = scaleY.to(device)
        if train:
            # Set gradients to zero of optimizer
            optimizer.zero_grad()
            # Calculate loc and scale parameters of output distribution
            mean, variance = model(X_lag, X_cov, X_idx, dim_outputseqlen)
            scale = (variance / factor).sqrt()
            loc = mean
            distr = StudentT(v, loc, scale)
            loss_batch = -distr.log_prob(Y).mean()
            # Backward pass
            loss_batch.backward()
            # Update parameters
            optimizer.step()
        else:                        
            with torch.no_grad():
                mean_prev = X_lag[dim_inputseqlen, :, [-1]].clone().detach()
                for t in range(dim_outputseqlen):
                    X_lag[dim_inputseqlen + t, :, [-1]] = mean_prev
                    mean, variance = model(X_lag[:dim_inputseqlen + t + 1], X_cov, X_idx, t + 1)
                    mean_prev = mean[-1].clone().detach().clamp(data_min, data_max)
                # Calculate loss
                scale = (variance / factor).sqrt()
                loc = mean                
                distr = StudentT(v, loc, scale)
                loss_batch = -distr.log_prob(Y).mean()
        
        # Append loss, calculate quantiles
        loss += loss_batch.item()
        yhat = distr.sample([n_samples_dist])
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q.detach().cpu().numpy()
        
    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.2f}s')      
    yhat_tot = np.clip(yhat_tot, 0, 1e9)
    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
            
    return model, loss, yhat_tot, y_tot, x_tot, df
