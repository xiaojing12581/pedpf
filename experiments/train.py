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
#%% Import packages
import torch
import torch.optim as optim
import numpy as np
import os
from lib.utils import fix_seed, instantiate_model, read_table, get_emb
from lib.train import loop
from data.datasets import timeseries_dataset
import pandas as pd
torch.backends.cudnn.benchmark = False
num_cores = 2
torch.set_num_threads(2)
#%% Initialize parameters for datasets
datasets = ['uci_electricity','uci_traffic','kaggle_favorita', 'kaggle_webtraffic', 'kaggle_m5']
dim_inputseqlens = [168, 168, 90, 90, 90]
dim_outputseqlens = [24, 24, 28, 30, 28]
dim_maxseqlens = [500, 500, 150, 150, 119]
#%% Initiate experiment
dataset_id = 1
cuda = 0
seed = 0
fix_seed(seed)
num_samples_train = 1500000 if datasets[dataset_id] == 'kaggle_m5' else 500000
num_samples_validate = 30000 if datasets[dataset_id] == 'kaggle_m5' else 10000
early_stopping_patience = 5
scaling = True
epochs = 100
#%% Load data
dataset_name = datasets[dataset_id]
experiment_dir = 'experiments/'+dataset_name
dim_inputseqlen = dim_inputseqlens[dataset_id] # Input sequence length
dim_outputseqlen = dim_outputseqlens[dataset_id]  # Output prediction length
dim_maxseqlen = dim_maxseqlens[dataset_id]
# Import data
dset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen)
training_set = dset.load('train')
validation_set = dset.load('validate')
# Initialize sample sets初始化样本集
id_samples_train = torch.randperm(len(training_set))[:num_samples_train]
id_samples_validate = torch.randperm(len(validation_set))[:num_samples_validate]
#%% Algorithm parameters算法参数
file_experiments = experiment_dir + f'/experiments_{dataset_name}.csv'
table = read_table(file_experiments)
d_emb = get_emb(dataset_name)
while table[table['in_progress'] == -1].isnull()['score'].sum() > 0:
    # Read experiment table, set hyperparameters读取实验表格，设置超参数
    idx = table[table['in_progress'] == -1].isnull()['score'].idxmax()
    algorithm = table.loc[idx, 'algorithm']
    learning_rate = table.loc[idx, 'learning_rate']
    batch_size = int(table.loc[idx, 'batch_size'])
    d_hidden = int(table.loc[idx, 'd_hidden'])
    # Following paper of TransformerConv, hidden dimension is defined by covariates, lags and embedding dims
    # 根据TransformerConv的论文，隐维是由协变量、滞后和嵌入维数定义的
    if algorithm == 'transformer_conv':
        d_hidden = training_set.d_cov + training_set.d_lag + d_emb[:, 1].sum()
    kernel_size = int(table.loc[idx, 'kernel_size'])
    N = int(table.loc[idx, 'N'])
    dropout = table.loc[idx, 'dropout']
    seed = int(table.loc[idx, 'seed'])
    table.loc[idx, 'in_progress'] = cuda
    table.to_csv(file_experiments, sep=';', index=False)
    device = torch.device(cuda)
    params = eval(table.loc[idx, 'params_train'])
    # Training loop训练循环
    filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_hidden={d_hidden}_lr={learning_rate}_bs={batch_size}"
    #os.path.isdir判断某路径是否为目录、os.makedirs用于递归创建目录
    if not os.path.isdir(f"{experiment_dir}/{algorithm}"): os.makedirs(f"{experiment_dir}/{algorithm}")
    fix_seed(seed)#在使用模型进行训练的时候，通常为了保证模型的可复现性，会设置固定随机种子
    #训练集批次：训练集样本数加滑动batch_size减1 —— 训练集被分为n批次进行训练（batch_size滑倒id_samples_train的最后一行）
    n_batch_train = (len(id_samples_train) + batch_size - 1) // batch_size #地板除，只去除完之后的整数部分
    n_batch_validate = (len(id_samples_validate) + batch_size - 1) // batch_size
    if 'model' in locals(): del model
    model = instantiate_model(algorithm)(*params).to(device)   
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_train = np.zeros((epochs))
    loss_validate = np.zeros((epochs))
    loss_validate_best = 1e6
    early_stopping_counter = 0
    best_epoch = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model, loss_train[epoch], _, _, _, _ = loop(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
        _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)    
        if loss_validate[epoch] < loss_validate_best:
            torch.save({'epoch':epoch, 
                       'model_state_dict':model.state_dict(),#存放模型训练过程中需要学习的权重和偏执系数
                       'optimizer_state_dict':optimizer.state_dict()}, filename)#存放优化器中学习率、动量等系数
            df_validate.to_csv(filename + '_validate.csv')
            loss_validate_best = loss_validate[epoch]
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if (early_stopping_counter == early_stopping_patience) | (epoch == epochs - 1):
            loss_train = loss_train / n_batch_train
            loss_validate = loss_validate / n_batch_validate
            df_loss = pd.DataFrame({'Validation_loss':loss_validate,'Training_loss':loss_train})
            df_loss.to_csv(filename + '_loss.csv')
            break
    # Write new table
    table = read_table(file_experiments)
    table.loc[idx, 'score'] = loss_validate_best / n_batch_validate
    table.loc[idx, 'in_progress'] = -1
    table.to_csv(file_experiments, sep=';', index=False)
