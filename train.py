from stock_data import get_batch
import torch
import torch.nn as nn
from tqdm import tqdm
import streamlit as st
from sklearn.model_selection import ParameterGrid
from model import TransformerModel

# パラメータグリッドの定義
param_grid = {
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'epochs': [100, 200, 300, 500],
    'lr': [0.001, 0.01, 0.05],
    'batch_size': [32, 64, 128]
}

def train_model(model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num):
    
    train_loss_list = []
    valid_loss_list = []    

    model.train()
    total_loss_train = 0.0 # ロスの合計
    for batch, i in enumerate(range(0, len(train_data), batch_size)):
        data, targets = get_batch(
            train_data, i, batch_size, observation_period_num)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()
    scheduler.step()
    total_loss_train = total_loss_train/len(train_data)

    model.eval()
    total_loss_valid = 0.0
    for i in range(0, len(valid_data), batch_size):
        data, targets = get_batch(
            valid_data, i, batch_size, observation_period_num)
        output = model(data)
        total_loss_valid += len(data[0]) * \
            criterion(output, targets).cpu().item()
    total_loss_valid = total_loss_valid/len(valid_data)

    return model, total_loss_valid

# モデルのトレーニングとバリデーションのためのグリッドサーチ関数
def grid_search(train_data, valid_data, observation_period_num):
    best_model = None
    best_loss = float('inf')
    best_params = None

    # パラメータグリッドからすべての組み合わせを試行
    for params in ParameterGrid(param_grid):
        st.write(f"Trying params: {params}")

        # モデルの定義
        model = TransformerModel(feature_size=90, num_layers=params['num_layers'], dropout=params['dropout']).to('cpu')

        # モデルのトレーニング
        model, train_loss_list, valid_loss_list = train_model(
            train_data, valid_data, model,
            epochs=params['epochs'], 
            lr=params['lr'], 
            batch_size=params['batch_size'], 
            observation_period_num=observation_period_num
        )

        # 最終エポックのバリデーション損失で最適なモデルを評価
        final_valid_loss = valid_loss_list[-1]
        if final_valid_loss < best_loss:
            best_loss = final_valid_loss
            best_model = model
            best_params = params

        st.write(f"Params {params} achieved validation loss: {final_valid_loss}")

    return best_model, best_params
