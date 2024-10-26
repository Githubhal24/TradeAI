import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from stock_data2 import search_stock_code, load_data, get_batch, normalization
from model import TransformerModel, EarlyStop
from train_sample import train_model
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import os

st.title("Trade AI")

url = "https://finance.yahoo.co.jp/"
st.markdown("参考：  [Yahoo!ファイナンス](%s)" % url)

ticker_or_code = st.text_input("ティッカーシンボルまたは株式コードを入力してください(例　AMZN, GOOGL, 9613)", placeholder="AMZN")
if ticker_or_code:
    stock_code = search_stock_code(ticker_or_code)
    st.session_state.stock_code = stock_code
    if stock_code:
        st.write(f"検索されたティッカーシンボル/株式コード: {stock_code}")
    else:
        st.error(f"シンボルやコード '{ticker_or_code}' から株式コードが見つかりませんでした")

start_date = st.date_input("データ取得開始日")
end_date = st.date_input("データ取得終了日")
st.session_state.start_date = start_date
st.session_state.end_date = end_date

observation_period_num = 90 # 観測期間
predict_period_num = 30 # 予測期間
train_rate = 0.7 # 訓練データの割合
lr = 0.0001 # 学習率
epochs = 500    # エポック数
batch_size = 32 # バッチサイズ
feature_size = 90 # 特徴量の次元数
patience = 3 # 早期終了のためのパラメータ

predict_period_num = st.slider(label='予測期間', min_value=0, max_value=30)
st.write(f"予測期間: {predict_period_num}")

st.write("オプション")
api_use = st.checkbox("Gemini APIを使用する(ニュースデータの感情分析)")
st.session_state.api = api_use

if st.button('データ取得'):
    df = yf.download(stock_code, start=start_date, end=end_date)
    st.write(df)
    st.write("株価の推移(終値)")
    data = df['Close']
    st.line_chart(data)
    data_norm, data_mean, data_std = normalization(data)
    # セッションにデータを保存(予測時に使用)
    st.session_state.data = data
    st.session_state.data_norm = data_norm
    st.session_state.data_mean = data_mean
    st.session_state.data_std = data_std

if st.button("予測"):
    # セッションからデータを取得できれば予測処理を実行
    if st.session_state.data_norm is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data, valid_data = load_data(data_norm=st.session_state.data_norm,
                                            stock_code=st.session_state.stock_code,
                                            start_date=st.session_state.start_date,
                                            end_date=st.session_state.end_date,
                                            device=device,
                                            api_use=st.session_state.api)
        
        # train_data, valid_data = load_data(data_norm=st.session_state.data_norm,
        #                                     observation_period_num=observation_period_num,
        #                                     predict_period_num=predict_period_num, 
        #                                     train_rate=train_rate, 
        #                                     device=device)

        model_path = f'tradeai_model_{stock_code}.pth'
        model = TransformerModel().to(device)
        #モデルが存在する場合はモデルをロード
        if os.path.exists(model_path):
            #model.load_state_dict(torch.load(model_path), weights_only=True)
            model.load_state_dict(torch.load(model_path))
        #モデルが存在しない場合はモデルを訓練
        else:
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
            earlystopping = EarlyStop(patience)

            progress_text = "Train"
            my_bar = st.progress(0, text=progress_text)

            # モデルの訓練
            for epoch in range(epochs):
                model, total_loss_valid = train_model(model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num)
                earlystopping(total_loss_valid, model)
                if earlystopping.early_stop:
                    break
                my_bar.progress((epoch + 1) / epochs)

            #訓練したモデルを保存
            torch.save(model.state_dict(), model_path)
            
        if predict_period_num == 0:# 精度評価用
            model.eval()
            result = torch.Tensor(0) # 予測結果
            actual = torch.Tensor(0) # 実際の値
            output = None  # 予測の最後の値を保持するための変数

            with torch.no_grad():
                for i in range(0, len(valid_data) - 1):
                    data, target = get_batch(valid_data, i, 1, observation_period_num)
                    output = model(data)
                    result = torch.cat((result, output[-1].view(-1).cpu()), 0)
                    actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)

            # 予測グラフ表示
            st.header("予測結果")

            # 正規化解除（逆正規化）
            df = yf.download(stock_code, start=start_date, end=end_date)
            mean_price = df['Close'].mean()
            std_price = df['Close'].std()
            actual = actual * std_price + mean_price  # 逆正規化
            result = result * std_price + mean_price  # 逆正規化

            # valid_dataの期間に対応する日付リストを作成
            date_range = pd.date_range(end=end_date, periods=len(actual))

            # 予測グラフ表示
            plt.figure(figsize=(10, 5))
            plt.grid(color='b', linestyle=':', linewidth=0.3)
            plt.xticks(rotation=90)
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.plot(date_range, actual, color='blue', alpha=0.7, label='Real')  # 実際の値
            plt.plot(date_range, result, color='red', linewidth=1.0, linestyle='--', label='Predict')  # 予測値
            plt.legend()
            st.pyplot(plt)

            st.header("検証結果")
            rmse = root_mean_squared_error(actual, result)
            mae = mean_absolute_error(actual, result)
            r2 = r2_score(actual, result) 

            score_set = [[rmse, mae, r2]]
            df = pd.DataFrame(
                data=score_set,
                columns=['RMSE', 'MAE', 'R²スコア']
            )  
            st.write(df)        

        # 予測期間がある場合
        else:
            model.eval()
            with torch.no_grad(): # 勾配計算をしない
                # 予測期間のデータを取得
                future_input = st.session_state.data_norm[-30:].values.reshape(-1, 1)
                # 予測
                result = model(torch.FloatTensor(future_input).to(device))[-1].view(-1)
                result = result * st.session_state.data_std + st.session_state.data_mean

            #old_data_length は過去のデータの長さ
            old_data_length = 30
            show_data = st.session_state.data[-old_data_length:].values
            #show_resultをグラフ表示すると横ばいになってしまう
            show_result = result[-predict_period_num:].cpu().numpy()

            #日付の定義
            show_index_tmp = st.session_state.data[-old_data_length:].index.values
            show_index_1 = [str(i)[:10] for i in show_index_tmp]
            show_index_2 = ["after " + str(i)  for i in range(1, predict_period_num + 1)]

            # グラフ表示
            plt.figure(figsize=(10, 5))
            plt.grid(color='b', linestyle=':', linewidth=0.3)
            plt.xticks(rotation=90)
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.plot(show_index_1, show_data, color='blue', label="Stock")
            plt.plot(show_index_2, show_result, "--", color='red', label="Predict")
            plt.legend()
            st.pyplot(plt)

    else:
        st.error("データがありません。先にデータを取得してください。")

st.write("© 2024 Trade AI")
