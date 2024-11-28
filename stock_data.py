import yfinance as yf
from yahooquery import Ticker
import streamlit as st
import numpy as np
import pandas as pd
import torch
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os

#株式コードを検索する
def search_stock_code(ticker_or_code):
    if ticker_or_code.isdigit():
        return f"{ticker_or_code}.T"
    else:
        ticker = Ticker(ticker_or_code)
        symbols = ticker.symbols
        if symbols:
            return symbols[0]
        return None

#正規化
def normalization(df):
    mean_list = df.mean()
    std_list = df.std()
    df = (df-mean_list)/std_list
    return df, mean_list, std_list

#データをロードする
def load_data(data_norm, stock_code, start_date, end_date, device, api_use, feature_size=120):
    df = yf.download(stock_code, start=start_date, end=end_date)
    
    if df.empty:
        st.error(f"株価データが取得できませんでした。株式コード: {stock_code}、期間: {start_date} から {end_date}")
        return None, None
    
    #Gemini APIを使用する場合
    if api_use:
        #envファイルからAPIキーを取得
        load_dotenv()
        news_api_key = os.getenv("NEWSAPI_KEY")
        gemini_api_key = os.getenv("GOOGLE_API_KEY")

        # APIキーが設定されている場合
        if news_api_key and genai.api_key:
            # ニュースデータの取得
            news_data = get_news_data(news_api_key, '日本株', start_date, end_date)
            if news_data:
                # ニュースデータの感情分析
                sentiment_data = analyze_sentiment(news_data, gemini_api_key)
                # 感情スコアを日付ごとに集計
                sentiment_data = sentiment_data.groupby('date')['sentiment_score'].mean().reset_index()
                sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
                # 感情スコアの欠損値をスムージングして補完
                sentiment_data['sentiment_score'] = sentiment_data['sentiment_score'].replace(0, np.nan).fillna(method='ffill').fillna(0)
                # 株価データに感情スコアを結合
                df = df.merge(sentiment_data, left_on='Date', right_on='date', how='left').fillna(0)  # 感情スコアがない場合は0
            else:
                st.error("ニュースデータの取得に失敗しました。")    

    normalization(df)
    observation_period_num = 120
    predict_period_num = 30
    inout_data = []

    for i in range(len(df) - observation_period_num - predict_period_num):
        # data = df.iloc[i:i + observation_period_num, 4].values
        # label = df.iloc[i + predict_period_num:i + observation_period_num + predict_period_num, 4].values
        # data = np.pad(data, (0, feature_size - len(data)))
        data = data_norm[i:i+observation_period_num]
        label = data_norm[i+predict_period_num:i +
                          observation_period_num+predict_period_num]        
        inout_data.append((data, label))

    if len(inout_data) == 0:
        st.error("十分なデータがありません。もう一度、別の日付範囲を試してください。")
        return None, None

    #データの変換
    inout_data = torch.FloatTensor(inout_data)
    train_rate = 0.8
    train_data = inout_data[:int(np.shape(inout_data)[0] * train_rate)].to(device)
    valid_data = inout_data[int(np.shape(inout_data)[0] * train_rate):].to(device)
    
    st.write("train_data size:", train_data.size())#デバッグ用
    st.write("valid_data size:", valid_data.size())#デバッグ用

    return train_data, valid_data

#データをバッチに分割する
def get_batch(source, i, batch_size, observation_period_num):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(observation_period_num, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(observation_period_num, 1))
    return input, target

#ニュースデータを取得する
def get_news_data(api_key, query, start_date, end_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,  # ニュース検索キーワード
        'from': start_date,
        'to': end_date,
        'language': 'jp',  # ニュースの言語
        'country': 'jp',  # ニュースの国
        'apiKey': api_key
    }

    response = requests.get(url, params=params)
    news_data = response.json()

    # ニュースデータの取得に失敗した場合
    if news_data.get('status') != 'ok' or  not news_data.get('articles'):
        st.error("ニュースデータの取得に失敗しました")
        return None

    return news_data['articles']

#Gemini APIを使用してニュース記事の感情分析を行う
def analyze_sentiment(news_data, api_key):
    genai.configure(api_key=api_key) 
    sentiment_scores = []
    model = genai.GenerativeModel("gemini-1.5-flash")
    # ニュースデータの感情分析
    for article in news_data:
        prompt = f"""以下のニュース記事に対して感情分析を行ってください。        
            ・タイトル: {article['title']}
            ・説明: {article['description']}
            ただし、以下の点に注意してください。
            1. 感情分析の結果は感情スコアとして0~5の範囲で返してください。
            2. 値が大きいほどポジティブな感情を表し、値が小さいほどネガティブな感情を表します。
            3. 以下の回答例の形式で示す通りに感情スコアのみを回答するように順守してください。
            【回答例】
            0
            3
            5
        """
        try:
            response = model.generate_content(prompt)
            sentiment_score = float(response.text.strip())
            if sentiment_score < 0 or sentiment_score > 5:
                sentiment_score = 0            
        except Exception:
            sentiment_score = 0
            # 感情スコアをリストに追加
        sentiment_scores.append({'date': article['publishedAt'][:10], 
                                 'sentiment_score': sentiment_score})

    return pd.DataFrame(sentiment_scores)
