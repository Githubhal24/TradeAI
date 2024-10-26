# README
<div id="top"></div>

## 使用技術

<!-- シールド一覧 -->
<!-- 該当するプロジェクトの中から任意のものを選ぶ-->
<p style="display: inline">
  <!-- フロントエンドのフレームワーク一覧 -->
  <img src="https://img.shields.io/badge/-streamlit-000000.svg?logo=streamlit&style=for-the-badge">
  <!-- バックエンドのフレームワーク一覧 -->
  <img src="https://img.shields.io/badge/-Pytorch-092E20.svg?logo=Pytorch&style=for-the-badge">
  <img src="https://img.shields.io/badge/-sklearn-FF2465.svg?logo=sklearn&style=for-the-badge">
  <img src="https://img.shields.io/badge/-numpy-232F3E.svg?logo=numpy&style=for-the-badge">
  <img src="https://img.shields.io/badge/-pandas-20232A?style=for-the-badge&logo=pandas&logoColor=844EBA">
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <!-- その他 -->
  <img src="https://img.shields.io/badge/-Gemini API-1488C6.svg?&style=for-the-badge">

</p>

## 目次

1. [プロジェクトについて](#プロジェクトについて)
2. [環境](#環境)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [開発環境構築](#開発環境構築)
5. [動作確認](#動作確認)
6. [環境変数の一覧](#環境変数の一覧)

<br />

<!-- プロジェクト名を記載 -->

## プロジェクト名

Trade AI

<!-- プロジェクトについて -->

## プロジェクトについて
### 概要
株価の時系列データとGemini APIを統合したデータをもとにRNNにより株価予測を行うAI

<!-- プロジェクトの概要を記載 -->
| ソースコード               | 機能概要                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| TradeAI.py                | streamlitによりブラウザ上でデータ取得・予測を行うメイン処理                                    |
| model.py                  | AIモデルの定義                                                                                 |
| stock_data.py             | yahoofinanceからデータ取得やGemini APIによるニュース記事の感情分析スコアとの統合を行う処理の定義     |
| train.py                  | モデルの学習プロセスの定義                                                                       |
| .env                      | 各種APIキーを格納するenvファイル                                                                 |


<p align="right">(<a href="#top">トップへ</a>)</p>

## 環境

<!-- 言語、フレームワーク、ミドルウェア、インフラの一覧とバージョンを記載 -->

| 言語・フレームワーク  | バージョン |
| --------------------- | ---------- |
| Python                | 3.12.7     |
| pytorch               | 4.2.1      |
| numpy                 | 1.24.3     |
| pandas                | 2.0.3      |
| python-dotenv         | 16.4.5     |
| scikit-leran          | 1.3.0      |
| streamlit             | 1.38.0     |
| tqdm                  | 4.65.0     |
|yfinance               | 0.1.56     |
|yahooquery             | 2.3.7      |

<p align="right">(<a href="#top">トップへ</a>)</p>

## ディレクトリ構成

<!-- Treeコマンドを使ってディレクトリ構成を記載 -->
.
<p>├── .env</p>
<p>├── README.md</p>
<p>├── TradeAI.py</p>
<p>├── model.py</p>
<p>├── stock_data.py</p>
<p>├── train.py</p>

<p align="right">(<a href="#top">トップへ</a>)</p>

## 動作確認
<ol type="1">
<p>
  <li>[streamlit run TradeAI.py]を実行</li>
  <li>Webブラウザページに遷移されるか確認, アクセスできたら成功</li>
</p>
</ol>  

<p align="right">(<a href="#top">トップへ</a>)</p>

## 環境変数の一覧

| 変数名                 | 役割                      | デフォルト値                         |                
| ---------------------- | ------------------------ | ----------------------------------- | 
| GOOGLE_API_KEY         | Gemini APIキー           | xxxxx(任意のAPIキーを設定してください) |
| NEWS_API_KEY           | NEWS APIキー             | xxxxx(任意のAPIキーを設定してください) |

<p align="right">(<a href="#top">トップへ</a>)</p>


