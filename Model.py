import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    #d_model: 特徴量の次元数, max_len: 最大の,位置情報を付与するモデル
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    #入力データに位置情報を付与, x: 入力データ, 出力: 位置情報付与後のデータ
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])

#transformerモデルの定義
class TransformerModel(nn.Module):
    #feature_size: 特徴量の次元数, num_layers: トランスフォーマーの層数, dropout: ドロップアウト率
    def __init__(self, feature_size=90, num_layers=1, dropout=0.2):
        super().__init__()        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")        
        self.pos_encoder = PositionalEncoding(d_model=feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10,
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)

    #重みの初期化
    def init_weights(self):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform(-0.1, 0.1)

    #sz: マスクのサイズ, 出力: マスク
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask        

    #src: 入力データ, 出力: 予測値
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = self.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
    
# EarlyStopクラス
class EarlyStop:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    #val_loss: 検証データの損失, model: モデル
    def __call__(self, val_loss, model):
        score = (-val_loss)
        if self.best_score is None:
            self.best_score = score
        elif score == self.patience:
            self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
