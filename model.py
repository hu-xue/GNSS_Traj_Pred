import torch
import torch.nn as nn
import torch.nn.functional as F

class GNSS_Transformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=2, out_dim=3, seq_len=8, pred_len=5, enc_len=3):
        super().__init__()
        self.seq_len = seq_len
        self.enc_len = enc_len
        self.pred_len = pred_len
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=256,
            batch_first=True
        )
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: [B, 8, feature_dim]
        enc_in = x[:, :self.enc_len, :]  # [B, 3, feature_dim]
        dec_in = x[:, self.enc_len-1:self.seq_len-1, :]  # [B, 5, feature_dim]，用前一帧特征做teacher forcing
        # 投影到d_model
        enc_in = self.input_proj(enc_in)
        dec_in = self.input_proj(dec_in)
        # 加位置编码
        enc_in = self.pos_encoder(enc_in)
        dec_in = self.pos_encoder(dec_in)
        # Transformer
        memory = self.transformer.encoder(enc_in)
        out = self.transformer.decoder(dec_in, memory)
        # 输出定位结果
        out = self.output_proj(out)  # [B, 5, 3]
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
