import torch
import torch.nn as nn
import torch.nn.functional as F

class GNSS_Transformer(nn.Module):
    def __init__(self, feature_dim=120, d_model=128, nhead=4, num_layers=2, out_dim=3, seq_len=8, pred_len=5, enc_len=3):
        super().__init__()
        self.seq_len = seq_len
        self.enc_len = enc_len
        self.pred_len = pred_len
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_proj_gps = nn.Linear(3, d_model)
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
        self.current_weight_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim)
        )
        self.current_bias_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x, gps):
        # x: [B, 8, 120]，每时刻特征展平为[40,3]
        # gps: [B, 8, 3]
        enc_in = x[:, :self.enc_len, :]  # [B, 3, 120]
        dec_in = x[:, self.enc_len-1:self.seq_len-1, :]  # [B, 5, 120]
        # 投影到d_model
        enc_in = self.input_proj(enc_in)
        dec_in = self.input_proj(dec_in)
        # 加位置编码
        enc_in = self.pos_encoder(enc_in)
        dec_in = self.pos_encoder(dec_in)
        # Transformer
        # memory = self.transformer.encoder(enc_in)  # 已废弃，轨迹预测不再用特征分支
        # out = self.transformer.decoder(dec_in, memory)  # 已废弃，轨迹预测不再用特征分支
        # 轨迹预测输出，输入为历史gps轨迹
        gps_enc = gps[:, :self.enc_len, :]  # [B, 3, 3]
        gps_dec = gps[:, self.enc_len-1:self.seq_len-1, :]  # [B, 5, 3]
        gps_enc_proj = self.input_proj_gps(gps_enc)  # [B, 3, d_model]
        gps_dec_proj = self.input_proj_gps(gps_dec)  # [B, 5, d_model]
        gps_enc_proj = self.pos_encoder(gps_enc_proj)
        gps_dec_proj = self.pos_encoder(gps_dec_proj)
        memory_gps = self.transformer.encoder(gps_enc_proj)
        out_gps = self.transformer.decoder(gps_dec_proj, memory_gps)
        pred_traj = self.output_proj(out_gps)  # [B, 5, 3]
        # 当前时刻高精度定位（用输入序列最后一帧特征）
        last_feat = x[:, 3, :]  # [B, 120]
        last_feat_proj = self.input_proj(last_feat)  # [B, d_model]
        last_feat_proj = self.pos_encoder(last_feat_proj.unsqueeze(1)).squeeze(1)
        current_weight = self.current_weight_proj(last_feat_proj)  # [B, 3]
        current_bias = self.current_bias_proj(last_feat_proj)      # [B, 3]
        return pred_traj, current_weight, current_bias

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
