# EEG Movement Classification Model (Cross-Transformer with Spatio-Functional Encoding)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEmbedding(nn.Module):
    def __init__(self, num_channels, spatial_dim=2, embed_dim=32):
        super().__init__()
        self.num_channels = num_channels
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        

    def forward(self, positions):
        # positions: [channels, 3]
        dists = torch.cdist(positions[:, 0:self.spatial_dim], positions[:, 0:self.spatial_dim], p=2)  # [channels, channels]
        return dists  # [channels, embed_dim]


class RegionEmbedding(nn.Module):
    def __init__(self, num_regions=10, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_regions, embed_dim)

    def forward(self, region_ids):
        return self.embedding(region_ids)  # [channels, embed_dim]


class TemporalConvEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        # x: [batch, 1, channels, time]
        return self.conv(x)  # [batch, out_channels, channels, time]


class CrossTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.temporal_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.spatial_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)

    def forward(self, x):
        # x: [batch, channels, time, embed_dim]
        B, C, T, D = x.shape
        # Temporal attention (per canale)
        x_t = x.permute(1, 0, 2, 3).reshape(C, B*T, D)  # [channels, batch*time, D]
        x_t = self.temporal_attn(x_t)
        x_t = x_t.reshape(C, B, T, D).permute(1, 0, 2, 3)  # [batch, channels, time, D]

        # Aggregazione temporale (media)
        x_pool = x_t.mean(dim=2)  # [batch, channels, D]

        # Spatial attention (tra canali)
        x_s = x_pool.permute(1, 0, 2)  # [channels, batch, D]
        x_s = self.spatial_attn(x_s)
        x_s = x_s.permute(1, 0, 2)  # [batch, channels, D]

        return x_s  # embedding spaziale aggregato

import torch.nn.utils as utils

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.3, reg_rate=3.0):
        super().__init__()
        layers = []
        for units in hidden_units:
            dense = nn.Linear(input_dim, units)
            # Applichiamo un max-norm constraint dopo ogni forward (vedi sotto)
            layers.append(dense)
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        self.model = nn.Sequential(*layers)
        self.reg_rate = reg_rate  # per max-norm manuale

    def forward(self, x):
        return self.model(x)

class MECnet(nn.Module):
    def __init__(self, num_channels=64, num_regions=10, spatial_dim=2, embed_dim=64, num_classes=4, mlp_hidden=[128, 64], dropout=0.3):
        super().__init__()
        self.temporal_encoder = TemporalConvEncoder(in_channels=1, out_channels=embed_dim)
        self.region_embed = RegionEmbedding(num_regions=num_regions, embed_dim=embed_dim)
        self.spatial_embed = SpatialEmbedding(num_channels=num_channels, spatial_dim=spatial_dim, embed_dim=embed_dim)
        self.cross_transformer = CrossTransformerBlock(embed_dim=embed_dim, num_heads=4)
        self.mlp = MLP(embed_dim, hidden_units=mlp_hidden, dropout_rate=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, eeg, region_ids, positions):
        # eeg: [batch, channels, time]
        x = eeg.unsqueeze(1)  # [batch, 1, channels, time]
        x = self.temporal_encoder(x)  # [batch, embed_dim, channels, time]
        x = x.permute(0, 2, 3, 1)  # [batch, channels, time, embed_dim]

        # embeddings
        region_emb = self.region_embed(region_ids)  # [channels, embed_dim]
        spatial_emb = self.spatial_embed(positions)  # [channels, embed_dim]

        total_emb = region_emb + spatial_emb  # [channels, embed_dim]
        x = x + total_emb.unsqueeze(0).unsqueeze(2)  # broadcast to [batch, channels, time, embed_dim]
        x = self.cross_transformer(x)  # [batch, channels, embed_dim]
        x = x.mean(dim=1)  # Global average over channels
        x = self.mlp(x)    # [batch, mlp_hidden[-1]]
        return self.classifier(x)  # [batch, num_classes]
