import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG2DTokenizer(nn.Module):
    def __init__(self, chans, samples, projection_dim):
        super(EEG2DTokenizer, self).__init__()
        self.chans = chans
        self.samples = samples
        self.projection_dim = projection_dim

        self.temporal_embedding = nn.Embedding(samples, projection_dim)
        self.spatial_embedding = nn.Embedding(chans, projection_dim)
        self.input_projection = nn.Linear(1, projection_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.squeeze(1)                     # (batch, chans, samples)
        x = x.permute(0, 2, 1)               # (batch, samples, chans)
        x = x.reshape(batch_size, -1, 1)     # (batch, samples * chans, 1)

        t_pos = torch.arange(self.samples, device=x.device).repeat_interleave(self.chans)
        c_pos = torch.arange(self.chans, device=x.device).repeat(self.samples)

        t_emb = self.temporal_embedding(t_pos)
        c_emb = self.spatial_embedding(c_pos)
        pos_emb = t_emb + c_emb

        x = self.input_projection(x)         # (batch, sequence_len, projection_dim)
        x = x + pos_emb.unsqueeze(0)         # aggiunta positional embedding
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        layers = []
        in_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TransformerLayer(nn.Module):
    def __init__(self, projection_dim, num_heads, transformer_units, dropout_rate=0.7):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, dropout=0.5, batch_first=True)
        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.mlp = MLP(projection_dim, transformer_units, dropout_rate)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(x_norm, x_norm, x_norm, need_weights=True)
        x = x + attn_output
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = x + mlp_output
        return x, attn_weights

class ConTraNet2D(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, projection_dim, transformer_layers, num_heads,
                 transformer_units, mlp_head_units, training=True):
        super(ConTraNet2D, self).__init__()
        self.return_attention = not training

        self.tokenizer = EEG2DTokenizer(Chans, Samples, projection_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(projection_dim, num_heads, transformer_units, dropout_rate=0.7)
            for _ in range(transformer_layers)
        ])

        seq_len = Chans * Samples
        self.representation_norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.representation_dropout = nn.Dropout(0.5)

        flattened_dim = seq_len * projection_dim
        self.mlp_head = MLP(flattened_dim, mlp_head_units, dropout_rate=0.7)

        if isinstance(mlp_head_units, (list, tuple)):
            mlp_head_out = mlp_head_units[-1]
        else:
            mlp_head_out = mlp_head_units

        self.classifier_dense = nn.Linear(mlp_head_out, nb_classes)

    def forward(self, x):
        x = self.tokenizer(x)  # (batch, seq_len, projection_dim)

        attn_scores = None
        for transformer in self.transformer_layers:
            x, attn = transformer(x)
            attn_scores = attn

        representation = self.representation_norm(x)
        representation = representation.flatten(start_dim=1)
        representation = self.representation_dropout(representation)

        features = self.mlp_head(representation)
        logits = self.classifier_dense(features)
        logits = F.elu(logits)
        output = F.softmax(logits, dim=1)

        if self.return_attention:
            return output, attn_scores
        else:
            return output
