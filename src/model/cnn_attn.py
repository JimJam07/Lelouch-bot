import torch
from torch import nn
import torch.nn.functional as F

class _AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))  # [d]

    def forward(self, x):
        # x: [B, 64, d]
        scores = torch.matmul(x, self.query)  # [B, 64]
        weights = F.softmax(scores, dim=1)  # [B, 64]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, d]
        return pooled


class CNNAttention(nn.Module):
    def __init__(self, input_dims=17, embed_dim=128, num_heads=8):
        super(CNNAttention, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dims, 64, kernel_size=3, padding=1),  # [B, input_dim, 8, 8] -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),  # [B, 64, 8, 8] -> [B, embed_dim, 8, 8]
            nn.ReLU()
        )
        self.attn_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_pool = _AttentionPooling(embed_dim=embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, X):
        X = self.conv_layers(X)
        X = X.flatten(2).transpose(1, 2)  # converts [B, embed_dim, 8, 8] -> [B, 64, embed_dim]

        attn_out, _ = self.attn_layer(X, X, X)  # Self attention

        X = self.attn_pool(attn_out)  # [B, 64, embed_dim] -> [B, embed_dim]

        # Uncomment to use simple mean
        # X = attn_out.mean(dim=1)

        out = self.fc(X)  # [B, embed_dim] -> [B, 1]

        return out  # eval output