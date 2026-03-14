"""
model.py
Complete model:
  1. CNN backbone  – local frequency pattern learning
  2. Self-Attention – AudioLM-style token attention
  3. BiLSTM         – temporal sequence modeling
  4. Contrastive head (Siamese) – distance-based classification
  5. Classifier head – binary sigmoid output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CNN Block
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Self-Attention Module  (AudioLM-style)
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    """
    Multi-head self-attention on the time dimension.
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(inplace=True),
            nn.Linear(d * 2, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.ff(x))


# ─────────────────────────────────────────────────────────────────────────────
# Feature Encoder  (shared Siamese branch)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    """
    Shared branch used for BOTH real and fake audio (Siamese).
    CNN → Self-Attention → BiLSTM → Residual → embedding
    """
    def __init__(self, n_features=268, embed_dim=256):
        super().__init__()

        # CNN backbone
        self.cnn = nn.Sequential(
            ConvBlock(1,  32, dropout=0.20),
            ConvBlock(32, 64, dropout=0.25),
            ConvBlock(64,128, dropout=0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 16))   # → (B,128,1,16)

        # Project CNN output to d_model for attention
        self.proj = nn.Linear(128, 128)

        # Self-Attention (AudioLM-style token attention)
        self.attn = SelfAttention(d_model=128, n_heads=4)

        # BiLSTM – temporal modeling
        self.bilstm = nn.LSTM(128, 128, num_layers=2,
                               batch_first=True, bidirectional=True,
                               dropout=0.3)

        # Residual block + projection to embed_dim
        self.res  = ResidualBlock(256)
        self.proj_out = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x: (B, 1, n_features, T)
        x = self.cnn(x)                          # (B,128,H,W)
        x = self.gap(x).squeeze(2)               # (B,128,16)
        x = x.permute(0, 2, 1)                   # (B,16,128)
        x = self.proj(x)                          # (B,16,128)
        x = self.attn(x)                          # (B,16,128) self-attention
        x, _ = self.bilstm(x)                    # (B,16,256) BiLSTM
        x = self.res(x[:, -1, :])                # (B,256) last step + residual
        x = self.proj_out(x)                     # (B,embed_dim)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Loss  (Siamese training)
# L(y,D) = ½·y·D² + ½·(1-y)·max(0, m-D)²
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        D = F.pairwise_distance(emb1, emb2)
        loss = 0.5 * label * D**2 + \
               0.5 * (1 - label) * F.relu(self.margin - D)**2
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────
class FakeAudioDetector(nn.Module):
    """
    Full pipeline:
      Input audio features → FeatureEncoder → Classifier head → P(fake)

    During Siamese training, two encoders share weights and
    ContrastiveLoss is applied on their embeddings.
    """
    def __init__(self, n_features=268, embed_dim=256, dropout=0.4):
        super().__init__()
        self.encoder = FeatureEncoder(n_features=n_features,
                                       embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)


def build_model(n_features=268, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return FakeAudioDetector(n_features=n_features).to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    print(f"Parameters: {count_params(m):,}")
    x = torch.randn(4, 1, 268, 251)
    print("Output:", m(x).shape)
