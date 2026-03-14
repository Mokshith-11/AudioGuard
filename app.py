import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import time
import tempfile
from pathlib import Path

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AudioGuard — Deepfake Audio Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:         #0a0c10;
    --surface:    #111318;
    --border:     #1e2130;
    --accent:     #00e5a0;
    --accent-dim: #00e5a020;
    --danger:     #ff4d6d;
    --danger-dim: #ff4d6d20;
    --muted:      #6b7280;
    --text:       #e8eaf0;
    --mono:       'Space Mono', monospace;
    --sans:       'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: var(--mono);
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.6rem;
    color: #fff;
}
.hero h1 span { color: var(--accent); }
.hero p {
    font-size: 1.05rem;
    color: var(--muted);
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}

/* Stat pills */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.8rem;
}
.stat {
    text-align: center;
}
.stat-num {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    display: block;
}
.stat-label {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Upload zone */
.upload-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    display: block;
}

/* Result cards */
.result-real {
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
}
.result-fake {
    background: var(--danger-dim);
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
}
.result-verdict {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: block;
    margin-bottom: 0.4rem;
}
.verdict-real { color: var(--accent); }
.verdict-fake { color: var(--danger); }
.result-sub {
    font-size: 0.9rem;
    color: var(--muted);
    margin: 0;
}
.conf-bar-wrap {
    background: var(--border);
    border-radius: 2px;
    height: 6px;
    margin: 1.2rem 0 0.4rem;
    overflow: hidden;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
}
.bar-real { background: var(--accent); }
.bar-fake { background: var(--danger); }
.conf-label {
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--text);
}

/* Feature row */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2rem 0;
}
.feat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
}
.feat-icon {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}
.feat-val {
    font-size: 1.5rem;
    font-weight: 600;
    color: #fff;
    display: block;
}
.feat-desc {
    font-size: 12px;
    color: var(--muted);
    margin-top: 2px;
}

/* Section headers */
.section-head {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Info box */
.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.6;
    margin: 1rem 0;
}

/* Streamlit overrides */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #00c98a !important;
    transform: translateY(-1px);
}
div[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Architecture ──────────────────────────────────────────────────────
# Matches exactly: CNN + BiLSTM + AudioLM-style self-attention + Siamese head

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, v)


class FakeAudioDetector(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 32)),
        )
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=128 * 4,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        # AudioLM-style self-attention
        self.attention = SelfAttention(512)
        self.norm = nn.LayerNorm(512)

        # Siamese projection head
        self.siamese_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, freq, time)
        x = self.cnn(x)                         # (B, 128, 4, 32)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)  # (B, 32, 512)
        x, _ = self.bilstm(x)                   # (B, 32, 512)
        x = self.norm(x + self.attention(x))    # residual + attention
        x = x.mean(dim=1)                       # (B, 512) global avg pool
        embed = self.siamese_proj(x)            # (B, 128)
        logits = self.classifier(embed)         # (B, 2)
        return logits, embed


# ─── Feature Extraction ──────────────────────────────────────────────────────
# Matches your 268×T feature matrix pipeline

def extract_features(audio, sr, n_mfcc=40, n_mels=128, target_length=128):
    """
    Extract: MFCC (40) + Mel Spectrogram (128) + Chroma (12) +
             Spectral Contrast (7) + ZCR (1) → 268 × T
    """
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)

    # Align time dimension
    min_t = min(mfcc.shape[1], mel_db.shape[1], chroma.shape[1],
                contrast.shape[1], zcr.shape[1])
    features = np.vstack([
        mfcc[:, :min_t],
        mfcc_delta[:, :min_t],
        mfcc_delta2[:, :min_t],
        mel_db[:, :min_t],
        chroma[:, :min_t],
        contrast[:, :min_t],
        zcr[:, :min_t],
    ])  # (268, T)

    # Resize time to fixed length
    if features.shape[1] < target_length:
        pad = target_length - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)), mode='constant')
    else:
        features = features[:, :target_length]

    # Normalize
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features.astype(np.float32)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = FakeAudioDetector()
    model_path = "model.pth"
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location="cpu")
            # Support both raw state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            return model, True
        except Exception as e:
            st.warning(f"Could not load weights: {e}. Running in demo mode.")
            return model, False
    return model, False


# ─── Inference ───────────────────────────────────────────────────────────────
def predict(audio, sr, model):
    model.eval()
    features = extract_features(audio, sr)               # (268, 128)
    tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)  # (1,1,268,128)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = F.softmax(logits, dim=-1)[0]
    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    label = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(real_prob, fake_prob)
    return label, confidence, real_prob, fake_prob


# ─── Waveform + Mel Plot ──────────────────────────────────────────────────────
def make_plots(audio, sr, label):
    accent = "#00e5a0" if label == "REAL" else "#ff4d6d"
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), facecolor="#111318")
    for ax in axes:
        ax.set_facecolor("#111318")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2130")

    # Waveform
    times = np.linspace(0, len(audio) / sr, len(audio))
    axes[0].plot(times, audio, color=accent, linewidth=0.6, alpha=0.85)
    axes[0].set_xlabel("Time (s)", color="#6b7280", fontsize=9)
    axes[0].set_ylabel("Amplitude", color="#6b7280", fontsize=9)
    axes[0].tick_params(colors="#6b7280", labelsize=8)
    axes[0].set_title("Waveform", color="#e8eaf0", fontsize=10, pad=8)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                   y_axis="mel", ax=axes[1],
                                   cmap="magma")
    axes[1].set_title("Mel Spectrogram", color="#e8eaf0", fontsize=10, pad=8)
    axes[1].set_xlabel("Time (s)", color="#6b7280", fontsize=9)
    axes[1].set_ylabel("Hz", color="#6b7280", fontsize=9)
    axes[1].tick_params(colors="#6b7280", labelsize=8)

    plt.tight_layout(pad=1.5)
    return fig


# ─── UI ──────────────────────────────────────────────────────────────────────
model, weights_loaded = load_model()

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-badge">v1.0 · Deep Learning · Real-time</div>
  <h1>Audio<span>Guard</span></h1>
  <p>State-of-the-art deepfake audio detection powered by CNN + BiLSTM + self-attention. Upload any audio file and get an instant verdict.</p>
  <div class="stats-row">
    <div class="stat">
      <span class="stat-num">95.8%</span>
      <span class="stat-label">Accuracy</span>
    </div>
    <div class="stat">
      <span class="stat-num">268</span>
      <span class="stat-label">Features</span>
    </div>
    <div class="stat">
      <span class="stat-num">FoR</span>
      <span class="stat-label">Dataset</span>
    </div>
    <div class="stat">
      <span class="stat-num">&lt;2s</span>
      <span class="stat-label">Analysis</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not weights_loaded:
    st.markdown("""
    <div class="info-box">
    <strong style="color:#e8eaf0">Demo mode</strong> — Place your trained <code>model.pth</code> in the same folder as <code>app.py</code> to enable real predictions. Scores shown are illustrative.
    </div>
    """, unsafe_allow_html=True)

# Upload
st.markdown('<span class="section-head">Upload Audio File</span>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Supported: WAV, MP3, FLAC, OGG, M4A · Max 200 MB",
)

if uploaded:
    # Save to temp file (librosa needs a path for some formats)
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Load audio
    try:
        audio, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=30)
    except Exception as e:
        st.error(f"Could not load audio: {e}")
        os.unlink(tmp_path)
        st.stop()

    duration = len(audio) / sr
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration:.1f}s")
    col2.metric("Sample Rate", f"{sr:,} Hz")
    col3.metric("File", uploaded.name[:24])

    st.audio(uploaded)

    # Analyse button
    st.markdown('<span class="section-head">Run Analysis</span>', unsafe_allow_html=True)

    if st.button("Analyse Audio →"):
        with st.spinner("Extracting 268-dimensional features…"):
            time.sleep(0.4)  # let spinner render
            if weights_loaded:
                label, confidence, real_p, fake_p = predict(audio, sr, model)
            else:
                # Demo mode: plausible random scores
                real_p = float(np.random.uniform(0.55, 0.92))
                fake_p = 1.0 - real_p
                label = "REAL" if real_p > 0.5 else "FAKE"
                confidence = max(real_p, fake_p)

        # Result card
        st.markdown('<span class="section-head">Verdict</span>', unsafe_allow_html=True)
        card_class = "result-real" if label == "REAL" else "result-fake"
        verdict_class = "verdict-real" if label == "REAL" else "verdict-fake"
        bar_class = "bar-real" if label == "REAL" else "bar-fake"
        summary = (
            "The audio exhibits natural spectral patterns consistent with a genuine human voice."
            if label == "REAL" else
            "Anomalies detected in spectral features and temporal patterns — likely AI-synthesised."
        )
        st.markdown(f"""
        <div class="{card_class}">
          <span class="result-verdict {verdict_class}">{label}</span>
          <p class="result-sub">{summary}</p>
          <div class="conf-bar-wrap">
            <div class="conf-bar-inner {bar_class}" style="width:{confidence*100:.1f}%"></div>
          </div>
          <span class="conf-label">Confidence: {confidence*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown('<span class="section-head">Probability Breakdown</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="feat-card">
              <span class="feat-icon">Real probability</span>
              <span class="feat-val" style="color:#00e5a0">{real_p*100:.1f}%</span>
              <span class="feat-desc">Genuine human voice</span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="feat-card">
              <span class="feat-icon">Fake probability</span>
              <span class="feat-val" style="color:#ff4d6d">{fake_p*100:.1f}%</span>
              <span class="feat-desc">AI-synthesised audio</span>
            </div>
            """, unsafe_allow_html=True)

        # Visualisations
        st.markdown('<span class="section-head">Signal Analysis</span>', unsafe_allow_html=True)
        fig = make_plots(audio, sr, label)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Feature detail
        st.markdown('<span class="section-head">Feature Pipeline</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="feat-grid">
          <div class="feat-card">
            <span class="feat-icon">MFCC</span>
            <span class="feat-val">120</span>
            <span class="feat-desc">40 coeffs + Δ + ΔΔ</span>
          </div>
          <div class="feat-card">
            <span class="feat-icon">Mel Spectrogram</span>
            <span class="feat-val">128</span>
            <span class="feat-desc">128 mel bands (dB)</span>
          </div>
          <div class="feat-card">
            <span class="feat-icon">Chroma</span>
            <span class="feat-val">12</span>
            <span class="feat-desc">12 pitch classes</span>
          </div>
          <div class="feat-card">
            <span class="feat-icon">Spectral Contrast</span>
            <span class="feat-val">7</span>
            <span class="feat-desc">6 bands + valley</span>
          </div>
          <div class="feat-card">
            <span class="feat-icon">Zero-Crossing Rate</span>
            <span class="feat-val">1</span>
            <span class="feat-desc">Temporal noise proxy</span>
          </div>
          <div class="feat-card">
            <span class="feat-icon">Total Features</span>
            <span class="feat-val">268</span>
            <span class="feat-desc">Per time frame</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        os.unlink(tmp_path)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#6b7280">
      <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3">◈</div>
      <p style="font-family:'Space Mono',monospace;font-size:13px;letter-spacing:0.1em">
        DROP AN AUDIO FILE TO BEGIN ANALYSIS
      </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem;border-top:1px solid #1e2130;margin-top:4rem">
  <p style="font-family:'Space Mono',monospace;font-size:11px;color:#6b7280;letter-spacing:0.1em">
    AUDIOGUARD · CNN + BiLSTM + SELF-ATTENTION · TRAINED ON FoR DATASET
  </p>
</div>
""", unsafe_allow_html=True)
