"""
🎵 NeuralComposer — AI Music Generation Engine
CodeAlpha Task 3 | LSTM-based symbolic music generation with music21

Dark developer-style dashboard: live training metrics, piano-roll visualization,
terminal-style pipeline logs, downloadable MIDI output.
"""

import streamlit as st
import numpy as np
import pickle
import tempfile
import os
import time
import io

import plotly.graph_objects as go
from music21 import converter, instrument, note, chord, stream

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# ============================================================================
# PAGE CONFIG + DARK DEVELOPER THEME
# ============================================================================
st.set_page_config(
    page_title="NeuralComposer | AI Music Engine",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"]  { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

.stApp { background: radial-gradient(circle at top left, #10151c, #0a0d12 65%); }

/* glowing title */
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(90deg, #00E5A0, #00B4D8, #7B2FF7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { color: #8B949E; font-size: 0.95rem; margin-top: -8px; }

/* metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #161B22, #1c232c);
    border: 1px solid #2a3038;
    border-radius: 10px; padding: 14px 10px;
    box-shadow: 0 0 12px rgba(0,229,160,0.05);
}
div[data-testid="stMetricValue"] { color: #00E5A0; }

/* buttons */
.stButton>button {
    background: linear-gradient(90deg, #00E5A0, #00B4D8);
    color: #05130f; font-weight: 700; border: none; border-radius: 8px;
    padding: 0.6rem 1.2rem; transition: 0.2s;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 18px rgba(0,229,160,0.35); }

/* status / terminal box */
.terminal {
    background: #05070a; border: 1px solid #1f2630; border-radius: 8px;
    padding: 12px 16px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: #7CFFC4; overflow-x: auto;
}
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
    background: rgba(0,229,160,0.12); color: #00E5A0; border: 1px solid rgba(0,229,160,0.4);
}
hr { border-color: #1f2630 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="hero-title">🎹 NeuralComposer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">LSTM symbolic music generation engine &nbsp;'
            '<span class="badge">CodeAlpha · Task 3</span>&nbsp;'
            '<span class="badge">TensorFlow + music21</span></p>', unsafe_allow_html=True)
st.write("")

SEQ_LENGTH = 40

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    "model": None, "notes": None, "pitch_names": None, "n_vocab": None,
    "loss_history": [], "trained_epochs": 0, "generated_notes": None,
    "train_time": None, "n_params": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def extract_notes_from_files(files):
    notes = []
    for f in files:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            midi = converter.parse(tmp_path)
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        finally:
            os.unlink(tmp_path)
    return notes


def prepare_sequences(notes, seq_length=SEQ_LENGTH):
    pitch_names = sorted(set(notes))
    n_vocab = len(pitch_names)
    note_to_int = {n: i for i, n in enumerate(pitch_names)}

    X, y = [], []
    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        X.append([note_to_int[n] for n in seq_in])
        y.append(note_to_int[seq_out])

    X = np.reshape(X, (len(X), seq_length, 1)) / float(n_vocab)
    y = to_categorical(y, num_classes=n_vocab)
    return X, y, pitch_names, n_vocab


def build_model(seq_length, n_vocab):
    model = Sequential([
        LSTM(128, input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dense(128),
        Dropout(0.3),
        Dense(n_vocab),
        Activation('softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


class StreamlitLossLogger(Callback):
    """Feeds live loss values back into the UI during training."""
    def __init__(self, chart_placeholder, progress_bar, status_box, total_epochs):
        super().__init__()
        self.chart_placeholder = chart_placeholder
        self.progress_bar = progress_bar
        self.status_box = status_box
        self.total_epochs = total_epochs
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        self.losses.append(loss)
        st.session_state.loss_history.append(loss)
        self.progress_bar.progress((epoch + 1) / self.total_epochs)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.losses, mode="lines+markers",
            line=dict(color="#00E5A0", width=2),
            marker=dict(size=5, color="#00B4D8"),
            name="loss"
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B949E", family="JetBrains Mono"),
            xaxis_title="epoch", yaxis_title="loss",
            title=dict(text="Live Training Loss", font=dict(size=13, color="#E6EDF3")),
        )
        self.chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"loss_{epoch}")
        self.status_box.markdown(
            f'<div class="terminal">[epoch {epoch+1:03d}/{self.total_epochs}] '
            f'loss={loss:.4f} | vocab={self.model.output_shape[-1]} '
            f'| batch_size=64</div>', unsafe_allow_html=True
        )


def notes_to_midi(prediction_output):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    return stream.Stream(output_notes)


def pitch_to_midi_number(pattern):
    """Rough numeric pitch estimate for piano-roll visualization."""
    try:
        if '.' in pattern or pattern.isdigit():
            return int(pattern.split('.')[0])
        return note.Note(pattern).pitch.midi
    except Exception:
        return 60


# ============================================================================
# SIDEBAR — live dashboard
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Status")
    st.markdown('<span class="badge">1 · Ingest</span> MIDI upload via music21', unsafe_allow_html=True)
    st.markdown('<span class="badge">2 · Preprocess</span> Note/chord sequencing', unsafe_allow_html=True)
    st.markdown('<span class="badge">3 · Train</span> Stacked LSTM (128×2)', unsafe_allow_html=True)
    st.markdown('<span class="badge">4 · Generate</span> Temperature sampling', unsafe_allow_html=True)
    st.markdown('<span class="badge">5 · Export</span> music21 → .mid', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📈 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Vocab size", st.session_state.n_vocab or "—")
    c2.metric("Notes parsed", len(st.session_state.notes) if st.session_state.notes else "—")
    c3, c4 = st.columns(2)
    c3.metric("Epochs trained", st.session_state.trained_epochs or 0)
    c4.metric("Params", f"{st.session_state.n_params/1000:.1f}K" if st.session_state.n_params else "—")

    if st.session_state.train_time:
        st.caption(f"⏱ Last training run: {st.session_state.train_time:.1f}s")

    st.divider()
    st.markdown("### 🧠 Model Architecture")
    st.code(
        "Sequential([\n"
        "  LSTM(128, return_sequences=True),\n"
        "  Dropout(0.3),\n"
        "  LSTM(128),\n"
        "  Dense(128), Dropout(0.3),\n"
        "  Dense(n_vocab, activation='softmax')\n"
        "])",
        language="python"
    )
    st.caption("loss = categorical_crossentropy · optimizer = adam")


# ============================================================================
# MAIN TABS
# ============================================================================
tab_train, tab_generate, tab_viz, tab_about = st.tabs(
    ["📤  Upload & Train", "🎼  Generate", "📊  Piano Roll", "ℹ️  About"]
)

# ---------------- TAB 1: TRAIN ----------------
with tab_train:
    left, right = st.columns([2, 1])
    with left:
        uploaded_files = st.file_uploader(
            "Drop MIDI files here (.mid) — classical / jazz piano works best",
            type=["mid", "midi"], accept_multiple_files=True
        )
    with right:
        epochs = st.slider("Training epochs", 5, 60, 20)
        seq_len_display = st.metric("Sequence length", SEQ_LENGTH)

    if uploaded_files:
        st.markdown(f'<div class="terminal">$ found {len(uploaded_files)} midi file(s) staged for ingestion</div>',
                    unsafe_allow_html=True)

        if st.button("🚀  Run Training Pipeline", type="primary"):
            pipeline_status = st.status("Running pipeline...", expanded=True)
            t0 = time.time()

            with pipeline_status:
                st.write("▸ Parsing MIDI files with `music21`...")
                notes = extract_notes_from_files(uploaded_files)
                st.write(f"✓ Extracted **{len(notes)}** notes/chords")

                if len(notes) < SEQ_LENGTH + 10:
                    st.error("Not enough notes extracted — upload longer/more MIDI files.")
                    st.stop()

                st.write("▸ Encoding sequences + building vocabulary...")
                X, y, pitch_names, n_vocab = prepare_sequences(notes)
                st.write(f"✓ Vocabulary size: **{n_vocab}** unique tokens | training samples: **{len(X)}**")

                st.write("▸ Compiling LSTM model...")
                model = build_model(SEQ_LENGTH, n_vocab)
                n_params = model.count_params()
                st.write(f"✓ Model compiled — **{n_params:,}** trainable parameters")

                st.write(f"▸ Training for {epochs} epochs (live loss chart below)...")
                chart_ph = st.empty()
                progress_bar = st.progress(0)
                log_box = st.empty()
                st.session_state.loss_history = []

                logger = StreamlitLossLogger(chart_ph, progress_bar, log_box, epochs)
                model.fit(X, y, epochs=epochs, batch_size=64, verbose=0, callbacks=[logger])

                train_time = time.time() - t0
                st.write(f"✓ Training complete in **{train_time:.1f}s**")

                pipeline_status.update(label="✅ Pipeline complete", state="complete", expanded=False)

            st.session_state.update({
                "model": model, "notes": notes, "pitch_names": pitch_names,
                "n_vocab": n_vocab, "trained_epochs": epochs,
                "train_time": train_time, "n_params": n_params,
            })
            st.success("Model trained! Head to the **Generate** tab →")
            st.balloons()
    else:
        st.info("⬆️ Upload MIDI files to begin. Free datasets: piano-midi.de, Lakh MIDI Dataset.")

# ---------------- TAB 2: GENERATE ----------------
with tab_generate:
    if st.session_state.model is None:
        st.warning("⚠️ No trained model in this session yet — go train one in the first tab.")
    else:
        c1, c2, c3 = st.columns(3)
        num_notes = c1.slider("Notes to generate", 100, 800, 300)
        temperature = c2.slider("Creativity (temperature)", 0.2, 1.5, 0.8, 0.1)
        c3.metric("Vocab available", st.session_state.n_vocab)

        if st.button("🎼  Generate Composition", type="primary"):
            model = st.session_state.model
            notes = st.session_state.notes
            pitch_names = st.session_state.pitch_names
            n_vocab = st.session_state.n_vocab
            note_to_int = {n: i for i, n in enumerate(pitch_names)}
            int_to_note = {i: n for i, n in enumerate(pitch_names)}

            start = np.random.randint(0, len(notes) - SEQ_LENGTH - 1)
            pattern = [note_to_int[n] for n in notes[start:start + SEQ_LENGTH]]
            prediction_output = []

            progress = st.progress(0, text="Sampling sequence...")
            for i in range(num_notes):
                input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
                prediction = model.predict(input_seq, verbose=0)
                preds = np.log(prediction[0] + 1e-9) / temperature
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                index = np.random.choice(len(preds), p=preds)
                prediction_output.append(int_to_note[index])
                pattern.append(index)
                pattern = pattern[1:]
                if i % 10 == 0:
                    progress.progress(i / num_notes, text=f"Sampling note {i}/{num_notes}...")
            progress.progress(1.0, text="Done")

            midi_stream = notes_to_midi(prediction_output)
            out_path = os.path.join(tempfile.gettempdir(), "generated_output.mid")
            midi_stream.write('midi', fp=out_path)

            st.session_state.generated_notes = prediction_output

            st.success(f"🎉 Generated {num_notes} notes at temperature {temperature}")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download generated_output.mid", f,
                                    file_name="generated_output.mid", mime="audio/midi",
                                    type="primary")
            st.info("Open the .mid file in VLC, GarageBand, or MuseScore to listen.")

            with st.expander("🔍 Raw generated token sequence"):
                st.code(" ".join(prediction_output), language="text")

# ---------------- TAB 3: PIANO ROLL VISUALIZATION ----------------
with tab_viz:
    if st.session_state.generated_notes is None:
        st.info("Generate a composition first — the piano roll will appear here.")
    else:
        seq = st.session_state.generated_notes
        midi_nums = [pitch_to_midi_number(p) for p in seq]
        times = list(range(len(midi_nums)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=midi_nums, mode="markers+lines",
            marker=dict(size=6, color=midi_nums, colorscale="Tealgrn", showscale=True,
                        colorbar=dict(title="pitch")),
            line=dict(color="rgba(0,229,160,0.25)", width=1),
        ))
        fig.update_layout(
            height=420, title="Generated Piano Roll (pitch over time)",
            xaxis_title="step", yaxis_title="MIDI pitch",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B949E", family="JetBrains Mono"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # pitch distribution
        fig2 = go.Figure(data=[go.Histogram(x=midi_nums, marker_color="#00B4D8", nbinsx=30)])
        fig2.update_layout(
            height=280, title="Pitch Distribution",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B949E", family="JetBrains Mono"),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 4: ABOUT ----------------
with tab_about:
    st.markdown("""
#### How it works
1. **Ingest** — uploaded `.mid` files are parsed with `music21` into a flat stream of notes/chords.
2. **Preprocess** — the note stream is windowed into fixed-length sequences (`SEQ_LENGTH=40`) and
   integer-encoded against a vocabulary of unique pitches/chords.
3. **Train** — a stacked 2-layer LSTM (128 units each) with dropout learns to predict the
   next note given the previous 40, trained with categorical cross-entropy.
4. **Generate** — starting from a random seed window, the model autoregressively predicts
   one note at a time. **Temperature sampling** controls how "safe" vs "creative" the output is.
5. **Export** — the generated token sequence is converted back into a `music21` Stream and
   written out as a standard MIDI file.

#### Tech stack
`TensorFlow/Keras` · `music21` · `Streamlit` · `Plotly`
""")
    st.caption("Built for CodeAlpha AI Internship — Task 3: Music Generation with AI")
