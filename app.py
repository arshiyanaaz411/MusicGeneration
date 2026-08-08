import streamlit as st
import numpy as np
import pickle
import tempfile
import os
 
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
 
st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="centered")
st.title("🎵 AI Music Generator")
st.caption("Upload MIDI files, train a small LSTM, generate new music — CodeAlpha Task 3")
 
SEQ_LENGTH = 40
 
# ---------------- Session state ----------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.notes = None
    st.session_state.pitch_names = None
    st.session_state.n_vocab = None
 
 
# ---------------- Helpers ----------------
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
 
 
# ---------------- UI ----------------
uploaded_files = st.file_uploader(
    "Upload MIDI files (.mid) — classical/jazz piano works best", 
    type=["mid", "midi"], accept_multiple_files=True
)
 
col1, col2 = st.columns(2)
with col1:
    epochs = st.slider("Training epochs (demo)", 5, 50, 20)
with col2:
    num_notes = st.slider("Notes to generate", 100, 800, 300)
 
if uploaded_files:
    st.success(f"{len(uploaded_files)} MIDI file(s) uploaded")
 
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Parsing MIDI files with music21..."):
            notes = extract_notes_from_files(uploaded_files)
 
        if len(notes) < SEQ_LENGTH + 10:
            st.error("Not enough notes extracted. Upload longer/more MIDI files.")
        else:
            with st.spinner(f"Preparing sequences ({len(notes)} notes found)..."):
                X, y, pitch_names, n_vocab = prepare_sequences(notes)
 
            with st.spinner(f"Training LSTM for {epochs} epochs..."):
                model = build_model(SEQ_LENGTH, n_vocab)
                progress = st.progress(0)
                for e in range(epochs):
                    model.fit(X, y, epochs=1, batch_size=64, verbose=0)
                    progress.progress((e + 1) / epochs)
 
            st.session_state.model = model
            st.session_state.notes = notes
            st.session_state.pitch_names = pitch_names
            st.session_state.n_vocab = n_vocab
            st.success("Training complete! Scroll down to generate music.")
 
if st.session_state.model is not None:
    st.divider()
    if st.button("🎼 Generate Music", type="primary"):
        model = st.session_state.model
        notes = st.session_state.notes
        pitch_names = st.session_state.pitch_names
        n_vocab = st.session_state.n_vocab
        note_to_int = {n: i for i, n in enumerate(pitch_names)}
        int_to_note = {i: n for i, n in enumerate(pitch_names)}
 
        start = np.random.randint(0, len(notes) - SEQ_LENGTH - 1)
        pattern = [note_to_int[n] for n in notes[start:start + SEQ_LENGTH]]
        prediction_output = []
 
        with st.spinner("Generating new sequence..."):
            for _ in range(num_notes):
                input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
                prediction = model.predict(input_seq, verbose=0)
                temperature = 0.8
                preds = np.log(prediction[0] + 1e-9) / temperature
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                index = np.random.choice(len(preds), p=preds)
                prediction_output.append(int_to_note[index])
                pattern.append(index)
                pattern = pattern[1:]
 
            midi_stream = notes_to_midi(prediction_output)
            out_path = os.path.join(tempfile.gettempdir(), "generated_output.mid")
            midi_stream.write('midi', fp=out_path)
 
        st.success(f"Generated {num_notes} notes!")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download generated_output.mid", f,
                file_name="generated_output.mid", mime="audio/midi"
            )
        st.info("Open the downloaded .mid in any MIDI player (VLC, GarageBand, MuseScore) to listen.")
else:
    st.info("Upload MIDI files and click Train to get started.")
