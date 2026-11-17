import io, json, numpy as np, soundfile as sf, streamlit as st
from scipy.signal import resample_poly
from ai_edge_litert.interpreter import Interpreter
from features import log_spectrogram, crop_to_word

LABELS = json.load(open("labels.json"))
NORM = json.load(open("norm.json"))
MEAN, STD = np.array(NORM["mean"]), np.array(NORM["std"])
THRESHOLD = 0.60          # below this confidence the app says "Not recognised"

@st.cache_resource
def load_model():
    it = Interpreter(model_path="kws.tflite"); it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]

st.title("Keyword Spotter")
st.write("Say one word: " + ", ".join(LABELS))
audio = st.audio_input("Record a word")
if audio:
    wave, sr = sf.read(io.BytesIO(audio.getvalue()), dtype="float32")
    if wave.ndim > 1: wave = wave.mean(axis=1)                 # stereo -> mono
    if sr != 16000: wave = resample_poly(wave, 16000, sr)      # e.g. 48 kHz -> 16 kHz
    if np.max(np.abs(wave)) < 1e-4:
        st.warning("No sound detected. Please try again.")
    else:
        x = (log_spectrogram(crop_to_word(wave)) - MEAN) / STD
        it, inp, out = load_model()
        it.set_tensor(inp["index"], x[None].astype("float32")); it.invoke()
        probs = it.get_tensor(out["index"])[0]
        top = int(np.argmax(probs)); conf = float(probs[top])
        if conf >= THRESHOLD:
            st.success(f"**{LABELS[top]}**  ({conf:.0%} confident)")
        else:
            st.warning(f"Not recognised (best guess: {LABELS[top]}, {conf:.0%})")
        st.bar_chart(dict(zip(LABELS, probs.tolist())))
