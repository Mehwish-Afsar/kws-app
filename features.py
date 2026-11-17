import numpy as np
SR, N_FFT, HOP = 16000, 512, 160

def log_spectrogram(wave):
    wave = np.pad(wave[:SR], (0, max(0, SR - len(wave))))             # exactly 1 s
    frames = np.lib.stride_tricks.sliding_window_view(wave, N_FFT)[::HOP]
    spec = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1))    # (97, 257)
    spec = spec[:, :256].reshape(len(spec), 64, 4).mean(axis=2)        # (97, 64)
    return np.log(spec + 1e-6).astype("float32")

def crop_to_word(wave, sr=SR):
    """Return the 1-second window with the highest energy (padded if the recording is shorter)."""
    wave = np.asarray(wave, dtype="float32")
    if len(wave) <= sr:
        return np.pad(wave, (0, sr - len(wave)))
    csum = np.concatenate([[0.0], np.cumsum(wave.astype("float64") ** 2)])
    energy = csum[sr:] - csum[:-sr]                 # energy of every 1-second window
    start = int(np.argmax(energy))
    return wave[start:start + sr]
