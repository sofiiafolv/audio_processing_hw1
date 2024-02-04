import numpy as np
import scipy 
from typing import Any
from stft import stft
import  matplotlib.pyplot as plt
import librosa

def hz_to_mel(frequencies):
    frequencies = np.asanyarray(frequencies)
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)

def mel_to_hz(mels):
    mels = np.asanyarray(mels)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def mel(
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    
    if fmax is None:
        fmax = float(sr) / 2
    
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, n_fft), dtype=np.float32)

    fftfreqs = np.fft.rfftfreq(n=n_fft * 2 - 1, d = 1.0/sr)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights

def melspectrogram(
    y: np.ndarray = None,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int = None,
    window: str = "hann",
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    S, n_fft = stft(y, win_length, hop_length, sr, window=window)
    S = np.abs(S) ** power
    mel_filter = mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    result = np.einsum("ft,mf->mt", S, mel_filter, optimize=True)
    return result
