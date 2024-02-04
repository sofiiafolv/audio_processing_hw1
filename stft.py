import numpy as np
import scipy 
import matplotlib.pyplot as plt

def stft(
    y: np.ndarray,
    window_length: int,
    hop_length: int,
    sampling_rate: int = None,
    n_fft: int = None,
    window: str = "hann",
    ) -> np.ndarray:

    window = scipy.signal.get_window(window, window_length, True)
    if n_fft is None:
        n_fft = sampling_rate // 2 + 1
    signal_length = len(y)
    num_frames = np.floor((( signal_length - window_length)) / hop_length).astype(int) + 1
    stft_output = np.zeros((n_fft, num_frames), dtype="complex")
    for frame in range(num_frames):
        y_windowed = y[frame * hop_length: frame * hop_length + window_length] * window
        sample_numbers = np.arange(window_length)
        frequencies = np.arange(n_fft).reshape((n_fft, 1))
        e = np.exp(-2j * np.pi * frequencies * sample_numbers / window_length)
        stft_output[:, frame] = e@y_windowed

    return stft_output, n_fft


if __name__== "__main__":

    sampling_rate = 128
    duration = 10
    t = np.arange(0, duration, 1/sampling_rate)
    x = np.sin(2 * np.pi * 20 * t)
    X = stft(x, 128, 8, sampling_rate, window="hann")
    Y, _ = np.abs(X)

    plt.imshow(Y)
    plt.xlabel('Frames, N')
    plt.ylabel('Frequency, N')
    plt.gca().invert_yaxis()
    plt.show()

    
