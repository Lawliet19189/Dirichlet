from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
import streamlit as st
import torch
import torchaudio
import soundfile
import os

def signal_plots(signal, fs, lw=0.1, fmax=6e3):
    N = len(signal)
    delta_t = 1 / fs
    times = np.arange(0, N) / fs
    signalf = fft(signal)
    freqs = np.linspace(0.0, 1.0 / (2.0 * delta_t), N // 2)

    fig, axs = plt.subplots(1, 3, figsize=(30, 5))
    axs[0].plot(times, signal, linewidth=lw)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Time Domain Representation')

    axs[1].plot(freqs, 2.0 / N * np.abs(signalf[0:N // 2]), linewidth=0.4)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Frequency Domain Representation')
    axs[1].set_xlim([0, fmax])

    f_bins, t_bins, Sxx = spectrogram(signal, fs=fs,
                                      window='hanning', nperseg=80,
                                      noverlap=None, detrend=False,
                                      scaling='spectrum')

    axs[2].pcolormesh(t_bins, f_bins, 20 * np.log10(Sxx + 1e-100), cmap='magma', shading="auto")
    axs[2].set_ylabel('Frequency (Hz)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylim([0, 2e2])
    plt.show()
    st.pyplot(fig)

def np_audio(np_array, samplerate=44100):
    soundfile.write('temp.wav', np_array, samplerate, 'PCM_24')
    st.audio('temp.wav', format='audio/wav')
    os.remove('temp.wav')

@st.cache(ttl=3600, max_entries=10)
def get_sample_data():
    test_samples = torchaudio.datasets.LIBRISPEECH("Dataset/test", url="test-clean", download=True)
    test_loader = torch.utils.data.DataLoader(test_samples,
                                          batch_size=1,
                                          shuffle=False)
    return test_samples