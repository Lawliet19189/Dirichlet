import streamlit as st
import torch
import torch.nn as nn
import sounddevice as sd
from scipy.io.wavfile import write
import util
import numpy as np
import scipy
import scipy.signal
from annotated_text import annotated_text
import matplotlib.pyplot as plt
import soundfile
import requests, os


# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl

mpl.use("agg")

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

# -- Set page config
apptitle = 'Learnable inverse TFRs'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# -- Default detector list
detectorlist = ['H1', 'L1', 'V1']

# Title the app



@st.cache(ttl=3600, max_entries=10)
def load_model(checkpoint="models/v1/checkpoint-4.pt"):
    class ComplexConv(nn.Module):
        def __init__(self, in_channel, out_channel, kernel_size, stride=80 // 4, padding=0, dilation=1, groups=1,
                     bias=True):
            super(ComplexConv, self).__init__()
            #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device("cpu")
            self.padding = padding

            self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias).to(self.device)
            self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias).to(self.device)

        def forward(self, x):
            """
            Args
              x : input of shape [batch, 2, in_channel, axis1, axis2]
            Output
              out: CNN output of shape [batch, 2, out_channel, *, *]
            """

            real = self.conv_re(x[:, 0]) - self.conv_im(x[:, 1])
            imaginary = self.conv_re(x[:, 1]) + self.conv_im(x[:, 0])
            out = torch.stack((real, imaginary), dim=1)
            out = torch.swapaxes(out.squeeze()[0], 0, 1).flatten()
            return out

    nfft = 80
    overlap = None  # uses nfft//4
    onesided = True
    window = "hamming"  # Hamming window is preferred over hann when our goal is recreate the original signal
    sampling_rate = 16000  # As per the Dataset

    model = ComplexConv(in_channel=1, out_channel=nfft // 2,
                        kernel_size=(nfft // 2 + 1 if onesided else nfft, 2 if overlap == None else 1), stride=1)
    model.load_state_dict(torch.load("models/v1/checkpoint-3.pt"))
    model.eval()
    return model


model = load_model()

st.sidebar.markdown("## Select TFR Algorithm")

# -- Set time by GPS or event
select_event = st.sidebar.selectbox('TFR',
                                    ['Short-time Fourier Transform (STFT)'])

select_type = st.sidebar.selectbox('What do you want to see?',
                                   ['Do the Magic!!!', 'Explain the Magic!!!'])
if select_event == 'Short-time Fourier Transform (STFT)':
    st.title(select_event)
    if select_type == 'Do the Magic!!!':
        st.header("STFT Demo!")
        samples = util.get_sample_data()

        option = st.selectbox(
            'Select a Sample file to Analyse',
            range(len(samples)),
            format_func=lambda x: samples[x][2])

        if st.button("Click here to Analyse!"):
            fs = 16000
            seconds = 5

            recording = samples[option][0][0].numpy()

            annotated_text(("Original Audio!", "", "#afa"),
                           height=40)
            util.np_audio(recording, samplerate=16000)

            annotated_text(("Time, Frequency and Spectogram plots using original signal", "", "#faa"), height=40)
            util.signal_plots(recording, fs, lw=0.1, fmax=6e3)

            nfft = 80
            overlap = None  # uses nfft//4
            onesided = True
            window = "hamming"  # Hamming window is preferred over hann when our goal is recreate the original signal
            sampling_rate = 16000  # As per the Dataset

            _, _, Zxx = scipy.signal.stft(recording.flatten(), nperseg=nfft, noverlap=overlap, nfft=nfft, padded=True,
                                          fs=16000,
                                          return_onesided=onesided, window=window)
            X = Zxx.reshape(1, 1, Zxx.shape[0], -1)
            X = np.stack((X.real, X.imag), axis=1)
            X = torch.Tensor(X)
            dout = model(X)

            st.markdown("""---""")

            annotated_text(("Re-created Audio!", "", "#afa"),
                           height=40)
            util.np_audio(dout.detach().numpy(), samplerate=16000)
            annotated_text(("Time, Frequency and Spectogram plots after applying learned inverse transform", "", "#faa"),
                           height=40)
            util.signal_plots(dout.detach().numpy(), fs, lw=0.1, fmax=6e3)
    else:
        st.header("Approach Walkthrough")

        annotated_text(("What?!", "", "#afa"),
                       height=40)
        st.info(
            """
            Design a deep learning Algorithm that learns to perform the operation of Inverse STFT without handcrafting the specifics of the Algorithm.
            """
        )

        st.markdown("""---""")
        st.markdown("""---""")
        annotated_text(("How?!", "", "#afa"),
                       height=40)
        st.header("Let's Break it down")
        annotated_text(("Data?!", "", "#fea"),
                       height=40)
        st.info(
            """
            For this work, we use the LibriSpeech Corpus which contains 1000 hours of 16kHz read English Speech. As a proof-of-concept, 
            We take a short training set of 100 hours speech data.
            """
        )
        st.markdown("""---""")
        annotated_text(("Model?!", "", "#fea"),
                       height=40)
        st.info(
            """
            We use CNN architecture Model. Particulary, since we are working with complex data, we use ComplexCNN as introduced in this paper https://arxiv.org/pdf/1705.09792.pdf.

            """
        )
        st.image("images/model_code.png")
        st.markdown("""---""")
        annotated_text(("Additional design decisions?!", "", "#fea"),
                       height=40)
        annotated_text(("Optimizer:", "", "#faa"), ("AdamW with no weight decay", "", "#8ef"),
                       height=40)
        annotated_text(("loss function:", "", "#faa"), ("AdamW with no weight decay", "", "#8ef"),
                       height=40)
        annotated_text(("learning_rate:", "", "#faa"),
                       ("starting lr as 2e-1 and decays with every epoch to a maximum of 5e-5", "", "#8ef"),
                       height=80)
        annotated_text(("Total Epochs:", "", "#faa"),
                       ("10", "", "#8ef"),
                       height=40)
        annotated_text(("Batch_size:", "", "#faa"),
                       ("1 with Gradient Accumulation of 64 samples", "", "#8ef"),
                       height=80)

        st.markdown("""---""")
        annotated_text(("STFT parameter configurations!", "", "#fea"),
                       height=40)
        annotated_text(("Sampling rate:", "", "#faa"), ("16kHz", "", "#8ef"),
                       height=40)
        annotated_text(("nperseg/nfft:", "", "#faa"), ("80", "", "#8ef"),
                       height=40)
        annotated_text(("Window Algorithm:", "", "#faa"),
                       ("Hamming with overlap of 40", "", "#8ef"),
                       height=80)



