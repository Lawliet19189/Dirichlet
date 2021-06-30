# For Dataset Download
import torchaudio
import torch

def download_dataset()
    data_samples = torchaudio.datasets.LIBRISPEECH("Datasets/LibriSpeech", url="train-clean-100", download=True)
    dev_samples = torchaudio.datasets.LIBRISPEECH("Datasets/LibriSpeech", url="dev-clean", download=True)
    assert len(data_samples)>0


