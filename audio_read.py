import soundfile
import resampy
import numpy as np
import os
import math
#from dp2_model_tf import deepspeech_tf_model
from read_vocab import vocab_dict

def read_audo(file):
    
    samples, sample_rate = soundfile.read(file, dtype='float32')
    # resample
    if sample_rate != 16000:
        if samples.ndim >= 2:
            samples = np.mean(samples, 1)
        samples = resampy.resample(samples, sample_rate, 16000, filter='kaiser_best')
        sample_rate = 16000

    # normalize
    rms_db = 10 * np.log10(np.mean(samples ** 2))
    gain = -20 - rms_db
    samples *= 10.**(gain / 20.)

    stride_ms = 10
    window_ms = 20
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)

    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)

    return fft

if __name__ == '__main__':
    train_path = os.path.dirname(os.path.realpath(__file__))
    dict = vocab_dict('vocab.txt')

    fft = read_audo('trends.flac')
    np.save("./weight/second_fft.npy",fft)
    print(fft.shape)