import librosa
import numpy as np
from scipy.fftpack import dct
from librosa.feature import mfcc
from librosa.filters import mel

# If you want to see the spectrogram picture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_spectrogram(spec, note,file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """

    # 计算功率谱
    spectrum = spectrum*spectrum/fft_len
    num_frames = spectrum.shape[0]
    banks = np.zeros((int(fft_len / 2 + 1), num_filter))
    feats = np.zeros((num_frames, num_filter))

    # 线性频率转梅尔频率
    def hertz_to_mels(f):
        return 1127. * np.log(1. + f / 700.)

    # Mel频率转线性频率
    def mel_to_hertz(mel):
        return 700. * (np.exp(mel / 1127.) - 1.)

    fmin = 0
    fmax = fs / 2

    # Mel刻度上等间隔划分，第一个滤波器的起始频率+最后一个滤波器的截止频，总共 num_filt + 2 个点
    grid_mels = np.linspace(hertz_to_mels(fmin), hertz_to_mels(fmax), num_filter + 2, True)

    # Mel等间隔刻度转线性频率刻度
    grid_hertz = mel_to_hertz(grid_mels)

    # 连续频率值转离散频率点 f = k*fs/N
    grid_indices = (grid_hertz * fft_len / fs).astype(int)

    # 三角滤波器
    for i in range(0, banks.shape[1]):
        left = grid_indices[i]
        middle = grid_indices[i + 1]
        right = grid_indices[i + 2]
        banks[left:middle, i] = np.linspace(0., 1., middle - left, False)
        banks[middle:right, i] = np.linspace(1., 0., right - middle, False)

    # fbank
    feats = np.dot(spectrum, banks)

    # 取对数
    feats = 20*np.log10(feats)
    return feats

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    feats = np.zeros((fbank.shape[0],num_mfcc))
    # Type II DCT
    feats = dct(fbank, norm='ortho')[:, 1:num_mfcc+1]
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)  
    plot_spectrogram(fbank_feats.T, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()
