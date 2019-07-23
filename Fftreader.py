import numpy as np
import os
from matplotlib import pyplot as plt
def Dftreader(dftdir):
    dft = np.loadtxt(dftdir, comments=['p', '[', '#', 'l','n'])
    return dft

def Fftreader(fftdir):
    fft = np.loadtxt(fftdir, comments=['p', '[', '#', 'l','n'])
    return fft

def mDrawer(m,name,prefftdir):
    m=m.T
    normhist = np.sort(m.flatten())
    thresh = normhist[-1 * int(len(normhist) * 0.5)]
    #print(thresh)
    outIndexs = [(i,j) for i in range(m.shape[0]) for j in range (m.shape[1]) if m[i][j] > thresh]
    m=np.zeros_like(m)
    for index in outIndexs:
        m[index]=1
    plt.figure()
    plt.imshow(m.T, cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title(name)
    plt.ylabel('time index')
    plt.xlabel('frequency bin index')
    plt.savefig(prefftdir +name+ '_hotmap.png', dpi=150)
def mcolorDrawer(m,name,prefftdir):
    m=m.T
    plt.figure()
    plt.imshow(m.T, cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title(name)
    plt.ylabel('time index')
    plt.xlabel('frequency bin index')
    plt.savefig(prefftdir +'/'+name+ '_hotmap.png', dpi=150)

def FftDrawer(fft,name,prefftdir):
    fft=fft.T
    plt.figure()
    plt.imshow(fft.T ** 0.125, cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title(name)
    plt.ylabel('time index')
    plt.xlabel('frequency bin index')
    plt.savefig(prefftdir +name+ '_hotmap.png', dpi=150)

def getdft(matrix):
    matrix = matrix.T
    norm = np.zeros((matrix.shape[0], int(matrix.shape[1] / 2)))
    for i in range(0, matrix.shape[0]):
        frame = matrix[i]
        real = frame[0::2]
        imag = frame[1::2]
        comp2 = np.square(real) + np.square(imag)
        norm[i] = np.sqrt(comp2)
    return norm/32768
if __name__ == '__main__':
    #wholeDir='./2019_4_17/aboutM/1/'
    #folder='noiseinput'
    #files = os.listdir(os.path.join(wholeDir, folder))
    low = Fftreader('./2019_4_17/1/noiseinput/lowfft.txt')
    low = getdft(low)
    high = Fftreader('./2019_4_17/1/noiseinput/highfft.txt')
    high = getdft(high)
    '''
    for filename in files:
        noisefftdir = os.path.join(wholeDir, folder, filename)
        fft=Fftreader(noisefftdir)
        FftDrawer(fft-original, filename[:-4], './2019_4_17/')
        #print(noisefftdir)
    '''
    FftDrawer(low, 'low','./2019_4_17/')
    FftDrawer(high, 'high', './2019_4_17/')
    #FftDrawer(dft, '2','./2019_4_15/')
