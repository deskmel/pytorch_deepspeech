import argparse
from pylab import *
import os
import audio_utilities
import Fftreader
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
def getM_(mdir):
    '''
    to get M feature map
    :param txtdir: str, lastm.txt, in 2K*T shape
    :return: dimMeans(K-dim),frameMeans(T-dim)
    '''

    norm = np.abs(np.loadtxt(mdir).T)
    print(norm.shape)
    print(norm)
    print(np.max(norm))
    print(np.min(norm))
    dimMeans = np.mean(norm, axis=0)
    frameMeans = np.mean(norm, axis=1)


    plt.figure()
    plt.hist(norm.flatten(), bins=1000, normed=False)
    plt.title('hist for m')
    plt.text(-30, -10, 'μ=%.3e, δ=%.3e' % (np.mean(dimMeans), np.var(dimMeans)))
    plt.savefig(os.path.dirname(mdir) + '/m.png')

    plt.figure()
    plt.hist(dimMeans, bins=1000, normed=False)
    plt.title('hist for avg-m in each dim')
    plt.text(-30, -10, 'μ=%.3e, δ=%.3e' % (np.mean(dimMeans), np.var(dimMeans)))
    plt.savefig(os.path.dirname(mdir) + '/avgm-ondim.png')
    plt.figure()
    plt.hist(frameMeans, bins=1000, normed=False)
    plt.title('hist for avg-m in each frame')
    plt.text(-30, -10, 'μ=%.3e, δ=%.3e' % (np.mean(frameMeans), np.var(frameMeans)))
    plt.savefig(os.path.dirname(mdir) + '/avgm-onframe.png')
    return dimMeans, frameMeans, norm

def saveMusic(stft_modified_scaled,outdir):
    # Author: Brian K. Vogel
    # brian.vogel@gmail.com
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default="000000000.wav",
                        help='Input WAV file')
    parser.add_argument('--sample_rate_hz', default=16000, type=int,
                        help='Sample rate in Hz')
    parser.add_argument('--fft_size', default=320, type=int,
                        help='FFT siz')
    parser.add_argument('--iterations', default=300, type=int,
                        help='Number of iterations to run')
    parser.add_argument('--enable_filter', action='store_true',
                        help='Apply a low-pass filter')
    parser.add_argument('--enable_mel_scale', action='store_true',
                        help='Convert to mel scale and back')
    parser.add_argument('--cutoff_freq', type=int, default=1000,
                        help='If filter is enable, the low-pass cutoff frequency in Hz')
    args = parser.parse_args()

    hopsamp = 160   # stride 16000 Hz x 10 ms
    proposal_spectrogram = stft_modified_scaled
    x_reconstruct = audio_utilities.istft_for_reconstruction(proposal_spectrogram, args.fft_size, hopsamp)
    max_sample = np.max(abs(x_reconstruct))
    print(max_sample)
    if max_sample > 1.0:x_reconstruct = x_reconstruct / max_sample
    print(np.sum(x_reconstruct))
    audio_utilities.save_audio_to_flac(x_reconstruct, args.sample_rate_hz,outfile=outdir)
def m_blar(fft,m):
    T = fft.shape[1]
    K = fft.shape[0]
    range1 = np.array(list(range(K)) * K * T).reshape((K, T, K))
    range2 = np.array(list(range(K)) * K * T).reshape((K, T, K)).transpose()
    abs_m = np.abs(m).reshape((K,T,1))
    m_tile = np.tile(abs_m,(1, 1, K))
    out = np.maximum((m_tile - np.abs(range1 - range2)) / (np.square(m_tile)),0)
    blar = (np.multiply(out, (m_tile > 1)) + np.multiply((m_tile <= 1),(range1 == range2)))
    norm_index = np.tile(np.sum(blar, axis=2).reshape([K, T, 1]),(1, 1, K))
    blar = blar / norm_index
    inputtile = np.tile(fft.reshape([K, T, 1]),(1, 1, K))
    inputs = np.sum(np.multiply(blar, inputtile), axis=0).transpose().reshape([K, T])
    return inputs.T
def makeVoice(prefftdir, dimMeans, frameMeans, norm, dimThreshes=None, frameThreshes=None):
    '''

    :param prefftdir:str, preFft.txt
    :param dimMeans: got from getM()
    :param frameMeans: framehist got from getM()
    :param dimThreshes:list, default=[0.1,0.3,0.8]
    :param frameThreshes:list, default=[0.1,0.3,0.5]
    :return:
    '''

    dimhist = np.sort(dimMeans)
    framehist = np.sort(frameMeans)
    norm=np.abs(norm)
    normhist=np.sort(norm.flatten())

    normThreshes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    fft = np.loadtxt(prefftdir)
    fft = fft.T
    fftnorm = fft




    saveMusic(fftnorm, os.path.dirname(prefftdir) + '/original.flac' )
    Fftreader.FftDrawer(fftnorm, "/original", os.path.dirname(prefftdir))
    #加噪声的音乐
    noise_fft = m_blar(fft.T, norm)
    saveMusic(noise_fft, os.path.dirname(prefftdir) + '/noise_ori.flac')
    Fftreader.FftDrawer(noise_fft, "/noise_ori", os.path.dirname(prefftdir))

    for threshPercent in normThreshes:
        thresh = normhist[-1 * int(len(normhist) * threshPercent)]
        print(thresh)
        outIndexs = [(i,j) for i in range(norm.shape[0]) for j in range (norm.shape[1]) if norm[i][j] > thresh]
        outIndex_rev = [(i,j) for i in range(norm.shape[0]) for j in range(norm.shape[1]) if norm[i][j] <= thresh]
        print(len(outIndexs))
        print(len(outIndex_rev))
        #here is in T*K format
        newnorm = np.copy(fftnorm)
        newnorm_rev = np.copy(fftnorm)


        for ind in outIndexs:
            newnorm[ind] = 0
        for ind in outIndex_rev:
            newnorm_rev[ind] = 0
        #print(np.max(newnorm))
        #drawhotmap(newnorm,os.path.dirname(prefftdir)+'/%d' % (100-100 * threshPercent)+'high_dft_hotmap.png','%d'%(100-100 * threshPercent)+'high_dft_hotmap','frequency bin index','time index')
        #drawhotmap(newnorm_rev, os.path.dirname(prefftdir) + '/%d' % (100 * threshPercent) + 'low_dft_hotmap.png','%d'%(100 * threshPercent)+'low_dft_hotmap','frequency bin index','time index')
        Fftreader.FftDrawer(newnorm,'/throw%d' % (100 * threshPercent)+'high',os.path.dirname(prefftdir))
        Fftreader.FftDrawer(newnorm_rev,  '/throw%d' % (100-100 * threshPercent) + 'low',os.path.dirname(prefftdir))

        outdir = os.path.dirname(prefftdir) + '/throw%d' % (100 * threshPercent) + 'high.flac'
        outdir_rev = os.path.dirname(prefftdir)+ '/throw%d' % (100-100* threshPercent) + 'low.flac'

        saveMusic(newnorm,outdir)
        saveMusic(newnorm_rev,outdir_rev)

def drawgradgraph(wholedir):
    #print(wholedir)
    for root,dirs,files in os.walk(wholedir):
        for file in files:
            if 'txt' not in file: continue
            value= np.loadtxt(os.path.join(wholedir,file), comments=['p', '[', '#', 'l','f','n','s'])
            filename= file[:-4]
            Fftreader.mcolorDrawer(value,'/{} hot map'.format(filename),wholedir)
            print(filename,'mean:',np.mean(value),'std:',np.std(value))




def getDft(src,outdir):
    fft=np.load(src)
    print(fft.shape)
    fftnorm = np.zeros((int(fft.shape[0]/2),fft.shape[1]))
    for i in range(0, fft.shape[1]):
        frame = fft[:,i]
        real = frame[0::2]
        imag = frame[1::2]
        comp2 = np.square(real) + np.square(imag)
        fftnorm[:,i] = np.sqrt(comp2)
    fftnorm=fftnorm.T
    mean=np.mean(fftnorm)
    stdvar=np.sqrt(np.var(fftnorm))
    for i in range(fftnorm.shape[0]):
        for j in range(fftnorm.shape[1]):
            fftnorm[i][j]=(fftnorm[i][j]-mean)/stdvar

    name=os.path.basename(src)
    rawname=os.path.splitext(name)[0]+'dft'
    np.save(outdir+rawname,fftnorm)
def getgrad(dir):
    gradli=np.loadtxt(dir,comments=['p', '[', '#', 'l','x','d'])
    return gradli
def drawdirs(dirs,name,prefftdir):
    for index,dir in enumerate(dirs):
        fulldir = os.path.join(wholeDir,folder,dir)
        matrix = getgrad(fulldir)
        Fftreader.mcolorDrawer(matrix, name+str(index) , os.path.dirname(prefftdir))
def drawlinegraph(data,dir,name):
    plt.subplot()
    plt.plot(data)
    plt.title(name)
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.savefig(dir + '/'+ name + '_line_graph.png', dpi=150)


def debug():
    wholeDir = 'E:/work/code/deepSpeech/pytorch_deepspeech/result/07-17-23-08/'
    mdir= wholeDir + 'm.txt'
    prefftdir = wholeDir + "fft.txt"
    #gradfile = os.path.join(wholeDir,folder, 'output_now')
    dimMeans, frameMeans, norm = getM_(mdir)
    m = np.ndarray((norm.shape))
    m[:,:]=10
    print(m)
    fft = np.loadtxt(prefftdir)
    Fftreader.FftDrawer(fft, 'ori', os.path.dirname(prefftdir))
    print(np.mean(fft))
    print(fft.shape)
    new_fft = m_blar(fft, m)
    print(new_fft.shape)
    new_fft = new_fft.reshape((161,1501)).T
    print(np.mean(new_fft))
    print(np.sum(new_fft))
    Fftreader.FftDrawer(new_fft,'new_fft', os.path.dirname(prefftdir))
    saveMusic(new_fft, 'blar.flac')
def main():
    wholeDir = 'E:/work/code/deepSpeech/pytorch_deepspeech/result/07-23-12-47/'
    mdir= wholeDir + 'm.txt'
    prefftdir = wholeDir + "fft.txt"
    #gradfile = os.path.join(wholeDir,folder, 'output_now')
    dimMeans, frameMeans, norm = getM_(mdir)
    #drawgradgraph(gradfile)


    lossdir = wholeDir + "loss.txt"
    ys = np.loadtxt(lossdir)
    xs = np.zeros_like(ys)
    for i in range(xs.shape[0]): xs[i] = i + 1
    plt.scatter(xs, ys)
    plt.savefig(os.path.dirname(wholeDir) + '/loss.png', dpi=150)
    plt.clf()
    myloss = np.loadtxt(wholeDir + "loss.txt")
    m_loss = - np.loadtxt(wholeDir + "loss.txt")


    fig = plt.figure()
    host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
    par1 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.axis['right'].set_visible(False)
    par1.axis['right'].set_visible(True)
    par1.set_ylabel('m_entropy=∑log(m_ij))')
    par1.axis['right'].major_ticklabels.set_visible(True)
    par1.axis['right'].label.set_visible(True)
    fig.add_axes(host)
    host.set_xlabel('epoch')
    host.set_ylabel('|preOutput-TrueOutput|²')
    p1, = host.plot(xs, myloss, label='|preOutput-TrueOutput|²')
    p2, = par1.plot(xs, m_loss, label='m_entropy=∑log(m_ij)); ')
    plt.title("two parts of loss")
    host.legend()
    # 轴名称，刻度值的颜色
    host.axis['left'].label.set_color(p1.get_color())
    par1.axis['right'].label.set_color(p2.get_color())
    plt.savefig(os.path.join(os.path.dirname(wholeDir), 'loss_two_parts.png'), dpi=150)
    plt.clf()


    Fftreader.mcolorDrawer(norm, '/m hot map', os.path.dirname(prefftdir))
    #Fftreader.mcolorDrawer(norm, '/m hot map color', os.path.dirname(prefftdir))
    #print('m, mean:',np.mean(norm),'max:',np.max(norm),'min:',np.min(norm),'std:',np.std(norm,ddof=1))
    #print(np.max(norm))
    makeVoice(prefftdir, dimMeans, frameMeans, norm)






if __name__ == '__main__':
    main()
    #debug()


