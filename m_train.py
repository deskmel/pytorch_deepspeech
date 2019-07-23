import argparse
import json
import os
import random
import time
import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex.fp16_utils import FP16_Optimizer
from apex.parallel import DistributedDataParallel
from tqdm import tqdm
from warpctc_pytorch import CTCLoss
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from visdom import  Visdom
from model import DeepSpeech, supported_rnns
from utils import convert_model_to_half, reduce_tensor, check_loss
from torch.autograd import Variable
import datetime
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=1, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=1, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing', default=True)
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='m_models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='m_models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
#------------------------------------------------
#我们需要调参的对象
parser.add_argument('--continue-from', default='models/deepspeech_final.pth', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/libri_train_manifest_only_one.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/libri_val_manifest.csv')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')  # 70
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model', default=True)
parser.add_argument('--lr', '--learning-rate', default=1, type=float, help='initial learning rate')
parser.add_argument('--lamda', default=10000, type=float, help='value of lamda')
parser.add_argument('--beta', default =1000, type=float, help='value of beta')
parser.add_argument('--m', default=1, type=float, help='initailize the value of m')
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

def get_current_time():
    return datetime.datetime.now().strftime("%m-%d-%H-%M")

def to_np(x):
    return x.cpu().numpy()

def self_loss(logit,logit_star,m,lamda ,beta):
    loss_0 = beta * torch.sum(torch.pow(logit - logit_star,2)) / (logit.size()[0]*logit.size()[1]*logit.size()[2])
    loss_1 = lamda * torch.sum( torch.log(m)) / ( m.size()[0] * m.size()[1])
    loss = loss_0 - loss_1
    return loss, loss_0, loss_1



class VisdomLinePlotter(object):

    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def clear(self):
        self.viz.close(None,env=self.env)
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
class M_Noise_Deepspeech(torch.nn.Module):
    def __init__(self,package,input_size):
        super(M_Noise_Deepspeech, self).__init__()
        self.small_const = 1e-6
        self.T = input_size[3]
        self.K = input_size[2]
        self.m = torch.nn.Parameter(torch.Tensor(np.array([args.m]*self.T*self.K,dtype=np.float32).reshape(self.K,self.T,1)).cuda(),requires_grad=True)
        self.range1 = torch.Tensor(np.array(list(range(self.K)) * self.K * self.T).reshape((self.K, self.T, self.K))).cuda()
        self.range2 = torch.Tensor(np.array(list(range(self.K)) * self.K * self.T).reshape((self.K, self.T, self.K)).transpose()).cuda()
        self.relu = torch.nn.ReLU()
        self.deepspeech_net = DeepSpeech.load_model_package(package)

    def forward(self,input,input_length):
        #np.savetxt('./debug_result/self_m.txt', self.m.cpu().detach().numpy()[:,:,0])
        abs_m = torch.abs(self.m)
        #np.savetxt('./debug_result/abs_m.txt', abs_m.cpu().detach().numpy()[:,:,0])
        m_tile = abs_m.repeat([1, 1, self.K])
        #np.savetxt('./debug_result/m_tile.txt', m_tile.cpu().detach().numpy()[:,:,0])
        out = self.relu(m_tile - torch.abs(self.range1 - self.range2)) / (torch.pow(m_tile, 2))
        #np.savetxt('./debug_result/out.txt', out.cpu().detach().numpy()[:,:,0])
        blar = (torch.mul(out, (m_tile > 1).float()) + torch.mul((m_tile <= 1).float(), (self.range1 == self.range2).float())).cuda()
        #np.savetxt('./debug_result/blar.txt', blar.cpu().detach().numpy()[:,:,0])
        norm_index = torch.sum(blar, dim=2).reshape([self.K, self.T, 1]).repeat([1, 1, self.K])
        #np.savetxt('./debug_result/norm_index.txt', norm_index.cpu().detach().numpy()[:,:,0])
        blar = blar / norm_index
        #np.savetxt('./debug_result/input.txt', input.cpu().detach().numpy()[0, 0, :, :])
        inputtile = input[0, 0, :, :].reshape([self.K, self.T, 1]).repeat(1, 1, self.K).cuda()
        #np.savetxt('./debug_result/inputtile.txt', inputtile.cpu().detach().numpy()[:,:,0])
        inputs = torch.sum(torch.mul(blar, inputtile), dim=0).transpose(0,1).reshape([1, 1, self.K, self.T])
        #np.savetxt('./debug_result/inputs.txt', inputs.cpu().detach().numpy()[0,0,:,:])
        return self.deepspeech_net(inputs,input_length)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parser.parse_args()

    #not important
    print(args.m)


    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None
    if main_proc and args.visdom:
        visdom_logger = VisdomLinePlotter(env_name='m_trainer')
        visdom_logger.clear()
    if main_proc and args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None




    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)


        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], \
                                                     package['wer_results']
        print("Loading label from %s" % args.labels_path)
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=None,
                          noise_prob=1,
                          noise_levels=(0.5, 0.8))
    else:
        print("must load model!!!")
        exit()



    # decoder = GreedyDecoder(labels)
    original_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=False, augment=args.augment)
    original_train_sampler = BucketingSampler(original_dataset, batch_size=args.batch_size)

    original_train_loader = AudioDataLoader(original_dataset,
                                   num_workers=args.num_workers, batch_sampler=original_train_sampler)
    for i, (data) in enumerate(original_train_loader, start=start_iter): original_inputs,_,_,_ = data

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)


    for i, (data) in enumerate(train_loader, start=start_iter):
        # 获取初始输入
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        size = inputs.size()
        print(size)
        # 初始化模型
        model = M_Noise_Deepspeech(package, size)
        for para in model.deepspeech_net.parameters():
            para.requires_grad=False
        model = model.to(device)
        # 获取初始输出
        out_star = model.deepspeech_net(inputs,input_sizes)[0]
        out_star = out_star.transpose(0, 1)  # TxNxH
        float_out_star = out_star.float()
        break


    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)
    print(model)

    #criterion = self_loss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ##
    lamda = args.lamda

    result_dir = './result/'+get_current_time()+'/'
    saved_loss = np.ndarray((args.epochs), dtype=np.float32)
    saved_m_mean = np.ndarray((args.epochs), dtype=np.float32)
    saved_m_std = np.ndarray((args.epochs), dtype=np.float32)


    for epoch in range(0, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()


        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)
            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            float_out = out.float()  # ensure float32 for loss
            loss,loss_0,loss_1  = self_loss(float_out,float_out_star,torch.abs(model.m),args.lamda,args.beta)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_value = loss.item()
            m_mean = torch.mean(torch.abs(model.m)).item()
            m_std = torch.std(torch.abs(model.m)).item()

            # Check to ensure valid loss was calculated
            #valid_loss, error = check_loss(loss, loss_value)

            optimizer.zero_grad()
            # compute gradient
            loss.backward(retain_graph=True)
            #optimizer.backward(loss)
            optimizer.step()
            avg_loss += loss_value

            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} (MSE Loss {mseloss:})\t'
                      'm avg: {mean:.4f} std: {std:.4f}'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses,mseloss = loss_0,mean=m_mean,std=m_std))
                visdom_logger.plot('loss', 'loss_0', 'Class Loss', epoch, loss_0.item())
                visdom_logger.plot('loss', 'loss_1', 'Class Loss', epoch, -loss_1.item())
                visdom_logger.plot('loss', 'train', 'Class Loss', epoch, loss_value)
                visdom_logger.plot('m_mean', 'train', 'M', epoch, m_mean)
                visdom_logger.plot('m_std', 'train', 'M', epoch, m_std)
            saved_loss[epoch]= loss_value
            saved_m_mean[epoch] =m_mean
            saved_m_std[epoch] = m_std
            del loss, out, float_out
        losses.reset()

    if not (os.path.exists(result_dir)):
        os.mkdir(result_dir)
    print('Finish training\n'
          'saved m to {}\n'
          'saved fft to {}\n'
          'saved loss to {}\n'.format(
        result_dir+'m.txt',result_dir+'fft.txt',result_dir+'loss.txt'
    ))
    np.savetxt('logit.txt',float_out_star.cpu().detach().numpy().flatten())
    np.savetxt(result_dir+'m.txt',model.m.cpu().detach().numpy().reshape(model.K,model.T))
    np.savetxt(result_dir+'fft.txt',to_np(original_inputs).reshape(model.K,model.T))
    np.savetxt(result_dir+'loss.txt',saved_loss)
    np.savetxt(result_dir + 'm_mean.txt', saved_m_mean)
    np.savetxt(result_dir + 'm_std.txt', saved_m_std)
