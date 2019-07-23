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
from model import DeepSpeech, supported_rnns
from utils import convert_model_to_half, reduce_tensor, check_loss
from torch.autograd import Variable
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/libri_train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/libri_val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=1, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=12, type=int, help='Number of training epochs')  # 70
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='m_models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='m_models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='models/deepspeech_final.pth', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=1.0, help='Probability of noise being added per sample')   # 0.4
parser.add_argument('--noise-min', default=0.5,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.8,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)  # 0.5
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--mixed-precision', action='store_true',
                    help='Uses mixed precision to train a model (suggested with volta and above)')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale for mixed precision, ' +
                         'positive power of 2 values can improve FP16 convergence,' +
                         'however dynamic loss scaling is preferred.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling for mixed precision. If supplied, this argument supersedes ' +
                         '--static_loss_scale. Suggested to turn on for mixed precision')
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

def to_np(x):
    return x.cpu().numpy()

def self_loss(logit,logit_star,lamda,m):
    loss_0 = torch.reduce_sum(torch.square(logit - logit_star))
    loss_1 = torch.reduce_sum(lamda * torch.log(m))
    loss = loss_0 - loss_1
    return loss, loss_0, loss_1

class M_Noise_Deepspeech(torch.nn.Module):
    def __init__(self,package,input_size):
        super(M_Noise_Deepspeech, self).__init__()
        small_const = 1e-6
        self.T = input_size[3]
        self.K = input_size[2]
        self.m = torch.nn.Parameter(torch.Tensor(np.array([0.1]*self.T*self.K,dtype=np.float32).reshape(self.K,self.T,1)),requires_grad=True)
        self.m_tile = self.m.repeat([1, 1, self.K]).cuda()
        self.range1 = torch.Tensor(np.array(list(range(self.K)) * self.K * self.T).reshape((self.K, self.T, self.K))).cuda()
        self.range2 = torch.Tensor(np.array(list(range(self.K)) * self.K * self.T).reshape((self.K, self.T, self.K)).transpose()).cuda()
        self.relu = torch.nn.ReLU()
        out = self.relu(self.m_tile - torch.abs(self.range1 - self.range2)) / (torch.pow(self.m_tile, 2) + small_const)
        self.blar = torch.mul(out, (self.m_tile > 1).float()) + torch.mul((self.m_tile < 1).float(), (self.range1 == self.range2).float())
        self.deepspeech_net = DeepSpeech.load_model_package(package)

    def forward(self,input,input_length):
        inputtile = input[0, 0, :, :].reshape([self.K, self.T, 1]).repeat(1, 1, self.K).cuda()
        inputs = torch.sum(torch.mul(self.blar, inputtile), dim=2).reshape([1, 1, self.K, self.T])
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
def test_dataloader():
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.mixed_precision and not args.cuda:
        raise ValueError('If using mixed precision training, CUDA must be enabled!')
    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None
    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
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
            if main_proc and args.visdom:  # Add previous scores to visdom graph
                visdom_logger.load_previous_values(start_epoch, package)
            if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
                tensorboard_logger.load_previous_values(start_epoch, package)

        print("Loading label from %s" % args.labels_path)
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))
    else:
        print("must load model!!!")
        exit()

    # decoder = GreedyDecoder(labels)
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
if __name__ == '__main__':
    test_dataloader()

