#!/usr/bin/env python
# encoding: utf-8
'''Example codes for https://arxiv.org/abs/2310.12768'''

import csv
import os
import copy
import warnings
import imageio
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as datasets

import LDPC

warnings.filterwarnings("ignore")

epoch_len = 20
batch_size = 1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:', device)

seed = None
rng = np.random.RandomState(seed)


def scale_8bit_weight(x):
    n = x.size()[1]  # sequence length
    w = range(8, 0, -1)  # np.power(2, range(7, -1, -1))
    for i in range(n):
        x[0, i] = x[0, i] * w[i % 8]
    return x


def img2bin(x1):
    x = copy.deepcopy(x1).reshape(1, -1)  # convert to vector
    x = (x / 2 + 0.5) * 255  # inverse of regularization
    n = x.size()[1]  # sequence length
    y = torch.zeros([1, n * 8], dtype=int)
    for i in range(n):
        x2 = bin(int(min(max(x[0, i].item(), 0), 255)))[2:].zfill(8)
        # print(bin(int(x[0, i].item())),x2)
        for j in range(8):
            y[0, i * 8 + j] = int(x2[j])
    return y


def bin2img(y):
    n = int(y.size()[1] / 8)  # sequence length
    x = torch.zeros([1, n], dtype=torch.float)
    for i in range(n):
        arr = np.array(y[0, i * 8: (i + 1) * 8])
        y2 = ''.join(str(i) for i in arr)
        for j in range(8):
            x[0, i] = int(y2, 2)  # bin to digital
    x = (x / 255. - 0.5) * 2  # regularization again
    return x


def data_tf(x):
    x = x.resize((96, 96), 2)  # shape of x: (96, 96, 3)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def merge_images(sources, targets, k=10):
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row * h, row * w * 2])

    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t

    return merged.transpose(1, 2, 0) / 2 + 0.5  # inverse of regularization and change channel order


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        y = x.cpu()
    else:
        y = x
    return y.data.numpy()


def calc_exinfo(Lp1, La1, g1, n1, n, k):
    X = torch.zeros_like(Lp1)
    for gi in range(g1 - 1):
        si = gi * k
        ei = (gi + 1) * k
        sni = gi * n
        eni = gi * n + k
        X[sni:eni] = Lp1[sni:eni] - La1[si:ei]
    sni = (g1 - 1) * n
    eni = (g1 - 1) * (n - k) + n1
    X[sni:eni] = Lp1[sni:eni] - La1[(g1 - 1) * k:]
    return X


def LDPC_enc(G, X1):
    n1 = X1.size()[1]
    n, k = G.shape  # n: code length, k: information bits length
    g1 = int(np.ceil(n1 / k))  # divide into groups
    C1 = torch.zeros([n * g1, 1])
    for gi in range(g1):
        X_g = torch.zeros([k, 1])  # padding "0" at the end of the last group
        si = gi * k
        ei = min((gi + 1) * k, n1)
        X_g[0:ei - si, 0] = X1[0, si:ei]
        C1[gi * n:(gi + 1) * n] = torch.tensor(LDPC.encode(G, X_g))
    return C1


def LDPC_dec_LLR(Lp1, DEC_para1, g1, n1, n, k, La, maxiter):
    for gi in range(g1):
        si = gi * n
        ei = (gi + 1) * n
        Lp = Lp1[si:ei]
        if La is None:
            La1 = None
        else:
            ski = gi * k
            if gi < g1 - 1:
                La1 = torch.zeros(1, n)
                eki = (gi + 1) * k
                La1[0, :k] = La[0, ski:eki]
            else:
                La1 = torch.ones(1, n)  # last bits are all 0，LLR should be positive
                La1[0, :n1 - ski] = La[0, ski:n1]
        Lp1[si:ei] = LDPC.decode_LLR(Lp, **DEC_para1, La=La1, maxiter=maxiter)
    return Lp1


def hard_decision(Lp2, g1, n1, n, k):
    X = torch.zeros([1, n1], dtype=int)
    for gi in range(g1 - 1):
        si = gi * k
        ei = (gi + 1) * k
        sni = gi * n
        eni = gi * n + k
        X[0, si:ei] = torch.tensor((Lp2[sni:eni] < 0).T)
    X[0, (g1 - 1) * k:] = torch.tensor((Lp2[(g1 - 1) * n:(g1 - 1) * (n - k) + n1] < 0).T)
    return X


def LDPC_dec_init(H, Y1, snr1, g1, n):
    Lc1 = torch.zeros_like(Y1)
    for gi in range(g1):
        si = gi * n
        ei = (gi + 1) * n
        Lc1[si:ei], DEC_para1 = LDPC.decoder_init(H, Y1[si:ei], snr1)
    return np.array(Lc1), DEC_para1


def save_img(img, path):
    imageio.imwrite(path, Image.fromarray(np.uint8(img * 255)))


def E_distance(x, y):
    x1 = np.array(x.detach().cpu())
    y1 = np.array(y)
    return ((x1 - y1) ** 2).sum() / x1.size


class SemanticNN(nn.Module):
    def __init__(self, out_ch=16):
        # coders and AWGN channel
        super(SemanticNN, self).__init__()
        # channel = 2
        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)

        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=2, stride=1, padding=0)

    def enc(self, x):
        out = self.conv1(x.to(device))
        out = self.conv2(out)
        out = self.conv3(out)

        # scale and quantize
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = copy.deepcopy(torch.div(out, out_max))

        # quantize
        out_tmp = copy.deepcopy(torch.mul(out_tmp, 256))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
        out_tmp = copy.deepcopy(torch.div(out_tmp, 256))

        out = copy.deepcopy(torch.mul(out_tmp, out_max))
        out = img2bin(out)
        return out

    def dec(self, x):
        # convert bit streams to img
        out = bin2img(x)
        out = out.reshape([batch_size, 16, 23, 23])  # recover image from bit stream

        out = out.to(device)

        out = self.tconv3(out)
        out = self.tconv4(out)
        out = self.tconv5(out)

        # scale and quantize
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = copy.deepcopy(torch.div(out, out_max))

        # quantize
        out_tmp = copy.deepcopy(torch.mul(out_tmp, 256))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
        out_tmp = copy.deepcopy(torch.div(out_tmp, 256))

        out = copy.deepcopy(torch.mul(out_tmp, out_max))
        return out

    def forward(self, x):
        return x


def SemantIC(x, snr1):
    n = 900  # LDPC codeword length
    d_v = 2  # Number of parity-check equations including a certain bit
    d_c = 3  # Number of bits in the same parity-check equation

    imgdir = f'images/snr{snr1}'
    os.makedirs(imgdir, exist_ok=True)

    X1 = img2bin(x)  # original bit stream

    # LDPC PHY channel

    n1 = X1.size()[1]

    H, G = LDPC.make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
    n, k = G.shape  # n: code length, k: information bits length

    g1 = int(np.ceil(n1 / k))  # divide bit sequence into groups for encoding

    C1 = LDPC_enc(G, X1)

    # received signals with noise
    Y1 = LDPC.add_gaussian_noise(C1, snr1, seed=seed)

    Lc1, DEC_para1 = LDPC_dec_init(H, Y1, snr1, g1, n)

    Lp1 = copy.deepcopy(Lc1)
    Lp1s = copy.deepcopy(Lc1)

    La1 = None  # torch.zeros([1, Lp1.shape[0]])  # np.zeros([1, Lp1.shape[1]])

    for i in range(8):  # joint dec
        print(f'--------------------- LDPC joint dec [{i:d}] -----------------------------')
        # joint decoding
        Lp1 = LDPC_dec_LLR(Lp1, DEC_para1, g1, n1, n, k, La=La1, maxiter=1)
        # independent decoding
        Lp1s = LDPC_dec_LLR(Lp1s, DEC_para1, g1, n1, n, k, La=None, maxiter=1)

        X1_hat = hard_decision(Lp1, g1, n1, n, k)  # hard decision
        X1s_hat = hard_decision(Lp1s, g1, n1, n, k)  # hard decision
        j1 = LDPC.BER(X1, X1_hat)
        s1 = LDPC.BER(X1, X1s_hat)
        print(f'BER s: {s1 :g}, j: {j1 :g}')

        X1_img = bin2img(X1_hat).reshape([batch_size, 3, 96, 96])
        X1_data = to_data(X1_img)
        X1s_data = to_data(bin2img(X1s_hat).reshape([batch_size, 3, 96, 96]))

        X2 = semantic_coder.enc(copy.deepcopy(X1_img))
        X2 = semantic_coder.dec(X2)
        X2_data = to_data(X2.reshape([batch_size, 3, 96, 96]))

        merged = merge_images(to_data(x), X2_data)
        save_img(merged, f'{imgdir:s}/origin-semantic-{e:d}-{i:d}.png')

        merged = merge_images(X1s_data, X1_data)

        ed1s = E_distance(x, X1s_data)
        ed1 = E_distance(x, X1_data)
        ed2 = E_distance(x, X2_data)

        print(f'EDs: {ed1s:g}, EDj: {ed1:g}, ED2: {ed2:g}')

        save_img(merged,
                 os.path.join('%s/%d-%d-BER=%.9f-ED1s=%.9f-ED1=%.9f.png' % (imgdir, e, i, j1, ed1s, ed1)))

        ex_info2 = (img2bin(X2) * -2 + 1)  # LLR mapping 0->1, 1->-1

        Lp1_max = Lp1.max()

        La1 = LDPC.fc(ex_info2, 0.5 / (i + 1), LLR_limit=50)  # exchange ex_info
        La1 = scale_8bit_weight(La1) * (
                10 ** ((-5 + i - snr1 / 2 - 3) / 10))  # SNR1 smaller，X2 should give more ex_info to X1
        La1_max = La1.max()
        # La2_max = La2.max()
        print(
            f'Max Lp1: {Lp1_max :g}, ex_info2: {ex_info2.max() :g}, La1: {La1_max:g}')
        if Lp1_max > 200:
            Lp1 = Lp1 * (200 / Lp1_max)
        X1_hat = hard_decision(Lp1, g1, n1, n, k)  # hard decision

        with open(f'images/snr{snr1:d}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            data = [e, i, s1, j1, ed1s, ed1, ed2, Lp1_max, La1_max]
            writer.writerow(data)

    return bin2img(X1_hat).reshape([batch_size, 3, 96, 96])


semantic_coder = SemanticNN()
file_path = 'semantic_coder.pkl'
if os.path.exists(file_path):
    semantic_coder.load_state_dict(torch.load(file_path))
semantic_coder.to(device)

# load data
train_set = datasets.CIFAR10('./data', train=True, transform=data_tf, download=True)
train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.CIFAR10('./data', train=False, transform=data_tf, download=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

for e in range(epoch_len):
    counter = 0
    for im, _ in train_data:
        print('Epoch %d-%d:' % (e, counter))
        im = Variable(im)
        im = im.to(device)

        for snr1 in range(-5, 10):
            print(f'===================== snr={snr1:d} ====================')
            os.makedirs('images/', exist_ok=True)
            fname = f'images/snr{snr1:d}.csv'
            if not os.path.exists(fname):
                with open(fname, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    data = ['epoch', 'iter_round', 'BERs', 'BERj', 'EDs', 'EDj', 'ED_semantic', 'Lp1_max',
                            'La1_max', 'Lp2_max', 'La2_max']
                    writer.writerow(data)

            SemantIC(copy.deepcopy(im), snr1)

        counter += 1
        if counter >= 32:
            break
