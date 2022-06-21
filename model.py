import os
from typing import Any, Dict

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn.parallel import DataParallel

from models.select_network import init_net
from utils import utils_image as util

import matplotlib.pyplot as plt
class Model:
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt
        self.save_dir = opt['path']['models']
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        # self.device = torch.device( 'cpu')


        self.netG = init_net(opt).to(self.device)
        self.netG = DataParallel(self.netG)
        self.lr = 0.0001
        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
        self.loss = 0
        # self.netG.eval()

        self.metrics: Dict[str, Any] = {'psnr': 0, 'ssim': 0, 'psnr_k': 0}

    def init(self):
        self.opt_test = self.opt['test']
        self.load_model()

    def feed_data(self, data: Dict[str, Any]):
        self.y = data['y'].to(self.device)
        self.z = data['z'].to(self.device)
        self.y_gt = data['y_gt'].to(self.device)
        self.k_gt = data['k_gt'].to(self.device)
        self.r_gt = data['r_gt'].to(self.device)
        self.sigma = data['sigma'].to(self.device)
        self.sf = data['sf'][0].item()
        self.path = data['path']

    def test(self):
        with torch.no_grad():
            self.dx, self.k, self.r, self.d = self.netG(self.y, self.z, self.sigma, self.sf)
            # for i in range(8):
            #     print(torch.mean(self.dx[i].cpu().detach()))
            #     plt.imshow(self.dx[i].cpu().detach().numpy()[0, 1, :, :])
            #     plt.show()
        self.prepare_visuals()
    def train(self, epoch):

        if epoch < 4:
            self.Warm_up( epoch)
        self.optimizer.zero_grad()

        self.dx, self.k, self.r, self.d = self.netG(self.y, self.z, self.sigma, self.sf)
        loss = 0
        gama = 0.01
        assert (len(self.dx)==len(self.k))
        N = len(self.dx)

        # for i in range(N):
        #
        #     plt.imshow(self.dx[i].cpu().detach().numpy()[0, 1, :, :])
        #     plt.show()
        # plt.imshow(self.y.cpu().detach().numpy()[0, 1, :, :])
        # plt.show()
        # print(self.dx[0].shape)
        # print(self.y_gt.shape)
        if not (N==1):
            for n in range(N-1):
                loss = loss + (torch.mean(torch.abs((self.dx[n] - self.y_gt))) + gama * torch.mean(
                    torch.abs((self.k[n] - self.k_gt))) + gama * torch.mean(torch.abs((self.r[n] - self.r_gt))))*0.1
                # print(loss)
                # print(torch.mean(
                #     torch.abs((self.k[n] - self.k_gt))))
                # print(torch.mean(torch.abs((self.dx[n] - self.y_gt))))
                # print(torch.mean(torch.abs((self.r[n] - self.r_gt))))
        # print(torch.mean(
        #     torch.abs((self.k[N - 1] - self.k_gt))))
        # print(torch.mean(torch.abs((self.dx[N - 1] - self.y_gt))))
        # print(torch.mean(torch.abs((self.r[N - 1] - self.r_gt))))
        self.loss = loss + torch.mean(torch.abs((self.dx[N - 1] - self.y_gt))) + gama * torch.mean(
            torch.abs((self.k[N - 1] - self.k_gt))) + gama * torch.mean(torch.abs((self.r[N - 1] - self.r_gt)))

        # self.loss = loss + torch.mean(torch.abs((self.dx[N - 1] - self.y_gt))) + gama * torch.mean(
        #     torch.abs((self.k[N - 1] - self.k_gt))) + gama * torch.mean(torch.abs((self.r[N - 1] - self.r_gt)))
        # self.loss = loss + torch.mean(torch.abs((self.dx[0] - self.y_gt))) + gama * torch.mean(
        #     torch.abs((self.k[0] - self.k_gt)))
        # print(self.loss)


        self.loss.backward()
        self.optimizer.step()


        if (epoch) % 10 == 0:
            self.LR_Decay(epoch/10)
            # print('Adjusting the learning rate.')

    def Warm_up(self, epoch):
        lr_d = 0.00001 + epoch * ((self.lr - 0.00001) / 3)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_d
    def LR_Decay(self, iter):
        lr_d = self.lr * (0.9 ** iter)
        # print(lr_d)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_d
    def showloss(self):
        print(self.loss)

    def prepare_visuals(self):
        self.out_dict: Dict[str, Any] = {}
        self.out_dict['y'] = util.tensor2uint(self.y[0].detach().float().cpu())

        self.out_dict['dx'] = util.tensor2uint(
            self.dx[7].detach().float().cpu())
        self.out_dict['y_gt'] = util.tensor2uint(
            self.y_gt[0].detach().float().cpu())
        self.out_dict['path'] = self.path[0]
        self.out_dict['k'] = util.tensor2uint(self.k[7].detach().float().cpu())
        self.out_dict['k_gt'] = util.tensor2uint(
            self.k_gt[0].detach().float().cpu())

    def cal_metrics(self):
        print(self.out_dict['dx'].shape)
        print(self.out_dict['y_gt'].shape)
        self.metrics['psnr'] = peak_signal_noise_ratio(self.out_dict['dx'],
                                                       self.out_dict['y_gt'])
        self.metrics['ssim'] = structural_similarity(self.out_dict['dx'],
                                                     self.out_dict['y_gt'],
                                                     multichannel=True)
        self.metrics['psnr_k'] = peak_signal_noise_ratio(
            self.out_dict['k'], self.out_dict['k_gt'])

        return self.metrics['psnr'], self.metrics['ssim'], self.metrics[
            'psnr_k']

    def save_visuals(self, tag: str):
        y_img = self.out_dict['y']
        y_gt_img = self.out_dict['y_gt']
        dx_img = self.out_dict['dx']
        path = self.out_dict['path']

        img_name = os.path.splitext(os.path.basename(path))[0]
        img_dir = os.path.join(self.opt['path']['images'], img_name)
        os.makedirs(img_dir, exist_ok=True)

        save_img_path = os.path.join(img_dir, f"{img_name:s}_{tag}.png")
        util.imsave(dx_img, save_img_path)
        util.imsave(y_img, save_img_path.replace('.png', '_y.png'))
        util.imsave(y_gt_img, save_img_path.replace('.png', '_y_gt.png'))
        util.imsave(self.out_dict['k_gt'],
                    save_img_path.replace('.png', '_k_gt.png'))
        util.imsave(self.out_dict['k'],
                    save_img_path.replace('.png', '_k.png'))

    def load_model(self):
        load_path = os.path.join(self.opt['path']['root'],
                                 self.opt['path']['pretrained_netG'])

        print(f'Loading model from {load_path}')

        network = self.netG

        if isinstance(network, nn.DataParallel):
            network = network.module

        # load head
        network.head.load_state_dict(  # type: ignore
            torch.load(os.path.join(load_path, 'models', 'head.pth')),
            strict=True)


        # load x
        state_dict_x = torch.load(os.path.join(load_path, 'models', 'x.pth'))
        network.body.net_x.load_state_dict(  # type: ignore
            state_dict_x, strict=False)

        # load k
        path_k = os.path.join(load_path, 'models', 'k.pth')
        if os.path.exists(path_k) and network.body.net_k is not None:
            print('load kernel net')
            network.body.net_k.load_state_dict(  # type: ignore
                torch.load(path_k), strict=True)

        # load hypa
        #state_dict_hypa = torch.load(
           # os.path.join(load_path, 'models', 'hypa.pth'))
       # network.hypa_list.load_state_dict(  # type: ignore
            #state_dict_hypa, strict=True)

    def save_model(self):
        save_path = os.path.join(self.opt['path']['root'],
                                 self.opt['path']['pretrained_netG'])

        print(f'Saving model to {save_path}')

        network = self.netG

        if isinstance(network, nn.DataParallel):
            network = network.module

        # save head
        torch.save(network.head.state_dict(), os.path.join(save_path, 'models', 'head.pth'))
        # load x

        torch.save(network.body.net_x.state_dict(), os.path.join(save_path, 'models', 'x.pth'))
        # load k
        torch.save(network.body.net_k.state_dict(), os.path.join(save_path, 'models', 'k.pth'))

        # load hypa
        torch.save(network.hypa_list.state_dict(), os.path.join(save_path, 'models', 'hypa.pth'))

