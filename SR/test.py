import os
import cv2
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from config import parse_args
from torch.backends import cudnn
from tools.test_dataloader import TestDataset
from tools.PSNR_SSIM import calculate_psnr, calculate_ssim, tensor2img

class Tester(object):
    def __init__(self, args):

        self.args = args
        self.image_scale = self.args.up_scale

        self.test_loader = TestDataset(self.args)
        self.test_iter = len(self.test_loader)

    def __init_framework__(self):

        # ===============build models================#
        print("正在加载网络参数...")
        self.modelpackage = __import__("model."+args.model+".model", fromlist=True)
        modelClass = getattr(self.modelpackage, self.args.model)
        # self.network = modelClass(self.args)
        if self.args.model == "MyNet":
            self.network = modelClass(self.args)
        else:
            self.network = modelClass()
        # self.network = Net(args)
        model_path = os.path.join(args.save_path, args.model, args.save_model)
        # train in GPU
        if self.args.cuda >= 0:
            self.network = self.network.cuda()
        model_path = os.path.join(os.path.join(model_path, str(args.load_epoch) + ".pth"))

        print("正在加载预训练模型：{}".format(model_path))
        model_spec = torch.load(model_path, map_location=torch.device("cpu"))
        own_state = self.network.state_dict()
        # print(own_state.keys())
        for name, param in model_spec.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
        print("预训练模型加载完成")

        from thop import profile
        input = torch.randn(1, 3, args.patch_size, args.patch_size).cuda()
        flops, params = profile(self.network, inputs=(input,))
        print("模型总参数：{:.3f}G".format(flops / 10 ** 9))
        print("模型计算量：{:.3f}K".format(params / 10 ** 3))

        if self.args.cuda >= 0:
            self.network = self.network.cuda()

    def test(self):

        save_dir = os.path.join(args.save_path, args.model, args.save_hrimg)

        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)

        # models
        self.__init_framework__()

        # Start time
        import datetime
        print("Start to test at %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        total_psnr = 0
        total_ssim = 0
        total_num = 0
        self.network.eval()
        patch_test = True
        with torch.no_grad():
            for _ in tqdm(range(self.test_iter)):
                hr, lr, names = self.test_loader()
                if self.args.cuda >= 0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                if patch_test:
                    tile = 64
                    tile_overlap = 24
                    scale = self.args.up_scale
                    b, c, h, w = lr.size()
                    tile = min(tile, h, w)

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                    E = torch.zeros(b, c, h * scale, w * scale).type_as(lr)
                    W = torch.zeros_like(E)

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = lr[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                            out_patch = self.network(in_patch)
                            if isinstance(out_patch, list):
                                out_patch = out_patch[-1]
                            out_patch_mask = torch.ones_like(out_patch)

                            E[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(
                                out_patch)
                            W[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(
                                out_patch_mask)
                    res = E.div_(W)
                else:
                    res = self.network(lr)

                dataset_size = res.shape[0]

                res = tensor2img(res.cpu())
                hr = tensor2img(hr.cpu())

                for t in range(dataset_size):
                    temp_img = res[t, :, :, :]
                    psnr = calculate_psnr(temp_img, hr[t, :, :, :])
                    ssim = calculate_ssim(temp_img, hr[t, :, :, :])
                    # print("PSNR is %.3f"%psnr)
                    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                    i_name = names[t]
                    cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i_name)), temp_img)
                    total_num += 1
                    total_psnr += psnr
                    total_ssim += ssim
            final_psnr = total_psnr / total_num
            final_ssim = total_ssim / total_num

        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}], PSNR: {:.4f}, SSIM: {:.4f}".format(elapsed, final_psnr, final_ssim))

if __name__ == '__main__':
    args = parse_args()
    cudnn.benchmark = True
    tester = Tester(args)
    tester.test()