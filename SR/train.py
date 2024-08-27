import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from config import parse_args
from torch.backends import cudnn
from tools.train_dataloader import GetLoader
from tools.eval_dataloader import EvalDataset
from tools.PSNR_SSIM import calculate_psnr, calculate_ssim, tensor2img


class Trainer(object):

    def __init__(self, args):
        self.args = args
        print("加载训练集和测试集数据...")

        self.train_loader = GetLoader(self.args)
        self.test_loader = EvalDataset(self.args)

        self.test_iter = len(self.test_loader) // self.args.test_batch_size
        if len(self.test_loader) % self.args.test_batch_size > 0:
            self.test_iter += 1

    def __init_framework__(self):

        print("正在构建模型...")

        self.modelpackage = __import__("model."+args.model+".model", fromlist=True)
        modelClass = getattr(self.modelpackage, self.args.model)

        if self.args.model == "MyNet":
            self.network = modelClass(self.args)
        else:
            self.network = modelClass()

        if self.args.cuda >= 0:
            self.network = self.network.cuda()

        if self.args.phase == "finetune":
            model_path = os.path.join("LR.pth")
            self.network.load_state_dict(torch.load(model_path), strict=False)

    def __evaluation__(self, eval_loader, eval_iter, epoch):
        # Evaluate the checkpoint
        self.network.eval()
        total_psnr = 0
        total_ssim = 0
        total_num = 0
        dataset_name = eval_loader.dataset_name
        with torch.no_grad():
            for _ in tqdm(range(eval_iter)):
                hr, lr = eval_loader()
                if self.args.cuda >= 0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                res = self.network(lr)
                res = tensor2img(res.cpu())
                hr = tensor2img(hr.cpu())
                psnr = calculate_psnr(res[0], hr[0])
                ssim = calculate_ssim(res[0], hr[0])
                total_psnr += psnr
                total_ssim += ssim
                total_num += 1
        final_psnr = total_psnr / total_num
        final_ssim = total_ssim / total_num

        if final_psnr > self.best_psnr["psnr"]:
            self.best_psnr["psnr"] = final_psnr
            self.best_psnr["epoch"] = epoch

        print("[{}], Epoch [{}], Dataset: {}, PSNR: {:.4f}, SSIM: {:.4f}".format(self.args.model, epoch,
                                                                                 dataset_name, final_psnr, final_ssim))


    def train(self):
        # -- save
        #     -- MyNet
        #         -- rcan
        #         -- yuanhrlimg
        #     -- SwinIR
        #         -- rcan
        #         -- yuanhrlimg

        # -- save_path == save
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        # -- save.args.rcan
        if not os.path.exists(os.path.join(args.save_path, args.model)):
            os.mkdir(os.path.join(args.save_path, args.model))

        model_path = os.path.join(args.save_path, args.model, args.save_model)
        hrimg_path = os.path.join(args.save_path, args.model, args.save_hrimg)

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(hrimg_path):
            os.mkdir(hrimg_path)

        total_epoch = self.args.total_epoch

        self.best_psnr = {
            "epoch": -1,
            "psnr": -1
        }

        # ===============build framework================#
        self.__init_framework__()

        from thop import profile
        input = torch.randn(1, 3, args.patch_size, args.patch_size).cuda()
        flops, params = profile(self.network, inputs=(input,))
        print("模型计算量：{:.3f}G".format(flops  / 10 ** 9))
        print("模型总参数：{:.3f}K".format(params / 10 ** 3))

        start = 0

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr)

        l1 = nn.L1Loss()

        # Caculate the epoch number
        step_epoch = len(self.train_loader)
        print("Total step = %d in each epoch" % step_epoch)

        # Start time
        import datetime
        print("Start to train at %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        print('Start   ===========================  training...')
        start_time = time.time()

        for epoch in range(start, total_epoch):
            for step in range(step_epoch):

                self.network.train()

                self.optimizer.zero_grad()

                hr, lr = self.train_loader.next()

                generated_hr = self.network(lr)
                loss_l1 = l1(generated_hr, hr)

                loss_curr = loss_l1

                loss_curr.backward()

                self.optimizer.step()

                # Print out log info
                if (step + 1) % args.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    print("[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, l1: {:.4f}".
                          format(args.model, elapsed, epoch + 1, args.total_epoch, step + 1, step_epoch,
                                 loss_curr.item(), loss_l1.item()))

            if (epoch + 1) in args.lr_decay_step and args.lr_decay_enable:
                print("Learning rate decay")
                for p in self.optimizer.param_groups:
                    p['lr'] *= args.lr_decay
                    print("Current learning rate is %f" % p['lr'])

            # ===============save model================#
            if (epoch + 1) % args.save_epoch == 0:
                print("Save epoch %d checkpoint!" % (epoch + 1))
                save_path = os.path.join(model_path, str(epoch + 1) + ".pth")
                torch.save(self.network.state_dict(), save_path)

                self.__evaluation__(self.test_loader, self.test_iter, epoch + 1)

if __name__ == '__main__':

    args = parse_args()
    cudnn.benchmark = True
    trainer = Trainer(args)
    trainer.train()