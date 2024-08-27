import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 模块构建
    parser.add_argument("--model", type=str, default="MyNet",
                        choices=["MyNet", "CARN", "IMDN", "MAFFSRN", "SwinIR", "RLFN", "ESRT", "NGramSwin"])
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--block_num", type=int, default=7)
    parser.add_argument("--num_feat", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=8)
    # 数据处理
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--img_subffix", type=str, default="jpg")
    parser.add_argument("--patch_size", type=int, default=64)
    # 训练参数
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--total_epoch", type=int, default=180)
    parser.add_argument("--lr_decay_step", type=list, default=[40, 80, 120, 150])
    parser.add_argument("--lr_decay_enable", type=bool, default=True)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--model_save_epoch", type=int, default=1)
    parser.add_argument("--test_epoch", type=int, default=1)
    parser.add_argument("--log_step", type=int, default=200)
    parser.add_argument("--up_scale", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    # 数据增强
    parser.add_argument("--dataset_enlarge", type=int, default=32)
    parser.add_argument("--dataloader_workers", type=int, default=1)

    # 模型保存
    # 保存路径
    # -- save
    #     -- MyNet
    #         -- rcan
    #         -- yuanhrlimg
    #     -- SwinIR
    #         -- rcan
    #         -- yuanhrlimg

    parser.add_argument("--load_epoch", type=int, default=179)
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--save_model", type=str, default="model=7")
    parser.add_argument("--save_hrimg", type=str, default="yuanhrlimg=7")

    # general settings
    parser.add_argument("--phase", type=str, default="train", choices=['train', 'finetune', 'debug'],)
    parser.add_argument("--cuda", type=int, default=0)
    # parser.add_argument("--ckpt", type=int, default=84, help="checkpoint epoch for test phase or finetune phase")

    return parser.parse_args()