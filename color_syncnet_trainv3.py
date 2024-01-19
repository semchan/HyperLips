from os.path import dirname, join, basename, isfile
from tqdm import tqdm
# model文件夹 syncnet.py文件 class SyncNet_color
from models import SyncNet_color as SyncNet
# audio.py
import audio
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, random, cv2, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# hparams.py
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')


parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='Train_data/imgs')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="./checkpoints_lipsync_expert", type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)


args = parser.parse_args()

# 训练期间执行优化步骤的步数
global_step = 0
# 训练的总轮数
global_epoch = 0
# 是否有可用gpu
use_cuda = torch.cuda.is_available()

ema_decay = 0.5 ** (32 / (10 * 1000))
# 连续输入的帧数
syncnet_T = 5
# 设置处理梅尔频谱图帧的步幅大小
syncnet_mel_step_size = 16    

# 定义加载数据集类
class Dataset(object):
    def __init__(self, split):
        # 返回图片文件地址
        # 获取指定数据集拆分（训练/验证）中的所有视频文件地址
        self.all_videos = get_image_list(args.data_root, split)
        # 初始化音频和视频同步的偏移量
        self.av_offset_shift = 0

    def get_frame_id(self, frame):
        # 从图像文件路径中提取帧的编号
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        # 根据给定的起始帧，返回一个包含一定数量帧的窗口（用于同步音频和视频）
        start_id = self.get_frame_id(start_frame) # 获取起始帧的编号
        vidname = dirname(start_frame) # 获取视频文件所在目录的路径

        # 存储窗口内帧的文件名列表
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T): # 根据syncnet_T确定窗口的大小
            frame = join(vidname, '{}.jpg'.format(frame_id)) # 构建帧的完整文件路径
            if not isfile(frame): # 如果文件不存在
                return None
            window_fnames.append(frame) # 将帧文件路径添加到窗口列表中
        return window_fnames # 返回窗口内所有帧的文件路径列表

    # crop_audio_window 方法：从音频梅尔频谱图中裁剪与图像窗口对应的部分
    def crop_audio_window(self, spec, start_frame):
        # 获取起始帧的编号
        start_frame_num = self.get_frame_id(start_frame)
        # 加上音频和视频同步的偏移量
        start_frame_num = start_frame_num + self.av_offset_shift
        # 计算在梅尔频谱图中的起始索引
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        # 裁剪的终止索引
        end_idx = start_idx + syncnet_mel_step_size
        # 返回裁剪后的梅尔频谱图
        return spec[start_idx: end_idx, :]

    # read_window 方法：读取图像窗口中的所有帧
    def read_window(self, window_fnames, flip_flag=False):
        # 如果窗口文件路径为空，返回 None
        if window_fnames is None:
            return None
        # 存储图像窗口中的帧
        window = []
        # 遍历窗口中的每个文件路径
        for fname in window_fnames:
            # 使用 OpenCV 读取图像
            img = cv2.imread(fname)
            # 如果图像为 None，说明读取失败，返回 None
            if img is None:
                return None
            try:
                # 尝试将图像调整为指定大小
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                # 如果调整大小发生异常，返回 None
                return None
            # 如果 flip_flag 为 True，水平翻转图像
            if flip_flag:
                img = np.flip(img, axis=1).copy()
            # 将处理后的图像添加到窗口列表中
            window.append(img)
        # 返回图像窗口列表
        return window

    # __getitem__ 方法：获取数据集中的一个样本
    def __getitem__(self, idx):
        # 无限循环，直到找到合适的视频
        while 1:
            # 随机选择一个视频索引
            idx = random.randint(0, len(self.all_videos) - 1)
            # 获取视频路径
            vidname = self.all_videos[idx]

            # 获取视频中所有帧的文件路径列表
            img_names = list(glob(join(vidname, '*.jpg')))
            # 如果帧数小于等于 3 * syncnet_T，跳过该样本
            if len(img_names) <= 3 * syncnet_T:
                continue

            # 随机选择一帧作为正样本
            img_name = random.choice(img_names)
            # 从同一视频中随机选择一帧作为负样本
            wrong_img_name = random.choice(img_names)
            # 避免选择相同的帧作为正负样本
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            # 随机选择标签 y 为 1 或 0，表示正样本或负样本
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            # 获取图像窗口的文件路径
            window_fnames = self.get_window(chosen)
            # 如果获取失败，跳过该样本
            if window_fnames is None:
                continue

            # 使用 read_window 方法读取图像窗口
            window = self.read_window(window_fnames, flip_flag=True)

            try:
                # 获取音频文件路径
                wavpath = join(vidname, "audio.wav")
                # 加载音频文件
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                # 计算原始梅尔频谱图
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                # 如果发生异常，跳过该样本
                continue

            # 裁剪梅尔频谱图
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            # 如果裁剪后的梅尔频谱图大小不符合 syncnet_mel_step_size，跳过该样本
            if mel.shape[0] != syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            # 将图像窗口拼接为一个三通道的图像
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]
            # 转换为 PyTorch 张量
            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            # 返回样本
            return x, mel, y


# 二元交叉熵损失
logloss = nn.BCELoss()

# 余弦相似度损失，用于监督标签与预测标签之间的损失
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    # 全局训练步数和训练周期
    global global_step, global_epoch
    # 保存训练步数，用于计算当前会话的步数
    resumed_step = global_step

    # 当全局训练周期小于指定周期数时进行训练
    while global_epoch < nepochs:
        # 记录当前训练步数的损失
        running_loss = 0.
        # 使用 tqdm 进度条迭代训练数据集
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            # 设置模型为训练模式
            model.train()
            # 梯度清零
            optimizer.zero_grad()

            # 将数据转移到 CUDA 设备上
            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)

            # 模型前向传播
            a, v = model(mel, x)

            # 计算余弦相似度损失
            loss = cosine_loss(a, v, y)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 更新全局训练步数
            global_step += 1
            # 计算当前会话的步数
            cur_session_steps = global_step - resumed_step
            # 记录当前步数的损失
            running_loss += loss.item()

            # 在每个检查点间隔或第一步时保存模型检查点
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

            # 在不计算梯度的情况下评估模型
            with torch.no_grad():
                eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            # 在指定的评估间隔内再次评估模型
            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            # 获取当前学习率
            lr_temp = optimizer.state_dict()['param_groups'][0]['lr']
            # 在 tqdm 进度条中显示损失和学习率
            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)) + '  ' + 'lr: {}'.format(lr_temp))

        # 更新全局训练周期
        global_epoch += 1
        print(global_epoch)


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    # 指定评估步数
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    # 保存评估损失
    losses = []
    # 无限循环，每次从测试数据集中获取一个批次进行评估
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):
            # 将数据转移到 CUDA 设备上
            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)

            # 设置模型为评估模式
            model.eval()
            # 模型前向传播
            a, v = model(mel, x)

            # 使用模型的指数移动平均进行评估
            # model_ema.eval()
            # a, v = model_ema(mel, x)

            # 计算余弦相似度损失
            loss = cosine_loss(a, v, y)
            # 记录评估损失
            losses.append(loss.item())

            # 如果超过指定评估步数，退出循环
            if step > eval_steps:
                break

        # 计算平均评估损失
        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    # 构建保存模型检查点的路径
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    # 如果设置保存优化器状态，则获取当前优化器的状态字典
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    # 保存模型的状态字典、优化器状态、全局步数和周期数到文件
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    # 打印保存的检查点路径
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    # 如果使用 CUDA，直接加载检查点
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    # 否则，在 CPU 上加载检查点
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    # 全局训练步数和周期数
    global global_step
    global global_epoch

    # 打印加载检查点的信息
    print("Load checkpoint from: {}".format(path))
    # 调用 _load 函数加载检查点
    checkpoint = _load(path)
    # 加载模型的状态字典
    model.load_state_dict(checkpoint["state_dict"])
    # 如果不重置优化器，则加载优化器的状态
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        # 如果优化器状态不为空，则加载优化器状态
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    # 更新全局步数和周期数
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

# 主程序入口
if __name__ == "__main__":
    # 获取检查点目录和检查点路径
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    # 如果检查点目录不存在，则创建目录
    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # 设置数据集和数据加载器
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=1)

    # 设置设备为 CUDA 或 CPU
    device = torch.device("cuda" if use_cuda else "cpu")

    # 创建 SyncNet 模型并将其移动到设备上
    model = SyncNet().to(device)
    # 打印模型中可训练参数的总数
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # 创建 Adam 优化器，仅优化可训练参数
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    # 如果提供了检查点路径，则加载检查点
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    # 调用 train 函数进行模型训练
    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)

