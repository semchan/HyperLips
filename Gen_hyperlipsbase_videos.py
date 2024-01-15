from os import listdir, path
import numpy as np
import cv2, os, sys, argparse, audio
import subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from torchvision.transforms.functional import normalize
from models.model_hyperlips import HyperLipsBase,HyperLipsHR
from GFPGAN import *
from face_parsing import init_parser, swap_regions_img
import shutil
from torch import nn

# 本文件整体功能总结：实现了Base Face Generation的各项功能，核心是class Hyperlips()
# 通过撰写代码或者调用其他文件中的函数，实现了面部数据处理，音频数据处理，FaceEncoder，AudioEncoder，HyperConv，HyperNet的功能


# 使用 Python 的 argparse 库来解析命令行参数
#
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using HyperLipsBase or HyperLipsHR models')

# --checkpoint_path_BASE: 指定 HyperLipsBase 模型的检查点文件路径，用于从中加载模型权重。默认值是 "checkpoints/hyperlipsbase_mead.pth"。
parser.add_argument('--checkpoint_path_BASE', type=str,help='Name of saved HyperLipsBase checkpoint to load weights from', default="checkpoints/hyperlipsbase_mead.pth")

# --video: 提供包含面部的视频或图像文件的路径。默认值是 "datasets"
parser.add_argument('--video', type=str,
                    help='Filepath of video/image that contains faces to use', default="datasets")

# --outfile: 指定保存处理后视频的路径。默认值是 'hyperlips_base_results'
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='hyperlips_base_results')

# --pads: 设置面部检测时的填充值（上，下，左，右），以确保包括下巴等面部特征。默认值是 [0, 0, 0, 0]
parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

# --filter_window: 设置过滤窗口的大小，用于某些处理步骤。默认值是 None
parser.add_argument('--filter_window', default=None, type=int,
                    help='real window is 2*T+1')

# --face_det_batch_size: 面部检测的批处理大小。默认值是 1
parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=1)

# --hyper_batch_size: HyperLips 模型的批处理大小。默认值是 1
parser.add_argument('--hyper_batch_size', type=int, help='Batch size for hyperlips model(s)', default=1)

# --resize_factor: 减少视频分辨率的因数。有时，在较低的分辨率（如480p或720p）下可以获得更好的结果。默认值是 1
parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

# --box: 指定一个恒定的面部边界框。仅在面部未被检测到时作为最后手段使用。默认值是 [-1, -1, -1, -1]
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

# --segmentation_path: 指定面部分割网络的检查点文件路径。默认值是 "checkpoints/face_segmentation.pth"
parser.add_argument('--segmentation_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/face_segmentation.pth")

# --gpu_id: 指定用于运行程序的 GPU ID。默认值是 0
parser.add_argument('--gpu_id', type=float, help='gpu id (default: 0)',
                    default=0, required=False)

# 解析命令行输入的参数，并将这些参数值存储在 args 变量中。设置了 args.img_size 为 128
args = parser.parse_args()
args.img_size = 128



# 作用：在一批图像中检测人脸，并对检测到的人脸进行裁剪和定位。服务于datagen函数。
# 参数：
# images: 一个图像列表，其中每个图像都是进行人脸检测的候选图像。
# detector: 一个人脸检测器对象，用于在图像中识别人脸。
# pad: 一个包含四个整数的列表，指定在人脸周围应用的填充量（上、下、左、右）。
def face_detect(images, detector,pad):
    batch_size = 1
    # 该函数一次只能处理一个图像
    if len(images) > 1:
        print('error')
        raise RuntimeError('leng(imgaes')
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                # 调用detector对象的get_detections_for_batch方法来检测当前批次中的图像的人脸，并将检测结果添加到predictions列表中
                predictions.extend(
                    detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError as e:
            # 如果在人脸检测过程中出现RuntimeError（可能是由于内存不足引起的），则执行 except 块。
            print(e)
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    # 空列表results，用于存储计算出的人脸边界框的坐标
    results = []
    pady1, pady2, padx1, padx2 = pad  # [0, 10, 0, 0]

    # 这个循环同时遍历检测结果predictions和原始图像images。rect是检测到的人脸边界框，image是对应的原始图像
    for rect, image in zip(predictions, images):
        # 如果在某个图像中没有检测到人脸
        if rect is None:
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        # 计算裁剪坐标：使用rect（人脸边界框）的坐标和pad参数计算裁剪区域的坐标
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)

    # 结果列表中每个元素包含两个部分：裁剪后的人脸图像和该人脸在原始图像中的坐标
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

# 作用：数据生成器，结合音频（Mel频谱）和视频（人脸图像）数据准备待处理的批次数据
# 参数：
# mels:这个参数代表 Mel 频谱的集合。mels 包含了与视频帧相关联的音频特征
# detector:detector 是一个用于人脸检测的对象或函数。它的作用是在给定的视频帧中识别人脸。
# face_path:这个参数指定一个路径，用于定位包含人脸的视频文件。函数将从这个路径读取视频帧。
# resize_factor:resize_factor 参数用于调整视频帧的大小。这个因子指定了帧的缩放比例，通常用于减小图像尺寸以减少计算复杂度或适应模型的输入要求。
def datagen(mels, detector,face_path, resize_factor):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    bbox_face, frame_to_det_list, rects, frame_to_det_batch = [], [], [], []
    img_size = 128
    hyper_batch_size = args.hyper_batch_size

    # 使用 read_frames 函数读取由 face_path 和 resize_factor 参数指定的视频帧
    reader = read_frames(face_path, resize_factor)

    # 用于逐步读取视频帧，执行人脸检测，然后将结果与相应的音频特征（Mel频谱）结合起来，以形成用于深度学习模型的批次数据
    for i, m in enumerate(mels):
        try:
            # try 块中，函数尝试使用 next(reader) 从 reader（一个视频帧生成器）获取下一个视频帧 frame_to_save
            frame_to_save = next(reader)
        except StopIteration:
            # 如果没有更多的帧可读取，捕获 StopIteration 异常，重新初始化 reader 以再次开始读取帧。然后，再次尝试获取下一个视频帧
            reader = read_frames(face_path, resize_factor)
            frame_to_save = next(reader)
        h, w, _ = frame_to_save.shape
        # 在捕获的视频帧上执行人脸检测
        face, coords = face_detect([frame_to_save], detector,args.pads)[0]
        # 将检测到的人脸调整到指定的图像大小
        face = cv2.resize(face, (img_size, img_size))

        # 将处理后的人脸图像、对应的 Mel 频谱、原始帧和人脸坐标添加到各自的批次列表中
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= hyper_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # 创建图像副本并遮蔽一半像素
            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            # 合并原始图像和修改后的图像，将像素值归一化到 [0, 1] 范围
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

            # 调整 Mel 频谱数组的形状以适应模型的输入要求
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            # 生成批次数据，分别是人脸图像、对应的 Mel 频谱、原始帧和人脸坐标
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # 处理遗漏的图像数据，具体细节与上述函数一致
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

# 加载一个保存在磁盘上的检查点（checkpoint）
def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

# 加载和准备 HyperLipsBase 模型用于评估或预测
def load_HyperLipsBase(path, device):
    # 创建模型实例
    model = HyperLipsBase()

    # 加载检查点
    checkpoint = _load(path, device)

    # 提取状态字典并加载到模型中
    s = checkpoint["state_dict"]
    model.load_state_dict(s)

    # 将模型移动到指定设备
    model = model.to(device)
    print("HyperLipsBase model loaded")
    return model.eval()

# 从一个视频文件中逐帧读取视频帧，并根据需要调整每帧的大小。服务于datagen函数。
def read_frames(face_path, resize_factor):
    video_stream = cv2.VideoCapture(face_path)

    print('Reading video frames from start...')
    read_frames_index = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
        yield frame

def main():
    Hyperlips_executor = Hyperlips()
    Hyperlips_executor._HyperlipsLoadModels()
    Hyperlips_executor._HyperlipsInference()

# Hyperlips 类封装了与 HyperLipsBase 模型相关的操作，用于生成具有同步唇部运动的高保真度说话面部视频
class Hyperlips():
    # 初始化
    def __init__(self):
        # 检查点文件路径
        self.checkpoint_path_BASE = args.checkpoint_path_BASE
        # 面部分割网络的路径
        self.parser_path = args.segmentation_path
        # 处理数据时的批处理大小
        self.batch_size = args.hyper_batch_size #128
        # Mel 频谱的步长
        self.mel_step_size = 16

    # 加载模型
    # 负责加载 HyperLipsBase 模型和面部分割网络
    # 它首先检查 GPU 可用性和配置，然后加载并将模型移到相应的设备（CPU 或 GPU）上。
    def _HyperlipsLoadModels(self):
        gpu_id = args.gpu_id
        if not torch.cuda.is_available() or (gpu_id > (torch.cuda.device_count() - 1)):
            raise ValueError(
                f'Existing gpu configuration problem.(gpu.is_available={torch.cuda.is_available()}| gpu.device_count={torch.cuda.device_count()})')
        self.device = torch.device(f'cuda:{gpu_id}')
        print('Using {} for inference.'.format(self.device))
        self.model = load_HyperLipsBase(self.checkpoint_path_BASE, self.device)
        self.seg_net = init_parser(self.parser_path, self.device)
        print(' models init successed...')

    # 负责处理整个唇部同步的推理过程
    # 该文件的核心部分。分为以下几个阶段：初始化和设置阶段，视频文件处理准备阶段，视频和音频处理准备阶段，音频数据处理阶段，生成批次数据阶段，模型推理和视频帧合成阶段，视频音频合成和清理阶段
    def _HyperlipsInference(self):
        # 使用 OpenCV 的 FaceAlignment 创建一个人脸检测器，用于在视频帧中定位人脸
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False, device='cuda')

        # 从给定的视频目录(args.video)中读取视频文件列表
        videos = args.video

        # 读取 videos 目录中的所有文件名，并将它们存储在变量 file_name_list 中
        file_name_list = os.listdir(videos)

        # 设置输出目录
        hyper_base_dir = args.outfile

        # 创建输出目录
        os.makedirs(hyper_base_dir, exist_ok=True)

        # 处理视频文件，并生成具有同步唇部运动的视频
        for i in tqdm(range(len(file_name_list))):
            # 从 file_name_list 中获取当前迭代的目录名。这个列表包含了所有待处理的视频文件或子目录的名称
            origion_dirname = file_name_list[i]

            # 来创建输出目录的路径，这个路径用于存储处理后的视频文件
            out_dirname_path = path.join(hyper_base_dir,origion_dirname)

            # 检查并创建输出目录
            if not path.exists(out_dirname_path):
                os.mkdir(out_dirname_path)

            # 获取视频文件列表
            video_name_list = glob(os.path.join(args.video,origion_dirname, "*.mp4"))

            for j in video_name_list:
                # 视频文件路径赋值给变量 face和变量 audiopath
                face = j
                audiopath = j

                # 创建输出目录和临时目录
                outfile = os.path.join(out_dirname_path,face.split('/')[-1])
                rest_root_path = "temp/rest"
                temp_save_path ="temp/temp"
                os.makedirs(rest_root_path, exist_ok=True)
                os.makedirs(temp_save_path, exist_ok=True)

                if not os.path.isfile(face):
                    raise ValueError('--face argument must be a valid path to video/image file')
                else:
                    # 读取视频各属性
                    video_stream = cv2.VideoCapture(face)
                    fps = video_stream.get(cv2.CAP_PROP_FPS)
                    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_stream.release()
                # 使用 OpenCV 的 VideoWriter 类创建一个视频写入对象，用于生成和保存处理后的视频文件
                out = cv2.VideoWriter(os.path.join(temp_save_path, 'result.avi'), cv2.VideoWriter_fourcc(*'DIVX'),fps, (frame_width, frame_height))

                # 把音频格式转换为 .wav
                if not audiopath.endswith('.wav'):
                    command = 'ffmpeg -y -i {} -strict -2 {} -loglevel quiet'.format(
                        audiopath, os.path.join(temp_save_path, 'temp.wav'))
                    subprocess.call(command, shell=True)
                    audiopath = os.path.join(temp_save_path, 'temp.wav')

                # 加载 .wav 音频文件
                wav = audio.load_wav(audiopath, 16000)
                # 提取 Mel 频谱
                mel = audio.melspectrogram(wav)

                # 数据校验：用于检查从音频文件中提取的 Mel 频谱中是否存在非数值
                if np.isnan(mel.reshape(-1)).sum() > 0:
                    raise ValueError(
                        'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
                # 空列表 mel_chunks，用于存储即将生成的 Mel 频谱的小块
                mel_chunks = []
                # 设置 Mel 频谱索引的乘数
                mel_idx_multiplier = 80. / fps
                i = 0

                # 整个 Mel 频谱分割成一系列较小的块，每个块对应视频的一段时间
                while 1:
                    # 计算 Mel 频谱的起始索引
                    start_idx = int(i * mel_idx_multiplier)

                    # 检查索引范围并截取 Mel 频谱
                    if start_idx + self.mel_step_size > len(mel[0]):
                        mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                        break
                    mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
                    i += 1

                # 核心阶段：生成批次数据阶段。生成处理视频帧和相应 Mel 频谱块的批次数据
                gen = datagen(mel_chunks, detector, face, args.resize_factor)

                # 核心阶段：遍历批次数据，进行模型推理和视频帧合成阶段
                for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                                total=int(
                                                                                    np.ceil(
                                                                                        float(len(mel_chunks))/ self.batch_size)))):
                    # 将图像批次和 Mel 频谱批次转换为 PyTorch 张量，并调整维度以适应模型的输入格式
                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)#([122, 6, 96, 96])
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

                    # 使用 with torch.no_grad(): 上下文管理器禁用梯度计算
                    with torch.no_grad():
                        pred = self.model(mel_batch, img_batch)  # mel_batch([122, 1, 80, 16]) img_batch([128, 6, 128, 128])

                    for p, f, c in zip(pred, frames, coords):
                        y1, y2, x1, x2 = c
                        # 创建遮罩
                        mask_temp = np.zeros_like(f)
                        # 将预测结果 p（通常是一个张量）转移到 CPU，转换为 NumPy 数组，并调整通道顺序（从 CHW 到 HWC），然后将像素值缩放到 [0, 255] 范围
                        p = p.cpu().numpy().transpose(1,2,0) * 255.

                        # 使用 swap_regions_img 函数处理原始帧的人脸区域和模型的预测结果
                        p,mask_out = swap_regions_img(f[y1:y2, x1:x2], p, self.seg_net)

                        # 调整图像和遮罩的大小和类型
                        p = cv2.resize(p, (x2 - x1, y2 - y1)).astype(np.uint8)
                        mask_out = cv2.resize(mask_out.astype(np.float)*255.0, (x2 - x1, y2 - y1)).astype(np.uint8)

                        # 应用遮罩
                        mask_temp[y1:y2, x1:x2] = mask_out

                        # 图像融合处理
                        kernel = np.ones((5,5),np.uint8)
                        mask_temp = cv2.erode(mask_temp,kernel,iterations = 1)
                        mask_temp = cv2.GaussianBlur(mask_temp, (75, 75), 0,0,cv2.BORDER_DEFAULT)

                        # 应用预测到原始帧
                        f_background = f.copy()
                        f[y1:y2, x1:x2] = p
                        f = f_background*(1-mask_temp/255.0)+f*(mask_temp/255.0)

                        # 转换帧格式并写入视频
                        f = f.astype(np.uint8)
                        out.write(f)

                out.release()

                # 构建并执行 ffmpeg 命令，用于将处理后的视频（无声）和原始音频合并
                command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
                    audiopath, os.path.join(temp_save_path, 'result.avi'), outfile)
                subprocess.call(command, shell=True)

                # 清理临时文件和目录
                if os.path.exists(temp_save_path):
                    shutil.rmtree(temp_save_path)

if __name__ == '__main__':
    main()
