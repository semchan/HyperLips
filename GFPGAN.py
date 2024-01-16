import cv2
import os
import torch
from basicsr.utils import img2tensor
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
import time
from gfpgan.gfpganv1_clean_arch import GFPGANv1Clean
import time
import numpy as np
import torch.nn.functional as F

# 将Tensor张量转换为Numpy数组
def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """
    会将值归一化到[0,1]之间，输入的Tensor向量为RGB格式

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:(B:批次大小, H:图像高, W:图像宽度)
            1.4D mini-batch张量，形状为(B x 3/1 x H x W)；
            2.3D张量，形状为(3/1 x H x W)；
            3.2D张量，形状为(H x W)。
        rgb2bgr (bool): 是否将RGB通道顺序转换为BGR
        out_type (numpy type): 输出的NumPy数组类型。
                                如果为np.uint8，则将输出转换为uint8类型，范围为[0, 255]。
                                否则，输出将是浮点数类型，范围为[0, 1]。
                                默认值为np.uint8。
        min_max (tuple[int])：clamp（夹取）操作的最小和最大值
        形式为元组，包含两个整数值
    Returns:
        三维阵列(H×W×C)或二维阵列，形状(H × W)，通道顺序为BGR
        这里的C指的是通道数
    """
    # 检查输入参数tensor是否为PyTorch张量或张量列表，如果不是则引发TypeError
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    result = []
    _tensor = tensor
    import time
    start = time.time()
    # 把批处理维度去掉同时把原值夹成[0,1]
    _tensor = _tensor.squeeze(0).float().detach().clamp_(*min_max)
    end = time.time()

    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
    _tensor = (_tensor.permute(1, 2, 0))
    print("tensor:")
    print(_tensor)
    img_np = (_tensor * 255.0).round().cpu().numpy()[:, :, ::-1]


    img_np = img_np.astype(out_type)
    result.append(img_np)
    # 消除列表嵌套
    if len(result) == 1:
        result = result[0]
    end = time.time()

    return result
# 创建用于图像恢复的类
class GFPGANer():
    """使用GFPGAN进行恢复

    检测裁剪，将尺寸调为512*512
    背景使用bg_upsampler进行上采样.

    Args:
        model_path (str): GFPGAN的模型路径. It can be urls (will first download it automatically).
        upscale (float): 最终输出的缩放比例. Default: 2.
        arch (str): GFPGAN的架构. Option: clean | original. Default: clean.
        channel_multiplier (int): StyleGAN2大型网络的通道乘法器. Default: 2.
        bg_upsampler (nn.Module): 背景的上采样器. Default: None.
    """

    def __init__(self, device,model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = device#torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)

        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device)
        loadnet = torch.load(model_path, map_location=device)
        #loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)
        print('GFPGAN model loaded')

    @torch.no_grad()
    # 对整个图像进行加强，适用于多张人脸的图像
    def enhance_allimg(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        # 清除
        self.face_helper.clean_all()
        import time
        start = time.time()
        # 如果图像已对齐，将其调整大小为(512, 512)，作为单个人脸存储；
        # 如果图像未对齐，获取每张脸的关键点，对每张脸进行对齐
        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()
        end = time.time()
        # print('got face: ', (end - start)*1000)
        # face restoration
        start = time.time()
        # 迭代处理每一张人脸
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            start = time.time()
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            end = time.time()

            try:
                import time
                start = time.time()
                output = self.gfpgan(cropped_face_t, return_rgb=False)[0]  # 15ms #NCHW
                end = time.time()


                start = time.time()
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))  # 18msms
                end = time.time()

            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face
            start = time.time()
            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)
            end = time.time()

        end = time.time()


        start = time.time()

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)

            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img

        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
        end = time.time()


    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        # has_aligned: 输入图像是否已经对齐，默认为False
        # only_center_face: 仅处理中心脸，默认为False
        # paste_back: 是否将处理后的脸粘贴回原图，默认为True
        self.face_helper.clean_all()
        # 如果输入图像已经对齐，则直接将图像添加到self.face_helper.cropped_faces列表
        if has_aligned:  # the inputs are already aligned

            self.face_helper.cropped_faces = [img]
        # 否则，使用FaceRestoreHelper类对输入图像进行人脸检测、对齐和裁剪
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        #print('got face: ', (end - start)*1000)
        # 对每个裁剪的人脸进行GFPGAN模型的推理和图像的还原
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data

            #cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True) ##CHW
            cropped_face = cropped_face[[2,1,0], :, :]#bgr to RGB
            normalize(cropped_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face.unsqueeze(0)#.to(self.device)##NCHW

            cropped_face_t = F.interpolate(cropped_face_t, (512, 512), mode='bilinear', align_corners=True)

            try:

                output = self.gfpgan(cropped_face_t, return_rgb=False)[0] # #NCHW  33ms


            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = output
           # restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        # 如果图像没有对齐，并且需要将脸粘贴回原图，则进行背景的上采样
        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None
            # 获取逆仿射变换参数，将还原的脸粘贴回原图
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)

            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img

        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None



def GFPGANInit(device,face_enhancement_path):
    """Inference demo for GFPGAN (for users).
    """
    upscale = 1

    # ------------------------ input & output ------------------------
    import numpy as np
    bg_upsampler = None
    # ------------------------ set up GFPGAN restorer ------------------------
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    model_path = face_enhancement_path
    restorer = GFPGANer(
        device = device,
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    return restorer

def GFPGANInfer(img, restorer, aligned):
    # restorer：GFPGANer类的实例，用于对图像进行增强或人脸修复
    # aligned：一个布尔值，指示输入的图像是否已经对齐
    only_center_face = True
    start = time.time()
    # 如果输入图像已对齐，那么调用 restorer.enhance 方法对单个人脸图像进行增强，并返回还原后的人脸
    # 如果输入图像未对齐，那么调用 restorer.enhance_allimg 方法对整个图像进行增强，并返回整个合成图像
    if aligned:
        cropped_faces, restored_faces, restored_img = restorer.enhance(
                img, has_aligned=aligned, only_center_face=only_center_face, paste_back=True)
    else:
        cropped_faces, restored_faces, restored_img = restorer.enhance_allimg(
                img, has_aligned=aligned, only_center_face=only_center_face, paste_back=True)

    end = time.time()
    #print(end - start) #600ms
    if aligned==False:
        return restored_img
    else:
        return restored_faces[0]



