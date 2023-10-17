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

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    result = []
    _tensor = tensor
    import time
    start = time.time()
    _tensor = _tensor.squeeze(0).float().detach().clamp_(*min_max)
    end = time.time()

    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
    _tensor = (_tensor.permute(1, 2, 0))

    img_np = (_tensor * 255.0).round().cpu().numpy()[:, :, ::-1]


    img_np = img_np.astype(out_type)
    result.append(img_np)
    if len(result) == 1:
        result = result[0]
    end = time.time()

    return result

class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
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
    def enhance_allimg(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        self.face_helper.clean_all()
        import time
        start = time.time()
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
        self.face_helper.clean_all()
        if has_aligned:  # the inputs are already aligned

            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        #print('got face: ', (end - start)*1000)
        # face restoration

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
    only_center_face = True
    start = time.time()
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


