import os, random, cv2, argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                        (17, 314), (314, 405), (405, 321), (321, 375),
                        (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                        (37, 0), (0, 267),
                        (267, 269), (269, 270), (270, 409), (409, 291),
                        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                        (14, 317), (317, 402), (402, 318), (318, 324),
                        (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                        (82, 13), (13, 312), (312, 311), (311, 310),
                        (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                            (374, 380), (380, 381), (381, 382), (382, 362),
                            (263, 466), (466, 388), (388, 387), (387, 386),
                            (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                (295, 285), (300, 293), (293, 334),
                                (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                        (4, 45), (45, 220), (220, 115), (115, 48),
                        (4, 275), (275, 440), (440, 344), (344, 278), ])
ROI =  frozenset().union(*[FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, 
FACEMESH_RIGHT_EYE,FACEMESH_RIGHT_EYEBROW,FACEMESH_FACE_OVAL,FACEMESH_NOSE])

def get_smoothened_landmarks(all_landmarks, image,windows_T):

    sketch = [] 
    for i in range(len(all_landmarks)):  # frame i
        if i>windows_T-1 and i+windows_T<len(all_landmarks):
            window = all_landmarks[i-windows_T: i + windows_T]
            for j in range(len(all_landmarks[i].landmark)):  # landmark j
                all_landmarks[i].landmark[j].x = np.mean([frame_landmarks.landmark[j].x for frame_landmarks in window])
                all_landmarks[i].landmark[j].y = np.mean([frame_landmarks.landmark[j].y for frame_landmarks in window])
                all_landmarks[i].landmark[j].z = np.mean([frame_landmarks.landmark[j].z for frame_landmarks in window])

            canvas = np.zeros_like(image.copy())
            mp_drawing.draw_landmarks(
                    image=canvas,   
                    landmark_list=all_landmarks[i],
                    connections= ROI,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=6, circle_radius=1,color=(255,255,255)))
            sketch.append(canvas)

        else:
            window = all_landmarks[i]
            canvas = np.zeros_like(image.copy())
            mp_drawing.draw_landmarks(
                image=canvas,   
                landmark_list=window,
                connections= ROI,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=6, circle_radius=1,color=(255,255,255)))
            sketch.append(canvas)
    return sketch


if __name__ == '__main__':
    # from src.video_portrait_matting_model import VideoPortraitMattingModel
    from mobilenetv3 import MobileNetV3LargeEncoder
    from resnet import ResNet50Encoder
    from lraspp import LRASPP
    from decoder import RecurrentDecoder, Projection
    from hyperlayers import partialclass
    from hyperlayers import HyperLayer
    from hyperlayers import HyperLinear
    import layers
    from hypernetwork import HyperNetwork
else:
    from .mobilenetv3 import MobileNetV3LargeEncoder
    from .resnet import ResNet50Encoder
    from .lraspp import LRASPP
    from .decoder import RecurrentDecoder, Projection
    from .hyperlayers import partialclass
    from .hyperlayers import HyperLayer
    from .hyperlayers import HyperLinear
    from . import layers
    from .hypernetwork import HyperNetwork







class FastGuidedFilterRefiner(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.guilded_filter = FastGuidedFilter(1)
    
    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha):
        fine_src_gray = fine_src.mean(1, keepdim=True)#([10, 1, 512, 512])
        base_src_gray = base_src.mean(1, keepdim=True)#([10, 1, 256, 256])
        
        fgr, pha = self.guilded_filter(
            torch.cat([base_src, base_src_gray], dim=1),
            torch.cat([base_fgr, base_pha], dim=1),
            torch.cat([fine_src, fine_src_gray], dim=1)).split([3, 1], dim=1)
        
        return fgr, pha
    
    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1))
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha
    
    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha)


class FastGuidedFilter(nn.Module):
    def __init__(self, r: int, eps: float = 1e-5):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        mean_x = self.boxfilter(lr_x)#([10, 4, 256, 256])
        mean_y = self.boxfilter(lr_y)#([10, 4, 256, 256])
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        A = F.interpolate(A, hr_x.shape[2:], mode='bilinear', align_corners=False)
        b = F.interpolate(b, hr_x.shape[2:], mode='bilinear', align_corners=False)
        return A * hr_x + b


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        # Note: The original implementation at <https://github.com/wuhuikai/DeepGuidedFilter/>
        #       uses faster box blur. However, it may not be friendly for ONNX export.
        #       We are switching to use simple convolution for box blur.
        kernel_size = 2 * self.r + 1
        kernel_x = torch.full((x.data.shape[1], 1, 1, kernel_size), 1 / kernel_size, device=x.device, dtype=x.dtype)
        kernel_y = torch.full((x.data.shape[1], 1, kernel_size, 1), 1 / kernel_size, device=x.device, dtype=x.dtype)
        x = F.conv2d(x, kernel_x, padding=(0, self.r), groups=x.data.shape[1])
        x = F.conv2d(x, kernel_y, padding=(self.r, 0), groups=x.data.shape[1])
        return x



class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class HyperFCNet(nn.Module):
    '''Builds a hypernetwork that predicts a fully connected neural network.
    '''
    def __init__(self,
                 hnet_hdim=64, # MLP input dim 
                 residual= True, # MLP output dim         
                 use_batchnorm=True, # Hyper input dim (embedding)
                 ):
        super().__init__()

        self.audio_encoder = MobileNetV3LargeEncoder(pretrained=False,in_ch=1)
        self.aspp_a = LRASPP(960, 128)
        self.hnet0 = HyperNetwork(in_dim=320*16, h_dim=hnet_hdim)
        self.hnet1 = HyperNetwork(in_dim=80*24, h_dim=hnet_hdim)
        self.hnet2 = HyperNetwork(in_dim=20*40, h_dim=hnet_hdim)
        self.hnet3 = HyperNetwork(in_dim=5*128, h_dim=hnet_hdim)

        self.residual = residual
        self.use_batchnorm = use_batchnorm

        self.dconv_down0 = self.double_conv(in_channels=16, out_channels=16,hnet_hdim=hnet_hdim)
        self.dconv_down1 = self.double_conv(in_channels=24, out_channels=24,hnet_hdim=hnet_hdim)
        self.dconv_down2 = self.double_conv(in_channels=40, out_channels=40,hnet_hdim=hnet_hdim)
        self.dconv_down3 = self.double_conv(in_channels=128, out_channels=128,hnet_hdim=hnet_hdim) 



    def forward(self, x,f1, f2, f3, f4):#([1, 512])
        '''
        :param  style_f_m0,style_f_m1,style_f_m2,style_f_m3: Input to hypernetwork.
        :return: nn.Module; Predicted fully connected neural network.
        '''
        # f1 torch.Size([8, 16, 128, 128]) torch.Size([8, 24, 64, 64]) torch.Size([8, 40, 32, 32]) torch.Size([8, 128, 16, 16])
        fa0,fa1,fa2,fa3 = self.audio_encoder(x)#([8, 1, 80, 16])->([8, 16, 40, 8]);([8, 24, 20, 4]);([8, 40, 10, 2]);([8, 960, 5, 1])
        fa3 = self.aspp_a(fa3)#([8, 128, 5, 1])


        fa0 = fa0.contiguous().view(fa0.size()[0],-1)#([8, 320, 16])
        fa1 = fa1.contiguous().view(fa1.size()[0],-1)#([8, 80, 24])
        fa2 = fa2.contiguous().view(fa2.size()[0],-1)#([8, 20, 40])
        fa3 = fa3.contiguous().view(fa3.size()[0],-1)#([8, 5, 128])

        hyp_out = self.hnet0(fa0)#([8, 320, 16])->([8, 64])
        f1_temp = self.dconv_down0(f1, hyp_out)
        
        hyp_out = self.hnet1(fa1)#([8, 320, 16])->([8, 64])
        f2_temp = self.dconv_down1(f2, hyp_out)  
        
        hyp_out = self.hnet2(fa2)#([8, 320, 16])->([8, 64])
        f3_temp = self.dconv_down2(f3, hyp_out)  

        hyp_out = self.hnet3(fa3)#([8, 320, 16])->([8, 64])
        f4_temp = self.dconv_down3(f4, hyp_out)   
                
                
        return f1_temp,f2_temp,f3_temp,f4_temp

    def double_conv(self, in_channels, out_channels,hnet_hdim):
        if hnet_hdim is not None:
            if self.use_batchnorm:
                return layers.MultiSequential(
                layers.BatchConv2d(in_channels, out_channels, hnet_hdim, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                layers.BatchConv2d(out_channels, out_channels, hnet_hdim, padding=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
                nn.Sigmoid()
                )   
            else:
                return layers.MultiSequential(
                layers.BatchConv2d(in_channels, out_channels, hnet_hdim, padding=1),
                nn.ReLU(inplace=True),
                layers.BatchConv2d(out_channels, out_channels, hnet_hdim, padding=1),
                # nn.ReLU(inplace=True)
                nn.Sigmoid()
                )   
        else:
            if self.use_batchnorm:
                return layers.MultiSequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
                nn.Sigmoid()
                )   
            else:
                return layers.MultiSequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                # nn.ReLU(inplace=True)
                nn.Sigmoid()
                )   


        # return net



class HyperLipsBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv = nn.Sequential(
                # Conv2dTranspose(16*4, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 16*4, kernel_size=3, stride=1, padding=1, residual=True), )  # 96,96


        self.output_block = nn.Sequential(
            Conv2d(16*4, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

        self.face_encoder = MobileNetV3LargeEncoder(pretrained=False,in_ch=6)
        self.aspp = LRASPP(960, 128)

        self.hyper_control_net = HyperFCNet(hnet_hdim=64,residual= True,use_batchnorm=True)
        self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16*4])

    def forward(self,audio_sequences: Tensor,face_sequences: Tensor):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])
        r1 = None
        r2 = None
        r3 = None
        r4 = None
        src_sm = face_sequences
        
        fc0, fc1, fc2, fc3 = self.face_encoder(src_sm)#([4, 3, 256, 256])->([4, 16, 128, 128]);([4, 24, 64, 64]);([4, 40, 32, 32]);([4, 960, 16, 16])
        fc3 = self.aspp(fc3)#([4, 960, 16, 16])->([4, 128, 16, 16])
        fh0, fh1, fh2, fh3 = self.hyper_control_net(audio_sequences,fc0,fc1,fc2,fc3)
        hid, *rec = self.decoder(src_sm, fh0, fh1, fh2, fh3, r1, r2, r3, r4)#hid([40, 64, 128, 128])
        x1 = self.up_conv(hid)#([20, 64, 64, 64])->([20, 64, 128, 128])   
        x1 = self.output_block(x1)#([20, 64, 128, 128])->([20, 3, 128, 128])
        if input_dim_size > 4:
            x1 = torch.split(x1, B, dim=0) # [(B, C, H, W)]  ([10, 3, 512, 512])->[10,3,512,512]
            outputs1 = torch.stack(x1, dim=2) # (B, C, T, H, W)  [10,3,512,512]->[2, 3, 5, 512, 512])

        else:
            outputs1 = x1
        return outputs1


    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
    
class HRDecoder(nn.Module):
    def __init__(self,rescaling=1):
        super().__init__()
        self.rescaling = rescaling
            
        self.conv_base = nn.Sequential(
                Conv2d(3*2, 64, kernel_size=3, stride=1, padding=1, residual=False),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), )  # 96,96        
        if rescaling==4:
            self.up_conv = nn.Sequential(
                    Conv2dTranspose(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
                    Conv2dTranspose(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
                    )  
        elif rescaling==2:
            self.up_conv = nn.Sequential(
                    Conv2dTranspose(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
                    )  
        else:
            self.up_conv = nn.Sequential(
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
                    )  
        self.output_block = nn.Sequential(
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 
    def forward(self,x):
        output = self.conv_base(x)
        output = self.up_conv(output)
        output = self.output_block(output)
        return output



class HRDecoder_disc_qual(nn.Module):
    def __init__(self):
        super(HRDecoder_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 128, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 128, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),

            # nn.Sequential(nonorm_Conv2d(128, 128, kernel_size=3, stride=2, padding=1),     # 3,3
            # nonorm_Conv2d(128, 128, kernel_size=3, stride=1, padding=1),),
            
            # nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            # nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            nn.AdaptiveAvgPool2d(1),
            ])

        self.binary_pred = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, face_sequences):
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)



class HyperLipsHR(nn.Module):
    def __init__(self,window_T,rescaling=1,base_model_checkpoint="",HRDecoder_model_checkpoint = ""):
        super().__init__()
        self.base_size = 128
        self.rescaling = rescaling
        if not (window_T == None):
            self.window_T = window_T
        else:
            self.window_T = 9999999
        self.base_model = HyperLipsBase()
        checkpoint = torch.load(base_model_checkpoint,map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        self.base_model.load_state_dict(s)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False


        self.HRDecoder = HRDecoder(self.rescaling)
        checkpoint = torch.load(HRDecoder_model_checkpoint,map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        self.HRDecoder.load_state_dict(s)
        # self.base_model.load_state_dict(torch.load(base_model_checkpoint))
        self.HRDecoder.eval()
        for param in self.HRDecoder.parameters():
            param.requires_grad = False

        
    
    def forward(self,
                audio_sequences: Tensor,
                face_sequences: Tensor):
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])
               
        src = face_sequences
        if self.rescaling != 1:
            # src_sm = self._interpolate(src, scale_factor=self.rescaling)#([1, 1, 3, 2160, 3840])->([1, 1, 3, 288, 512])
            src_sm = torch.nn.functional.interpolate(src,(self.base_size, self.base_size), mode='bilinear', align_corners=False)
        else:
            src_sm = src
        output = self.base_model(audio_sequences,src_sm)
        # for HRDecoder
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            img = []
            all_landmarks = []
            for p in output:
                image = p.cpu().numpy().transpose(1,2,0) * 255.
                image = image.astype(np.uint8)
                results = face_mesh.process(image)
                if results.multi_face_landmarks==None:
                    print("***********")
                face_landmarks =  results.multi_face_landmarks[0]
                all_landmarks.append(face_landmarks)
                img.append(image)
            sketch = get_smoothened_landmarks(all_landmarks,img[0],windows_T=self.window_T)
            img_batch = np.concatenate((img, sketch), axis=3) / 255.
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).cuda()

        output = self.HRDecoder(img_batch)
        return output


  



class HyperCtrolDiscriminator(nn.Module):
    def __init__(self):
        super(HyperCtrolDiscriminator, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            nn.AdaptiveAvgPool2d(1),
            ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]#取下半部分

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)#([2, 3, 5, 512, 512])->([10, 3, 512, 512])
        false_face_sequences = self.get_lower_half(false_face_sequences)#([10, 3, 512, 512])->([10, 3, 256, 512])

        false_feats = false_face_sequences#([10, 3, 256, 512])
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)#([10, 32, 256, 512]);([10, 64, 256, 256]);([10, 128, 128, 128]):([10, 256, 64, 64]);([10, 512, 32, 32]);([10, 512, 16, 16]);([10, 512, 14, 14])

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)#([10, 3, 512, 512])
        face_sequences = self.get_lower_half(face_sequences)#([10, 3, 512, 512])->([10, 3, 256, 512])

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
    




