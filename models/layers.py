"""
Layers for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Upsample(nn.Module):
  """Upsample a multi-channel input image"""
  def __init__(self, scale_factor, mode, align_corners):
    super(Upsample, self).__init__()
    self.scale_factor = scale_factor
    self.mode = mode
    self.align_corners = align_corners
  def forward(self, x):
    return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class MultiSequential(nn.Sequential):
  def forward(self, *inputs):
    x = inputs[0]
    hyp_out = inputs[1]
    for module in self._modules.values():
      if type(module) == BatchConv2d:
        x = module(x, hyp_out)
      else:
        x = module(x)
    return x

class Conv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, padding=0):
    super(Conv2d, self).__init__()
    self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
  def forward(self, x, hyp_out=None):
    return self.layer(x)

class BatchConv2d(nn.Module):
  """
  Conv2D for a batch of images and weights
  For batch size B of images and weights, convolutions are computed between
  images[0] and weights[0], images[1] and weights[1], ..., images[B-1] and weights[B-1]

  Takes hypernet output and transforms it to weights and biases
  """
  def __init__(self, in_channels, out_channels, hyp_out_units, stride=1,
         padding=0, dilation=1, kernel_size=3):
    super(BatchConv2d, self).__init__()

    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.out_channels = out_channels

    kernel_units = np.prod(self.get_kernel_shape())
    bias_units = np.prod(self.get_bias_shape())
    self.hyperkernel = nn.Linear(hyp_out_units, kernel_units)
    self.hyperbias = nn.Linear(hyp_out_units, bias_units)

  def forward(self, x, hyp_out, include_bias=True):
    assert x.shape[0] == hyp_out.shape[0], 'dim=0 of x ({}) must be equal in size to dim=0 ({}) of hypernet output'.format(x.shape[0], hyp_out.shape[0])

    x = x.unsqueeze(1)
    b_i, b_j, c, h, w = x.shape
    # b_i, c, h, w = x.shape

    # Reshape input and get weights from hyperkernel
    out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
    
    self.kernel = self.hyperkernel(hyp_out)
    
    kernel = self.kernel.view(b_i * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
    out = F.conv2d(out, weight=kernel, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,padding=self.padding)
    
    # out = F.conv2d(x, weight=kernel, bias=None, stride=self.stride, padding=self.padding)

    out = out.view(b_j, b_i, self.out_channels, out.shape[-2], out.shape[-1])
    out = out.permute([1, 0, 2, 3, 4])
    
    if include_bias:
      # Get weights from hyperbias
      self.bias = self.hyperbias(hyp_out)
      
      out = out + self.bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

    out = out[:,0,...]

    return out

  def get_kernel(self):
    return self.kernel
  def get_bias(self):
    return self.bias
  def get_kernel_shape(self):
    return [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size]
  def get_bias_shape(self):
    return [self.out_channels]

class ClipByPercentile(object):
  """Divide by specified percentile and clip values in [0, 1]."""
  def __init__(self, perc=99):
    self.perc = perc

  def __call__(self, img):
    val = np.percentile(img, self.perc)
    if val == 0:
      val = 1
    img_divide = img / val
    img_clip = np.clip(img_divide, 0, 1) 
    return img_clip

class ZeroPad(object):
  def __init__(self, final_size):
    self.final_size = final_size

  def __call__(self, img):
    '''
    '''
    final_img = np.zeros(self.final_size)
    size = img.shape
    pad_row = self.final_size[0] - size[0]
    pad_col = self.final_size[1] - size[1]
    final_img[pad_row//2:-pad_row//2, pad_col//2:-pad_col//2] = img
    return final_img