import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F
from .model import BiSeNet
from torchvision.transforms.functional import normalize
from torchvision.transforms import Resize
def init_parser(pth_path, device):

    n_classes = 19
    net = BiSeNet(device,n_classes=n_classes)
    net.to(device)

    net.load_state_dict(torch.load(pth_path,map_location=device))
    net.eval().to(device)
    print('Parser model loaded')
    return net

def image_to_parsing_img(img, net):
    import time
    start = time.time()
    img = cv2.resize(img, (512, 512))
    img_copy = img.copy()
    img = img[:,:,::-1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img.copy())
    img = torch.unsqueeze(img, 0)
    end = time.time()


    start1 = time.time()
    with torch.no_grad():
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).argmax(0)
        parsing = parsing.cpu().numpy()
        end1 = time.time()

        return parsing


def image_to_parsing(img, net):
    img = cv2.resize(img, (512, 512))
    img_copy = img.copy()
    img = img[:,:,::-1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img.copy())
    img = torch.unsqueeze(img, 0)
    device = next(net.parameters()).device
    with torch.no_grad():
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).argmax(0)
        parsing = parsing.cpu().numpy()

        return parsing


def image_to_parsing2(img, net):
    img = img.to(dtype=torch.float32).div(255)
    normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    img = torch.unsqueeze(img, 0)
    img = F.interpolate(img, (512, 512), mode='bilinear', align_corners=True)
    with torch.no_grad():
        out = net(img)[0]   #15ms
        parsing = out.squeeze(0).argmax(0)
        return parsing


def get_mask(parsing, classes):
    res = parsing == classes[0]
    for val in classes[1:]:
        res += parsing == val
    return res

def swap_regions_img(source, target, net):
    import time
    parsing = image_to_parsing_img(source, net)  #13ms
    source = cv2.resize(source,(512,512))
    face_classes = [1, 11, 12, 13]
    mask = get_mask(parsing, face_classes)
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, 2)
    mask = mask.astype(np.float)
    result = (1 - mask) * cv2.resize(source, (512, 512)) + mask * cv2.resize(target, (512, 512))
    result = cv2.resize(result.astype(np.uint8), (source.shape[1], source.shape[0]))
    return result,mask



def swap_regions(source, target, net):
    parsing = image_to_parsing(source, net)  #13ms
    face_classes = [1, 11, 12, 13]
    mask = get_mask(parsing, face_classes)
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, 2)
    result = (1 - mask) * cv2.resize(source, (512, 512)) + mask * cv2.resize(target, (512, 512))
    result = cv2.resize(result.astype(np.uint8), (source.shape[1], source.shape[0]))
    mask = cv2.resize(mask.astype(np.uint8), (source.shape[1], source.shape[0]))
    return result,mask
