from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models.model_hyperlips import HRDecoder,HRDecoder_disc_qual
import audio
import lpips
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from torchvision.models.vgg import vgg19
from glob import glob
mseloss = nn.MSELoss()
import os, random, cv2, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from hparams_HR import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument("-hyperlips_trian_dataset", help="Root folder of the preprocessed LRS2 dataset", default='Train_data/HR_Train_Dateset')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="checkpoints_hyperlips_HR", type=str)
parser.add_argument('--batch_size', type=int, help='Batch size for hyperlips model(s)', default=28)
parser.add_argument('--img_size', type=int, help='imgsize for hyperlips model(s)', default=128)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16




class Dataset(object):
    def __init__(self, split):
        gt_img_root = os.path.join(args.hyperlips_trian_dataset,'GT_IMG')
        self.gt_img      =  get_image_list(gt_img_root,split) 




    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (args.img_size, args.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window


    def read_window_base(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (128, 128))
            except Exception as e:
                return None

            window.append(img)

        return window


    def read_window_sketch(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                if args.img_size == 128:
                    kenerl_size = 5
                elif args.img_size == 256:
                    kenerl_size = 7
                elif args.img_size == 512:
                    kenerl_size = 11
                else:
                    print("Please input rigtht img_size!")
                img = cv2.resize(img, (args.img_size, args.img_size))
                img = cv2.GaussianBlur(img, (kenerl_size, kenerl_size), 0,0,cv2.BORDER_DEFAULT)
                ret, img= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                cv2.imwrite("test_skech.png",img)
            except Exception as e:
                return None

            window.append(img)

        return window

    def read_window_sketch_base(self, window_fnames):
        if window_fnames is None: return None
        window = []
        img_size = 128
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                if img_size == 128:
                    kenerl_size = 5
                elif img_size == 256:
                    kenerl_size = 7
                elif img_size == 512:
                    kenerl_size = 11
                else:
                    print("Please input rigtht img_size!")
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.GaussianBlur(img, (kenerl_size, kenerl_size), 0,0,cv2.BORDER_DEFAULT)
                ret, img= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            except Exception as e:
                return None

            window.append(img)

        return window
    def read_coord(self,window_fnames):
        if window_fnames is None: return None

        coords =  []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (args.img_size, args.img_size))
            except Exception as e:
                return None
            index = np.argwhere(img[:,:,0] == 255)
            x_max =max(index[:,0])
            x_min =min(index[:,0])
            y_max =max(index[:,1])
            y_min =min(index[:,1])
            coords.append([x_min,x_max,y_min,y_max])
        return coords
    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.gt_img)

    def __getitem__(self, idx):
        while 1:
 
            idx = random.randint(0, len(self.gt_img) - 1)
            vidname = os.path.join(self.gt_img[idx].split('/')[-2],self.gt_img[idx].split('/')[-1])
            gt_img_root = os.path.join(args.hyperlips_trian_dataset,'GT_IMG')
            gt_sketch_data_root = os.path.join(args.hyperlips_trian_dataset,'GT_SKETCH')
            gt_mask_root = os.path.join(args.hyperlips_trian_dataset,'GT_MASK')
            hyper_img_root = os.path.join(args.hyperlips_trian_dataset,'HYPER_IMG')
            hyper_sketch_data_root = os.path.join(args.hyperlips_trian_dataset,'HYPER_SKETCH')

            gt_img_names       =   list(glob(join(gt_img_root,vidname, '*.jpg')))
            gt_sketch_names    =   list(glob(join(gt_sketch_data_root,vidname, '*.jpg')))
            gt_mask_names      =   list(glob(join(gt_mask_root,vidname, '*.jpg')))
            hyper_img_names    =   list(glob(join(hyper_img_root,vidname, '*.jpg')))
            hyper_sketch_names =   list(glob(join(hyper_sketch_data_root,vidname, '*.jpg')))
            if not(len(gt_img_names)==len(gt_sketch_names)==len(gt_mask_names)==len(hyper_img_names)==len(hyper_sketch_names)):
                continue
            if len(gt_img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(gt_img_names).split('/')[-1]
            gt_img_name        = join(gt_img_root,vidname,img_name)
            gt_sketch_name     = join(gt_sketch_data_root,vidname,img_name)
            gt_mask_name       = join(gt_mask_root,vidname,img_name)
            hyper_img_name     = join(hyper_img_root,vidname,img_name)
            hyper_sketch_name  = join(hyper_sketch_data_root,vidname,img_name)


            gt_img_name_window_frames         = self.get_window(gt_img_name)
            gt_sketch_name_window_frames      = self.get_window(gt_sketch_name)
            gt_mask_name_window_frames        = self.get_window(gt_mask_name)
            hyper_img_name_window_frames      = self.get_window(hyper_img_name)
            hyper_sketch_name_window_frames   = self.get_window(hyper_sketch_name)

            coords = self.read_coord(gt_mask_name_window_frames)
            
            if gt_img_name_window_frames is None :
                continue

            gt_img_window           =   self.read_window(gt_img_name_window_frames)
            gt_sketch_window        =   self.read_window_sketch(gt_sketch_name_window_frames)
            gt_mask_window          =   self.read_window(gt_mask_name_window_frames)
            hyper_img_window        =   self.read_window_base(hyper_img_name_window_frames)
            hyper_sketch_window     =   self.read_window_sketch_base(hyper_sketch_name_window_frames)


            gt_img_window          =   self.prepare_window(gt_img_window)
            gt_sketch_window       =   self.prepare_window(gt_sketch_window)
            gt_mask_window         =   self.prepare_window(gt_mask_window)
            hyper_img_window       =   self.prepare_window(hyper_img_window)
            hyper_sketch_window    =   self.prepare_window(hyper_sketch_window)

            gt_img          =   torch.FloatTensor(gt_img_window)
            gt_sketch       =   torch.FloatTensor(gt_sketch_window) 
            gt_mask         =   torch.FloatTensor(gt_mask_window)
            hyper_img       =   torch.FloatTensor(hyper_img_window)
            hyper_sketch    =   torch.FloatTensor(hyper_sketch_window)
            coords          =   torch.FloatTensor(coords)
            return gt_img, gt_sketch, gt_mask,hyper_img,hyper_sketch,coords



def save_sample_images(x, g, gt,m, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    m = (m.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((x, g, gt,m), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss   

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
recon_loss = nn.L1Loss()


def train(device, model, disc,train_data_loader, test_data_loader, optimizer,disc_optimizer, checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion = nn.L1Loss().to(device)
    perception_criterion = PerceptualLoss().to(device)

    
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_lip_c_loss, running_l1_loss, disc_loss, running_lip_l_loss = 0., 0., 0., 0.
        running_con_loss, running_mse_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (gt_img, gt_sketch, gt_mask,hyper_img,hyper_sketch,coords) in prog_bar:
            disc.train()
            model.train()
            hyper_img = hyper_img.to(device)
            hyper_sketch = hyper_sketch.to(device)
            gt_mask = gt_mask.to(device)
            gt_sketch = gt_sketch.to(device)
            gt_img = gt_img.to(device)
            B = hyper_img.size(0)

            
            input_dim_size = len(hyper_img.size())
            if input_dim_size > 4:
                hyper_img       = torch.cat([hyper_img[:, :, i] for i in range(hyper_img.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])
                hyper_sketch    = torch.cat([hyper_sketch[:, :, i] for i in range(hyper_sketch.size(2))], dim=0)
                gt_mask         = torch.cat([gt_mask[:, :, i] for i in range(gt_mask.size(2))], dim=0)
                gt_sketch       = torch.cat([gt_sketch[:, :, i] for i in range(gt_sketch.size(2))], dim=0)
                gt_img          = torch.cat([gt_img[:, :, i] for i in range(gt_img.size(2))], dim=0)
                coords_t = torch.cat([( coords)[ :, i] for i in range(coords.size(1))], dim=0)
            real_labels = torch.ones((gt_img.size()[0], 1)).to(device)#[4,1]
            fake_labels = torch.zeros((gt_img.size()[0], 1)).to(device)#[4,1]   
                
            input_temp = torch.cat((hyper_img,hyper_sketch), dim=1)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            optimizer.zero_grad()
            g = model(input_temp)

            lip_lpips_loss = 0
            lip_recons_loss_temp = 0
            for i in range(gt_img.shape[0]):
                x_min,x_max,y_min,y_max = int(coords_t[i,0]),int(coords_t[i,1]),int(coords_t[i,2]),int(coords_t[i,3])
                gt_t_i = gt_img[i,:,x_min:x_max,y_min:y_max]
                g_t_i = g[i,:,x_min:x_max,y_min:y_max]
                recons_loss_temp_i = recon_loss(g_t_i, gt_t_i)
                lip_recons_loss_temp = lip_recons_loss_temp+recons_loss_temp_i
                
                lpips_loss_i = loss_fn_vgg(g_t_i, gt_t_i)
                lip_lpips_loss = lip_lpips_loss+lpips_loss_i
            lip_lpips_loss = lip_lpips_loss/gt_img.shape[0]
            lip_recons_loss_temp = lip_recons_loss_temp/gt_img.shape[0]
            
            score_real = disc(gt_img)#[4,1]
            score_fake = disc(g)#[4,1]
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()

            adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
            adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            perceptual_loss = perception_criterion(gt_img, g)
            content_loss = content_criterion(g, gt_img)

            loss = adversarial_loss  + perceptual_loss  + content_loss +lip_lpips_loss+lip_recons_loss_temp

            loss.backward()
            optimizer.step()

            ##########################
             # training discriminator #
            ##########################            


            disc_optimizer.zero_grad()

            score_real = disc(gt_img)
            score_fake = disc(g.detach())
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()

            adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
            adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
            discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            discriminator_loss.backward()
            disc_optimizer.step()

            if global_step % checkpoint_interval == 0:
                hyper_img_temp = torch.nn.functional.interpolate(hyper_img,(gt_img.size()[2], gt_img.size()[3]), mode='bilinear', align_corners=False)
                hyper_sketch_temp = torch.nn.functional.interpolate(hyper_sketch,(gt_img.size()[2], gt_img.size()[3]), mode='bilinear', align_corners=False)
                if input_dim_size > 4:#训练时输入为5维，测试时输入为4维（把T与B进行了合并）
                    output = torch.split(g, B, dim=0) 
                    outputs1 = torch.stack(output, dim=2) 
                    
                    hyper_img_temp = torch.split(hyper_img_temp, B, dim=0) 
                    hyper_img_temp = torch.stack(hyper_img_temp, dim=2) 
                    
                    hyper_sketch_temp = torch.split(hyper_sketch_temp, B, dim=0) 
                    hyper_sketch_temp = torch.stack(hyper_sketch_temp, dim=2) 
                    
                    gt_img = torch.split(gt_img, B, dim=0) 
                    gt_img = torch.stack(gt_img, dim=2) 
                else:
                    outputs1 = output

                save_sample_images(hyper_img_temp, hyper_sketch_temp, outputs1,gt_img, global_step, checkpoint_dir)

            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step                

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)  
                save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')             
            
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr_disc = disc_optimizer.state_dict()['param_groups'][0]['lr']
            running_l1_loss+= adversarial_loss.item()
            running_mse_loss+= perceptual_loss.item()
            running_con_loss+= content_loss.item()
            running_lip_c_loss+= lip_recons_loss_temp.item()
            running_lip_l_loss +=lip_lpips_loss.item()#+lip_recons_loss_temp
            
            
            
            disc_loss+= discriminator_loss.item()

            prog_bar.set_description('ad_loss: {}, perc_loss: {},cont_loss: {},lipc_loss: {},lipl_loss: {},disc_loss: {}'.format(running_l1_loss / (step + 1),
                                                                                        running_mse_loss / (step + 1),
                                                                                        running_con_loss / (step + 1),
                                                                                        running_lip_c_loss / (step + 1),
                                                                                        running_lip_l_loss / (step + 1),
                                                                                        
                                                                                        disc_loss / (step + 1),
                                                                                        # running_disc_fake_loss / (step + 1),
                                                                                        # running_disc_real_loss / (step + 1)
                                                                                        ))
        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
    while 1:
        for step, (x, indiv_mels, mel, gt,m,coords) in enumerate((test_data_loader)):
        # for step, (x, indiv_mels, mel, gt,m,coords) in prog_bar:
            model.eval()
            # disc.eval()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)



            g = model(indiv_mels, x)



            l1loss = recon_loss(g, gt)



            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            


            if step > eval_steps: break

        print('L1: {}, Sync: {}'.format(sum(running_l1_loss) / len(running_l1_loss),
                                                            sum(running_sync_loss) / len(running_sync_loss),
                                                            # sum(running_perceptual_loss) / len(running_perceptual_loss),
                                                            # sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                                                            #  sum(running_disc_real_loss) / len(running_disc_real_loss)
                                                             ))
        return sum(running_sync_loss) / len(running_sync_loss)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s,strict=False)
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    
    if args.img_size==512:
        rescaling = 4
    elif args.img_size==256:
        
        rescaling = 2
    else:
        rescaling = 1
    model = HRDecoder(rescaling)
    if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
    model = model.to(device)

    disc = HRDecoder_disc_qual()
    if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                disc = nn.DataParallel(disc)
    disc = disc.to(device)




    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, reset_optimizer=False)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, reset_optimizer=False, overwrite_global_states=False)
        


    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))

    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))


    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model,disc, train_data_loader, test_data_loader, optimizer,disc_optimizer, checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)