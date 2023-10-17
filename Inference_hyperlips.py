import cv2, os, sys, argparse, audio
import subprocess, random, string
from tqdm import tqdm
import torch, face_detection
from models.model_hyperlips import HyperLipsBase,HyperLipsHR
from GFPGAN import *
from face_parsing import init_parser, swap_regions_img
import shutil


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using HyperLipsBase or HyperLipsHR models')

parser.add_argument('--checkpoint_path_BASE', type=str,help='Name of saved HyperLipsBase checkpoint to load weights from', default="checkpoints/hyperlipsbase_mead.pth")
parser.add_argument('--checkpoint_path_HR', type=str,help='Name of saved HyperLipsHR checkpoint to load weights from', default="checkpoints/hyperlipshr_mead_128.pth")
parser.add_argument('--modelname', type=str,
                    help='Choosing HyperLipsBase or HyperLipsHR', default="HyperLipsHR")
parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', default="test/video/M003-002.mp4")
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', default="test/audio/M003-002.mp4")
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='results/result_voice.mp4')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--filter_window', default=2, type=int,
                    help='real window is 2*T+1')
parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=8)
parser.add_argument('--hyper_batch_size', type=int, help='Batch size for hyperlips model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument('--segmentation_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/face_segmentation.pth")
parser.add_argument('--face_enhancement_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/GFPGANv1.3.pth")
parser.add_argument('--no_faceenhance', default=True, action='store_true',
					help='Prevent using face enhancement')
parser.add_argument('--gpu_id', type=float, help='gpu id (default: 0)',
                    default=0, required=False)
args = parser.parse_args()
args.img_size = 128


def get_smoothened_mels(mel_chunks, T):
    for i in range(len(mel_chunks)):
        if i > T-1 and i<len(mel_chunks)-T:
            window = mel_chunks[i-T: i + T]
            mel_chunks[i] = np.mean(window, axis=0)
        else:
            mel_chunks[i] = mel_chunks[i]
        
    return mel_chunks

def face_detect(images, detector,pad):
    batch_size = 16
    if len(images) > 1:
        print('error')
        raise RuntimeError('leng(imgaes')
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(
                    detector.get_detections_for_batch(np.array(images[i:i + batch_size])))  
        except RuntimeError as e:
            print(e)
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pad  # [0, 10, 0, 0]
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(mels, detector,face_path, resize_factor):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    bbox_face, frame_to_det_list, rects, frame_to_det_batch = [], [], [], []
    img_size = 128
    hyper_batch_size = args.hyper_batch_size
    reader = read_frames(face_path, resize_factor)
    for i, m in enumerate(mels):
        try:
            frame_to_save = next(reader)
        except StopIteration:
            reader = read_frames(face_path, resize_factor)
            frame_to_save = next(reader)
        h, w, _ = frame_to_save.shape
        face, coords = face_detect([frame_to_save], detector,args.pads)[0] 
        face = cv2.resize(face, (img_size, img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= hyper_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_HyperLipsHR(path,path_hr,device):
    model = HyperLipsHR(window_T =args.filter_window ,rescaling=1,base_model_checkpoint=path,HRDecoder_model_checkpoint =path_hr)
    model = model.to(device)
    print("HyperLipsHR model loaded")
    return model.eval()

def load_HyperLipsBase(path, device):
    model = HyperLipsBase()
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    model.load_state_dict(s)
    model = model.to(device)
    print("HyperLipsBase model loaded")
    return model.eval()

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

class Hyperlips():
    def __init__(self):
        self.checkpoint_path_BASE = args.checkpoint_path_BASE
        self.checkpoint_path_HR = args.checkpoint_path_HR
        self.parser_path = args.segmentation_path
        self.batch_size = args.hyper_batch_size #128
        self.mel_step_size = 16

    def _HyperlipsLoadModels(self):
        gpu_id = args.gpu_id
        if not torch.cuda.is_available() or (gpu_id > (torch.cuda.device_count() - 1)):
            raise ValueError(
                f'Existing gpu configuration problem.(gpu.is_available={torch.cuda.is_available()}| gpu.device_count={torch.cuda.device_count()})')
        self.device = torch.device(f'cuda:{gpu_id}')
        print('Using {} for inference.'.format(self.device))
        self.restorer = GFPGANInit(self.device, args.face_enhancement_path)
        if args.modelname == "HyperLipsBase":
            self.model = load_HyperLipsBase(self.checkpoint_path_BASE, self.device)
        elif args.modelname == "HyperLipsHR":
            self.model = load_HyperLipsHR(self.checkpoint_path_BASE, self.checkpoint_path_HR,self.device)
        self.seg_net = init_parser(self.parser_path, self.device)
        print(' models init successed...')

    def _HyperlipsInference(self):
        face = args.face
        audiopath = args.audio
        print("The input video path is {}， The output audio path is {}".format(face, audiopath))

        outfile = args.outfile
        outfile = os.path.abspath(outfile)
        rest_root_path = os.path.dirname(os.path.realpath(outfile))
        temp_save_path = outfile.rsplit('.', 1)[0]
        # rest_root_path = '/'.join(outfile.split('/')[:-1])
        # temp_save_path = os.path.join(rest_root_path, outfile.split('/')[-1][:-4])
        if not os.path.exists(rest_root_path):
            os.mkdir(rest_root_path)
        if not os.path.exists(temp_save_path):
            os.mkdir(temp_save_path)
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False, device='cuda:{}'.format(args.gpu_id))

        if not os.path.isfile(face):
            raise ValueError('--face argument must be a valid path to video/image file')
        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_stream.release()

        out = cv2.VideoWriter(os.path.join(temp_save_path, 'result.avi'), cv2.VideoWriter_fourcc(*'DIVX'),
                                      fps, (frame_width, frame_height))

        if not audiopath.endswith('.wav'):
            print('Extracting raw audio...')

            command = 'ffmpeg -y -i {} -strict -2 {}'.format(
                audiopath, os.path.join(temp_save_path, 'temp.wav'))
            subprocess.call(command, shell=True)
            audiopath = os.path.join(temp_save_path, 'temp.wav')
        wav = audio.load_wav(audiopath, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1
        if not (args.filter_window == None):
            mel_chunks = get_smoothened_mels(mel_chunks,T=args.filter_window)
        print("Length of mel chunks: {}".format(len(mel_chunks)))

        gen = datagen(mel_chunks, detector, face, args.resize_factor)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(
                                                                            np.ceil(
                                                                                float(len(mel_chunks))/ self.batch_size)))):

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)#([122, 6, 96, 96])
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)  # mel_batch([122, 1, 80, 16]) img_batch([128, 6, 128, 128])
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                mask_temp = np.zeros_like(f)
                p = p.cpu().numpy().transpose(1,2,0) * 255.

                if not args.no_faceenhance: 
                    ori_f = f.copy()
                    p = cv2.resize(p, (x2 - x1, y2 - y1)).astype(np.uint8)
                    f[y1:y2, x1:x2] = p
                    Code_img = GFPGANInfer(f, self.restorer, aligned=False)  # 33ms
                    p,mask_out = swap_regions_img(ori_f[y1:y2, x1:x2], Code_img[y1:y2, x1:x2], self.seg_net)
                    p = cv2.resize(p, (x2 - x1, y2 - y1)).astype(np.uint8)
                    mask_out = cv2.resize(mask_out.astype(np.float)*255.0, (x2 - x1, y2 - y1)).astype(np.uint8)
                    f[y1:y2, x1:x2] = p
                        
                else:
                    p,mask_out = swap_regions_img(f[y1:y2, x1:x2], p, self.seg_net)
                    p = cv2.resize(p, (x2 - x1, y2 - y1)).astype(np.uint8)
                    mask_out = cv2.resize(mask_out.astype(np.float)*255.0, (x2 - x1, y2 - y1)).astype(np.uint8)
                        

                mask_temp[y1:y2, x1:x2] = mask_out
                kernel = np.ones((5,5),np.uint8)  
                mask_temp = cv2.erode(mask_temp,kernel,iterations = 1)
                mask_temp = cv2.GaussianBlur(mask_temp, (75, 75), 0,0,cv2.BORDER_DEFAULT) 
                f_background = f.copy()
                f[y1:y2, x1:x2] = p
                f = f_background*(1-mask_temp/255.0)+f*(mask_temp/255.0)
                f = f.astype(np.uint8)
                out.write(f)

        out.release()
        outfile_dfl = os.path.join(rest_root_path, args.outfile.split('/')[-1]) 
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
            audiopath, os.path.join(temp_save_path, 'result.avi'), outfile_dfl)
        subprocess.call(command, shell=True)
        if os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
