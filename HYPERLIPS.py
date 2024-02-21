import cv2, os, sys,audio
import subprocess, random, string
from tqdm import tqdm
import torch, face_detection
from models.model_hyperlips import HyperLips_inference
from GFPGAN import *
from face_parsing import init_parser,swap_regions
import shutil


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, detector,pad):
    batch_size = 8

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
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(mels, detector,frames,img_size,hyper_batch_size,pads):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    face_det_results = face_detect(frames,detector,pads)
    for i, m in enumerate(mels):
        frame_to_save = frames[i].copy()
        face, coords = face_det_results[i].copy()
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

    
def load_HyperLips(window,rescaling,path,path_hr,device):
    model = HyperLips_inference(window_T =window ,rescaling=rescaling,base_model_checkpoint=path,HRDecoder_model_checkpoint =path_hr)
    model = model.to(device)
    print("HyperLipsHR model loaded")
    return model.eval()
    
def main():
    Hyperlips_executor = Hyperlips()
    Hyperlips_executor._HyperlipsLoadModels()
    Hyperlips_executor._HyperlipsInference()

class Hyperlips():
    def __init__(self,checkpoint_path_BASE=None,
                 checkpoint_path_HR=None,
                 segmentation_path=None,
                 face_enhancement_path = None,
                 gpu_id = None,
                 window =None,
                 hyper_batch_size=128,
                 img_size = 128,
                 resize_factor = 1,
                 pad = [0, 10, 0, 0]
                 ):
        self.checkpoint_path_BASE = checkpoint_path_BASE
        self.checkpoint_path_HR = checkpoint_path_HR
        self.parser_path = segmentation_path
        self.face_enhancement_path = face_enhancement_path
        self.batch_size = hyper_batch_size #128
        self.mel_step_size = 16
        self.gpu_id = gpu_id
        self.img_size = img_size
        self.resize_factor = resize_factor
        self.pad =pad
        if (128==self.img_size):
            self.rescaling = 1
        elif(256==self.img_size):
             self.rescaling = 2
        elif(512==self.img_size):
            self.rescaling = 4
        else:
            raise ValueError(
                f'Init error! img_size should be 128 256 or 512!')
        self.window = window

    def _HyperlipsLoadModels(self):
        gpu_id = self.gpu_id
        if not torch.cuda.is_available() or (gpu_id > (torch.cuda.device_count() - 1)):
            raise ValueError(
                f'Existing gpu configuration problem.(gpu.is_available={torch.cuda.is_available()}| gpu.device_count={torch.cuda.device_count()})')
        self.device = torch.device(f'cuda:{gpu_id}')
        print('Using {} for inference.'.format(self.device))
        if self.face_enhancement_path is not None:
            self.restorer = GFPGANInit(self.device, self.face_enhancement_path)
        self.model = load_HyperLips(self.window,self.rescaling,self.checkpoint_path_BASE, self.checkpoint_path_HR,self.device)
        self.seg_net = init_parser(self.parser_path, self.device)
        print(' models init successed...')

    def _HyperlipsInference(self,face_path,audio_path,outfile_path):
        face = face_path
        audiopath =audio_path
        print("The input video path is {}， The intput audio path is {}".format(face_path, audio_path))

        outfile =outfile_path
        outfile = os.path.abspath(outfile)
        rest_root_path = os.path.dirname(os.path.realpath(outfile))
        temp_save_path = outfile.rsplit('.', 1)[0]
        if not os.path.exists(rest_root_path):
            os.mkdir(rest_root_path)
        if not os.path.exists(temp_save_path):
            os.mkdir(temp_save_path)
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False, device='cuda:{}'.format(self.gpu_id))

        if not os.path.isfile(face):
            raise ValueError('--face argument must be a valid path to video/image file')
        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//self.resize_factor, frame.shape[0]//self.resize_factor))
                full_frames.append(frame)
            video_stream.release()
        print ("Number of frames available for inference: "+str(len(full_frames)))
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
        print("Length of mel chunks: {}".format(len(mel_chunks)))
        full_frames = full_frames[:len(mel_chunks)]
        gen = datagen(mel_chunks, detector, full_frames, self.img_size,self.batch_size,self.pad)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(
                                                                            np.ceil(
                                                                                float(len(mel_chunks))/ self.batch_size)))):

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
            for p, f, c in zip(pred, frames, coords):

                y1, y2, x1, x2 = c
                mask_temp = np.zeros_like(f)
                p = p.cpu().numpy().transpose(1,2,0) * 255.
                f_background = f.copy()
                p,mask_out = swap_regions(f[y1:y2, x1:x2], p, self.seg_net) #
                p = cv2.resize(p, (x2 - x1, y2 - y1)).astype(np.uint8)
                mask_out=mask_out*255
                mask_out[:mask_out.shape[0]//2, :, :] = 0.
                mask_out[:,:int(mask_out.shape[1]*0.15),:] = 0.
                mask_out[:,int(mask_out.shape[1]*0.85):,:] = 0.
                mask_temp[y1:y2, x1:x2] = mask_out.astype(np.float)
                kernel = np.ones((5,5),np.uint8)  
                mask_temp = cv2.erode(mask_temp,kernel,iterations = 1)
                mask_temp = cv2.GaussianBlur(mask_temp, (75, 75), 0,0,cv2.BORDER_DEFAULT) 
                f[y1:y2, x1:x2] = p
                f = f_background*(1-mask_temp/255.0)+f*(mask_temp/255.0)
                if self.face_enhancement_path is not None:
                    Code_img = GFPGANInfer(f, self.restorer,aligned=False) 
                    f=Code_img
                f = f.astype(np.uint8)
                out.write(f)
                i = i+1

        out.release()
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
            audiopath, os.path.join(temp_save_path, 'result.avi'), outfile)
        subprocess.call(command, shell=True)
        if os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
