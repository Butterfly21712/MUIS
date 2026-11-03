import argparse
from pathlib import Path
import re
import cv2
import numpy as np
import os
import torch
from geometry.projection import numpy_to_torch_image, torch_to_numpy_image, warp_image
from geometry.alignment import mulpic
import torch.nn.functional as F
from composition.fusion import Fusion
import glob
import time
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    image_paths = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}], key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', p.name)])
    video_num = len(image_paths)
    if video_num < 2:
        raise ValueError('At least two images are required')
    refi = video_num // 2
    print(f'使用参考图像索引: {refi}')
    first_frames = []
    frames = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
            resized = cv2.resize(frame, (args.width, args.height))
            first_frames.append(resized)
        else:
            raise ValueError(f'Cannot read {path}')
    v_all, u_all, m_v1_, m_v0_, m_u1_, m_u0_, mosaich, mosaicw = mulpic(video_num, first_frames, refi, 'xfeat')
    mosaich, mosaicw = (int(mosaich), int(mosaicw))
    u0_ronhe = []
    u1_ronhe = []
    for i in range(2):
        u0_ronhe.append(m_u0_[i])
        u1_ronhe.append(m_u1_[i])
    im_p = [None] * video_num
    ronhe = [None] * video_num
    mask = [None] * video_num
    for i in range(video_num):
        ronhe[i] = torch.zeros((3, mosaich, mosaicw), device=device)
    for ki in range(video_num):
        i = (ki + refi) % video_num
        im_p[i] = warp_image(numpy_to_torch_image(first_frames[i]), u_all[ki], v_all[ki]).squeeze(0)
        ronhe[i][:, m_v0_[i]:m_v1_[i] + 1, m_u0_[i]:m_u1_[i] + 1] = im_p[i]
        mask_i = (ronhe[i].abs().sum(dim=0) > 0).float()
        mask[i] = mask_i.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, 5, 5), device=mask[i].device)
        dilated = F.conv2d(mask[i], kernel, padding=2) > 0
        eroded = F.conv2d(dilated.float(), kernel, padding=2) == kernel.sum()
        mask[i] = eroded.squeeze().float()
    stitched_tensor = (ronhe[0].float() / 127.5 - 1.0).unsqueeze(0).to(device)
    stitched_mask = mask[0].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    for i in range(1, video_num):
        warp_tensor = (ronhe[i].float() / 127.5 - 1.0).unsqueeze(0).to(device)
        warp_mask = mask[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        learned_mask1, learned_mask2 = Fusion(stitched_tensor, warp_tensor, stitched_mask, warp_mask, args.checkpoint)
        stitched_tensor = (stitched_tensor + 1.0) * learned_mask1 + (warp_tensor + 1.0) * learned_mask2 - 1.0
        stitched_mask = torch.clamp(learned_mask1 + learned_mask2, 0, 1)
        stitched_image = ((stitched_tensor[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.output, np.clip(stitched_image, 0, 255).astype(np.uint8)):
        raise OSError('Cannot write panorama')
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--checkpoint', default=str(Path(__file__).parent / 'epoch050_newmodel.pth'))
    parser.add_argument('--output', default='results/panorama.jpg')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    main(parser.parse_args())
    end_time = time.time()
    print(f'总运行时间: {end_time - start_time:.2f} 秒')
