import torch
import torch.nn as nn
from torchvision.models import vgg16
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim_measure
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 2채널 데이터를 3채널로 변환 (채널 복제)
        if pred.size(1) == 2:
            pred = pred.repeat(1, 2, 1, 1)[:, :3, :, :]  # 2채널을 4채널로 확장 후 3채널 자름
        if target.size(1) == 2:
            target = target.repeat(1, 2, 1, 1)[:, :3, :, :]  # 동일하게 변환
        pred_vgg = self.vgg(pred)
        target_vgg = self.vgg(target)
        return self.mse(pred_vgg, target_vgg)

def calculate_metrics(pred, target):
    """
    Calculate PSNR and SSIM metrics between predicted and target images.
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf'), 1.0
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    # Use win_size=3 and specify channel_axis=2 for [H, W, C] format
    ssim = ssim_measure(pred, target, data_range=1.0, win_size=3, channel_axis=2, multichannel=True)
    return psnr, ssim

def visualize_images(inputs, fake_imgs, targets, epoch, save_dir='visualizations/valid', batch_idx=0, dataset_name='val'):
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(inputs.shape[0], 5)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f'{dataset_name} Images - Epoch {epoch} - Batch {batch_idx}', fontsize=16)
    for i in range(num_images):
        input_img = inputs[i]
        if input_img.ndim == 3:
            input_img = input_img[0]
        input_img = np.clip(input_img, 0, 1)

        fake_img = fake_imgs[i]
        if fake_img.ndim == 3:
            fake_img = fake_img[0]
        print(f"Raw fake_img[{i}] min: {fake_img.min()}, max: {fake_img.max()}")
        fake_img = np.clip(fake_img, 0, 1)  # targets와 동일한 방식
        print(f"Normalized fake_img[{i}] min: {fake_img.min()}, max: {fake_img.max()}")

        target_img = targets[i]
        if target_img.ndim == 3:
            target_img = target_img[0]
        target_img = np.clip(target_img, 0, 1)

        axes[0, i].imshow(input_img, cmap='gray')
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(fake_img, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')

        axes[2, i].imshow(target_img, cmap='gray')
        axes[2, i].set_title(f'Ground Truth {i+1}')
        axes[2, i].axis('off')

    for i in range(num_images, 5):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_epoch_{epoch}_batch_{batch_idx}.png'))
    plt.close()
    
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_images_c(inputs, fake_imgs, targets, epoch, save_dir='visualizations/valid', batch_idx=0, dataset_name='val'):
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(inputs.shape[0], 7)  # 7개 모드 지원
    modes = ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']  # 모드 리스트

    fig, axes = plt.subplots(3, 7, figsize=(21, 9))  # 3x7 그리드, 크기 조정
    fig.suptitle(f'{dataset_name} Images - Epoch {epoch} - Batch {batch_idx}', fontsize=16)
    for i in range(num_images):
        input_img = inputs[i]
        if input_img.ndim == 3:
            input_img = input_img[0]  # 첫 번째 채널 사용 (예: [C, H, W] -> [H, W])
        input_img = np.clip(input_img, 0, 1)

        fake_img = fake_imgs[i]
        if fake_img.ndim == 3:
            fake_img = fake_img[0]
        print(f"Raw fake_img[{i}] min: {fake_img.min()}, max: {fake_img.max()}")
        fake_img = np.clip(fake_img, 0, 1)
        print(f"Normalized fake_img[{i}] min: {fake_img.min()}, max: {fake_img.max()}")

        target_img = targets[i]
        if target_img.ndim == 3:
            target_img = target_img[0]
        target_img = np.clip(target_img, 0, 1)

        axes[0, i].imshow(input_img, cmap='gray')
        axes[0, i].set_title(f'Input {i+1} ({modes[i]})')
        axes[0, i].axis('off')

        axes[1, i].imshow(fake_img, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1} ({modes[i]})')  # 모드 이름 추가
        axes[1, i].axis('off')

        axes[2, i].imshow(target_img, cmap='gray')
        axes[2, i].set_title(f'Ground Truth {i+1}')
        axes[2, i].axis('off')

    for i in range(num_images, 7):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_epoch_{epoch}_batch_{batch_idx}.png'))
    plt.close()