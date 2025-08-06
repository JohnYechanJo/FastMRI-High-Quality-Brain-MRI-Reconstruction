import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from cnn_dataset import MRIDataset
from cnn_utils import VGGPerceptualLoss, visualize_images
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing as mp

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(-1).unsqueeze(-1)
        return x * y

class RefinementBlock(nn.Module):
    """SRGAN 스타일의 Residual Block으로 디테일 정제"""
    def __init__(self, channels):
        super(RefinementBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class DRCAMGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(DRCAMGenerator, self).__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc1 = DilatedResidualBlock(64, 64, dilation=1)
        self.enc2 = DilatedResidualBlock(64, 128, dilation=2)
        self.enc3 = DilatedResidualBlock(128, 256, dilation=3)
        self.enc4 = DilatedResidualBlock(256, 256, dilation=3)
        self.pool = nn.MaxPool2d(2, 2, padding=0)
        self.cam1 = ChannelAttention(64)
        self.cam2 = ChannelAttention(128)
        self.cam3 = ChannelAttention(256)
        self.cam4 = ChannelAttention(256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DilatedResidualBlock(256, 128, dilation=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DilatedResidualBlock(128, 64, dilation=1)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        # Refinement 모듈 추가
        self.refine1 = RefinementBlock(out_channels)
        self.refine2 = RefinementBlock(out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor with shape {x.size()}")
        if x.size(2) < 4 or x.size(3) < 4:
            padding_h = max(0, 4 - x.size(2))
            padding_w = max(0, 4 - x.size(3))
            x = F.pad(x, (0, padding_w, 0, padding_h), mode='constant', value=0)

        e1 = F.relu(self.init_conv(x))
        e1 = self.enc1(e1)
        e1 = self.cam1(e1)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        e2 = self.cam2(e2)
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        e3 = self.cam3(e3)
        e4 = self.enc4(e3)
        e4 = self.cam4(e4)
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = torch.tanh(self.out_conv(d2))
        # Refinement 단계
        out = F.relu(self.refine1(out))
        out = F.relu(self.refine2(out))
        out = torch.tanh(self.final_conv(out))
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([128, 192, 192])
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.ln3 = nn.LayerNorm([256, 96, 96])
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm([512, 48, 48])
        self.out_conv = nn.Conv2d(512, 1, 3, stride=1, padding=1)

    def forward(self, x):
        if x.dim() != 4 or x.size(2) < 3 or x.size(3) < 3:
            raise ValueError(f"Expected 4D tensor with height and width >= 3, got {x.size()}")
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.ln2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.ln3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.ln4(self.conv4(x)), 0.2)
        x = self.out_conv(x)
        return x

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Gradient Penalty 계산"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    if torch.isnan(gradients).any() or torch.isinf(gradients).any():
        print("Warning: NaN or Inf in gradients!")
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    print(f"Gradient Penalty: {gradient_penalty.item():.4f}")
    return gradient_penalty

def weights_init(m):
    """모델 가중치 초기화"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_wgan_gp():
    # 하이퍼파라미터
    batch_size = 16
    lr_g = 0.0002
    lr_d = 0.00005
    beta1 = 0.5
    n_epochs = 50
    n_critic = 3
    lambda_gp = 5
    lambda_pixel = 1.0
    lambda_vgg = 0.1
    lambda_adv = 0.1
    lambda_content = 0.1  # Content Loss 추가
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 및 데이터로더
    data_dir = '/home/compu/john/MAI-LAB/brain/train/image'
    dataset = MRIDataset(data_dir=data_dir, mode='train', transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 모델 및 옵티마이저
    generator = DRCAMGenerator(in_channels=2, out_channels=2).to(device)
    discriminator = PatchDiscriminator(in_channels=2).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.9))

    # 손실 함수 및 스케줄러
    mse_loss = nn.MSELoss()
    vgg_loss = VGGPerceptualLoss(device)
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=2)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=2)

    # 출력 디렉토리
    save_dir = 'visualizations/wgan_gp'
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = 'checkpoints/wgan_gp'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 학습 루프
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 데이터 정규화 확인
            print(f"Inputs: min={inputs.min().item()}, max={inputs.max().item()}, mean={inputs.mean().item()}")
            print(f"Targets: min={targets.min().item()}, max={targets.max().item()}, mean={targets.mean().item()}")

            # Discriminator 학습
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                fake_imgs = generator(inputs).detach()
                real_validity = discriminator(targets)
                fake_validity = discriminator(fake_imgs)
                gradient_penalty = compute_gradient_penalty(discriminator, targets, fake_imgs, device)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizer_D.step()

            # Generator 학습
            optimizer_G.zero_grad()
            fake_imgs = generator(inputs)
            fake_validity = discriminator(fake_imgs)
            pixel_loss = mse_loss(fake_imgs, targets)
            vgg_loss_val = vgg_loss(fake_imgs, targets)
            adv_loss = -torch.mean(fake_validity)
            # Content Loss 추가 (VGG의 중간 레이어 활용)
            content_loss = vgg_loss(fake_imgs, targets)
            g_loss = lambda_pixel * pixel_loss + lambda_vgg * vgg_loss_val + lambda_adv * adv_loss + lambda_content * content_loss
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # 진행 상황 출력 및 시각화
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                visualize_images(
                    inputs.cpu().numpy(),
                    fake_imgs.detach().cpu().numpy(),
                    targets.cpu().numpy(),
                    epoch=epoch,
                    save_dir=save_dir,
                    batch_idx=i,
                    dataset_name='wgan_train'
                )

        # 에포크 종료
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)

        scheduler_G.step(avg_g_loss)
        scheduler_D.step(avg_d_loss)

        current_lr_g = optimizer_G.param_groups[0]['lr']
        current_lr_d = optimizer_D.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{n_epochs}] Avg G_Loss: {avg_g_loss:.4f}, LR_G: {current_lr_g:.6f}, LR_D: {current_lr_d:.6f}')

        # 체크포인트 저장
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'wgan_generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'wgan_discriminator_epoch_{epoch}.pth'))

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    train_wgan_gp()