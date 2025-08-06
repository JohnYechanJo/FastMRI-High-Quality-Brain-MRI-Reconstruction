import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from wgan import DRCAMGenerator
from cnn_dataset import MRIDataset
from cnn_utils import VGGPerceptualLoss, visualize_images
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing as mp

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([128, 192, 192])  # BatchNorm 대신 LayerNorm
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_model(generator, discriminator, train_loader, device, num_epochs=10):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=4e-4, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.9))
    mse_loss = nn.MSELoss()
    vgg_loss = VGGPerceptualLoss(device)
    lambda_pixel, lambda_vgg, lambda_adv, lambda_gp = 1.0, 0.1, 0.01, 10.0
    n_critic = 5

    scheduler_g = ReduceLROnPlateau(g_optimizer, mode='min', factor=0.1, patience=2)
    scheduler_d = ReduceLROnPlateau(d_optimizer, mode='min', factor=0.1, patience=2)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Discriminator 학습
            for _ in range(n_critic):
                d_optimizer.zero_grad()
                fake_imgs = generator(inputs).detach()
                real_validity = discriminator(targets)
                fake_validity = discriminator(fake_imgs)
                gradient_penalty = compute_gradient_penalty(discriminator, targets, fake_imgs, device)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                d_loss.backward()
                d_optimizer.step()

            # Generator 학습
            g_optimizer.zero_grad()
            fake_imgs = generator(inputs)
            fake_validity = discriminator(fake_imgs)
            pixel_loss = mse_loss(fake_imgs, targets)
            vgg_loss_val = vgg_loss(fake_imgs, targets)
            adv_loss = -torch.mean(fake_validity)  # Wasserstein 손실
            g_loss = lambda_pixel * pixel_loss + lambda_vgg * vgg_loss_val + lambda_adv * adv_loss
            g_loss.backward()
            g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # 로그 및 시각화
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}: G_Loss {g_loss.item():.4f}, D_Loss {d_loss.item():.4f}")
                visualize_images(inputs.cpu().numpy(), fake_imgs.detach().cpu().numpy(), targets.cpu().numpy(), epoch,
                                 save_dir='/home/compu/john/MAI-LAB/visualizations/train', batch_idx=batch_idx, dataset_name='train')

        # 에포크 종료
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)

        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)

        current_lr_g = g_optimizer.param_groups[0]['lr']
        current_lr_d = d_optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{num_epochs}] Avg G_Loss: {avg_g_loss:.4f}, LR_G: {current_lr_g:.6f}, LR_D: {current_lr_d:.6f}')

        # 체크포인트 저장
        torch.save(generator.state_dict(), f'checkpoints/wgan_generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'checkpoints/wgan_discriminator_epoch_{epoch}.pth')

def main():
    data_dir = '/home/compu/john/MAI-LAB/brain/train/image'
    batch_size = 16
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('checkpoints', exist_ok=True)
    transform = None
    train_dataset = MRIDataset(data_dir, mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Number of batches per epoch: {len(train_loader)}")

    generator = DRCAMGenerator(in_channels=2, out_channels=2).to(device)
    discriminator = PatchDiscriminator(in_channels=2).to(device)
    train_model(generator, discriminator, train_loader, device, num_epochs)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()