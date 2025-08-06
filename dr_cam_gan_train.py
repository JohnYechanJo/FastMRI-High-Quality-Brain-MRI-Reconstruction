import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from cnn_model import DRCAMGenerator, PatchDiscriminator
from cnn_dataset import MRIDataset
from cnn_utils import VGGPerceptualLoss, visualize_images
from cnn_utils import calculate_metrics  # NMSE는 사용하지 않음
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing as mp

def train_model(generator, discriminator, train_loader, device, num_epochs=50):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=4e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    mse_loss = nn.MSELoss()
    vgg_loss = VGGPerceptualLoss(device)
    lambda_pixel, lambda_vgg, lambda_adv = 1.0, 0.1, 0.01  # (이미지 도메인)

    # ReduceLROnPlateau 스케줄러 설정
    scheduler_g = ReduceLROnPlateau(g_optimizer, mode='min', factor=0.1, patience=2)
    scheduler_d = ReduceLROnPlateau(d_optimizer, mode='min', factor=0.1, patience=2)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            d_optimizer.zero_grad() # Discriminator 업데이트 초기화
            # Discriminator 학습
            real_output = discriminator(targets) # targets는 실제 이미지 #범위: [-1, 1]
            fake_output = discriminator(generator(inputs).detach()) # generator의 출력을 사용하여 가짜 이미지 생성 후 detach
            # Discriminator 손실 계산
            # real_output은 실제 이미지에 대한 판별 결과, fake_output은 가짜 이미지에 대한 판별 결과
            # Binary Cross Entropy 손실을 사용하여 Discriminator 손실 계산
            # real_output과 fake_output에 대한 손실을 합산하여 최종 손실 계산
            # real_output에 대한 손실은 1로, fake_output에 대한 손실은 0으로 설정
            # 이진 교차 엔트로피 손실을 사용하여 실제 이미지와 가짜 이미지에 대한 판별 결과를 비교
            d_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output)) + \
                     F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
            d_loss.backward()
            d_optimizer.step()

        
            g_optimizer.zero_grad()
            fake_imgs = generator(inputs)
            fake_output = discriminator(fake_imgs)
            pixel_loss = mse_loss(fake_imgs, targets)
            vgg_loss_val = vgg_loss(fake_imgs, targets)
            adv_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
            g_loss = lambda_pixel * pixel_loss + lambda_vgg * vgg_loss_val + lambda_adv * adv_loss
            g_loss.backward()
            g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()  # 누락된 업데이트 추가

            # 로그 및 시각화
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}: G_Loss {g_loss.item():.4f}, D_Loss {d_loss.item():.4f}")
                visualize_images(inputs.cpu().numpy(), fake_imgs.detach().cpu().numpy(), targets.cpu().numpy(), epoch,
                                 save_dir='/home/compu/john/MAI-LAB/visualizations/train', batch_idx=batch_idx, dataset_name='train')

        # 에포크 종료
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)

        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)  # Discriminator 손실 기준

        # 학습률 모니터링 (수동 출력)
        current_lr_g = g_optimizer.param_groups[0]['lr']
        current_lr_d = d_optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{num_epochs}] Avg G_Loss: {avg_g_loss:.4f}, LR_G: {current_lr_g:.6f}, LR_D: {current_lr_d:.6f}')
        # Save checkpoint
        torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch}.pth')

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