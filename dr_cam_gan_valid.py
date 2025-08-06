import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from cnn_model import DRCAMGenerator
from cnn_dataset import MRIDataset
from cnn_utils import VGGPerceptualLoss, calculate_metrics, visualize_images
import numpy as np
import multiprocessing as mp
import random

def validate_model(generator, val_loader, device):
    """
    Validation 데이터로 모델 성능 평가 (PSNR, SSIM 계산)
    """
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Generator로 복원
            fake_imgs = generator(inputs)

            print(f"Batch {batch_idx} - Inputs: shape={inputs.shape}, min={inputs.min().item()}, max={inputs.max().item()}, mean={inputs.mean().item()}, std={inputs.std().item()}")
            print(f"Batch {batch_idx} - Fake_imgs: shape={fake_imgs.shape}, min={fake_imgs.min().item()}, max={fake_imgs.max().item()}, mean={fake_imgs.mean().item()}, std={fake_imgs.std().item()}")
            print(f"Batch {batch_idx} - Targets: shape={targets.shape}, min={targets.min().item()}, max={targets.max().item()}, mean={targets.mean().item()}, std={targets.std().item()}")

            # # 시각화 (첫 5개 샘플)
            # if batch_idx % 100 == 0:  # 매 100번째 배치마다 시각화
            #     visualize_images(
            #         inputs.cpu().numpy(),
            #         fake_imgs.cpu().numpy(),
            #         targets.cpu().numpy(),
            #         epoch=0,  # Validation에서는 epoch 정보가 없으므로 0으로 설정
            #         save_dir='visualizations/valid',
            #         batch_idx=batch_idx,
            #         dataset_name='val'
            #     )
                
            # 메트릭 계산 (배치 단위)
            for i in range(inputs.size(0)):
                pred_np = fake_imgs[i].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                target_np = targets[i].cpu().numpy().transpose(1, 2, 0)
                psnr_val, ssim_val = calculate_metrics(pred_np, target_np)
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1

            if batch_idx % 100 == 0:
                print(f'Validation Batch [{batch_idx}/{len(val_loader)}] Processed')

    # 평균 메트릭 계산
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    # 전체 배치 처리 후 5개 샘플 랜덤 선택 및 시각화
    dataset = val_loader.dataset
    selected_indices = random.sample(range(len(dataset)), 5)
    vis_inputs = []
    vis_fakes = []
    vis_targets = []
    with torch.no_grad():
        for idx in selected_indices:
            input_img, target_img = dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)  # 배치 차원 추가
            fake_img = generator(input_img)
            vis_inputs.append(input_img.squeeze(0).cpu())
            vis_fakes.append(fake_img.squeeze(0).cpu())
            vis_targets.append(target_img.cpu())

    vis_inputs = torch.stack(vis_inputs)  # [5, C, H, W]
    vis_fakes = torch.stack(vis_fakes)
    vis_targets = torch.stack(vis_targets)
    # 3x5 시각화 (visualize_images 함수가 3x5 그리드를 지원한다고 가정)
    visualize_images(
        vis_inputs.numpy(),
        vis_fakes.numpy(),
        vis_targets.numpy(),
        epoch=0,
        save_dir='visualizations/valid',
        batch_idx='summary',
        dataset_name='validation'
    )
    return avg_psnr, avg_ssim

def find_best_checkpoint(checkpoint_dir='checkpoints'):
    """
    체크포인트 디렉토리에서 최고 에포크를 찾아 반환
    """
    best_psnr = -float('inf')
    best_epoch = 0
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('generator_epoch_') and filename.endswith('.pth'):
            epoch = int(filename.split('_')[2].split('.')[0])
            with open('validation_metrics.txt', 'r') as f:
                for line in f:
                    if f'Epoch {epoch}' in line:
                        psnr = float(line.split('PSNR: ')[1].split(',')[0])
                        if psnr > best_psnr:
                            best_psnr = psnr
                            best_epoch = epoch
    return f'checkpoints/generator_epoch_{best_epoch}.pth' if best_epoch > 0 else None

def main():
    data_dir = '/home/compu/john/MAI-LAB/brain/val/image'
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 최고 에포크 체크포인트 자동 선택 (validation_metrics.txt 기반)
    checkpoint_dir = '/home/compu/john/MAI-LAB/checkpoints'
    model_path = find_best_checkpoint(checkpoint_dir)
    if not model_path:
        print("No valid checkpoint found or validation_metrics.txt not available. Using default epoch 0.")
        model_path = 'checkpoints/generator_epoch_9.pth'

    # 모델 로드 및 freeze
    generator = DRCAMGenerator(in_channels=2, out_channels=2).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False  # Freeze weights

    # 데이터 로드
    transform = None
    val_dataset = MRIDataset(data_dir, mode='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Number of validation batches: {len(val_loader)}")

    # 검증 실행
    avg_psnr, avg_ssim = validate_model(generator, val_loader, device)

    # 결과 출력 및 저장 (튜닝 피드백 제공)
    print(f'Validation Results - Model: {os.path.basename(model_path)}, '
          f'Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}')
    with open('validation_metrics.txt', 'a') as f:
        f.write(f'{os.path.basename(model_path)} - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

    # 튜닝 피드백 로그
    with open('validation_feedback.txt', 'a') as f:
        f.write(f'Model: {os.path.basename(model_path)}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
        f.write(f'Suggestion: Adjust lambda_adv if PSNR < 25, consider increasing epochs if SSIM < 0.9\n')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()