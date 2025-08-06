import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from wgan import DRCAMGenerator
from cnn_dataset import MRIDataset
from cnn_utils import VGGPerceptualLoss, calculate_metrics, visualize_images
import numpy as np
import multiprocessing as mp
import random

def test_model(generator, test_loader, device):
    """
    Test 데이터로 모델 성능 평가 (PSNR, SSIM 계산)
    """
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Generator로 복원
            fake_imgs = generator(inputs)

            print(f"Batch {batch_idx} - Inputs: shape={inputs.shape}, min={inputs.min().item()}, max={inputs.max().item()}, mean={inputs.mean().item()}, std={inputs.std().item()}")
            print(f"Batch {batch_idx} - Fake_imgs: shape={fake_imgs.shape}, min={fake_imgs.min().item()}, max={fake_imgs.max().item()}, mean={fake_imgs.mean().item()}, std={fake_imgs.std().item()}")
            print(f"Batch {batch_idx} - Targets: shape={targets.shape}, min={targets.min().item()}, max={targets.max().item()}, mean={targets.mean().item()}, std={targets.std().item()}")

            # 시각화 (매 100번째 배치)
            if batch_idx % 100 == 0 or batch_idx == 0:
                vis_inputs = np.abs(inputs.cpu().numpy())  # 복소수 절대값 적용
                vis_fakes = np.abs(fake_imgs.cpu().numpy())
                vis_targets = np.abs(targets.cpu().numpy())
                visualize_images(
                    vis_inputs,
                    vis_fakes,
                    vis_targets,
                    epoch=0,
                    save_dir='/home/compu/john/MAI-LAB/visualizations/wgan_test',
                    batch_idx=batch_idx,
                    dataset_name='wgan_test'
                )

            # 메트릭 계산 (배치 단위)
            for i in range(inputs.size(0)):
                pred_np = fake_imgs[i].cpu().numpy().transpose(1, 2, 0)
                target_np = targets[i].cpu().numpy().transpose(1, 2, 0)
                psnr_val, ssim_val = calculate_metrics(pred_np, target_np)
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1

            if batch_idx % 100 == 0:
                print(f'Test Batch [{batch_idx}/{len(test_loader)}] Processed')

    if num_samples == 0:
        print("Error: No test samples processed. Check test dataset.")
        return 0.0, 0.0

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    # 전체 배치 처리 후 5개 샘플 랜덤 선택 및 시각화
    dataset = test_loader.dataset
    selected_indices = random.sample(range(len(dataset)), 5)
    vis_inputs = []
    vis_fakes = []
    vis_targets = []
    with torch.no_grad():
        for idx in selected_indices:
            input_img, target_img = dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)
            fake_img = generator(input_img)
            vis_inputs.append(input_img.squeeze(0).cpu())
            vis_fakes.append(fake_img.squeeze(0).cpu())
            vis_targets.append(target_img.cpu())

    vis_inputs = torch.stack(vis_inputs)
    vis_fakes = torch.stack(vis_fakes)
    vis_targets = torch.stack(vis_targets)

    # 3x5 시각화
    vis_inputs = np.abs(vis_inputs.cpu().numpy())  # 복소수 절대값 및 CPU 이동
    vis_fakes = np.abs(vis_fakes.cpu().numpy())
    vis_targets = np.abs(vis_targets.cpu().numpy())  # targets.cpu() 추가
    visualize_images(
        vis_inputs,
        vis_fakes,
        vis_targets,
        epoch=0,
        save_dir='/home/compu/john/MAI-LAB/visualizations/wgan_test',
        batch_idx='summary',
        dataset_name='wgan_test'
    )
    return avg_psnr, avg_ssim

def find_best_checkpoint(checkpoint_dir='/home/compu/john/MAI-LAB/checkpoints'):
    """
    체크포인트 디렉토리에서 최고 에포크를 찾아 반환
    """
    best_epoch = 0
    try:
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('wgan_generator_epoch_') and filename.endswith('.pth'):
                epoch = int(filename.split('_')[3].split('.')[0])
                if epoch > best_epoch:
                    best_epoch = epoch
        return f'/home/compu/john/MAI-LAB/checkpoints/wgan_generator_epoch_{best_epoch}.pth' if best_epoch > 0 else None
    except FileNotFoundError:
        print("No checkpoints found in directory. Using default epoch 2.")
        return '/home/compu/john/MAI-LAB/checkpoints/wgan_generator_epoch_2.pth'

def main():
    data_dirs = [
        '/home/compu/john/MAI-LAB/brain/leaderboard/acc4/image',
        '/home/compu/john/MAI-LAB/brain/leaderboard/acc8/image'
    ]
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 최고 에포크 체크포인트 자동 선택
    checkpoint_dir = '/home/compu/john/MAI-LAB/checkpoints'
    model_path = find_best_checkpoint(checkpoint_dir)
    if not model_path:
        model_path = '/home/compu/john/MAI-LAB/checkpoints/wgan_generator_epoch_2.pth'

    # 모델 로드 및 freeze
    generator = DRCAMGenerator(in_channels=2, out_channels=2).to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded checkpoint: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint {model_path} not found.")
        return
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Attempting to load compatible keys only...")
        checkpoint = torch.load(model_path, map_location=device)
        model_dict = generator.state_dict()
        checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint_dict)
        generator.load_state_dict(model_dict)
        print(f"Loaded compatible keys from checkpoint: {model_path}")
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False  # Freeze weights

    # 각 데이터 경로에 대해 테스트
    for data_dir in data_dirs:
        print(f"\nProcessing data directory: {data_dir}")

        # 데이터 디렉토리 확인
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} does not exist.")
            continue
        files = os.listdir(data_dir)
        if not files:
            print(f"Error: Data directory {data_dir} is empty.")
            continue
        print(f"Found {len(files)} files in {data_dir}")

        # 데이터 로드
        transform = None
        test_dataset = MRIDataset(data_dir, mode='test', transform=transform)
        print(f"Test dataset size: {len(test_dataset)}")
        if len(test_dataset) == 0:
            print(f"Error: Test dataset is empty for {data_dir}. Check MRIDataset implementation.")
            continue

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"Number of test batches: {len(test_loader)}")

        # 테스트 실행
        avg_psnr, avg_ssim = test_model(generator, test_loader, device)

        # 결과 출력 및 저장
        print(f'WGAN Test Results - Model: {os.path.basename(model_path)}, '
              f'Data: {os.path.basename(data_dir)}, Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}')
        with open('wgan_test_metrics.txt', 'a') as f:
            f.write(f'{os.path.basename(model_path)} - Data: {os.path.basename(data_dir)}, '
                    f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n')

        # 테스트 피드백 로그
        with open('wgan_test_feedback.txt', 'a') as f:
            f.write(f'Model: {os.path.basename(model_path)}, Data: {os.path.basename(data_dir)}, '
                    f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n')
            f.write(f'Final Evaluation: Consider adjusting lambda_gp or n_critic if PSNR < 25, or retraining if SSIM < 0.9\n')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()