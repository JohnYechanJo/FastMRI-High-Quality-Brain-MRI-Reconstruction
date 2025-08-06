import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import random
import multiprocessing as mp
from cnn_model import DRCAMGenerator
from cnn_dataset import MRIDataset
from cnn_utils import VGGPerceptualLoss, calculate_metrics, visualize_images_c
from scipy.ndimage import gaussian_filter
from skimage.transform import warp, AffineTransform

def add_noise(image, noise_level=0.05):
    """Gaussian Noise 추가."""
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)

def add_blur(image, kernel_size=5, sigma=2.0):
    """Gaussian Blur 적용."""
    return gaussian_filter(image, sigma=sigma, mode='constant', cval=0.0, truncate=kernel_size / sigma)

def add_motion_artifact(image, period=20, amplitude=0.2):
    """주기적 모션 아티팩트 추가."""
    h, w = image.shape[:2]
    y, x = np.indices((h, w))
    motion = amplitude * np.sin(2 * np.pi * y / period)  # 수직 방향 주기적 이동
    avg_motion = np.mean(motion)
    transform = AffineTransform(translation=(0, avg_motion))
    return np.clip(warp(image, transform, mode='constant', cval=0.0), 0, 1)

def corrupt_image(image, mode='nbm'):
    """이미지에 noise, blur, motion artifact 적용 (mode에 따라 선택적 적용)."""
    image = np.abs(image)  # magnitude 추출
    # 채널별로 처리
    channels = [image[i] for i in range(image.shape[0])]  # [C, H, W] -> [H, W] 리스트
    corrupted_channels = []
    
    for channel in channels:
        corrupted = channel.copy()
        if 'n' in mode:
            corrupted = add_noise(corrupted, noise_level=0.05)
        if 'b' in mode:
            corrupted = add_blur(corrupted, kernel_size=5, sigma=2.0)
        if 'm' in mode:
            corrupted = add_motion_artifact(corrupted, period=20, amplitude=0.2)
        corrupted_channels.append(corrupted)
    
    # 채널 결합
    corrupted = np.stack(corrupted_channels, axis=0)  # [C, H, W]로 복원
    return corrupted

def test_model(generator, test_loader, device):
    """
    Corrupted Test 데이터로 모델 성능 평가 (PSNR, SSIM 계산)
    """
    generator.eval()
    total_psnr = {mode: 0.0 for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']}
    total_ssim = {mode: 0.0 for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']}
    num_samples = {mode: 0 for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 각 모드에 대해 corrupt 적용
            for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']:
                inputs_corrupted = []
                for i in range(inputs.size(0)):
                    input_np = inputs[i].cpu().numpy()
                    corrupted_np = corrupt_image(input_np, mode=mode)
                    corrupted_tensor = torch.from_numpy(corrupted_np).float().to(device)
                    inputs_corrupted.append(corrupted_tensor)
                inputs_corrupted = torch.stack(inputs_corrupted)

                # Generator로 복원
                fake_imgs = generator(inputs_corrupted)

                print(f"Batch {batch_idx} - Mode: {mode}, Inputs: shape={inputs.shape}, min={inputs.min().item():.4f}, max={inputs.max().item():.4f}, mean={inputs.mean().item():.4f}, std={inputs.std().item():.4f}")
                print(f"Batch {batch_idx} - Mode: {mode}, Fake_imgs: shape={fake_imgs.shape}, min={fake_imgs.min().item():.4f}, max={fake_imgs.max().item():.4f}, mean={fake_imgs.mean().item():.4f}, std={fake_imgs.std().item():.4f}")
                print(f"Batch {batch_idx} - Mode: {mode}, Targets: shape={targets.shape}, min={targets.min().item():.4f}, max={targets.max().item():.4f}, mean={targets.mean().item():.4f}, std={targets.std().item():.4f}")

                # 메트릭 계산 (배치 단위)
                for i in range(inputs.size(0)):
                    pred_np = fake_imgs[i].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                    target_np = targets[i].cpu().numpy().transpose(1, 2, 0)
                    psnr_val, ssim_val = calculate_metrics(pred_np, target_np)  # NMSE는 사용하지 않음
                    total_psnr[mode] += psnr_val
                    total_ssim[mode] += ssim_val
                    num_samples[mode] += 1

                if batch_idx % 100 == 0:
                    print(f'Test Batch [{batch_idx}/{len(test_loader)}] Processed - Mode: {mode}')
                    # 매 100번째 배치마다 3x7 시각화
                    sample_idx = random.randint(0, inputs.size(0) - 1)  # 배치 내 랜덤 샘플 선택
                    vis_inputs_all = []
                    vis_fakes_all = []
                    vis_targets_all = []
                    with torch.no_grad():
                        input_img = inputs[sample_idx].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                        vis_targets_all.append(targets[sample_idx].cpu())  # 타겟
                        for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']:
                            corrupted_np = corrupt_image(input_img.cpu().numpy(), mode=mode)
                            corrupted_tensor = torch.from_numpy(corrupted_np).float().to(device)
                            fake_img = generator(corrupted_tensor)
                            vis_inputs_all.append(corrupted_tensor.squeeze(0).cpu())  # corrupt된 입력
                            vis_fakes_all.append(fake_img.squeeze(0).cpu())  # 복원된 결과

                    vis_inputs_all = torch.stack(vis_inputs_all)  # [7, C, H, W]
                    vis_fakes_all = torch.stack(vis_fakes_all)    # [7, C, H, W]
                    vis_targets_all = torch.stack([vis_targets_all[0]] * 7)  # [7, C, H, W], 타겟 반복

                    # 3x7 시각화
                    visualize_images_c(
                        vis_inputs_all.numpy(),
                        vis_fakes_all.numpy(),
                        vis_targets_all.numpy(),
                        epoch=0,
                        save_dir='visualizations/test_all',
                        batch_idx=batch_idx,
                        dataset_name=f'test_all_modes_batch_{batch_idx}'
                    )

    # 평균 계산
    avg_metrics = {}
    for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']:
        if num_samples[mode] == 0:
            print(f"Error: No test samples processed for mode {mode}. Check test dataset.")
            avg_metrics[mode] = (0.0, 0.0)
        else:
            avg_metrics[mode] = (total_psnr[mode] / num_samples[mode], total_ssim[mode] / num_samples[mode])

    # 전체 배치 처리 후 3x7 시각화
    selected_index = random.sample(range(len(test_loader.dataset)), 1)[0]  # 1개 샘플 랜덤 선택
    vis_inputs_all = []
    vis_fakes_all = []
    vis_targets_all = []
    
    with torch.no_grad():
        input_img, target_img = test_loader.dataset[selected_index]
        input_img = input_img.unsqueeze(0).to(device)  # 배치 차원 추가
        vis_targets_all.append(target_img.cpu())  # 타겟은 모든 모드에서 동일
        
        for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']:
            corrupted_np = corrupt_image(input_img.cpu().numpy(), mode=mode)
            corrupted_tensor = torch.from_numpy(corrupted_np).float().to(device)
            fake_img = generator(corrupted_tensor)
            vis_inputs_all.append(corrupted_tensor.squeeze(0).cpu())  # corrupt된 입력
            vis_fakes_all.append(fake_img.squeeze(0).cpu())  # 복원된 결과

    vis_inputs_all = torch.stack(vis_inputs_all)  # [7, C, H, W]
    vis_fakes_all = torch.stack(vis_fakes_all)    # [7, C, H, W]
    vis_targets_all = torch.stack([vis_targets_all[0]] * 7)  # [7, C, H, W], 타겟 반복

    # 3x7 시각화 (모든 모드 한 번에 출력)
    visualize_images_c(
        vis_inputs_all.numpy(),
        vis_fakes_all.numpy(),
        vis_targets_all.numpy(),
        epoch=0,
        save_dir='visualizations/test_all',
        batch_idx='summary_all_modes',
        dataset_name='test_all_modes'
    )

    return avg_metrics

def find_best_checkpoint(checkpoint_dir='/home/compu/john/MAI-LAB/checkpoints'):
    """
    체크포인트 디렉토리에서 최고 에포크를 찾아 반환
    """
    best_psnr = -float('inf')
    best_epoch = 0
    try:
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('generator_epoch_') and filename.endswith('.pth'):
                epoch = int(filename.split('_')[2].split('.')[0])
                with open('test_metrics.txt', 'r') as f:
                    for line in f:
                        if f'Epoch {epoch}' in line:
                            psnr = float(line.split('PSNR: ')[1].split(',')[0])
                            if psnr > best_psnr:
                                best_psnr = psnr
                                best_epoch = epoch
        return f'checkpoints/generator_epoch_{best_epoch}.pth' if best_epoch > 0 else None
    except FileNotFoundError:
        print("test_metrics.txt not found. Using default epoch 0.")
        return None

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
        print("No valid checkpoint found. Using default epoch 9.")
        model_path = 'checkpoints/generator_epoch_9.pth'

    # 모델 로드 및 freeze
    generator = DRCAMGenerator(in_channels=2, out_channels=2).to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint {model_path} not found.")
        return
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

        # 각 corrupt 모드에 대해 테스트
        metrics = test_model(generator, test_loader, device)

        # 결과 출력 및 저장 (모든 모드)
        for mode in ['n', 'b', 'm', 'nb', 'nm', 'bm', 'nbm']:
            avg_psnr, avg_ssim = metrics[mode]
            print(f'Test Results - Model: {os.path.basename(model_path)}, '
                  f'Data: {os.path.basename(data_dir)}, Mode: {mode}, '
                  f'Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}')
            with open('test_metrics.txt', 'a') as f:
                f.write(f'{os.path.basename(model_path)} - Data: {os.path.basename(data_dir)}, '
                        f'Mode: {mode}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n')

            # 테스트 피드백 로그
            with open('test_feedback.txt', 'a') as f:
                f.write(f'Model: {os.path.basename(model_path)}, Data: {os.path.basename(data_dir)}, '
                        f'Mode: {mode}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n')
                f.write(f'Final Evaluation: Consider retraining if PSNR < 25\n')

if __name__ == '__main__':
    mp.set_start_method('spawn')  # 멀티프로세싱 시작 방법 설정
    main()