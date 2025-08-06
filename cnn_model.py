import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
from cnn_dataset import MRIDataset

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

class DRCAMGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(DRCAMGenerator, self).__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1) # padding=1 유지, 크기 3x3, stride 1
         # Dilated Residual Blocks
        self.enc1 = DilatedResidualBlock(64, 64, dilation=1)
        self.enc2 = DilatedResidualBlock(64, 128, dilation=2)
        self.enc3 = DilatedResidualBlock(128, 256, dilation=3)
        self.enc4 = DilatedResidualBlock(256, 256, dilation=3)
        self.pool = nn.MaxPool2d(2, 2, padding=0)  # padding=0 유지
        self.cam1 = ChannelAttention(64)
        self.cam2 = ChannelAttention(128)
        self.cam3 = ChannelAttention(256)
        self.cam4 = ChannelAttention(256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DilatedResidualBlock(256, 128, dilation=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DilatedResidualBlock(128, 64, dilation=1)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 입력 크기 검증 (4D 텐서)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor with shape {x.size()}")
        if x.size(2) < 4 or x.size(3) < 4:
            padding_h = max(0, 4 - x.size(2))
            padding_w = max(0, 4 - x.size(3))
            x = F.pad(x, (0, padding_w, 0, padding_h), mode='constant', value=0)

        e1 = F.relu(self.init_conv(x))  # [batch_size, 64, 384, 384]
        e1 = self.enc1(e1)
        e1 = self.cam1(e1)
        e2 = self.pool(e1)  # [batch_size, 64, 192, 192]
        e2 = self.enc2(e2)
        e2 = self.cam2(e2)
        e3 = self.pool(e2)  # [batch_size, 128, 96, 96]
        e3 = self.enc3(e3)
        e3 = self.cam3(e3)
        e4 = self.enc4(e3)
        e4 = self.cam4(e4)
        d1 = self.up1(e4)  # [batch_size, 128, 192, 192]
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)  # [batch_size, 64, 384, 384]
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = torch.tanh(self.out_conv(d2)) # [batch_size, out_channels, 384, 384] # tanh 활성화 함수 사용

        return out # 범위: [-1, 1]

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)  # 커널 크기 3x3, 패딩 1
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.out_conv = nn.Conv2d(512, 1, 3, stride=1, padding=1)

    def forward(self, x):
        if x.dim() != 4 or x.size(2) < 3 or x.size(3) < 3:
            raise ValueError(f"Expected 4D tensor with height and width >= 3, got {x.size()}")
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.out_conv(x)
        return x
    
    
def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 디렉토리
    data_dir = '/home/compu/john/MAI-LAB/brain/train/image'

    # 데이터셋 생성 (cnn_dataset.py의 MRIDataset 사용)
    dataset = MRIDataset(data_dir=data_dir, mode='train', transform=None)

    # 첫 번째 샘플 로드 (aliased image: image_input_img)
    inputs, _ = dataset[0]  # [2, 384, 384]

    # 모델 생성 및 디바이스 이동
    model = DRCAMGenerator().to(device)

    # Weight 로드 (checkpoints/generator_epoch_7.pth)
    checkpoint_path = 'checkpoints/generator_epoch_7.pth'  # 경로 확인 (필요 시 절대 경로 사용)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)  # 만약 키가 'state_dict'라면 checkpoint['state_dict']

    # 입력 텐서 배치 차원 추가 및 디바이스 이동
    inputs = inputs.unsqueeze(0).to(device)  # [1, 2, 384, 384]

    # Generator 실행 (inference 모드)
    model.eval()
    with torch.no_grad():
        fake_imgs = model(inputs)

    # 출력 범위 확인
    print("Fake min:", fake_imgs.min().item())
    print("Fake max:", fake_imgs.max().item())

    # 폴더 생성 및 이미지 저장 (첫 채널 저장, 정규화 적용)
    save_dir = 'visualizations/testing_fake'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fake_epoch_7.png')

    fake_np = fake_imgs[0, 0].detach().cpu().numpy()  # 첫 채널
    fake_np = (fake_np - fake_np.min()) / (fake_np.max() - fake_np.min() + 1e-8)  # [0,1] 정규화
    plt.imshow(fake_np, cmap='gray')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

    print(f"Generator 출력 이미지가 {save_path}에 저장되었습니다!")
if __name__ == '__main__':
    main()