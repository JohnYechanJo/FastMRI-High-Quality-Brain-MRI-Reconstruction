import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import h5py
from torchvision import transforms
from sklearn.cluster import KMeans
from skimage.morphology import binary_closing, binary_opening

# Masked Autoencoder (MAE) Implementation for Brain Segmentation
class MAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), # 1 input channel for grayscale images 
            nn.ReLU(inplace=True), # 64 output channels
            nn.MaxPool2d(2, 2), # Downsample by 2 
            nn.Conv2d(64, 128, 3, padding=1), # 64 input channels, 128 output channels
            nn.ReLU(inplace=True), # 128 output channels
            nn.MaxPool2d(2, 2), # Downsample by 2
            nn.Conv2d(128, 256, 3, padding=1), # 128 input channels, 256 output channels
            nn.ReLU(inplace=True) # 256 output channels
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), # Upsample by 2 
            nn.ReLU(inplace=True), # 128 output channels
            nn.ConvTranspose2d(128, 64, 2, stride=2), # Upsample by 2
            nn.ReLU(inplace=True), # 64 output channels
            nn.Conv2d(64, out_channels, 3, padding=1), # 64 input channels, out_channels output channels
            nn.Sigmoid()  # Binary mask [0,1]
        )

    def forward(self, x):
        # 마스크 생성 (50% 픽셀 마스킹)
        mask = torch.rand(x.shape, device=x.device) > 0.5 # 50% 확률로 마스킹
        x_masked = x * mask # 마스크 적용
        encoded = self.encoder(x_masked) # 인코딩
        decoded = self.decoder(encoded) # 디코딩
        return decoded 

class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode='train', transform=None): # 'train' or 'val' or 'test'
        self.data_dir = Path(data_dir) # 데이터 디렉토리
        self.mode = mode # 'train' or 'val' or 'test'
        self.transform = transform # 변환 함수
        self.files = [f for f in self.data_dir.glob('*.h5') if not f.name.startswith('._')] # .h5 파일만 선택 # ._ 파일 제외
        self.samples = [] # 샘플 리스트 초기화
        for f in self.files: # 각 파일에 대해
            with h5py.File(f, 'r') as h5_file: # h5 파일 열기
                num_slices = h5_file['image_input'].shape[1] # 이미지 슬라이스 수 가져오기
                self.samples.extend([(f, i) for i in range(num_slices)]) # 각 슬라이스에 대해 (파일, 슬라이스 인덱스) 튜플 추가
                # 왜 슬라이스 인덱스가 필요한가?
                # 슬라이스 인덱스는 각 이미지의 특정 슬라이스를 식별하기 위해 필요합니다.
                # 3D MRI 데이터는 여러 슬라이스로 구성되어 있으며, 각 슬라이스는 2D 이미지로 표현됩니다.
                # 따라서 슬라이스 인덱스를 사용하여 특정 슬라이스를 선택하고 처리할 수 있습니다.
                # 튜플은 왜 사용되는가?
                # 튜플은 (파일 경로, 슬라이스 인덱스) 형태로 데이터를 저장하기 위해 사용됩니다.
                # 이렇게 하면 각 샘플에 대한 파일 경로와 슬라이스 인덱스를 쉽게 참조할 수 있습니다.
                # 또한, 튜플은 불변(immutable) 자료형이므로 데이터의 무결성을 보장할 수 있습니다.
        # Initialize MAE and load pre-trained weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 사용 가능 여부 확인
        self.mae = MAE(in_channels=1, out_channels=1).to(self.device) # MAE 모델 초기화
        try:
            self.mae.load_state_dict(torch.load('mae_brain_segmentation.pth', map_location=self.device)) # 사전 훈련된 가중치 로드
            self.mae.eval() # 평가 모드로 설정
        except FileNotFoundError:
            print("Warning: mae_brain_segmentation.pth not found. Using untrained MAE (results may be poor).") 

    def __len__(self):
        return len(self.samples) # 데이터셋의 샘플 수 반환

    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx] # 인덱스에 해당하는 파일 경로와 슬라이스 인덱스 가져오기
        # HDF5 파일에서 이미지 데이터 읽기
        with h5py.File(file_path, 'r') as f:
            image_input = np.array(f['image_input'])  # [12, 384, 384]
            image_label = np.array(f['image_label'])  # [12, 384, 384]

        # Multi-coil 처리 (RSS 결합)
        image_input_rss = torch.sqrt(torch.sum(torch.tensor(image_input, dtype=torch.float32) ** 2, dim=0))  # [384, 384] # RSS 결합
        image_label_rss = torch.sqrt(torch.sum(torch.tensor(image_label, dtype=torch.float32) ** 2, dim=0))  # [384, 384] # RSS 결합
        # RSS 결합은 여러 코일에서 얻은 MRI 데이터를 하나의 이미지로 결합하는 방법입니다.
        # image_input_rss와 image_label_rss는 [384, 384] 형태의 텐서입니다.
        # image_input_rss와 image_label_rss는 각각 입력 이미지와 레이블 이미지의 RSS 결합 결과입니다.
        
        # MAE 세그멘테이션 for image_input and image_label
        input_np = image_input_rss.numpy()
        label_np = image_label_rss.numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)  # Normalize [0,1]
        label_np = (label_np - label_np.min()) / (label_np.max() - label_np.min() + 1e-8)  # Normalize [0,1]

        # MAE 입력 준비
        input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 384, 384]
        label_tensor = torch.tensor(label_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 384, 384]
        # unsqueeze(0)는 배치 차원 추가, unsqueeze(0)는 채널 차원 추가
        # input_np와 label_np는 [384, 384] 형태로, unsqueeze(0) 후 [1, 1, 384, 384] 형태가 됨
        
        # MAE로 마스크 생성
        with torch.no_grad():
            mask_input = self.mae(input_tensor)  # [1, 1, 384, 384]
            mask_label = self.mae(label_tensor)  # [1, 1, 384, 384]
        
        # KMeans 클러스터링으로 마스크 후처리
        mask_input_np = mask_input.squeeze().cpu().numpy() # [384, 384]
        mask_label_np = mask_label.squeeze().cpu().numpy() # [384, 384]
        # squeeze()는 차원 축소, cpu()는 CPU로 이동, numpy()는 NumPy 배열로 변환
        mask_input_flat = mask_input_np.flatten().reshape(-1, 1) # [N, 1] 형태로 변환
        mask_label_flat = mask_label_np.flatten().reshape(-1, 1) # [N, 1] 형태로 변환
        kmeans_input = KMeans(n_clusters=2, random_state=0).fit(mask_input_flat) # KMeans 클러스터링
        kmeans_label = KMeans(n_clusters=2, random_state=0).fit(mask_label_flat) # KMeans 클러스터링
        # n_clusters=2는 2개의 클러스터로 분할, random_state=0은 재현성을 위해 설정
        # mask_input_flat과 mask_label_flat은 각각 [N, 1] 형태로, KMeans는 각 픽셀을 2개의 클러스터로 분할합니다.
        # reshape(-1, 1)은 1차원 배열을 2차원 배열로 변환하여 KMeans에 입력합니다.
        # KMeans 클러스터링은 이미지의 픽셀 값을 기반으로 두 개의 클러스터로 분할합니다.
        # 클러스터링 결과는 mask_input.labels_와 mask_label.labels_에 저장됩니다.
        # KMeans는 픽셀 값을 기반으로 클러스터를 형성하며, 각 클러스터는 뇌 영역과 배경 영역을 나타냅니다.
        # 클러스터링 후, mask_input_np와 mask_label_np는 각각 [384, 384] 형태로 유지됩니다.
        # KMeans 레이블은 각 픽셀에 대해 0 또는 1로 레이블링됩니다.
        # KMeans 레이블을 원래 이미지 형태로 변환
        mask_input = kmeans_input.labels_.reshape(mask_input_np.shape) # [384, 384]
        mask_label = kmeans_label.labels_.reshape(mask_label_np.shape) # [384, 384]

        # 뇌 영역을 1, 배경을 0으로 설정 (KMeans 레이블은 무작위이므로 조정)
        if mask_input.mean() > 0.5:  # 뇌 영역이 0으로 잘못 레이블링된 경우 반전
            mask_input = 1 - mask_input # [384, 384]
        if mask_label.mean() > 0.5:
            mask_label = 1 - mask_label # [384, 384]

        # 형태학적 연산으로 마스크 정제
        mask_input = binary_closing(mask_input, footprint=np.ones((5, 5))) # [384, 384]
        mask_input = binary_opening(mask_input, footprint=np.ones((5, 5))) # [384, 384]
        mask_label = binary_closing(mask_label, footprint=np.ones((5, 5))) # [384, 384]
        mask_label = binary_opening(mask_label, footprint=np.ones((5, 5))) # [384, 384]

        mask_input = mask_input.astype(np.float32) # [384, 384]
        mask_label = mask_label.astype(np.float32) # [384, 384]

        # 마스크 적용
        input_np = input_np * mask_input # [384, 384]
        label_np = label_np * mask_label # [384, 384]

        # 텐서로 변환 및 2채널 확장
        image_input_img = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)  # [2, 384, 384]
        image_label_img = torch.tensor(label_np, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)  # [2, 384, 384]

        # transform 적용
        if self.transform:
            from torchvision.transforms import Lambda # Lambda 변환을 사용하여 identity_transform 정의
            identity_transform = Lambda(lambda x: x) if isinstance(self.transform, transforms.ToTensor) else self.transform # identity_transform
            image_input_img = identity_transform(image_input_img) # [2, 384, 384]
            image_label_img = identity_transform(image_label_img) # [2, 384, 384]

        return image_input_img, image_label_img