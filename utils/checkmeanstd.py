import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image

from pathlib import Path

# 이미지 폴더 경로
image_folder_path = './../../Dataset/train_data/image'

# 이미지 파일 경로 리스트
image_paths = [str(p) for p in Path(image_folder_path).glob('*.png')]

# 평균과 표준편차를 계산하는 함수
def calculate_mean_std(image_paths):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img_path in image_paths:
        image = read_image(img_path) / 255.0  # 이미지를 [0, 1] 범위로 스케일링
        mean += image.mean([1, 2])
        std += image.std([1, 2])
    mean /= len(image_paths)
    std /= len(image_paths)
    return mean, std

mean, std = calculate_mean_std(image_paths)
print("평균:", mean)
print("표준편차:", std)
