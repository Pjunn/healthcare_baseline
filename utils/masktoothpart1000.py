import torch
from torchvision.io import read_image
from pathlib import Path
import random

# 이미지 폴더 경로
image_folder_path = './../../Dataset/train_data/imagemaskfast'

# 이미지 파일 경로 리스트
image_paths = [str(p) for p in Path(image_folder_path).glob('*.png')]

# 무작위로 1000개의 이미지 선택
print("image_paths 길이:", len(image_paths))  # 리스트 길이 확인
sample_size = min(1000, len(image_paths))
random_selected_paths = random.sample(image_paths, sample_size)

# 평균과 표준편차를 계산하는 함수
def calculate_mean_std(image_paths):
    pixel_sum = torch.zeros(3)
    pixel_count = torch.zeros(3)
    for img_path in image_paths:
        image = read_image(img_path).float() / 255.0  # 이미지를 [0, 1] 범위로 스케일링
        mask = image.sum(dim=0) != 0  # 검정색 마스크 부분 제외
        for i in range(3):
            pixel_sum[i] += image[i, mask].mean()
            pixel_count[i] += 1

    mean = pixel_sum / pixel_count
    std = torch.zeros(3)

    for img_path in image_paths:
        image = read_image(img_path).float() / 255.0
        mask = image.sum(dim=0) != 0
        for i in range(3):
            std[i] += ((image[i, mask] - mean[i])**2).mean()
    std = torch.sqrt(std / pixel_count)

    return mean, std

mean, std = calculate_mean_std(random_selected_paths)
print("평균:", mean)
print("표준편차:", std)
