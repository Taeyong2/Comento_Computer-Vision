import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def edge_based_exists(img, close_ksize=20, edge_thresh=0.1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 15, 130)

    # Closing (팽창 → 침식)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=1)

    # 컨투어 검출
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 내부 채우기용 마스크 생성
    mask = np.zeros_like(edges_closed)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)

    # 흰색 픽셀 비율 계산
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    white_ratio = white_pixels / total_pixels

    exists = white_ratio > edge_thresh  # 비율 기준으로 객체 존재 여부 판단

    return edges, edges_closed, mask, white_ratio, exists

# ------------------------
# 실행 + 시각화
# ------------------------
folder = "samples"
save_dir = "preprocessed_samples"
os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(folder):
    if not filename.endswith(".jpg"):
        continue

    img_path = os.path.join(folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    edges, edges_closed, mask, white_ratio, exists = edge_based_exists(img)

    print(f"{filename} → 객체 존재 여부: {exists}, 흰색 비율: {white_ratio:.2%}")

    # 시각화 (matplotlib)
    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("Closed")
    plt.imshow(edges_closed, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Mask (Contours filled)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()
