import cv2
import numpy as np
import os

def brightness(img, brightness_thresh=50): # 평균 밝기가 threshold보다 높으면 True 반환

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > brightness_thresh

def edge_base_filtering(img, edge_thresh=0.01):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # i번째 이미지에  블러 적용

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 15, 130)  # 임계값은 상황에 맞게 조정 가능
    close_ksize = 20


    # Dilation → Erosion
    # 이미지에서 발견된 엣지들이 원래는 이어져있지만 끊어진 부분이 많아서 하나의 형태로 알아보기 힘들다
    # 따라서 엣지에 각 픽셀에 팽창을 하고, 이후에 침식을 통하여 서로 가까이에 존재하는 성분끼리는 이어지도록 한다.
    #
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=1)


    # 컨투어 검출
    contours, hierarchy = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 내부 채우기용 마스크 생성
    mask = np.zeros_like(edges_closed)

    # 컨투어 내부 흰색으로 채우기
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)


    # # 흰색픽셀의 개수
    white_pixels = np.sum(mask == 255)

    # 전체 픽셀 개수
    total_pixels = mask.size

    # 비율 (0 ~ 1)
    white_ratio = white_pixels / total_pixels

    # print(f"흰색 비율: {white_ratio:.4f} ({white_ratio * 100:.2f}%)")

    exists = white_ratio > 10 # 10% 이상히면 객체가 있다고 판단

    return exists

folder=("samples")
save_dir = "preprocessed_samples"

images = []
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):   # jpg 파일만 읽기
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        images.append(img)
        print(f"Loaded: {filename}, shape={img.shape}")

print(f"총 {len(images)}장의 이미지를 불러왔습니다.")


for idx, img in enumerate(images):

    if brightness(img) and edge_base_filtering(img):
        print(f" Skip: {filename} (조건 불충족)")
        continue  # 여기서 바로 다음 이미지로


    # 1. 데이터 증강(색상변화)후, 다른 처리를 하는것이 더 효과적 이라고 생각하여 증강 먼저 진행



    aug_list = [img]  # 원본 포함
    aug_list.append(cv2.flip(img, 1))  # 좌우 반전
    aug_list.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))  # 90도 회전
    aug_list.append(cv2.rotate(img, cv2.ROTATE_180))  # 180도 회전
    # 색상 변환
    # 이미지의 각 채널에 존재하는 색상채널의 평균 값을 다 구한다.
    B_mean = np.mean(img[:, :, 0])
    G_mean = np.mean(img[:, :, 1])
    R_mean = np.mean(img[:, :, 2])
    # 각 채널의 평균값에 20%씩 밝기를 증가 시키기 위한 연산
    B_delta = int(B_mean * 0.2)
    G_delta = int(G_mean * 0.2)
    R_delta = int(R_mean * 0.2)

    bright = cv2.add(img, (B_delta, G_delta, R_delta, 0))
    # 이미지에 채널별 delta를 더래서 밝기를 증가시킨다.
    aug_list.append(bright)



    # 2. 노이즈 제거
    # 위에서 데이터 증강을 진행한 데이터에 가우시안블러를 이용해 고주파 성분을 제거
    blurred_list = []
    for i in aug_list:

        blurred = cv2.GaussianBlur(i, (5, 5), 0)  # i번째 이미지에  블러 적용
        # 가우시칸 분포의 표준편자는 0으로 둬서 자동으로 정해지게끔
        # 커널 사이즈는 5x5로 설정
        blurred_list.append(blurred)  # 결과를 리스트에 추가한다,

    aug_list = blurred_list  # 기존 리스트의 값을 노이즈가 제거된 값으로 대체한다.

    #3.색상변환(Grayscale & Normalize)

    aug_gray = []
    for i in aug_list:

        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) # 3차원 이미지를 흑백 1차원 이미지로 바꾼다.
        # gray = gray.astype("float32") / 255.0  # [0,1] 정규화 # 값을 0~1 사이로 매칭 되도록 정규화 한다.
        aug_gray.append(gray)# 결과를 리스트에 추가한다,

    aug_list = aug_gray  # 기존 리스트의 값을 이진화 & 정규화된 값으로 대체한다.

    aug_resize = []

    for i in aug_list:
        resized = cv2.resize(i, (224, 224)) # 이미지의 크기를 224,224 사이즈로 resize 한다.
        aug_resize.append(resized)  # 결과를 리스트에 추가한다,

    aug_list = aug_resize

    filename = f"{idx}_image.jpg"  # 예시: 0_image.jpg, 1_image.jpg
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, aug_list[4])  # 5번째 결과만 저장
    print(f"{save_path} 저장 완료!")




















