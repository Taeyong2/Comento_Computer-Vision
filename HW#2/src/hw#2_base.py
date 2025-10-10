#### 기본적인 Depth Map 생성 코드 (OpenCV 활용)
import cv2
import numpy as np
# 이미지 로드
image = cv2.imread('../data/test_img1.JPG')
# 이미지 읽어오기

# 3차원 이미지에서 1차원 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 깊이 맵 생성
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
# 이미지의 밝기 값을 이용해서 컬러맵에 대응시키는 이미지를 반환


# 결과 출력
cv2.imshow('Original Image', gray)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()