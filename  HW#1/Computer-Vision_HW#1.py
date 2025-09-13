import cv2
import numpy as np
# 이미지 로드
image = cv2.imread('sample.JPG') # 분석할 이미지 파일
# BGR에서 HSV 색상 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 빨간색 범위 지정 (두 개의 범위를 설정해야 함)
lower_green = np.array([35, 100, 50])   # 초록색 최소값 (H, S, V)
upper_green = np.array([85, 255, 255])  # 초록색 최대값 (H, S, V)
# 각 배열의 요소 ([색상 (H),채도(s),명도(V)])


# 마스크 생성
mask = cv2.inRange(hsv, lower_green, upper_green)

# 원본 이미지에서 빨간색 부분만 추출
result = cv2.bitwise_and(image, image, mask=mask)
# 결과 이미지 출력
cv2.imshow('Original', image)
cv2.imshow('Red Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#test
