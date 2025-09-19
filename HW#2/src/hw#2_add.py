#### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# 이미지 로드
image = cv2.imread('../data/test_img1.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 원본 이미지를 출력
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
# 원본에서 엣지 성분만 출력

# 시각화 (원본 vs 에지)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.show()


# Depth Map 생성
# 이미지의 밝기 정보를 Depth값 처럼 사용해 컬러맵에 대응
depth_map_gray = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

depth_map_edges = cv2.applyColorMap(edges, cv2.COLORMAP_JET)


# 3D 포인트 클라우드 변환
h, w = depth_map_edges.shape[:2] # Depth값이 저장된 이미지의 배열 형태 ex) (480,640,3) 에서 앞에 두가지 원소만 가져옴
# 즉 이미지의 크기를 반환


X, Y = np.meshgrid(np.arange(w), np.arange(h))
#해당 함수를 사용해서 위에서 구한 원본이미지의 크기와 동일한 3차원 공간을 그리기 위해 각 값을 표현한 x,y 좌표쌍을 생성한다.
#즉. x,y 좌표에 (0,0) , (0,1) ... (110,300)을 가지는 값을 가지게 한 후,
# (0,0) 좌표에 depth 값 Z 를 각각 대응시키기 위해사용

###################원 본 이 미 지 ####################
Z = edges.astype(np.float32) # Depth 값을 Z 축으로 사용
####################################################


###################엣 지 이 미 지 ####################
Z = edges.astype(np.float32) # Depth 값을 Z 축으로 사용
####################################################

# 3D 좌표 생성
points_3d = np.dstack((X, Y, Z))
# X, Y, Z 를 하나의 배열로 쌓는다.

pts = points_3d.reshape(-1, 3)
#그리고 하나의 배열에서 x,y 값을 펴서 점 하나가 각 각 하나의 요소를 가지도록 한다.

# matplotlib 기반 시각화
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='jet', s=5)
#
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()

###

pts = points_3d.reshape(-1, 3)

# Open3D 기반 포인트 클라우드 객체 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# 색상도 Z값 기준으로 지정 (옵션)
colors = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].ptp() + 1e-8)
colors = plt.cm.jet(colors)[:, :3]  # matplotlib colormap 사용 (RGB)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])


# 결과 출력
cv2.imshow('Depth Map', depth_map_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()