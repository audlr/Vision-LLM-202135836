import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이전 셀에서 가져온 코드 (TODO 실행에 필요한 mask_enhanced 생성)
basketball = cv2.imread('../src/images/basketball_crop.jpg')
basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(basketball_lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)


# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Opening으로 노이즈 제거 및 공과 주변 영역 분리 후, Closing으로 공 내부 빈 공간 채움
ball_mask_opened2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

# 최종 binary mask
final_ball_mask = ball_mask_closed3

# Contour를 그릴 원본 이미지 copy
final_img = basketball_rgb.copy()

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    final_ball_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# TODO: 최종 농구공 영역 및 외곽선 선택
if len(contours) > 0:
    # 가장 큰 contour를 농구공 영역으로 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    final_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
plt.subplot(1,3,2), plt.imshow(final_ball_mask, cmap="gray"), plt.title("Opening x2 & Closing x3")
plt.subplot(1,3,3), plt.imshow(final_img), plt.title("Basketball Contour")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()
