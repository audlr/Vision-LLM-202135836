import cv2
import numpy as np
from numpy.linalg import inv

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma

# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])


# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
def get_mask(frame):
    # TODO: 각 프레임 이미지를 전처리
    frame_blur = cv2.GaussianBlur(frame, (7,7), 0)
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    # TODO: Color Segmentation으로 Binary Mask 생성
    mask_hsv = cv2.inRange(frame_lab, green_lower, green_upper)

    # TODO: Morphological Operation을 적용하여 객체 영역 추출
    ball_mask_opened2 = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel, iterations=2)
    ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

    final_ball_mask = ball_mask_closed3

    return final_ball_mask


cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# 객체 최초 검출 여부 확인용 boolean
found = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    contour_lst, _ = cv2.findContours(
        get_mask(frame),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
        _, radius = cv2.minEnclosingCircle(contour)
        # 무게중심
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = tuple(contour[0][0])

        center = (cX, cY)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255,255,0), 2)
        cv2.circle(frame, center, 5, (255,255,0), -1)

        # 객체 최초 검출
        if not found:
            found = True

    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 예측한 객체 위치에 노란 원 overlay
    if found:
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (255,255,0), 2)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (255,255,0), -1)

    cv2.imshow("VideoCapture with GStreamer", frame)

cap.release()
cv2.destroyAllWindows()