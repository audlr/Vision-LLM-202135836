import cv2
import math
import time
import numpy as np
import mediapipe as mp
from collections import deque
from numpy.linalg import inv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ======================== 설정 ========================
MODEL_PATH = "src/models/MediaPipe/pose_landmarker_full.task"

# 판정
WARN_SCORE, BAD_SCORE = 0.15, 0.20  # 기준 대비 변화량 (목 짧아짐 + 얼굴 커짐) 이상이면 주의 / 거북목
CALIB_SEC = 10                      # 캘리브레이션 시간(초)
ALERT_SEC = 5                       # 거북목이 이 시간(초) 이상 지속되면 경고

# 전처리
ALPHA, BETA = 1.1, 10               # 대비, 밝기
CLAHE_CLIP, CLAHE_TILE = 2.0, 8     # CLAHE 대비 제한, 타일 크기
BLUR_K, MORPH_K = 7, 5              # 가우시안 블러, 모폴로지 커널 크기
OPEN_ITER, CLOSE_ITER = 2, 3        # 열림 / 닫힘 반복 횟수

# 어깨 마커
LAB_LOW = np.array([30, 60, 90], dtype=np.uint8)       # 초록 LAB 범위
LAB_HIGH = np.array([230, 115, 180], dtype=np.uint8)
SEARCH_RATIO = 0.35                 # Pose 어깨에서 어깨 너비의 이 비율 이내만 마커 탐색
MIN_AREA = 80                       # 마커로 인정할 최소 윤곽선 면적
LOST_LIMIT = 15                     # 마커를 이 프레임 수 넘게 놓치면 Pose 어깨로 대체

# 표시 (BGR)
GREEN, YELLOW, RED, WHITE, BLACK = (0, 200, 0), (0, 200, 255), (0, 0, 255), (255, 255, 255), (0, 0, 0)
LINE, DOT = 3, 5                    # 선 두께, 점 반지름
TEXT, TEXT_L = 0.5, 0.6             # 글자 크기 (보통, 강조)
PAD = 15                            # 화면 여백
WIN = "Turtle Neck Detector"


# ======================== 초기화 ========================
base_option = python.BaseOptions(model_asset_path=MODEL_PATH)                   # 모델 경로 지정하는 옵션
options = vision.PoseLandmarkerOptions(base_options=base_option, num_poses=1)   # 모델 경로와 최대 사람 수 지정
pose_detector = vision.PoseLandmarker.create_from_options(options)              # 해당 옵션으로 자세 검출하는 객체 생성
PL = vision.PoseLandmark                                                        # landmark 이름 (NOSE, LEFT_EYE 등)

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)    # Jetson
# cap = cv2.VideoCapture(0)                            # 노트북 웹캠

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))


# ======================== 1. 전처리 ========================
def preprocess(frame):
    """밝기/대비 조정 → LAB 변환 → L 채널 CLAHE. 표시용 BGR과 색 분할용 LAB 반환"""
    frame = cv2.convertScaleAbs(frame, alpha=ALPHA, beta=BETA)
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), lab


# ======================== 2. 신체 인식 ========================
def detect_body(frame):
    """코, 양쪽 눈, 양쪽 어깨(화면 기준 왼쪽/오른쪽) 좌표. 사람이 없으면 None"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        return None
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in result.pose_landmarks[0]]
    s1, s2 = pts[PL.LEFT_SHOULDER], pts[PL.RIGHT_SHOULDER]
    return {"nose": pts[PL.NOSE], "eye_l": pts[PL.LEFT_EYE], "eye_r": pts[PL.RIGHT_EYE],
            "sh": {"L": min(s1, s2), "R": max(s1, s2)}}


# ======================== 3. 마커 검출 ========================
def find_marker(lab, center, r):
    """center 주변 r 범위만 잘라 블러 → 색 분할 → 모폴로지 → center에 가장 가까운 윤곽선 무게중심"""
    h, w = lab.shape[:2]
    x1, y1 = max(int(center[0] - r), 0), max(int(center[1] - r), 0)
    x2, y2 = min(int(center[0] + r), w), min(int(center[1] + r), h)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = cv2.GaussianBlur(lab[y1:y2, x1:x2], (BLUR_K, BLUR_K), 0)
    mask = cv2.inRange(roi, LAB_LOW, LAB_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x1, y1))

    best = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > MIN_AREA:                 # m00 = 윤곽선 면적
            p = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            d = math.dist(p, center)
            if d < r and (best is None or d < best[0]):
                best = (d, p)
    return best[1] if best else None


# ======================== 4. 객체 추적 (Kalman Filter) ========================
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

# Kalman filter 변수 정의 (상태 mu: [[x, y], [vx, vy]])
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])


def track(t, z):
    """측정값 있으면 Prediction & Update, 없으면 Prediction만. LOST_LIMIT 넘게 놓치면 None"""
    if t["mu"] is None:
        if z is None:
            return None
        t["mu"], t["sigma"] = np.array([[z[0], z[1]], [0, 0]]), np.zeros((2, 2))    # 최초 검출 위치에서 시작
    t["lost"] = 0 if z is not None else t["lost"] + 1
    if t["lost"] > LOST_LIMIT:
        t["mu"] = None
        return None
    t["mu"], t["sigma"] = KalmanFilter(t["mu"], t["sigma"], None if z is None else np.array([list(z)]))
    return int(t["mu"][0][0]), int(t["mu"][0][1])


def get_shoulders(lab, body, trackers):
    """양쪽 어깨: 마커 검출 → Kalman 추적, 놓치면 Pose 어깨로 대체. (어깨 좌표, 대체된 쪽 목록) 반환"""
    pose_sh = body["sh"]
    r = SEARCH_RATIO * math.dist(pose_sh["L"], pose_sh["R"])
    sh, fallback = {}, []
    for k in ("L", "R"):
        sh[k] = track(trackers[k], find_marker(lab, pose_sh[k], r))
        if sh[k] is None:
            sh[k] = pose_sh[k]
            fallback.append(k)
    return sh, fallback


# ======================== 5. 측정 / 판정 ========================
def measure(body, sh):
    """어깨 너비로 정규화한 목 길이(코~어깨 중점 세로 거리)와 얼굴 크기(눈 사이 거리)"""
    sw = max(math.dist(sh["L"], sh["R"]), 1)
    mid = ((sh["L"][0] + sh["R"][0]) // 2, (sh["L"][1] + sh["R"][1]) // 2)
    neck = (mid[1] - body["nose"][1]) / sw
    face = math.dist(body["eye_l"], body["eye_r"]) / sw
    return neck, face, mid


def judge(score):
    if score >= BAD_SCORE:
        return "TURTLE NECK", RED
    if score >= WARN_SCORE:
        return "WARNING", YELLOW
    return "GOOD", GREEN


# ======================== 6. 화면 표시 ========================
def put_text(img, text, org, color=WHITE, scale=TEXT):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_body(img, body, sh, mid, fallback, color):
    """어깨선, 목 길이 선, 눈 사이 선 (마커 검출 어깨는 채운 점, Pose 대체 어깨는 빈 원)"""
    nose, eye_l, eye_r = body["nose"], body["eye_l"], body["eye_r"]
    cv2.line(img, sh["L"], sh["R"], color, LINE, cv2.LINE_AA)
    cv2.line(img, mid, (mid[0], nose[1]), color, LINE, cv2.LINE_AA)
    cv2.line(img, eye_l, eye_r, color, LINE, cv2.LINE_AA)
    for k, p in sh.items():
        cv2.circle(img, p, DOT + 2, color, 2 if k in fallback else -1, cv2.LINE_AA)
    for p in (nose, eye_l, eye_r):
        cv2.circle(img, p, DOT, color, -1, cv2.LINE_AA)


def button_rect(shape):
    h, w = shape[:2]
    return w // 2 - 70, h // 2 - 22, w // 2 + 70, h // 2 + 22


def draw_button(img):
    x1, y1, x2, y2 = button_rect(img.shape)
    cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, -1)
    put_text(img, "READY?", (x1 + 32, y2 - 14), BLACK, 0.8)


def draw_progress(img, ratio, y, color):
    w = img.shape[1]
    cv2.rectangle(img, (PAD, y), (PAD + int((w - 2 * PAD) * ratio), y + 4), color, -1)


def draw_alert(img, sec):
    cv2.addWeighted(np.full_like(img, RED), 0.3, img, 0.7, 0, img)
    put_text(img, f"STRAIGHTEN UP!  {sec:.0f}s", (PAD, 80), scale=TEXT_L)


def draw_ratio_bar(img, ratio):
    h, w = img.shape[:2]
    cv2.rectangle(img, (PAD, h - 20), (w - PAD, h - 12), RED, -1)
    cv2.rectangle(img, (PAD, h - 20), (PAD + int((w - 2 * PAD) * ratio), h - 12), GREEN, -1)
    put_text(img, f"Good posture {ratio * 100:.0f}%", (PAD, h - 28))


# ======================== 7. 상태별 처리 ========================
# 상태: READY(버튼 대기) → CALIB(10초 기준 측정) → RUN(판정), c 키로 READY 복귀
def new_state():
    return {"mode": "READY", "shape": (480, 640),
            "calib": [], "calib_start": None, "base": None,           # 캘리브레이션 샘플, 시작 시각, 기준값
            "scores": deque(maxlen=10),                               # 점수 이동평균용
            "good_t": 0.0, "bad_t": 0.0, "bad_since": None,           # 바른/거북목 누적 시간, 거북목 시작 시각
            "trackers": {k: {"mu": None, "sigma": None, "lost": 0} for k in ("L", "R")}}


def on_click(event, x, y, flags, S):
    x1, y1, x2, y2 = button_rect(S["shape"])
    if S["mode"] == "READY" and event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
        S["mode"] = "CALIB"


def handle_key(S, key):
    """q 종료(False 반환), Space 시작, c 처음으로"""
    if key == ord("q"):
        return False
    if key == ord(" ") and S["mode"] == "READY":
        S["mode"] = "CALIB"
    if key == ord("c"):
        S.update(new_state())
    return True


def calibrate(S, frame, neck, face, now):
    """CALIB_SEC 동안 값을 모아 중앙값을 바른 자세 기준으로 저장"""
    S["calib_start"] = S["calib_start"] or now
    S["calib"].append((neck, face))
    remain = max(CALIB_SEC - (now - S["calib_start"]), 0)
    put_text(frame, f"Keep good posture...  {remain:.0f}s", (PAD, 25), YELLOW, TEXT_L)
    draw_progress(frame, 1 - remain / CALIB_SEC, 35, YELLOW)
    if remain == 0:
        S["base"] = np.median(S["calib"], axis=0)
        S["mode"] = "RUN"


def evaluate(S, frame, neck, face, fallback):
    """기준 대비 목이 짧아진 비율 + 얼굴이 커진 비율로 판정. (상태, 색) 반환"""
    neck_drop = 1 - neck / S["base"][0]
    face_grow = face / S["base"][1] - 1
    S["scores"].append(neck_drop + face_grow)
    status, color = judge(sum(S["scores"]) / len(S["scores"]))
    put_text(frame, status + (f" (pose shoulder: {','.join(fallback)})" if fallback else ""), (PAD, 25), color, TEXT_L)
    put_text(frame, f"Neck {-neck_drop * 100:+.0f}%  Face {face_grow * 100:+.0f}%", (PAD, 50))
    return status, color


def update_stats(S, frame, status, now, dt):
    """판정 시간 누적, 거북목 ALERT_SEC 이상 지속 시 경고, 바른 자세 비율 바"""
    if status == "TURTLE NECK":
        S["bad_t"] += dt
        S["bad_since"] = S["bad_since"] or now
        if now - S["bad_since"] >= ALERT_SEC:
            draw_alert(frame, now - S["bad_since"])
    elif status:
        S["good_t"] += dt
        S["bad_since"] = None
    total = S["good_t"] + S["bad_t"]
    if S["mode"] == "RUN" and total > 0:
        draw_ratio_bar(frame, S["good_t"] / total)


# ======================== 8. 프레임 처리 ========================
def process(S, frame, now, dt):
    """한 프레임: 전처리 → 신체 인식 → 어깨 추적 → 측정 → 상태별 처리 → 표시"""
    frame, lab = preprocess(frame)
    S["shape"] = frame.shape[:2]
    body = detect_body(frame)
    status = None

    if body is None:
        put_text(frame, "No person detected", (PAD, 25), RED, TEXT_L)
    else:
        sh, fallback = get_shoulders(lab, body, S["trackers"])
        neck, face, mid = measure(body, sh)
        color = YELLOW
        if S["mode"] == "CALIB":
            calibrate(S, frame, neck, face, now)
        elif S["mode"] == "RUN":
            status, color = evaluate(S, frame, neck, face, fallback)
        draw_body(frame, body, sh, mid, fallback, color)

    if S["mode"] == "READY":
        put_text(frame, "Click READY (or SPACE) and keep good posture", (PAD, 50 if body is None else 25), YELLOW, TEXT_L)
        draw_button(frame)

    update_stats(S, frame, status, now, dt)
    return frame


def main():
    S = new_state()
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click, S)
    print(f"turtle_neck 시작 - READY 버튼 클릭 또는 Space")

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or not handle_key(S, cv2.waitKey(1) & 0xFF):
            break
        now = time.time()
        cv2.imshow(WIN, process(S, frame, now, now - prev))
        prev = now

    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()