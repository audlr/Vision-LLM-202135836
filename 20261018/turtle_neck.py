"""
거북목 판정기
- 측면 카메라, C7(목 뒤 튀어나온 뼈)과 귀(이주)에 서로 다른 색 스티커 부착
- 마커: 색 분할 + 윤곽선 + 칼만 추적, MediaPipe Pose로 탐색 영역 제한
- 판정: CVA = C7→귀 직선과 수평선의 각도 (작을수록 거북목)

실행
  Jetson CSI : python turtle_neck.py
  웹캠       : python turtle_neck.py --source webcam
  영상 파일  : python turtle_neck.py --source file --path side.mp4
  화면 없이  : python turtle_neck.py --headless
키: q 종료, r 통계 초기화
"""
import argparse
import math
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from camera import add_camera_args, camera_from_args

GOOD_DEG = 55
WARN_DEG = 50
ALERT_SEC = 5
MIN_AREA = 40
LOST_LIMIT = 15
POSE_WIDTH = 480    # Pose 추론용 축소 폭 (Jetson 부하 감소)
SIDE_RATIO = 0.5    # 양 어깨 x 간격이 귀-어깨 거리의 이 비율보다 크면 측면 아님

# 마커별 HSV 기본값 (하한, 상한)
MARKERS = {
    "c7":  ((35, 80, 60), (85, 255, 255)),     # 초록
    "ear": ((140, 80, 80), (175, 255, 255)),   # 분홍
}
MARKER_COLOR = {"c7": (255, 255, 0), "ear": (255, 0, 255)}

if not hasattr(mp, "solutions"):
    raise ImportError(f"mediapipe {mp.__version__}에는 solutions API가 없습니다. "
                      "pip install mediapipe==0.10.14 로 설치하세요.")
mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
KERNEL = np.ones((5, 5), np.uint8)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class HSVControl:
    """GUI 모드는 트랙바, headless 모드는 기본값 사용"""
    NAMES = ("H_low", "S_low", "V_low", "H_high", "S_high", "V_high")
    MAX = (179, 255, 255, 179, 255, 255)

    def __init__(self, key, gui):
        self.win, self.gui = f"mask_{key}", gui
        lo, hi = MARKERS[key]
        self.default = (*lo, *hi)
        if gui:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            for n, v, mx in zip(self.NAMES, self.default, self.MAX):
                cv2.createTrackbar(n, self.win, v, mx, lambda _: None)

    def range(self):
        v = [cv2.getTrackbarPos(n, self.win) for n in self.NAMES] if self.gui else self.default
        return np.array(v[:3]), np.array(v[3:])


class MarkerTracker:
    """등속 모델 칼만 필터. 잠깐 가려지면 예측값 유지"""

    def __init__(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf, self.on, self.lost = kf, False, 0
        self.trail = deque(maxlen=30)

    def update(self, meas):
        if not self.on:
            if meas is None:
                return None
            self.kf.statePost = np.array([[meas[0]], [meas[1]], [0], [0]], np.float32)
            self.on = True

        est = self.kf.predict()
        if meas is not None:
            est = self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            self.lost = 0
        else:
            self.lost += 1
            if self.lost > LOST_LIMIT:
                self.on = False
                self.trail.clear()
                return None

        pt = (float(est[0, 0]), float(est[1, 0]))
        self.trail.append(pt)
        return pt


# ---------------- 영상 처리 ----------------
def enhance(frame):
    adj = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    l, a, b = cv2.split(cv2.cvtColor(adj, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def find_marker(img, lo, hi, roi):
    """ROI 안에서만 색 분할 후 가장 큰 윤곽선의 중심 반환"""
    x1, y1, x2, y2 = roi
    full = np.zeros(img.shape[:2], np.uint8)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, full, None

    hsv = cv2.cvtColor(cv2.GaussianBlur(img[y1:y2, x1:x2], (7, 7), 0), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL)
    full[y1:y2, x1:x2] = m

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x1, y1))
    cnts = [c for c in cnts if cv2.contourArea(c) > MIN_AREA]
    if not cnts:
        return None, full, None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]), full, c


def run_pose(pose, img):
    h, w = img.shape[:2]
    if w > POSE_WIDTH:
        img = cv2.resize(img, (POSE_WIDTH, int(h * POSE_WIDTH / w)))
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return res.pose_landmarks.landmark if res.pose_landmarks else None


def body_points(lm, w, h):
    """카메라 쪽 귀/어깨 좌표, 귀-어깨 거리, 측면 여부"""
    left = lm[PL.LEFT_EAR].visibility >= lm[PL.RIGHT_EAR].visibility
    ear = lm[PL.LEFT_EAR if left else PL.RIGHT_EAR]
    sh = lm[PL.LEFT_SHOULDER if left else PL.RIGHT_SHOULDER]
    ear_p, sh_p = (ear.x * w, ear.y * h), (sh.x * w, sh.y * h)
    d = max(math.dist(ear_p, sh_p), 20)
    gap = abs(lm[PL.LEFT_SHOULDER].x - lm[PL.RIGHT_SHOULDER].x) * w
    return ear_p, sh_p, d, gap < SIDE_RATIO * d


def box(cx, cy, half, w, h):
    return (max(int(cx - half), 0), max(int(cy - half), 0),
            min(int(cx + half), w), min(int(cy + half), h))


# ---------------- 판정 ----------------
def calc_cva(ear, c7):
    return math.degrees(math.atan2(c7[1] - ear[1], abs(ear[0] - c7[0])))


def classify(angle):
    if angle >= GOOD_DEG:
        return "GOOD", (0, 200, 0)
    if angle >= WARN_DEG:
        return "WARNING", (0, 200, 255)
    return "TURTLE NECK", (0, 0, 255)


# ---------------- 표현 ----------------
def draw_marker(img, roi, cnt, trail, color):
    cv2.rectangle(img, roi[:2], roi[2:], color, 1)
    if cnt is not None:
        cv2.drawContours(img, [cnt], -1, color, 2)
    pts = [tuple(map(int, p)) for p in trail]
    for p1, p2 in zip(pts, pts[1:]):
        cv2.line(img, p1, p2, color, 1)


def draw_overlay(img, ear, c7, angle, status, color, ear_fallback):
    e, c = tuple(map(int, ear)), tuple(map(int, c7))
    front = e[0] >= c[0]
    cv2.line(img, c, (c[0] + (100 if front else -100), c[1]), (200, 200, 200), 2)
    cv2.line(img, c, e, color, 3)
    cv2.circle(img, e, 7, color, -1)
    cv2.circle(img, c, 7, color, -1)
    cv2.ellipse(img, c, (50, 50), 0, 0 if front else 180,
                -angle if front else 180 + angle, color, 2)
    cv2.putText(img, f"{angle:.1f} deg", (c[0] + 10, c[1] + 30), FONT, 0.7, color, 2)
    label = status + (" (ear: pose)" if ear_fallback else "")
    cv2.putText(img, label, (20, 40), FONT, 1.0, color, 3)


def draw_stats(img, good_t, bad_t, fps):
    h, w = img.shape[:2]
    total = good_t + bad_t
    ratio = good_t / total if total > 0 else 1.0
    x0, y0, bw = 20, h - 40, w - 40
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + 20), (0, 0, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + int(bw * ratio), y0 + 20), (0, 200, 0), -1)
    cv2.putText(img, f"Good posture {ratio * 100:.0f}%  ({total:.0f}s)",
                (x0, y0 - 8), FONT, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"{fps:.1f} FPS", (w - 130, 30), FONT, 0.6, (255, 255, 255), 2)


def draw_alert(img, sec):
    red = np.full_like(img, (0, 0, 255))
    cv2.addWeighted(red, 0.3, img, 0.7, 0, img)
    cv2.putText(img, f"STRAIGHTEN UP!  {sec:.0f}s", (20, 90), FONT, 1.1, (255, 255, 255), 3)


# ---------------- 메인 ----------------
def main(cam, gui=True, complexity=0):
    hsv_ctl = {k: HSVControl(k, gui) for k in MARKERS}
    trackers = {k: MarkerTracker() for k in MARKERS}
    angles = deque(maxlen=10)
    good_t = bad_t = 0.0
    bad_since = None
    prev, last_log = time.time(), 0.0
    fps = 0.0

    with mp_pose.Pose(model_complexity=complexity,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            frame = cam.read()
            if frame is None:
                break
            now = time.time()
            dt, prev = now - prev, now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            img = enhance(frame)
            h, w = img.shape[:2]
            out = img.copy()
            masks = {k: np.zeros((h, w), np.uint8) for k in MARKERS}
            status, msg, angle = None, "", None

            lm = run_pose(pose, img)
            if lm is None:
                msg = "NO PERSON"
                for t in trackers.values():
                    t.update(None)
            else:
                ear_p, sh_p, d, side = body_points(lm, w, h)
                rois = {"ear": box(*ear_p, 0.5 * d, w, h),
                        "c7": box(sh_p[0], sh_p[1] - 0.4 * d, 0.6 * d, w, h)}
                pts = {}
                for k in MARKERS:
                    meas, masks[k], cnt = find_marker(img, *hsv_ctl[k].range(), rois[k])
                    pts[k] = trackers[k].update(meas)
                    draw_marker(out, rois[k], cnt, trackers[k].trail, MARKER_COLOR[k])

                if pts["c7"] is None:
                    msg = "C7 MARKER LOST"
                elif not side:
                    msg = "TURN SIDEWAYS"
                else:
                    ear_pt = pts["ear"] or ear_p    # 귀 마커 없으면 Pose 귀로 대체
                    angles.append(calc_cva(ear_pt, pts["c7"]))
                    angle = sum(angles) / len(angles)
                    status, color = classify(angle)
                    draw_overlay(out, ear_pt, pts["c7"], angle, status, color, pts["ear"] is None)

            if status is None:
                # 측정 불가 구간은 통계에서 제외
                angles.clear()
                bad_since = None
                cv2.putText(out, msg, (20, 40), FONT, 1.0, (0, 200, 255), 3)
            elif status == "TURTLE NECK":
                bad_t += dt
                bad_since = bad_since or now
                if now - bad_since >= ALERT_SEC:
                    draw_alert(out, now - bad_since)
            else:
                good_t += dt
                bad_since = None

            if not gui:
                if now - last_log >= 1:
                    a = f"{angle:.1f}deg" if angle is not None else "-"
                    print(f"[{status or msg}] {a}  {fps:.1f}fps")
                    last_log = now
                continue

            draw_stats(out, good_t, bad_t, fps)
            cv2.imshow("Turtle Neck Detector", out)
            for k, m in masks.items():
                cv2.imshow(hsv_ctl[k].win, m)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                good_t = bad_t = 0.0
                bad_since = None

    if gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = add_camera_args(argparse.ArgumentParser(description="거북목 판정기"))
    parser.add_argument("--headless", action="store_true", help="화면 없이 터미널 출력만")
    parser.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                        help="Pose 모델 크기 (Jetson은 0 권장)")
    args = parser.parse_args()

    with camera_from_args(args) as cam:
        print(f"[camera] {cam.name}")
        try:
            main(cam, gui=not args.headless, complexity=args.complexity)
        except KeyboardInterrupt:
            pass
