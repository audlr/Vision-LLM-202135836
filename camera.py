"""카메라 입력 모듈: webcam / csi(Jetson GStreamer) / file 을 같은 인터페이스로 제공"""
import re
import cv2


class Camera:
    def __init__(self, cap, name):
        self.cap, self.name = cap, name

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def gstreamer_available():
    m = re.search(r"GStreamer:\s*(\w+)", cv2.getBuildInformation())
    return bool(m) and m.group(1).upper() == "YES"


def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    # 센서는 1280x720으로 받고, nvvidconv(GPU)에서 출력 크기로 변환
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )


def _open_csi(sensor_id=0, width=1280, height=720, fps=30, flip=0, **_):
    if not gstreamer_available():
        raise RuntimeError(
            "현재 OpenCV에 GStreamer 지원이 없습니다.\n"
            "pip로 설치한 opencv-python이 JetPack OpenCV를 덮어쓴 경우가 대부분입니다.\n"
            "venv를 --system-site-packages로 만들고 opencv 계열 pip 패키지를 제거하세요."
        )
    pipeline = gstreamer_pipeline(sensor_id, width, height, fps, flip)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "CSI 카메라를 열 수 없습니다.\n"
            "연결 확인: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! fakesink\n"
            f"파이프라인: {pipeline}"
        )
    return Camera(cap, f"csi:{sensor_id}")


def _open_webcam(index=0, width=1280, height=720, fps=30, **_):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠 {index}번을 열 수 없습니다. --index 값을 바꿔보세요.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return Camera(cap, f"webcam:{index}")


def _open_file(path=None, **_):
    if not path:
        raise ValueError("file 소스는 --path가 필요합니다.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {path}")
    return Camera(cap, f"file:{path}")


_SOURCES = {"csi": _open_csi, "webcam": _open_webcam, "file": _open_file}


def open_camera(source="csi", **opts):
    if source not in _SOURCES:
        raise ValueError(f"알 수 없는 소스: {source} (가능: {list(_SOURCES)})")
    return _SOURCES[source](**opts)


def add_camera_args(parser):
    g = parser.add_argument_group("camera")
    g.add_argument("--source", choices=list(_SOURCES), default="csi")
    g.add_argument("--index", type=int, default=0, help="웹캠 번호")
    g.add_argument("--sensor-id", type=int, default=0, help="CSI 센서 번호")
    g.add_argument("--path", help="영상 파일 경로")
    g.add_argument("--width", type=int, default=960)
    g.add_argument("--height", type=int, default=540)
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--flip", type=int, default=0, help="CSI 회전 (0 그대로, 2 180도, 4 좌우, 6 상하)")
    return parser


def camera_from_args(args):
    return open_camera(args.source, index=args.index, sensor_id=args.sensor_id,
                       path=args.path, width=args.width, height=args.height,
                       fps=args.fps, flip=args.flip)
