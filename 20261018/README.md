# 거북목 판정기

측면 카메라로 CVA(두개척추각)를 측정해 거북목 여부를 실시간으로 판정합니다.

## 원리
- 목 뒤 C7(고개를 숙이면 튀어나오는 뼈)에 **초록**, 귀 앞 연골(이주)에 **분홍** 스티커 부착
- 색 분할 → 윤곽선 → 칼만 필터로 두 마커 추적
- MediaPipe Pose로 머리·목 주변만 탐색하고, 측면 자세인지 확인
- CVA = C7→귀 직선과 수평선 사이 각도 (55° 이상 정상, 50~55° 주의, 50° 미만 거북목)

## 파일
| 파일 | 내용 |
|---|---|
| `camera.py` | 카메라 입력 (CSI / 웹캠 / 영상 파일) |
| `turtle_neck.py` | 검출·추적·판정·화면 출력 |

## 설치

**Jetson**
```bash
python3 -m venv .venv --system-site-packages   # JetPack OpenCV(GStreamer) 사용
source .venv/bin/activate
pip install mediapipe==0.10.14 --no-deps        # opencv 덮어쓰기 방지
pip install absl-py attrs flatbuffers protobuf==4.25.3 matplotlib
```
Jetson에서 pip로 MediaPipe 설치가 안 되면, 수업에서 안내한 빌드 방법을 따르세요.

**노트북**
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행
```bash
python turtle_neck.py                                   # Jetson CSI (기본)
python turtle_neck.py --flip 2                          # 카메라가 뒤집혀 달렸을 때
python turtle_neck.py --source webcam                   # 노트북 웹캠
python turtle_neck.py --source file --path side.mp4     # 녹화 영상
python turtle_neck.py --headless                        # 화면 없이 (SSH)
```
키: `q` 종료, `r` 통계 초기화

## 화면 메시지
| 메시지 | 의미 / 조치 |
|---|---|
| `C7 MARKER LOST` | 초록 마커 미검출 → `mask_c7` 창 트랙바로 HSV 조정 |
| `(ear: pose)` | 분홍 마커 미검출, Pose 귀 좌표로 대신 측정 |
| `TURN SIDEWAYS` | 몸이 카메라를 향함 → 옆으로 서기 |
| `NO PERSON` | 사람 미검출 |

## 주의
- MediaPipe는 **0.10.14**로 고정해야 합니다. 이후 버전에는 `mp.solutions` API가 없습니다.
- Jetson에서 `opencv-python`을 pip로 설치하면 GStreamer 지원이 사라져 CSI 카메라가 열리지 않습니다.
- 판정 기준 각도는 `turtle_neck.py` 상단 `GOOD_DEG`, `WARN_DEG` 값에서 조정합니다.
