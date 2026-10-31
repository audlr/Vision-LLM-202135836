## 20261025 과제03
- 파일: `20261025/Assignment03_202135836_정혜인.py`

## 내용
- NumPy(CPU)와 CuPy(GPU)로 이미지 잔상 효과를 구현
  - CPU: 출력 픽셀을 반복문으로 하나씩 계산
  - GPU: 이동 거리별로 이미지 전체를 배열 slicing으로 병렬 계산

## 실행 환경
- Python 3, NumPy, CuPy, OpenCV, Matplotlib
- NVIDIA GPU 및 CUDA 필요 (CuPy 버전은 CUDA 버전에 맞게 설치, 예: `cupy-cuda12x`)

## 실행 방법
```bash
python Assignment03_202135836_정혜인.py
