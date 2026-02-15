"""전체 코드 분석 문서 PDF 생성 스크립트"""

import markdown
from weasyprint import HTML

md_content = r"""
# HY-eat 식당 대기시간 추정 시스템 - 전체 코드 분석서

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [main.py - 엔트리포인트](#3-mainpy---엔트리포인트)
4. [src/core/camera.py - 카메라 관리](#4-srccorecamerapy---카메라-관리)
5. [src/core/detector.py - YOLOv8 검출기](#5-srccoredetectorpy---yolov8-검출기)
6. [src/core/tracker.py - ByteTrack 추적기](#6-srccoretrackerpy---bytetrack-추적기)
7. [src/core/roi_manager.py - ROI 관리자](#7-srccoreroimanagerpy---roi-관리자)
8. [src/core/wait_time_estimator.py - 대기시간 예측](#8-srccorewait_time_estimatorpy---대기시간-예측)
9. [src/web/app.py - Flask 웹 서버](#9-srcwebapppy---flask-웹-서버)
10. [src/cloud/dynamodb_sender.py - DynamoDB 전송](#10-srcclouddynamodb_senderpy---dynamodb-전송)
11. [tests/test_dynamodb_sender.py - 단위 테스트](#11-teststest_dynamodb_senderpy---단위-테스트)
12. [frontend/src/types/hyeat.ts - TypeScript 타입](#12-frontendsrctypeshyeatts---typescript-타입)
13. [설정 파일](#13-설정-파일)
14. [데이터 흐름도](#14-데이터-흐름도)

---

## 1. 프로젝트 개요

### 시스템 목적
NVIDIA Jetson Orin Nano 위에서 동작하는 실시간 식당 대기시간 추정 시스템이다. CSI 카메라로 촬영한 영상에서 YOLOv8 + TensorRT로 사람을 검출하고, ByteTrack으로 추적하며, ROI 기반 대기열 인원을 측정하여 예상 대기시간을 예측한다. 예측 결과는 AWS DynamoDB로 전송되어 프론트엔드 대시보드에서 소비된다.

### 기술 스택
| 분류 | 기술 |
|------|------|
| 하드웨어 | NVIDIA Jetson Orin Nano (8GB) |
| 카메라 | Arducam IMX219 CSI 카메라 |
| GPU 추론 | TensorRT (FP16), CUDA Runtime (ctypes) |
| 객체 검출 | YOLOv8n (Ultralytics ONNX) |
| 객체 추적 | ByteTrack + Kalman Filter (cv2.KalmanFilter) |
| 웹 서버 | Flask + Waitress (WSGI) |
| 클라우드 | AWS DynamoDB (boto3) |
| 프론트엔드 | TypeScript 타입 정의 (향후 React/Next.js) |
| 영상 인코딩 | TurboJPEG (libjpeg-turbo NEON SIMD) |

### 개발 이력 (Phase)
| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | CSI 카메라 + GStreamer 파이프라인 | 완료 |
| Phase 2 | YOLOv8 + TensorRT 검출기 | 완료 |
| Phase 3 | ROI 관리 (다각형, 웹 UI) | 완료 |
| Phase 4 | ByteTrack 추적기 + 칼만 필터 | 완료 |
| Phase 5 | 대기시간 측정/예측 알고리즘 | 완료 |
| Phase 6 | DynamoDB 데이터 전송 | 완료 |
| Phase 7 | TypeScript 타입 정의 | 완료 |

---

## 2. 디렉토리 구조

```
aidea/
├── main.py                          # 엔트리포인트 (167줄)
├── requirements.txt                 # 의존성 (flask, pyyaml, boto3)
├── config/
│   ├── roi_config.json              # ROI 다각형 설정
│   └── aws_config.json              # DynamoDB 설정 (Phase 6)
├── models/
│   ├── yolov8n.onnx                 # YOLOv8n ONNX 모델
│   └── yolov8n_fp16.engine          # TensorRT FP16 캐시
├── src/
│   ├── core/
│   │   ├── camera.py                # CSI 카메라 관리 (217줄)
│   │   ├── detector.py              # TensorRT 검출기 (521줄)
│   │   ├── tracker.py               # ByteTrack 추적기 (432줄)
│   │   ├── roi_manager.py           # ROI 관리자 (244줄)
│   │   └── wait_time_estimator.py   # 대기시간 예측 (450줄)
│   ├── cloud/
│   │   └── dynamodb_sender.py       # DynamoDB 전송 (286줄)
│   └── web/
│       ├── app.py                   # Flask 서버 (517줄)
│       └── templates/
│           └── index.html           # 웹 대시보드 UI
├── tests/
│   └── test_dynamodb_sender.py      # 단위 테스트 (285줄)
├── frontend/
│   └── src/types/
│       └── hyeat.ts                 # TypeScript 타입 정의 (148줄)
├── docs/                            # 문서
├── scripts/                         # 셋업 스크립트
└── data/                            # 스냅샷/통계 저장
```

**전체 Python 코드**: 약 2,900줄 (10개 파일)

---

## 3. main.py - 엔트리포인트

**파일**: `main.py` (167줄)

### 역할
시스템의 모든 컴포넌트를 초기화하고 Flask 웹 서버를 시작하는 엔트리포인트이다.

### 초기화 순서

```
1. CameraManager       → CSI/USB 카메라 열기
2. YOLOv8Detector      → ONNX→TensorRT 엔진 로드/빌드
3. ByteTracker         → 칼만 필터 기반 추적기
4. ROIManager          → JSON에서 ROI 다각형 로드
5. WaitTimeEstimator   → 대기시간 예측기 (--start-roi 지정 시)
6. DynamoDBSender      → DynamoDB 클라이언트 (--no-dynamodb 미지정 시)
7. Flask init_app()    → 모든 컴포넌트를 Flask에 주입
8. waitress.serve()    → WSGI 서버 시작 (8스레드)
```

### CLI 인자 (주요)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--host` | 0.0.0.0 | 서버 바인드 주소 |
| `--port` | 5000 | 서버 포트 |
| `--model` | models/yolov8n.onnx | YOLO 모델 경로 |
| `--conf-threshold` | 0.5 | 검출 신뢰도 임계값 |
| `--start-roi` | None | 대기시간 측정 시작 ROI 이름 |
| `--end-roi` | None | 대기시간 측정 종료 ROI 이름 |
| `--no-dynamodb` | False | DynamoDB 전송 비활성화 |
| `--aws-config` | config/aws_config.json | AWS 설정 파일 경로 |
| `--min-dwell` | 30 | ROI 최소 체류 프레임 수 |

### 종료 처리 (SIGINT/SIGTERM)
```
shutdown() 호출 순서:
1. DynamoDBSender.stop()  → 남은 큐 플러시
2. CameraManager.stop()   → 캡처 스레드 종료
3. YOLOv8Detector.destroy()→ GPU 메모리 해제
```

### 핵심 코드 분석

- **라인 102-110**: `WaitTimeEstimator`는 `--start-roi` 인자가 지정된 경우에만 생성된다. 단일 ROI 모드(end_roi=None)와 플로우 모드를 모두 지원한다.
- **라인 112-121**: `DynamoDBSender`는 `--no-dynamodb`가 아니고 `aws_config.json`이 존재할 때만 초기화된다. 초기화 실패 시 경고만 출력하고 시스템은 계속 동작한다 (graceful degradation).
- **라인 113-119**: `waitress.serve()`를 먼저 시도하고, waitress가 설치되지 않은 경우 Flask 개발 서버로 폴백한다.

---

## 4. src/core/camera.py - 카메라 관리

**파일**: `src/core/camera.py` (217줄)
**클래스**: `CameraManager`

### 역할
Jetson Orin Nano의 CSI 카메라(Arducam IMX219)를 GStreamer 파이프라인으로 관리한다. USB 카메라 폴백, 자동 재연결, 하드웨어 JPEG 인코딩을 지원한다.

### 아키텍처

```
nvarguscamerasrc (CSI)
    │
    ▼
NV12 → nvvidconv (GPU 스케일링 + 플립)
    │
    ▼
BGRx → videoconvert → BGR (OpenCV 호환)
    │
    ▼
appsink (drop=1, 최신 프레임만)
    │
    ▼
_capture_loop() [백그라운드 스레드]
    │
    ▼
self._frame (Lock으로 보호)
    │
    ▼
get_frame() → Inference 스레드에서 폴링
```

### GStreamer 파이프라인 상세

**일반 캡처 파이프라인** (`_build_gstreamer_pipeline()`):
```
nvarguscamerasrc sensor-id=0 !
video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 !
nvvidconv flip-method=0 !
video/x-raw, width=640, height=480, format=BGRx !
videoconvert !
video/x-raw, format=BGR !
appsink drop=1
```

- `nvarguscamerasrc`: Jetson 전용 CSI 카메라 소스 (ISP 하드웨어 활용)
- `memory:NVMM`: NVIDIA Multimedia Memory (GPU 메모리에서 직접 처리)
- `nvvidconv`: GPU 가속 색상 변환 + 스케일링 (1280x720 → 640x480)
- `appsink drop=1`: 소비가 느릴 때 이전 프레임 자동 드롭

**HW JPEG 파이프라인** (`_build_jpeg_pipeline()`):
- `nvjpegenc`: Jetson NVJPEG 하드웨어 인코더 활용
- 현재는 참조용으로만 존재 (실제 사용은 TurboJPEG)

### 스레드 안전 설계

- `_frame`: 최신 프레임을 저장하는 단일 슬롯. `threading.Lock()`으로 보호.
- `get_frame()`: Lock 획득 후 프레임을 반환하고 `None`으로 교체. 소유권이 호출자에게 이전되어 동일 프레임의 중복 처리를 방지한다.
- `_capture_loop()`: 백그라운드 데몬 스레드에서 무한 루프로 프레임을 캡처한다.

### 자동 재연결 (`_reconnect()`)

```
연속 30프레임 읽기 실패
    ↓
기존 VideoCapture 해제 (1초 대기)
    ↓
CSI 파이프라인 재시도
    ↓ (실패 시)
USB 카메라(0번) 폴백
    ↓ (실패 시)
3초 대기 후 재시도
```

### 주요 메서드

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `start()` | bool | 카메라 열기 + 캡처 스레드 시작 |
| `stop()` | None | 스레드 종료 + 리소스 해제 |
| `get_frame()` | ndarray/None | 최신 BGR 프레임 (소유권 이전) |
| `get_jpeg_frame()` | bytes/None | JPEG 인코딩된 프레임 |
| `is_running` | bool | 카메라 동작 상태 (property) |

---

## 5. src/core/detector.py - YOLOv8 검출기

**파일**: `src/core/detector.py` (521줄)
**클래스**: `CudaRT`, `YOLOv8Detector`

### 역할
TensorRT 엔진으로 YOLOv8n 객체 검출을 수행한다. ONNX → TensorRT 자동 변환, FP16 추론, NMS 후처리를 포함한다.

### CudaRT 클래스 (ctypes CUDA 래퍼)

PyCUDA 없이 CUDA Runtime API를 직접 호출하는 래퍼다. Jetson에서 PyCUDA 설치가 불안정한 문제를 우회한다.

**래핑하는 CUDA API**:

| API | 메서드 | 설명 |
|-----|--------|------|
| `cudaMalloc` | `malloc(size)` | GPU 메모리 할당 |
| `cudaFree` | `free(d_ptr)` | GPU 메모리 해제 |
| `cudaMemcpy` | `memcpy_htod/dtoh()` | 동기 메모리 복사 |
| `cudaMemcpyAsync` | `memcpy_htod/dtoh_async()` | 비동기 메모리 복사 |
| `cudaMallocHost` | `malloc_host(size)` | Pinned 호스트 메모리 |
| `cudaFreeHost` | `free_host(h_ptr)` | Pinned 메모리 해제 |
| `cudaStreamCreate` | `stream_create()` | CUDA 스트림 생성 |
| `cudaStreamSynchronize` | `stream_synchronize()` | 스트림 동기화 |
| `cudaStreamDestroy` | `stream_destroy()` | 스트림 파괴 |

**libcudart.so 검색 순서**:
1. `/usr/local/cuda-12.6/lib64/libcudart.so`
2. `/usr/local/cuda/lib64/libcudart.so`
3. `libcudart.so` (시스템 경로)

### YOLOv8Detector 클래스

#### 추론 파이프라인

```
BGR 프레임 (640x480)
    ↓ _preprocess()
Letterbox Resize (640x640) + RGB 변환 + Normalize [0,1]
    ↓ np.copyto()
Pinned Host Memory (입력 버퍼)
    ↓ cudaMemcpyAsync HtoD
GPU Input Buffer
    ↓ execute_async_v3() 또는 execute_v2()
GPU Output Buffer
    ↓ cudaMemcpyAsync DtoH
Pinned Host Memory (출력 버퍼)
    ↓ _postprocess()
검출 결과: [{bbox, confidence, class_id}, ...]
```

#### Letterbox 전처리 (`_preprocess()`)

```python
# 원본 프레임의 종횡비를 유지하면서 640x640에 맞춤
scale = min(640/img_w, 640/img_h)
new_w, new_h = int(w*scale), int(h*scale)
img = cv2.resize(img, (new_w, new_h))

# 나머지 영역을 114(회색)로 채움
canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
canvas[dy:dy+new_h, dx:dx+new_w] = img

# CHW 변환 + float32 + normalize
tensor = canvas / 255.0
tensor = tensor.transpose(2, 0, 1)  # HWC → CHW
tensor = np.expand_dims(tensor, 0)  # 배치 차원 추가
```

#### 후처리 (`_postprocess()`)

YOLOv8 출력 형태: `[1, 84, 8400]`
- 84 = 4 (cx, cy, w, h) + 80 (COCO 클래스 score)
- 8400 = 검출 후보 수

```
1. Transpose: (84, 8400) → (8400, 84)
2. 클래스별 최대 score 추출 + class_id 결정
3. 신뢰도 임계값 필터링 (conf_threshold)
4. 타겟 클래스 필터링 (target_classes=[0] → person만)
5. cx,cy,w,h → x1,y1,x2,y2 변환
6. Letterbox 역변환 (스케일/오프셋 보정)
7. 좌표 클리핑 (화면 범위)
8. NMS (cv2.dnn.NMSBoxes)
```

#### Pinned Memory vs 일반 메모리

| 항목 | Pinned Memory | 일반 메모리 |
|------|--------------|------------|
| 할당 | `cudaMallocHost()` | `np.zeros()` |
| 전송 | `cudaMemcpyAsync()` (비동기 DMA) | `cudaMemcpy()` (동기) |
| 장점 | CPU-GPU 병렬 실행 가능 | 항상 사용 가능 |
| 단점 | 할당 실패 가능 | GPU 전송 중 CPU 블로킹 |

코드에서는 Pinned Memory를 먼저 시도하고 (`_use_async=True`), 실패 시 일반 메모리로 폴백한다.

#### 엔진 캐싱

ONNX → TensorRT 변환은 Jetson에서 수분이 소요된다. 따라서 한 번 빌드한 엔진을 파일로 캐싱한다:

```
yolov8n.onnx → yolov8n_fp16.engine (FP16)
yolov8n.onnx → yolov8n_fp32.engine (FP32)
```

캐시 파일이 존재하면 `_load_engine()`으로 즉시 로드한다.

#### FPS 측정

- `deque(maxlen=30)`: 최근 30프레임의 소요 시간을 저장
- FPS = 30 / sum(최근 30프레임 소요시간)
- 이동 윈도우 방식으로 안정적인 FPS 측정

#### 리소스 해제 (`destroy()`)

해제 순서가 중요하다:
```
1. CUDA 스트림 해제
2. Execution Context 해제 (None 할당)
3. Engine 해제 (None 할당)
4. Pinned 호스트 메모리 해제
5. GPU 디바이스 메모리 해제
```

---

## 6. src/core/tracker.py - ByteTrack 추적기

**파일**: `src/core/tracker.py` (432줄)
**클래스**: `KalmanBoxTracker`, `ByteTracker`, `ROIDwellFilter`

### 역할
프레임 간 동일 인물을 추적하여 고유 track_id를 할당한다. 일시적 가림(occlusion)에도 ID를 유지한다.

### KalmanBoxTracker 클래스

cv2.KalmanFilter를 래핑한 단일 바운딩박스 추적기다.

**상태 벡터 (7D)**: `[cx, cy, s, r, dx, dy, ds]`
- cx, cy: 바운딩박스 중심 좌표
- s: 면적 (w * h)
- r: 종횡비 (w / h) - 상수로 가정
- dx, dy, ds: 속도 (등속 모델)

**측정 벡터 (4D)**: `[cx, cy, s, r]`

**좌표 변환**:
```
bbox [x1,y1,x2,y2] ↔ state [cx,cy,s,r]

_bbox_to_z(): 바운딩박스 → 칼만 상태
  w = x2 - x1, h = y2 - y1
  cx = x1 + w/2, cy = y1 + h/2
  s = w * h, r = w / h

_z_to_bbox(): 칼만 상태 → 바운딩박스
  w = sqrt(s * r)
  h = s / w
  x1 = cx - w/2, y1 = cy - h/2
  x2 = cx + w/2, y2 = cy + h/2
```

**칼만 필터 행렬 설정**:
- F (전이행렬): 7x7 항등행렬 + 속도 반영 (cx += dx, cy += dy, s += ds)
- H (측정행렬): 4x7, 위치 성분만 관측
- Q (프로세스 노이즈): 위치 1e-2, 속도 1e-2, 면적속도 5e-3
- R (측정 노이즈): 1e-1
- P (초기 오차 공분산): 위치 1.0, 속도 10.0

**트랙 ID**: 클래스 변수 `_next_id`로 단조 증가. 절대 재사용하지 않는다.

### IoU 배치 계산 (`_iou_batch()`)

numpy 브로드캐스팅을 사용한 벡터화 IoU 계산:

```
입력: bboxes_a (N,4), bboxes_b (M,4)
출력: IoU 행렬 (N,M)

방법:
1. a를 (N,1,4)로, b를 (1,M,4)로 reshape
2. 교차 영역 계산 (element-wise max/min)
3. 합집합 영역 계산
4. IoU = 교차 / 합집합
```

scipy 없이 순수 numpy로 구현하여 Jetson 의존성을 최소화했다.

### 그리디 매칭 (`_greedy_assignment()`)

```
1. IoU 행렬의 모든 값을 내림차순 정렬
2. 높은 IoU부터 탐욕적으로 (row, col) 쌍 매칭
3. 이미 매칭된 row/col은 건너뜀
4. IoU < threshold이면 중단
```

헝가리안 알고리즘(O(n^3)) 대비 O(n*m*log(n*m))으로 약간 차선이지만, 사람 수가 적은 식당 환경에서는 충분하다.

### ByteTracker 클래스

**2단계 연관 알고리즘**:

```
Step 1: 칼만 예측 (모든 기존 트랙)
    ↓
Step 2: 검출 분류
    고신뢰도 (≥ high_thresh=0.5)
    저신뢰도 (≥ low_thresh=0.1)
    ↓
Step 3: 1단계 매칭
    고신뢰도 검출 ↔ 전체 트랙 (IoU)
    → 매칭 성공: 칼만 보정
    → 미매칭 트랙: 2단계로
    ↓
Step 4: 2단계 매칭
    저신뢰도 검출 ↔ 1단계 미매칭 트랙 (IoU)
    → 매칭 성공: 칼만 보정
    ↓
Step 5: 미매칭 고신뢰도 검출 → 새 트랙 생성
    ↓
Step 6: time_since_update > max_age → 트랙 삭제
    ↓
Step 7: 출력 필터링
    hits >= min_hits 인 트랙만 출력
    이번 프레임에 매칭된 트랙만 출력
```

**왜 2단계인가?**: 부분적으로 가려진 사람은 검출 신뢰도가 낮다. 1단계에서 고신뢰도 검출을 먼저 매칭하고, 남은 트랙에 저신뢰도 검출을 매칭하면 가림 상황에서도 트랙을 유지할 수 있다.

### ROIDwellFilter 클래스

**역할**: ROI를 잠깐 스치고 지나가는 사람을 인원수에서 제외한다.

**로직**:
```
매 프레임:
1. ROI 안의 (track_id, roi_name) 쌍 수집
2. 각 쌍의 연속 체류 프레임 수 카운트
3. ROI를 벗어난 쌍은 카운터 즉시 리셋
4. min_dwell_frames(기본 30 ≈ 1초) 이상 체류한 트랙만 카운팅
```

---

## 7. src/core/roi_manager.py - ROI 관리자

**파일**: `src/core/roi_manager.py` (244줄)
**클래스**: `ROIManager`

### 역할
다각형 ROI를 생성/수정/삭제하고, 검출 결과를 ROI별로 분류한다.

### 데이터 구조

```python
self._rois = [
    {
        "name": "대기구역",
        "points": [[x1,y1], [x2,y2], ...],  # 다각형 꼭짓점
        "color": [0, 255, 0],                # BGR 색상
    },
    ...
]
```

### 점-다각형 판별 (Point-in-Polygon)

검출된 사람이 ROI 안에 있는지 판별하는 두 가지 방법이 구현되어 있다:

1. **cv2.pointPolygonTest()** (실제 사용): OpenCV의 C++ 구현. GIL 해제 상태에서 실행되어 성능이 좋다.
2. **Ray-casting 알고리즘** (`_point_in_polygon()`): 순수 Python 폴백. 반직선과 다각형 변의 교차 횟수로 내부/외부 판별.

**판별 기준점**: 바운딩박스의 하단 중심 `(cx, y2)`. 사람의 발 위치가 실제 위치를 가장 잘 반영한다.

### 색상 팔레트

8색 자동 할당 팔레트 (BGR):
```
green(0,255,0) → blue(255,0,0) → red(0,0,255) →
yellow(0,255,255) → magenta(255,0,255) → cyan(255,255,0) →
orange(0,165,255) → pink(203,192,255)
```

새 ROI 추가 시 `len(rois) % 8`로 색상을 자동 선택한다.

### 반투명 오버레이 (`draw_rois()`)

```python
# 성능 최적화: 오버레이 버퍼 재사용
if self._overlay is None or self._overlay.shape != frame.shape:
    self._overlay = np.empty_like(frame)
np.copyto(self._overlay, frame)

cv2.fillPoly(self._overlay, [pts], color)       # 반투명 채우기
cv2.polylines(frame, [pts], True, color, 2)     # 불투명 테두리
cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)    # 알파 블렌딩
```

매 프레임 `np.empty_like()`를 호출하지 않고 버퍼를 재사용하여 메모리 할당을 최소화한다.

### 스레드 안전

모든 ROI 접근은 `threading.Lock()`으로 보호된다. `get_all_rois()`는 `dict(r)`로 복사본을 반환하여 외부에서 원본을 변경할 수 없다.

### JSON 영속화

ROI 추가/수정/삭제 시 자동으로 `config/roi_config.json`에 저장된다. 서버 재시작 후에도 ROI 설정이 유지된다.

---

## 8. src/core/wait_time_estimator.py - 대기시간 예측

**파일**: `src/core/wait_time_estimator.py` (450줄)
**클래스**: `EMAPredictor`, `MovingAveragePredictor`, `HybridPredictor`, `WaitTimeEstimator`

### 역할
ROI 진입/퇴출 이벤트를 감지하고, 실제 대기시간을 측정하며, 다음 고객의 예상 대기시간을 예측한다.

### 예측 알고리즘 비교

| 알고리즘 | 공식 | 메모리 | 연산 | 적합 상황 |
|----------|------|--------|------|-----------|
| EMA | EMA[t] = α×sample + (1-α)×EMA[t-1] | O(1) | O(1) | 초기 데이터 부족, 빠른 반응 |
| Moving Average | MA = sum(window) / len(window) | O(N) | O(1) | 안정적 환경 |
| **Hybrid** (기본) | EMA × (1 + β×(queue-avg)/avg) | O(N) | O(1) | 동적 식당 환경 |

### HybridPredictor 상세

Little's Law (L = λW) 기반 대기열 보정:

```
predicted = EMA × (1 + β × correction)

correction = (current_queue - avg_queue) / avg_queue

예시:
- 평균 대기열 5명, 현재 10명 (2배)
- correction = (10-5)/5 = 1.0
- β=0.3이면 30% 상향 보정
- EMA=300초일 때 predicted = 300 × 1.3 = 390초
```

### 동작 모드

**단일 ROI 모드** (`end_roi=None`):
```
사람이 ROI에 들어옴 → entry_time 기록
사람이 ROI에서 나감 → exit_time - entry_time = 대기시간
```

**플로우 모드** (`end_roi` 지정):
```
사람이 start_roi에 들어옴 → entry_time 기록
사람이 end_roi에 들어옴   → entry_time 제거, 대기시간 완료
```

### 이벤트 감지 메커니즘

상태 전이 기반 (이전 프레임 vs 현재 프레임 비교):

```python
prev_state = {roi_name: set(track_ids)}  # 이전 프레임
current_state = {roi_name: set(track_ids)}  # 현재 프레임

# 진입: 현재에 있고 이전에 없음
entries = current - previous

# 퇴출: 이전에 있고 현재에 없음
exits = previous - current
```

### 이상치 필터링 (IQR)

```
정렬된 대기시간 히스토리에서:
Q1 = 25% 백분위수
Q3 = 75% 백분위수
IQR = Q3 - Q1

이상치 조건:
value < Q1 - 1.5 × IQR  또는  value > Q3 + 1.5 × IQR
```

- 최소 10개 샘플이 모여야 활성화
- 3σ 방식 대비 분포 가정이 없어 강건

### 스테일 트랙 정리

ByteTracker가 트랙을 잃으면 `_entry_times`에 영구 잔류하여 메모리 누수가 발생한다. 1초마다 `stale_timeout`(기본 300초) 초과 항목을 자동 정리한다.

### update() 반환값

```python
{
    'events': [{'type': 'ENTRY'|'EXIT', 'track_id': int, ...}],
    'predicted_wait': float,   # 예측 대기시간 (초)
    'current_queue': int,      # 현재 대기 인원
    'completed': [float, ...], # 이번 프레임 완료 대기시간들
    'active_waiters': {track_id: elapsed_seconds},
}
```

---

## 9. src/web/app.py - Flask 웹 서버

**파일**: `src/web/app.py` (517줄)

### 역할
3-Thread 파이프라인 기반 웹 서버. MJPEG 스트리밍, REST API, ROI 관리 UI를 제공한다.

### 3-Thread 아키텍처

```
Thread 1 (Camera)
    CameraManager._capture_loop()
    └→ _frame에 최신 프레임 저장
    │
    ▼ get_frame()
Thread 2 (Inference)
    _inference_loop()
    └→ YOLO 검출 → ByteTrack 추적
    └→ ROI 카운팅 → 체류 필터
    └→ WaitTimeEstimator 업데이트
    └→ DynamoDB 주기적 전송
    └→ JPEG 인코딩 → FrameBuffer.put()
    │
    ▼ FrameBuffer.get()
Thread 3 (Network)
    generate_frames()
    └→ MJPEG 스트림으로 yield
    └→ 느린 클라이언트는 프레임 스킵
```

**설계 의도**: 네트워크가 느려도(와이파이 끊김 등) Inference 스레드는 영향 없이 계속 동작한다.

### FrameBuffer 클래스

```
특성:
- 최신 1프레임만 유지 (오래된 프레임 자동 폐기)
- put(): 항상 논블로킹 (덮어쓰기)
- get(): Condition.wait()로 새 프레임 대기
- frame_id: 중복 수신 방지 (단조 증가)
- 여러 MJPEG 클라이언트가 동시에 읽어도 안전
```

### Inference Loop 상세 흐름

```
매 프레임:
1. 카메라 프레임 획득 (get_frame)
2. YOLO 검출 → detections
3. ByteTrack 추적 → tracked (track_id 포함)
4. ROI 분류 + 체류 필터 → roi_counts
5. [Phase 5] WaitTimeEstimator.update(roi_dets)
6. [Phase 6] DynamoDB 전송 조건 체크
   - 10초 주기 OR 예측값 변경 시 (2초 쿨다운)
7. JPEG 인코딩 (TurboJPEG 우선, cv2.imencode 폴백)
8. FrameBuffer.put()
9. FPS 조절 sleep (최소 1ms GIL 양보)
```

### DynamoDB 전송 조건 (Phase 6 통합)

```python
# 두 가지 조건 중 하나 충족 시 전송:
time_elapsed >= 10.0              # 10초 주기
OR
(value_changed AND elapsed >= 2.0) # 예측값 변경 + 2초 쿨다운
```

### API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 웹 대시보드 HTML |
| GET | `/video_feed` | MJPEG 실시간 스트림 |
| GET | `/api/stats` | FPS, 검출수, 트랙ID, ROI 카운트 |
| GET | `/api/tracks` | 현재 활성 추적 결과 |
| GET | `/api/wait_time` | 대기시간 예측 + 통계 |
| GET | `/api/dynamodb/stats` | DynamoDB 전송 통계 |
| GET | `/api/roi` | 전체 ROI 목록 |
| POST | `/api/roi` | ROI 추가 |
| GET | `/api/roi/stats` | ROI별 인원수 |
| PUT | `/api/roi/<name>` | ROI 수정 |
| DELETE | `/api/roi/<name>` | ROI 삭제 |
| GET | `/health` | 헬스체크 |

### TurboJPEG vs cv2.imencode

```python
try:
    from turbojpeg import TurboJPEG
    _turbojpeg = TurboJPEG()
except:
    _turbojpeg = None
```

TurboJPEG (libjpeg-turbo)는 ARM NEON SIMD 명령어를 활용하여 cv2.imencode 대비 2-3배 빠르다. Jetson의 ARM 프로세서에서 특히 유리하다.

### 성능 모니터링

100프레임마다 평균 성능 로그를 출력한다:
```
[Perf/100f] jpeg(TurboJPEG)=2.1ms total=15.3ms (65.4fps)
```

---

## 10. src/cloud/dynamodb_sender.py - DynamoDB 전송

**파일**: `src/cloud/dynamodb_sender.py` (286줄)
**클래스**: `DynamoDBSender`

### 역할
Jetson에서 측정한 대기시간 데이터를 AWS DynamoDB로 비동기 전송한다.

### 아키텍처

```
send() [논블로킹]
    ↓ _transform()
snake_case → camelCase 변환
    ↓ append
내부 큐 (deque, Lock 보호)
    ↓
_worker_loop() [백그라운드 스레드]
    ↓ _drain_queue(25)
최대 25개 배치 추출
    ↓ _write_batch()
boto3 Table.batch_writer()
    ↓ 실패 시
exponential backoff 재시도 (최대 3회)
    ↓ 최종 실패 시
큐에 재적재 (데이터 유실 방지)
```

### 데이터 변환 (`_transform()`)

```
입력 (snake_case):                    출력 (camelCase DynamoDB):
─────────────────                     ────────────────────────
restaurant_id: "hanyang_plaza"    →   pk: "CORNER#hanyang_plaza#korean"
corner_id: "korean"               →   sk: "1770349800000" (string)
queue_count: 15                   →   restaurantId: "hanyang_plaza"
est_wait_time_min: 8              →   cornerId: "korean"
timestamp: 1770349800000          →   queueLen: 15
                                      estWaitTimeMin: 8
                                      dataType: "observed"
                                      source: "jetson_nano"
                                      timestampIso: "2026-02-15T17:00:00+09:00"
                                      createdAtIso: "2026-02-15T17:00:01+09:00"
                                      ttl: 1772941800 (30일 후)
```

### DynamoDB 테이블 설계

| 속성 | 역할 | 형식 |
|------|------|------|
| `pk` (PK) | Partition Key | `CORNER#{restaurantId}#{cornerId}` |
| `sk` (SK) | Sort Key | epoch milliseconds (문자열) |
| `ttl` | 자동 삭제 | epoch 초 + 30일 |

**PK/SK 패턴**: 단일 테이블 설계 (Single Table Design). 코너별 시계열 데이터를 하나의 테이블에 저장한다. SK가 타임스탬프이므로 시간 범위 쿼리가 자연스럽다.

### 재시도 전략 (Exponential Backoff)

```
시도 1: 0.5초 + jitter(0~0.5초) 대기
시도 2: 1.0초 + jitter(0~0.5초) 대기
시도 3: 2.0초 + jitter(0~0.5초) 대기
최종 실패: 큐에 재적재 (다음 배치에서 재시도)
```

jitter(랜덤 지연)는 여러 클라이언트의 동시 재시도(thundering herd)를 방지한다.

### 보안

- AWS 자격증명은 환경 변수(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)만 사용
- 코드에 하드코딩 없음
- boto3의 credential chain을 그대로 활용 (환경변수 → AWS config → IAM 역할)

### 종료 시 플러시

`stop()` 호출 시 `_stop_event.set()` 후 남은 큐를 모두 전송한다. 데이터 유실을 최소화한다.

---

## 11. tests/test_dynamodb_sender.py - 단위 테스트

**파일**: `tests/test_dynamodb_sender.py` (285줄)
**테스트 수**: 23개 (모두 통과)

### 테스트 구성

| 클래스 | 테스트 수 | 대상 |
|--------|-----------|------|
| `TestDynamoDBSenderConfig` | 3 | 설정 파일 로드/검증 |
| `TestTransform` | 11 | snake_case→camelCase 변환 |
| `TestSendQueue` | 3 | 전송 큐 관리 |
| `TestBatchWrite` | 2 | 배치 쓰기 + 재시도 |
| `TestWorkerLifecycle` | 2 | 워커 시작/종료/플러시 |
| `TestDrainQueue` | 2 | 큐 드레인 로직 |

### 주요 테스트 케이스

**TestTransform**: 변환 로직의 정확성을 검증한다.
- `test_pk_format`: PK 형식 `CORNER#hanyang_plaza#korean` 검증
- `test_sk_is_string_timestamp`: SK가 문자열 타입인지 검증
- `test_camel_case_keys`: 모든 키가 camelCase인지 검증
- `test_timestamp_iso_kst`: ISO 8601 형식 + KST(+09:00) 검증
- `test_ttl_30_days`: TTL = timestamp + 30일 검증
- `test_default_restaurant_corner_from_config`: 미지정 시 config 기본값 사용 검증

**TestBatchWrite**: boto3를 mock하여 실제 AWS 연결 없이 테스트한다.
- `test_write_batch_success`: 정상 배치 쓰기 3건
- `test_write_batch_retry_on_failure`: 2회 실패 후 3회째 성공 (재시도 검증)

**TestWorkerLifecycle**: 워커 스레드의 생명주기를 테스트한다.
- `test_start_stop`: 시작 → 데이터 전송 → 종료
- `test_flush_on_stop`: 종료 시 남은 큐 플러시 검증

### 테스트 헬퍼

```python
_make_config(tmpdir, **overrides)  # 임시 디렉토리에 설정 파일 생성
_sample_data(timestamp_ms, ...)    # 테스트용 입력 데이터 생성
```

---

## 12. frontend/src/types/hyeat.ts - TypeScript 타입

**파일**: `frontend/src/types/hyeat.ts` (148줄)

### 역할
DynamoDB 데이터 구조를 TypeScript 인터페이스로 정의한다. 프론트엔드에서 type-safe하게 데이터를 다룰 수 있다.

### 인터페이스 정의

**DdbWaitingItem** - DynamoDB 아이템 전체 구조:
```typescript
{
  pk: string;            // "CORNER#{restaurantId}#{cornerId}"
  sk: string;            // epoch ms (문자열)
  restaurantId: string;
  cornerId: string;
  queueLen: number;
  estWaitTimeMin: number;
  dataType: "observed" | "predicted" | "dummy";
  source?: string;       // "jetson_nano"
  timestampIso: string;  // ISO 8601 KST
  createdAtIso: string;  // ISO 8601 KST
  ttl?: number;          // epoch 초
}
```

**WaitingDataResponse** - API 응답용 (간소화):
```typescript
{
  cornerId: string;
  queueLen: number;
  estWaitTimeMin: number;
  lastUpdated: string;
}
```

**CornerStatus** - 코너별 실시간 현황:
```typescript
{
  cornerId: string;
  cornerName: string;    // "한식 코너"
  queueLen: number;
  estWaitTimeMin: number;
  status: "available" | "crowded" | "full";
  lastUpdated: string;
}
```

### 사용 예시
```typescript
import type { DdbWaitingItem, WaitingDataResponse, CornerStatus } from './types/hyeat';
```

---

## 13. 설정 파일

### config/roi_config.json
```json
{
  "rois": [
    {
      "name": "2번째",
      "points": [[146,89],[515,87],[533,407],[132,413],[147,92]],
      "color": [0, 255, 0]
    }
  ]
}
```

### config/aws_config.json
```json
{
  "region": "ap-northeast-2",
  "table_name": "hyeat-waiting-data",
  "restaurant_id": "hanyang_plaza",
  "corner_id": "korean"
}
```

### requirements.txt
```
flask>=3.0
pyyaml>=6.0
boto3>=1.34
```

---

## 14. 데이터 흐름도

### 전체 파이프라인 (End-to-End)

```
[CSI 카메라]
    │ GStreamer (nvarguscamerasrc)
    ▼
[CameraManager] ─── Thread 1: _capture_loop()
    │ get_frame()
    ▼
[YOLOv8Detector] ─── TensorRT FP16 추론
    │ detections: [{bbox, confidence, class_id}]
    ▼
[ByteTracker] ─── 칼만 필터 + 2단계 IoU 매칭
    │ tracked: [{bbox, confidence, track_id}]
    ▼
[ROIManager] ─── 점-다각형 판별 (cv2.pointPolygonTest)
    │ roi_dets: {roi_name: [tracked_det, ...]}
    ▼
[ROIDwellFilter] ─── 체류 시간 필터 (≥30 프레임)
    │ roi_counts: {roi_name: int}
    ▼
[WaitTimeEstimator] ─── EMA + 대기열 보정 (Hybrid)
    │ predicted_wait, current_queue
    ▼
[DynamoDBSender] ─── 비동기 배치 전송 (10초 주기)
    │ snake_case → camelCase
    ▼
[AWS DynamoDB] ─── 테이블: hyeat-waiting-data
    │ PK: CORNER#hanyang_plaza#korean
    ▼
[프론트엔드] ─── TypeScript (hyeat.ts)
    │ DdbWaitingItem, CornerStatus
    ▼
[사용자 대시보드]
```

### 스레드 간 데이터 교환

```
Thread 1 (Camera)
    │
    │ self._frame (Lock)
    │
    ▼
Thread 2 (Inference)
    │
    │ FrameBuffer (Condition)        → Thread 3 (MJPEG Network)
    │ _latest_roi_counts (global)    → GET /api/stats
    │ _latest_tracked (global)       → GET /api/tracks
    │ _latest_wait_result (global)   → GET /api/wait_time
    │ DynamoDBSender._queue (Lock)   → Thread 4 (DynamoDB Worker)
```

---

*문서 생성일: 2026-02-15*
"""

css = """
@page {
    size: A4;
    margin: 2cm 1.5cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #888;
    }
}

body {
    font-family: "Noto Serif CJK KR", "Noto Sans CJK KR", serif;
    font-size: 10.5pt;
    line-height: 1.7;
    color: #1a1a1a;
}

h1 {
    font-size: 22pt;
    color: #1a237e;
    border-bottom: 3px solid #1a237e;
    padding-bottom: 8px;
    margin-top: 40px;
    page-break-before: avoid;
}

h2 {
    font-size: 16pt;
    color: #283593;
    border-bottom: 2px solid #c5cae9;
    padding-bottom: 5px;
    margin-top: 30px;
    page-break-before: always;
}

h2:first-of-type {
    page-break-before: avoid;
}

h3 {
    font-size: 13pt;
    color: #3949ab;
    margin-top: 20px;
}

h4 {
    font-size: 11pt;
    color: #5c6bc0;
    margin-top: 15px;
}

code {
    font-family: "Noto Sans Mono CJK KR", "Noto Sans Mono", monospace;
    font-size: 9pt;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
    color: #c62828;
}

pre {
    background: #263238;
    color: #eeffff;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 8.5pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: none;
    color: #eeffff;
    padding: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th {
    background: #e8eaf6;
    color: #1a237e;
    font-weight: bold;
    padding: 8px 10px;
    text-align: left;
    border: 1px solid #c5cae9;
}

td {
    padding: 6px 10px;
    border: 1px solid #e0e0e0;
}

tr:nth-child(even) td {
    background: #fafafa;
}

hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 20px 0;
}

blockquote {
    border-left: 4px solid #7986cb;
    margin: 10px 0;
    padding: 8px 16px;
    background: #e8eaf6;
    color: #283593;
}

strong {
    color: #1a237e;
}
"""

html_body = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'toc', 'nl2br'],
)
full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""

output_path = "/home/iimjuhong/projects/aidea/docs/HY-eat_전체코드분석서.pdf"
HTML(string=full_html).write_pdf(output_path)
print(f"PDF 생성 완료: {output_path}")
