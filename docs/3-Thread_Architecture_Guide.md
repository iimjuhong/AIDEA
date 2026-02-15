# ⚡ 3-Thread 아키텍처 상세 설명

> 왜 멀티스레드를 사용하고, 어떻게 동작하는지 쉽게 이해하기

---

## 🤔 왜 멀티스레드를 쓰나요?

### 문제 상황: 싱글 스레드의 한계

만약 모든 작업을 하나의 스레드에서 처리하면:

```
카메라 읽기 → YOLO 검출 → 웹 전송 → 다시 카메라 읽기 → ...
     ↓          ↓          ↓
   10ms      50ms      100ms (느린 네트워크!)
```

**문제점**:
- 😱 네트워크가 느리면 카메라 프레임 누락
- 😱 웹 클라이언트가 느리면 YOLO 검출 중단
- 😱 하나가 막히면 전체 시스템 멈춤

### 해결책: 멀티스레드로 분리!

```
Thread 1 (카메라): 계속 프레임 캡처 ───┐
                                    │
Thread 2 (AI 추론): YOLO + 추적 ────┼──→ 각자 독립적으로 동작!
                                    │
Thread 3 (웹 전송): MJPEG 스트림 ────┘
```

**장점**:
- ✅ 카메라는 느린 네트워크와 무관하게 계속 동작
- ✅ AI 추론은 웹 전송 속도와 무관
- ✅ 한 부분이 느려도 다른 부분은 계속 작동

---

## 🏗️ 3-Thread (실제로는 4-Thread) 아키텍처

### Thread 1: 카메라 캡처 스레드 📷

**위치**: `src/core/camera.py` → `CameraManager._capture_loop()`

**하는 일**:
```python
while True:
    # GStreamer에서 프레임 읽기
    frame = cap.read()
    
    # thread-safe 변수에 저장
    self._frame = frame
    
    # 계속 반복 (30fps)
```

**특징**:
- 백그라운드에서 계속 실행
- 최신 프레임을 `_frame` 변수에 저장
- 다른 스레드가 읽어갈 때까지 대기 (논블로킹)

**왜 독립적인가?**
- 카메라는 일정한 속도로 계속 캡처해야 함
- YOLO가 느리든, 네트워크가 느리든 상관없이 동작

---

### Thread 2: AI 추론 스레드 🤖

**위치**: `src/web/app.py` → `_inference_loop()`

**하는 일**:
```python
while True:
    # 1. Thread 1에서 최신 프레임 가져오기
    frame = camera.get_frame()
    
    # 2. YOLO 객체 검출 (50ms)
    detections = detector.detect(frame)
    
    # 3. ByteTracker로 추적 (5ms)
    tracks = tracker.update(detections)
    
    # 4. ROI 분석 (1ms)
    roi_counts = roi_manager.count_in_rois(tracks)
    
    # 5. 대기시간 계산 (if 활성화)
    if wait_estimator:
        wait_time = wait_estimator.update(tracks)
    
    # 6. DynamoDB 전송 (비동기, 10초마다 or 변경 시)
    if dynamodb_sender and should_send():
        dynamodb_sender.send(data)
    
    # 7. 웹 스트리밍용 JPEG 인코딩
    jpeg_frame = encode_jpeg(annotated_frame)
    
    # 8. Thread 3를 위한 프레임 버퍼에 저장
    frame_buffer.put(jpeg_frame)
```

**특징**:
- 가장 무거운 작업 (YOLO 추론)
- 고정된 속도로 실행 (예: 20-27 FPS)
- 웹 전송과 완전히 독립적

**왜 독립적인가?**
- AI 추론은 웹 클라이언트 속도와 무관
- 느린 브라우저가 있어도 추론은 계속 진행

---

### Thread 3: 웹 스트리밍 스레드 🌐

**위치**: `src/web/app.py` → `generate_frames()`

**하는 일**:
```python
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            # Thread 2가 만든 JPEG 프레임 가져오기
            jpeg_frame = frame_buffer.get()
            
            # MJPEG 형식으로 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg_frame + b'\r\n')
    
    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
```

**특징**:
- HTTP 클라이언트마다 별도 스레드 생성
- 느린 클라이언트는 프레임 스킵 (최신 것만 전송)
- Flask가 자동으로 관리

**왜 독립적인가?**
- 웹 브라우저가 느려도 AI 추론에 영향 없음
- 여러 클라이언트가 동시 접속 가능

---

### Thread 4: DynamoDB 전송 스레드 ☁️

**위치**: `src/cloud/dynamodb_sender.py` → `DynamoDBSender._worker_loop()`

**하는 일**:
```python
while self._running:
    # 큐에서 전송할 아이템 꺼내기
    batch = []
    for _ in range(25):  # 최대 25개 배치
        if not self._queue.empty():
            batch.append(self._queue.get())
    
    # DynamoDB로 전송
    if batch:
        try:
            dynamodb.batch_write_item(batch)
            self._stats['sent'] += len(batch)
        except Exception as e:
            # 실패 시 재시도 큐에 추가
            retry_queue.extend(batch)
    
    time.sleep(0.1)  # CPU 과부하 방지
```

**특징**:
- 백그라운드에서 비동기 전송
- 배치 처리 (최대 25개)
- 실패 시 재시도
- Exponential backoff

**왜 독립적인가?**
- AWS 네트워크가 느려도 시스템 계속 동작
- 전송 실패해도 로컬 추론은 계속 진행

---

## 🔄 스레드 간 통신 방법

### 1. Thread 1 → Thread 2: 프레임 공유

```python
# Thread 1 (Camera)
self._frame = new_frame  # 최신 프레임 저장 (with lock)

# Thread 2 (Inference)
frame = camera.get_frame()  # 최신 프레임 읽기 (with lock)
```

**동기화 방법**: `threading.Lock` (뮤텍스)

---

### 2. Thread 2 → Thread 3: 프레임 버퍼

```python
# Thread 2 (Inference)
frame_buffer.put(jpeg_frame)  # 큐에 추가

# Thread 3 (Web)
jpeg_frame = frame_buffer.get()  # 큐에서 꺼내기
```

**동기화 방법**: `queue.Queue` (thread-safe)

**특별 처리**:
- 큐가 가득 차면 오래된 프레임 버림
- 느린 클라이언트는 최신 프레임만 수신

---

### 3. Thread 2 → Thread 4: DynamoDB 큐

```python
# Thread 2 (Inference)
dynamodb_sender.send(data)  # 큐에 추가

# Thread 4 (DynamoDB)
batch = queue.get()  # 큐에서 배치로 꺼내기
```

**동기화 방법**: `collections.deque` + `threading.Lock`

---

## 🎯 핵심 설계 원칙

### 1. 비블로킹 (Non-blocking) ✅

각 스레드는 다른 스레드를 기다리지 않습니다.

```python
# ❌ 잘못된 예 (블로킹)
frame = camera.get_frame()
while frame is None:  # 프레임 올 때까지 무한 대기
    frame = camera.get_frame()

# ✅ 올바른 예 (논블로킹)
frame = camera.get_frame()
if frame is None:
    return  # 다음 루프에서 다시 시도
```

### 2. 느린 소비자 무시 (Drop slow consumers) ✅

느린 웹 클라이언트가 시스템 전체를 막지 못하게 합니다.

```python
# FrameBuffer
if self._queue.full():
    self._queue.get()  # 오래된 프레임 버림
self._queue.put(new_frame)  # 최신 프레임 추가
```

### 3. Graceful Degradation (우아한 성능 저하) ✅

한 부분이 실패해도 시스템은 계속 동작합니다.

```python
# DynamoDB 전송 실패 시
try:
    dynamodb.batch_write_item(batch)
except Exception as e:
    logger.warning(f"DynamoDB 전송 실패: {e}")
    # 에러 로그만 남기고 계속 진행
    # AI 추론은 멈추지 않음!
```

---

## 📊 실행 흐름 예시

### 시간 축으로 보는 스레드 동작

```
시간(ms) | Thread 1 (Camera) | Thread 2 (AI)      | Thread 3 (Web)    | Thread 4 (DynamoDB)
---------|-------------------|--------------------|--------------------|--------------------
0        | 프레임1 캡처      |                    |                    |
10       |                   | 프레임1 YOLO 시작  |                    |
20       | 프레임2 캡처      |                    |                    |
30       |                   |                    | 클라이언트1 요청   |
40       | 프레임3 캡처      | YOLO 완료, 인코딩  |                    |
50       |                   | 버퍼에 추가        | JPEG 전송          | 큐에서 데이터 꺼냄
60       | 프레임4 캡처      | 프레임2 YOLO 시작  |                    |
70       |                   |                    | 클라이언트2 요청   | DynamoDB 전송 시작
80       | 프레임5 캡처      | YOLO 완료          | JPEG 전송          |
...      | (계속 반복)       | (계속 반복)        | (계속 반복)        | (계속 반복)
```

**핵심**:
- 각 스레드는 자기 속도대로 동작
- 서로 기다리지 않음
- 동시에 여러 작업 진행

---

## 🛡️ Thread Safety (스레드 안전성)

### 공유 자원 보호

```python
class CameraManager:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()  # 뮤텍스
    
    def get_frame(self):
        with self._lock:  # 락 획득
            return self._frame.copy()  # 복사본 반환
        # 락 자동 해제
```

**왜 필요한가?**
- Thread 1이 프레임 쓰는 중에 Thread 2가 읽으면 깨진 데이터
- Lock으로 동시 접근 방지

---

## 💡 성능 최적화 팁

### 1. CPU 코어 활용

```
Jetson Orin Nano: 6코어
- Core 1-2: Thread 1 (Camera)
- Core 3-4: Thread 2 (YOLO, 가장 무거움)
- Core 5: Thread 3 (Web)
- Core 6: Thread 4 (DynamoDB)
```

### 2. GIL (Global Interpreter Lock) 우회

Python의 GIL 때문에 CPU-bound 작업은 병렬화가 어렵지만:

- ✅ **YOLO 추론**: TensorRT (C++ 기반) → GIL 영향 없음
- ✅ **JPEG 인코딩**: TurboJPEG (C 기반) → GIL 영향 없음
- ✅ **네트워크 I/O**: 블로킹 I/O시 GIL 자동 해제

결과: 실제로 멀티코어를 잘 활용!

---

## 🔍 실제 코드에서 확인하기

### Thread 1 (Camera)
- 파일: `src/core/camera.py`
- 함수: `CameraManager._capture_loop()`

### Thread 2 (Inference)
- 파일: `src/web/app.py`
- 함수: `_inference_loop()`

### Thread 3 (Web)
- 파일: `src/web/app.py`
- 함수: `generate_frames()`
- Flask가 자동으로 스레드 생성

### Thread 4 (DynamoDB)
- 파일: `src/cloud/dynamodb_sender.py`
- 함수: `DynamoDBSender._worker_loop()`

---

## 📚 더 알아보기

- **전체 아키텍처**: [README.md](README.md) - "시스템 아키텍처" 섹션
- **폴더 구조**: [FOLDER_GUIDE.md](FOLDER_GUIDE.md)
- **빠른 실행**: [QUICKSTART.md](QUICKSTART.md)

---

## ❓ FAQ

### Q: 왜 4개 스레드인데 3-Thread라고 부르나요?
A: 원래 3개였는데 DynamoDB 기능 추가하면서 4개가 되었습니다. 관습적으로 3-Thread 아키텍처라고 부릅니다.

### Q: 스레드를 더 추가하면 더 빨라지나요?
A: 아닙니다. 병목 지점은 YOLO 추론 속도이고, 이미 최적화되어 있습니다. 스레드를 더 추가해도 의미 없습니다.

### Q: 멀티프로세싱을 쓰면 안 되나요?
A: 가능하지만 복잡도만 증가합니다. 현재 멀티스레드로 충분히 효율적입니다.

### Q: GIL 때문에 느리지 않나요?
A: 무거운 작업(YOLO, JPEG)이 모두 C/C++ 기반이라 GIL 영향 거의 없습니다.
