# Phase 5: 대기시간 측정 및 예측 알고리즘 추천

식당 대기시간 추정 시스템을 위한 대기시간 측정 및 예측 알고리즘 가이드입니다.

---

## 📋 목차

1. [시스템 구조 분석](#시스템-구조-분석)
2. [ROI 진입/퇴출 이벤트 감지](#roi-진입퇴출-이벤트-감지)
3. [실제 대기시간 계산](#실제-대기시간-계산)
4. [대기시간 예측 알고리즘](#대기시간-예측-알고리즘)
5. [구현 클래스 설계](#구현-클래스-설계)

---

## 시스템 구조 분석

### 현재 구현 상태

✅ **Phase 1-3**: 카메라 스트리밍, YOLO 검출, ROI 관리  
✅ **Phase 4**: ByteTrack 객체 추적, 칼만필터, ROI 체류시간 필터 (`tracker.py`)

### 기존 컴포넌트 활용

- `ByteTracker`: 고유 `track_id` 부여 및 유지
- `ROIDwellFilter`: ROI별 체류 프레임 수 추적
- `ROIManager`: Point-in-Polygon 판정, ROI별 검출 필터링

---

## ROI 진입/퇴출 이벤트 감지

### 1️⃣ 상태 전이 기반 이벤트 감지

기존의 `ROIDwellFilter`를 확장하여 **상태 전이(State Transition)**를 감지합니다.

#### 알고리즘 개요

```
각 (track_id, roi_name) 쌍에 대해:
  - 이전 프레임 상태: in_roi[t-1]
  - 현재 프레임 상태: in_roi[t]
  
  이벤트 감지:
    - 진입 (ENTRY): in_roi[t-1] = False, in_roi[t] = True
    - 퇴출 (EXIT):  in_roi[t-1] = True,  in_roi[t] = False
    - 체류 (STAY):  in_roi[t-1] = True,  in_roi[t] = True
```

#### 데이터 구조

```python
# 트랙별 이전 프레임 ROI 상태
previous_state = {
    track_id: {roi_name: bool}
}

# 이벤트 로그
events = [
    {
        'timestamp': datetime,
        'track_id': int,
        'roi_name': str,
        'event_type': 'ENTRY' | 'EXIT',
        'frame_number': int
    }
]
```

#### 구현 아이디어

```python
def detect_roi_events(current_roi_detections, previous_state):
    """ROI 진입/퇴출 이벤트 감지"""
    events = []
    current_state = {}
    
    # 현재 프레임에서 각 ROI 안에 있는 트랙 수집
    for roi_name, detections in current_roi_detections.items():
        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue
                
            # 진입 이벤트 감지
            if track_id not in previous_state.get(roi_name, set()):
                events.append({
                    'type': 'ENTRY',
                    'track_id': track_id,
                    'roi_name': roi_name,
                    'timestamp': time.time()
                })
            
            # 현재 상태 업데이트
            current_state.setdefault(roi_name, set()).add(track_id)
    
    # 퇴출 이벤트 감지
    for roi_name, prev_tracks in previous_state.items():
        curr_tracks = current_state.get(roi_name, set())
        for track_id in prev_tracks - curr_tracks:
            events.append({
                'type': 'EXIT',
                'track_id': track_id,
                'roi_name': roi_name,
                'timestamp': time.time()
            })
    
    return events, current_state
```

---

## 실제 대기시간 계산

### 2️⃣ ROI 기반 대기시간 정의

식당 시나리오에서 대기시간은 다음과 같이 정의할 수 있습니다:

#### 시나리오 A: 단일 대기 구역

```
대기시간 = "대기구역" 진입 시각 → "대기구역" 퇴출 시각
```

**적용 조건**: ROI가 "대기구역" 하나만 있는 경우

#### 시나리오 B: 입구 → 대기 → 카운터 플로우

```
대기시간 = "대기구역" 진입 시각 → "카운터" 진입 시각
```

**적용 조건**: 여러 ROI가 순차적으로 정의된 경우

#### 시나리오 C: 큐(Queue) 기반 시뮬레이션

```
대기시간 = 현재 대기열 인원수 × 평균 서비스 시간
```

**적용 조건**: 카운터/서비스 ROI가 있는 경우

---

### 3️⃣ 추천 알고리즘: **시나리오 B (진입→퇴출 추적)**

가장 정확하고 실용적인 방법입니다.

#### 알고리즘 상세

```python
class WaitTimeTracker:
    def __init__(self):
        # track_id → {roi_name: entry_timestamp}
        self.entry_times = {}
        
        # 완료된 대기시간 샘플 (초 단위)
        self.completed_wait_times = []
        
        # ROI 간 플로우 정의
        self.flow = {
            'start_roi': '대기구역',  # 대기 시작
            'end_roi': '카운터'        # 대기 종료
        }
    
    def process_event(self, event):
        track_id = event['track_id']
        roi_name = event['roi_name']
        timestamp = event['timestamp']
        
        if event['type'] == 'ENTRY':
            # 대기 시작 ROI 진입
            if roi_name == self.flow['start_roi']:
                if track_id not in self.entry_times:
                    self.entry_times[track_id] = {}
                self.entry_times[track_id][roi_name] = timestamp
            
            # 대기 종료 ROI 진입 → 대기시간 계산
            elif roi_name == self.flow['end_roi']:
                if track_id in self.entry_times:
                    start_time = self.entry_times[track_id].get(
                        self.flow['start_roi']
                    )
                    if start_time is not None:
                        wait_time = timestamp - start_time
                        self.completed_wait_times.append(wait_time)
                        
                        # 정리
                        del self.entry_times[track_id]
                        
                        return wait_time
        
        return None
```

#### 주요 특징

- ✅ **정확성**: 실제 사람의 진입/퇴출 시각을 추적
- ✅ **확장성**: 여러 ROI 플로우 설정 가능
- ✅ **강건성**: 트랙 손실 시 자동 정리

---

## 대기시간 예측 알고리즘

수집된 실제 대기시간 데이터를 기반으로 **다음 고객의 예상 대기시간**을 예측합니다.

### 4️⃣ 추천 알고리즘 1: **지수 이동 평균 (EMA)**

최근 샘플에 더 큰 가중치를 부여하여 트렌드 변화에 빠르게 반응합니다.

#### 공식

```
EMA[t] = α × 최신_샘플 + (1 - α) × EMA[t-1]
```

- `α` (alpha): 평활 계수 (0 < α ≤ 1)
  - **α = 0.3**: 최근 30% + 과거 70% (안정적)
  - **α = 0.5**: 균형 (추천)
  - **α = 0.7**: 최근 변화에 민감

#### 구현 예시

```python
class EMAPredictor:
    def __init__(self, alpha=0.5, initial_estimate=60.0):
        """
        Args:
            alpha: 평활 계수 (0 < α ≤ 1)
            initial_estimate: 초기 예측값 (초)
        """
        self.alpha = alpha
        self.ema = initial_estimate
        self.sample_count = 0
    
    def update(self, new_wait_time):
        """새로운 대기시간 샘플 추가"""
        if self.sample_count == 0:
            # 첫 샘플은 그대로 사용
            self.ema = new_wait_time
        else:
            # 지수 이동 평균 업데이트
            self.ema = self.alpha * new_wait_time + (1 - self.alpha) * self.ema
        
        self.sample_count += 1
    
    def predict(self):
        """현재 예상 대기시간 반환 (초)"""
        return self.ema
```

#### 장점

- ✅ **실시간 반영**: 최근 데이터에 빠르게 반응
- ✅ **메모리 효율**: O(1) 공간 복잡도
- ✅ **간단한 구현**: 복잡한 통계 불필요

---

### 5️⃣ 추천 알고리즘 2: **시간대별 이동 평균 (Time-windowed MA)**

최근 N개 샘플의 단순 평균을 사용합니다.

#### 공식

```
MA = (샘플[t-N+1] + ... + 샘플[t]) / N
```

- `N`: 윈도우 크기 (예: 10, 20, 50)

#### 구현 예시

```python
from collections import deque

class MovingAveragePredictor:
    def __init__(self, window_size=20, initial_estimate=60.0):
        """
        Args:
            window_size: 이동 평균 윈도우 크기
            initial_estimate: 초기 예측값 (초)
        """
        self.window = deque(maxlen=window_size)
        self.initial_estimate = initial_estimate
    
    def update(self, new_wait_time):
        """새로운 대기시간 샘플 추가"""
        self.window.append(new_wait_time)
    
    def predict(self):
        """현재 예상 대기시간 반환 (초)"""
        if len(self.window) == 0:
            return self.initial_estimate
        return sum(self.window) / len(self.window)
    
    def get_stats(self):
        """통계 정보 반환"""
        if len(self.window) == 0:
            return None
        return {
            'mean': sum(self.window) / len(self.window),
            'min': min(self.window),
            'max': max(self.window),
            'samples': len(self.window)
        }
```

#### 장점

- ✅ **안정성**: 이상치(outlier)의 영향 완화
- ✅ **직관적**: 평균 개념이 명확
- ✅ **통계 제공**: min/max/mean 등 부가 정보

---

### 6️⃣ 추천 알고리즘 3: **하이브리드 (EMA + 현재 대기열 보정)**

예측 정확도를 높이기 위해 **과거 대기시간 + 현재 상황**을 결합합니다.

#### 공식

```
예측 대기시간 = EMA × (1 + 대기열_보정_계수)

대기열_보정_계수 = (현재_대기인원 - 평균_대기인원) / 평균_대기인원 × β
```

- `β` (beta): 대기열 영향도 (예: 0.2 ~ 0.5)

#### 구현 예시

```python
class HybridPredictor:
    def __init__(self, alpha=0.5, beta=0.3, initial_estimate=60.0):
        """
        Args:
            alpha: EMA 평활 계수
            beta: 대기열 영향 계수
            initial_estimate: 초기 예측값 (초)
        """
        self.alpha = alpha
        self.beta = beta
        self.ema = initial_estimate
        
        # 대기열 크기 이동 평균
        self.avg_queue_size = deque(maxlen=50)
    
    def update(self, new_wait_time, queue_size):
        """
        Args:
            new_wait_time: 실제 측정된 대기시간 (초)
            queue_size: 해당 시점의 대기열 인원 수
        """
        # EMA 업데이트
        self.ema = self.alpha * new_wait_time + (1 - self.alpha) * self.ema
        
        # 대기열 크기 기록
        self.avg_queue_size.append(queue_size)
    
    def predict(self, current_queue_size):
        """
        Args:
            current_queue_size: 현재 대기열 인원 수
        
        Returns:
            예상 대기시간 (초)
        """
        if len(self.avg_queue_size) == 0:
            return self.ema
        
        # 평균 대기열 크기
        avg_queue = sum(self.avg_queue_size) / len(self.avg_queue_size)
        
        # 대기열 보정 계수 계산
        if avg_queue > 0:
            correction = ((current_queue_size - avg_queue) / avg_queue) * self.beta
        else:
            correction = 0
        
        # 최종 예측
        predicted = self.ema * (1 + correction)
        
        # 음수 방지
        return max(predicted, 0)
```

#### 장점

- ✅ **상황 반영**: 현재 대기열 크기를 고려
- ✅ **동적 조정**: 혼잡 시 자동으로 대기시간 증가
- ✅ **실전 최적**: 식당 같은 동적 환경에 적합

---

## 구현 클래스 설계

### 통합 클래스: `WaitTimeEstimator`

모든 기능을 통합한 클래스 구조입니다.

```python
class WaitTimeEstimator:
    """대기시간 측정 및 예측 통합 클래스"""
    
    def __init__(self, 
                 start_roi='대기구역',
                 end_roi='카운터',
                 predictor_type='hybrid',
                 fps=30):
        """
        Args:
            start_roi: 대기 시작 ROI 이름
            end_roi: 대기 종료 ROI 이름
            predictor_type: 'ema' | 'moving_average' | 'hybrid'
            fps: 카메라 FPS (시간 계산용)
        """
        self.start_roi = start_roi
        self.end_roi = end_roi
        self.fps = fps
        
        # 진입 시각 추적
        self.entry_times = {}  # {track_id: {roi_name: timestamp}}
        
        # 이전 프레임 ROI 상태
        self.previous_state = {}  # {roi_name: set(track_id)}
        
        # 완료된 대기시간 로그
        self.wait_time_history = []
        
        # 예측기 선택
        if predictor_type == 'ema':
            self.predictor = EMAPredictor(alpha=0.5)
        elif predictor_type == 'moving_average':
            self.predictor = MovingAveragePredictor(window_size=20)
        elif predictor_type == 'hybrid':
            self.predictor = HybridPredictor(alpha=0.5, beta=0.3)
        else:
            raise ValueError(f"Unknown predictor: {predictor_type}")
        
        self.predictor_type = predictor_type
    
    def update(self, roi_detections):
        """
        프레임 단위 업데이트
        
        Args:
            roi_detections: {roi_name: [tracked_det, ...]}
                ROIManager.filter_detections_by_roi() 결과
        
        Returns:
            dict: {
                'events': [...],  # 이번 프레임 이벤트 목록
                'current_wait_time': float,  # 예측 대기시간 (초)
                'completed_waits': int,  # 측정 완료 건수
            }
        """
        current_time = time.time()
        events = []
        completed_waits = []
        
        # 현재 상태 구축
        current_state = {}
        for roi_name, detections in roi_detections.items():
            current_state[roi_name] = set()
            for det in detections:
                track_id = det.get('track_id')
                if track_id is not None:
                    current_state[roi_name].add(track_id)
        
        # 이벤트 감지
        for roi_name, tracks in current_state.items():
            prev_tracks = self.previous_state.get(roi_name, set())
            
            # 진입 이벤트
            for track_id in tracks - prev_tracks:
                events.append({
                    'type': 'ENTRY',
                    'track_id': track_id,
                    'roi_name': roi_name,
                    'timestamp': current_time
                })
                
                # 진입 시각 기록
                if track_id not in self.entry_times:
                    self.entry_times[track_id] = {}
                self.entry_times[track_id][roi_name] = current_time
                
                # 대기 종료 감지
                if roi_name == self.end_roi:
                    if track_id in self.entry_times:
                        start_time = self.entry_times[track_id].get(self.start_roi)
                        if start_time is not None:
                            wait_time = current_time - start_time
                            completed_waits.append(wait_time)
                            self.wait_time_history.append({
                                'track_id': track_id,
                                'wait_time': wait_time,
                                'timestamp': current_time
                            })
                            
                            # 예측기 업데이트
                            if self.predictor_type == 'hybrid':
                                queue_size = len(current_state.get(self.start_roi, set()))
                                self.predictor.update(wait_time, queue_size)
                            else:
                                self.predictor.update(wait_time)
                            
                            # 정리
                            del self.entry_times[track_id]
            
            # 퇴출 이벤트
            for track_id in prev_tracks - tracks:
                events.append({
                    'type': 'EXIT',
                    'track_id': track_id,
                    'roi_name': roi_name,
                    'timestamp': current_time
                })
        
        # 상태 업데이트
        self.previous_state = current_state
        
        # 예측 대기시간
        if self.predictor_type == 'hybrid':
            queue_size = len(current_state.get(self.start_roi, set()))
            predicted_wait = self.predictor.predict(queue_size)
        else:
            predicted_wait = self.predictor.predict()
        
        return {
            'events': events,
            'current_wait_time': predicted_wait,
            'completed_waits': len(completed_waits),
            'wait_samples': completed_waits,
        }
    
    def get_statistics(self):
        """통계 정보 반환"""
        if not self.wait_time_history:
            return None
        
        wait_times = [x['wait_time'] for x in self.wait_time_history]
        return {
            'total_samples': len(wait_times),
            'mean': sum(wait_times) / len(wait_times),
            'min': min(wait_times),
            'max': max(wait_times),
            'recent_10_avg': sum(wait_times[-10:]) / min(len(wait_times), 10)
        }
```

---

## 📊 알고리즘 선택 가이드

### 상황별 추천

| 상황 | 추천 알고리즘 | 이유 |
|------|--------------|------|
| **단순 구현** | 이동 평균 (MA) | 간단하고 직관적 |
| **빠른 반응** | 지수 이동 평균 (EMA) | 최근 변화에 민감 |
| **실전 운영** | 하이브리드 | 현재 대기열 반영으로 정확도 향상 |
| **데이터 부족** | EMA (α=0.7) | 적은 샘플로도 빠르게 수렴 |
| **안정성 중시** | 이동 평균 (N=50) | 이상치 영향 최소화 |

---

## 🚀 통합 가이드

### 기존 코드와의 통합

```python
# main.py 또는 통합 루프
tracker = ByteTracker(...)
roi_manager = ROIManager(...)
wait_estimator = WaitTimeEstimator(
    start_roi='대기구역',
    end_roi='카운터',
    predictor_type='hybrid'
)

while True:
    frame = camera.get_frame()
    detections = detector.detect(frame)
    
    # 추적 업데이트
    tracked = tracker.update(detections)
    
    # ROI별 필터링
    roi_detections = roi_manager.filter_detections_by_roi(tracked)
    
    # 대기시간 추정
    wait_result = wait_estimator.update(roi_detections)
    
    print(f"예상 대기시간: {wait_result['current_wait_time']:.1f}초")
    print(f"측정 완료: {wait_result['completed_waits']}건")
```

---

## 💡 추가 고려사항

### 1. ROI 설정 전략

식당 환경에 맞는 ROI 구성:

- **입구**: 고객 진입 감지
- **대기구역**: 대기 시작 지점
- **카운터/주문대**: 대기 종료 지점
- **착석구역**: 최종 목적지 (선택)

### 2. 예외 처리

- **트랙 손실**: 칼만필터로 일시적 가림 대응 (Phase 4 완료)
- **역류**: 고객이 대기 중 이탈하는 경우 타임아웃 처리
- **이상치**: 대기시간 3σ 밖 샘플 제외 (예: 60초 ± 180초)

### 3. 성능 최적화

- **메모리**: 히스토리 최대 길이 제한 (예: 1000개)
- **DB 저장**: DynamoDB 배치 쓰기 (Phase 7)
- **실시간성**: 예측 계산은 O(1) 복잡도 유지

### 4. 시각화

웹 대시보드에 표시할 정보:

- 📊 **실시간 예측**: "예상 대기시간: 3분 20초"
- 📈 **추세 그래프**: 시간대별 대기시간 변화
- 👥 **현재 대기열**: "대기 중: 5명"
- 📉 **통계**: 평균/최소/최대 대기시간

---

## ✅ 최종 추천

### 권장 구성

```python
WaitTimeEstimator(
    start_roi='대기구역',
    end_roi='카운터',
    predictor_type='hybrid',  # 🎯 하이브리드 방식
    fps=30
)

HybridPredictor(
    alpha=0.5,   # EMA 평활 (균형)
    beta=0.3     # 대기열 영향 30%
)
```

이 구성은 **정확성**, **반응성**, **안정성**의 균형을 제공합니다.

---

## 📚 참고자료

- **칼만필터**: 이미 `tracker.py`에 구현됨
- **ByteTrack**: 이미 `tracker.py`에 구현됨
- **ROI 관리**: `roi_manager.py` 참조
- **이동 평균**: [Wikipedia - Moving Average](https://en.wikipedia.org/wiki/Moving_average)
