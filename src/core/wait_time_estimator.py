"""대기시간 측정 및 예측 모듈

Phase 5: ROI 진입/퇴출 이벤트 감지, 실제 대기시간 계산, 예측 알고리즘.

알고리즘 선택 근거
==================

1. **하이브리드(EMA + 대기열 보정)** 방식을 기본 예측기로 채택한 이유:

   (a) EMA vs 단순 이동평균(MA)
       - 식당 대기시간은 비정상(non-stationary) 시계열이다.
         점심 피크 → 한산 → 저녁 피크처럼 분포가 수시로 변한다.
       - EMA는 최근 관측치에 지수적으로 높은 가중치를 부여하여
         트렌드 변화에 2~3배 빠르게 수렴한다 (α=0.3 기준 half-life ≈ 2샘플).
       - 메모리 O(1), 연산 O(1)로 Jetson 실시간 파이프라인에 부담 없음.
       - MA는 안정적이지만 윈도우 크기(N) 튜닝이 필요하고,
         급변 시 N프레임 지연이 발생한다.

   (b) 대기열 보정 (Little's Law 기반)
       - L = λW (평균 대기열 길이 = 도착률 × 평균 대기시간)
       - 순수 과거 평균만으로는 "지금 줄이 긴데 짧게 예측"하는 오류 발생.
       - 현재 대기 인원이 과거 평균보다 많으면 β 비율만큼 상향 보정.
       - β=0.3: 대기열이 2배이면 예측이 30% 증가. 과민 반응 방지.

   (c) 이상치 필터링 (IQR 방식)
       - 3σ 방식은 정규분포를 가정하고 소규모 샘플에서 불안정.
       - IQR(사분위수 범위)는 분포 가정 없이 강건하게 이상치 판별.
       - 예: 대기 중 이탈, 장시간 사회적 대화 등 비정상 샘플 자동 제거.

   (d) 스테일 트랙 정리
       - ByteTracker가 트랙을 잃으면 entry_time이 영구 잔류 → 메모리 누수.
       - stale_timeout(기본 300초) 초과 시 자동 정리.

2. 듀얼 모드 지원
   - 단일 ROI 모드 (end_roi=None): start_roi 체류시간 = 대기시간
   - 플로우 모드 (end_roi 설정): start_roi 진입 → end_roi 진입 = 대기시간

기존 코드 연동
==============
- ByteTracker.update() → tracked detections (track_id 포함)
- ROIManager.filter_detections_by_roi(tracked) → {roi_name: [det, ...]}
- 위 결과를 WaitTimeEstimator.update()에 전달
"""

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================
#  예측 알고리즘
# ============================================================

class EMAPredictor:
    """지수 이동 평균 (Exponential Moving Average) 예측기

    공식: EMA[t] = α × sample + (1 - α) × EMA[t-1]

    장점: O(1) 메모리/연산, 최근 변화에 민감
    적합: 데이터 부족 초기, 빠른 반응이 필요한 경우
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._ema = 0.0
        self._count = 0

    def update(self, wait_time, **kwargs):
        if self._count == 0:
            self._ema = wait_time
        else:
            self._ema = self.alpha * wait_time + (1 - self.alpha) * self._ema
        self._count += 1

    def predict(self, **kwargs):
        return self._ema

    @property
    def sample_count(self):
        return self._count


class MovingAveragePredictor:
    """시간대별 이동 평균 (Simple Moving Average) 예측기

    공식: MA = sum(window) / len(window)

    장점: 이상치 영향 완화, 직관적
    적합: 안정적 예측이 필요한 경우
    """

    def __init__(self, window_size=20):
        self._window = deque(maxlen=window_size)

    def update(self, wait_time, **kwargs):
        self._window.append(wait_time)

    def predict(self, **kwargs):
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def sample_count(self):
        return len(self._window)


class HybridPredictor:
    """하이브리드 예측기: EMA + 대기열 크기 보정

    공식:
      predicted = EMA × (1 + β × (current_queue - avg_queue) / avg_queue)

    원리 (Little's Law 응용):
      - EMA: 과거 트렌드 반영
      - 보정항: 현재 대기열이 평균 대비 얼마나 큰지에 비례하여 조정
      - β: 보정 강도. 0이면 순수 EMA, 1이면 100% 보정

    장점: 과거 트렌드 + 현재 상황을 동시에 반영
    적합: 식당 등 대기열 길이가 수시로 변하는 동적 환경
    """

    def __init__(self, alpha=0.3, beta=0.3):
        self.alpha = alpha
        self.beta = beta
        self._ema = 0.0
        self._count = 0
        self._queue_sizes = deque(maxlen=50)

    def update(self, wait_time, queue_size=0, **kwargs):
        if self._count == 0:
            self._ema = wait_time
        else:
            self._ema = self.alpha * wait_time + (1 - self.alpha) * self._ema
        self._count += 1
        self._queue_sizes.append(queue_size)

    def predict(self, current_queue_size=0, **kwargs):
        if self._count == 0:
            return 0.0

        if not self._queue_sizes:
            return self._ema

        avg_queue = sum(self._queue_sizes) / len(self._queue_sizes)

        if avg_queue > 0:
            correction = ((current_queue_size - avg_queue) / avg_queue) * self.beta
        else:
            correction = 0.0

        return max(self._ema * (1 + correction), 0.0)

    @property
    def sample_count(self):
        return self._count


# ============================================================
#  대기시간 측정 및 예측 통합 클래스
# ============================================================

_PREDICTOR_CLASSES = {
    'hybrid': HybridPredictor,
    'ema': EMAPredictor,
    'moving_average': MovingAveragePredictor,
}


class WaitTimeEstimator:
    """대기시간 측정 및 예측 통합 클래스

    기능:
      1. ROI 진입/퇴출 이벤트 감지 (상태 전이 기반)
      2. 실제 대기시간 계산 (track_id별 진입~퇴출 시간 추적)
      3. 다음 고객 예상 대기시간 예측

    동작 모드:
      - 단일 ROI 모드 (end_roi=None):
          start_roi 진입 → start_roi 퇴출 = 대기시간
      - 플로우 모드 (end_roi 설정):
          start_roi 진입 → end_roi 진입 = 대기시간

    사용법:
        estimator = WaitTimeEstimator(start_roi='대기구역', end_roi='카운터')

        # 매 프레임 (inference loop 안에서)
        roi_dets = roi_manager.filter_detections_by_roi(tracked)
        result = estimator.update(roi_dets)
        print(f"예상 대기시간: {result['predicted_wait']:.0f}초")
    """

    def __init__(self, start_roi, end_roi=None,
                 predictor_type='hybrid',
                 alpha=0.3, beta=0.3,
                 stale_timeout=300.0,
                 max_history=500,
                 outlier_iqr_factor=1.5):
        """
        Args:
            start_roi: 대기 시작 ROI 이름
            end_roi: 대기 종료 ROI 이름 (None → 단일 ROI 모드)
            predictor_type: 'hybrid' | 'ema' | 'moving_average'
            alpha: EMA 평활 계수 (0 < α ≤ 1). 클수록 최근 값에 민감.
            beta: 대기열 보정 계수 (hybrid 전용). 클수록 현재 대기열 영향 증가.
            stale_timeout: 미완료 트랙 정리 타임아웃 (초). 0이면 비활성.
            max_history: 완료 대기시간 이력 최대 보관 수
            outlier_iqr_factor: IQR 이상치 판별 배수 (0이면 비활성)
        """
        if predictor_type not in _PREDICTOR_CLASSES:
            raise ValueError(
                f"Unknown predictor_type: {predictor_type}. "
                f"Choose from {list(_PREDICTOR_CLASSES.keys())}"
            )

        self.start_roi = start_roi
        self.end_roi = end_roi
        self.stale_timeout = stale_timeout
        self.outlier_iqr_factor = outlier_iqr_factor

        # 모드 결정
        self._flow_mode = end_roi is not None
        mode = (f"플로우 ({start_roi} → {end_roi})" if self._flow_mode
                else f"단일 ROI ({start_roi})")
        logger.info(f"WaitTimeEstimator 초기화: {mode}, predictor={predictor_type}")

        # 예측기 생성
        self._predictor_type = predictor_type
        cls = _PREDICTOR_CLASSES[predictor_type]
        if predictor_type == 'hybrid':
            self._predictor = cls(alpha=alpha, beta=beta)
        elif predictor_type == 'ema':
            self._predictor = cls(alpha=alpha)
        else:
            self._predictor = cls()

        # ── 상태 ──
        self._prev_state = {}       # {roi_name: set(track_id)} 이전 프레임
        self._entry_times = {}      # {track_id: float} start_roi 진입 시각
        self._wait_history = deque(maxlen=max_history)  # 완료 대기시간 (초)

        # ── 통계 ──
        self._total_completed = 0
        self._total_outliers = 0
        self._last_cleanup = 0.0

    # ----------------------------------------------------------
    #  공개 API
    # ----------------------------------------------------------

    def update(self, roi_detections):
        """프레임 단위 업데이트

        Args:
            roi_detections: {roi_name: [detection_dict, ...]}
                ROIManager.filter_detections_by_roi() 결과.
                각 detection_dict에는 'track_id' 키가 있어야 함.

        Returns:
            dict: {
                'events': list,          # [{type, track_id, roi_name, timestamp}]
                'predicted_wait': float,  # 예측 대기시간 (초). 데이터 없으면 0.0
                'current_queue': int,     # 현재 start_roi 대기 인원
                'completed': list,        # 이번 프레임에 완료된 대기시간 [초, ...]
                'active_waiters': dict,   # {track_id: elapsed_seconds}
            }
        """
        now = time.time()
        events = []
        completed = []

        # ── 현재 상태 구축 ──
        current_state = {}
        for roi_name, dets in roi_detections.items():
            ids = set()
            for det in dets:
                tid = det.get('track_id')
                if tid is not None:
                    ids.add(tid)
            current_state[roi_name] = ids

        # ── 이벤트 감지: 이전 vs 현재 상태 전이 ──
        all_rois = set(current_state) | set(self._prev_state)
        for roi_name in all_rois:
            curr = current_state.get(roi_name, set())
            prev = self._prev_state.get(roi_name, set())

            # 진입
            for tid in curr - prev:
                events.append({
                    'type': 'ENTRY',
                    'track_id': tid,
                    'roi_name': roi_name,
                    'timestamp': now,
                })
                wt = self._handle_entry(tid, roi_name, now)
                if wt is not None:
                    completed.append(wt)

            # 퇴출
            for tid in prev - curr:
                events.append({
                    'type': 'EXIT',
                    'track_id': tid,
                    'roi_name': roi_name,
                    'timestamp': now,
                })
                wt = self._handle_exit(tid, roi_name, now)
                if wt is not None:
                    completed.append(wt)

        self._prev_state = current_state

        # ── 스테일 트랙 정리 (1초마다) ──
        if self.stale_timeout > 0 and now - self._last_cleanup > 1.0:
            self._cleanup_stale(now)
            self._last_cleanup = now

        # ── 현재 대기열 크기 ──
        queue_size = len(current_state.get(self.start_roi, set()))

        # ── 완료 샘플로 예측기 업데이트 ──
        for wt in completed:
            if self._is_outlier(wt):
                self._total_outliers += 1
                logger.debug(f"이상치 제거: {wt:.1f}초")
                continue
            self._wait_history.append(wt)
            self._total_completed += 1
            if self._predictor_type == 'hybrid':
                self._predictor.update(wt, queue_size=queue_size)
            else:
                self._predictor.update(wt)

        # ── 예측 ──
        if self._predictor_type == 'hybrid':
            predicted = self._predictor.predict(current_queue_size=queue_size)
        else:
            predicted = self._predictor.predict()

        # ── 현재 대기 중인 사람들 ──
        active_waiters = {
            tid: now - t for tid, t in self._entry_times.items()
        }

        return {
            'events': events,
            'predicted_wait': predicted,
            'current_queue': queue_size,
            'completed': completed,
            'active_waiters': active_waiters,
        }

    def get_statistics(self):
        """누적 통계 정보 반환

        Returns:
            dict 또는 None (데이터 없음)
        """
        if not self._wait_history:
            return None

        history = list(self._wait_history)
        n = len(history)
        sorted_h = sorted(history)
        mean = sum(history) / n

        return {
            'total_completed': self._total_completed,
            'total_outliers_rejected': self._total_outliers,
            'recent_samples': n,
            'mean': mean,
            'median': sorted_h[n // 2],
            'min': sorted_h[0],
            'max': sorted_h[-1],
            'recent_10_avg': (sum(history[-10:]) / min(n, 10)),
            'active_waiting': len(self._entry_times),
        }

    def get_predicted_wait(self):
        """현재 예측 대기시간 (초) 반환. 데이터 없으면 0.0."""
        queue_size = len(self._prev_state.get(self.start_roi, set()))
        if self._predictor_type == 'hybrid':
            return self._predictor.predict(current_queue_size=queue_size)
        return self._predictor.predict()

    # ----------------------------------------------------------
    #  내부 메서드
    # ----------------------------------------------------------

    def _handle_entry(self, track_id, roi_name, timestamp):
        """진입 이벤트 처리. 대기 완료 시 초 단위 반환, 아니면 None."""
        # start_roi 진입 → 대기 시작 기록
        if roi_name == self.start_roi:
            self._entry_times[track_id] = timestamp
            return None

        # 플로우 모드: end_roi 진입 → 대기 완료
        if self._flow_mode and roi_name == self.end_roi:
            entry = self._entry_times.pop(track_id, None)
            if entry is not None:
                wait_time = timestamp - entry
                if wait_time > 0:
                    return wait_time

        return None

    def _handle_exit(self, track_id, roi_name, timestamp):
        """퇴출 이벤트 처리. 대기 완료 시 초 단위 반환, 아니면 None."""
        # 단일 ROI 모드: start_roi 퇴출 → 대기 완료
        if not self._flow_mode and roi_name == self.start_roi:
            entry = self._entry_times.pop(track_id, None)
            if entry is not None:
                wait_time = timestamp - entry
                if wait_time > 0:
                    return wait_time

        return None

    def _cleanup_stale(self, now):
        """타임아웃된 미완료 트랙 정리"""
        stale = [
            tid for tid, t in self._entry_times.items()
            if now - t > self.stale_timeout
        ]
        for tid in stale:
            del self._entry_times[tid]
        if stale:
            logger.debug(f"스테일 트랙 {len(stale)}개 정리: {stale}")

    def _is_outlier(self, value):
        """IQR 기반 이상치 판별. 샘플 10개 미만이면 항상 False."""
        if self.outlier_iqr_factor <= 0:
            return False
        if len(self._wait_history) < 10:
            return False

        sorted_h = sorted(self._wait_history)
        n = len(sorted_h)
        q1 = sorted_h[n // 4]
        q3 = sorted_h[3 * n // 4]
        iqr = q3 - q1

        lower = q1 - self.outlier_iqr_factor * iqr
        upper = q3 + self.outlier_iqr_factor * iqr

        return value < lower or value > upper
