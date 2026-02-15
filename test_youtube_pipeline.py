#!/usr/bin/env python3
"""YouTube 영상 파이프라인 End-to-End 테스트

유튜브 영상을 다운로드해서 실제 파이프라인을 테스트한다:
  detector → tracker → ROI → dwell_filter → wait_time → DynamoDB

사용법:
  # 1. 첫 프레임 확인 (ROI 없이 → 프레임 저장 후 종료)
  python test_youtube_pipeline.py "YOUTUBE_URL"

  # 2. ROI 좌표 지정 후 전체 파이프라인
  python test_youtube_pipeline.py "YOUTUBE_URL" \
      --start-roi queue \
      --roi-json '{"name":"queue","points":[[100,200],[500,200],[500,600],[100,600]]}'
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

# yt-dlp 의존성 체크
try:
    import yt_dlp
except ImportError:
    print("yt-dlp가 설치되어 있지 않습니다.")
    print("  pip install yt-dlp")
    sys.exit(1)

import cv2

from src.core.detector import YOLOv8Detector
from src.core.tracker import ByteTracker, ROIDwellFilter
from src.core.roi_manager import ROIManager
from src.core.wait_time_estimator import WaitTimeEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

_RESOLUTION_MAP = {
    '480p': 'best[height<=480][ext=mp4]/best[height<=480]',
    '720p': 'best[height<=720][ext=mp4]/best[height<=720]',
    '1080p': 'best[height<=1080][ext=mp4]/best[height<=1080]',
    'best': 'best[ext=mp4]/best',
}


# ==============================================================
#  1단계: 영상 다운로드
# ==============================================================

def download_video(url, output_dir, resolution='720p', skip_download=False):
    """yt-dlp로 YouTube 영상을 다운로드한다.

    Returns:
        다운로드된 파일 경로 (str) 또는 None
    """
    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, '%(title)s.%(ext)s')

    ydl_opts = {
        'format': _RESOLUTION_MAP.get(resolution, _RESOLUTION_MAP['720p']),
        'outtmpl': outtmpl,
        'quiet': True,
        'no_warnings': True,
    }

    # 이미 다운로드된 파일 찾기
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        filename = ydl.prepare_filename(info)
        # merge_output_format 적용 시 확장자가 mp4로 변경됨
        base, _ = os.path.splitext(filename)
        filename = base + '.mp4'

        if skip_download and os.path.exists(filename):
            logger.info(f"기존 파일 재사용: {filename}")
            return filename

        if os.path.exists(filename):
            logger.info(f"이미 다운로드됨, 재사용: {filename}")
            return filename

        logger.info(f"영상 다운로드 중: {info.get('title', url)}")
        ydl.download([url])

        if os.path.exists(filename):
            logger.info(f"다운로드 완료: {filename}")
            return filename

    logger.error("다운로드된 파일을 찾을 수 없습니다")
    return None


# ==============================================================
#  2단계: ROI 설정
# ==============================================================

def setup_roi(video_path, output_dir, roi_json_str, roi_config_path):
    """첫 프레임을 저장하고 ROI를 설정한다.

    Returns:
        (roi_manager, has_roi) 튜플
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"영상을 열 수 없습니다: {video_path}")
        return None, False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("첫 프레임을 읽을 수 없습니다")
        return None, False

    h, w = frame.shape[:2]
    logger.info(f"영상 해상도: {w}x{h}")

    # 첫 프레임 저장
    frame_path = os.path.join(output_dir, 'youtube_first_frame.jpg')
    cv2.imwrite(frame_path, frame)
    logger.info(f"첫 프레임 저장: {frame_path}")

    # ROI 매니저 초기화
    roi_manager = ROIManager(config_path=roi_config_path)
    roi_manager.load()

    # 인라인 ROI 추가
    if roi_json_str:
        roi_data = json.loads(roi_json_str)
        name = roi_data['name']
        points = roi_data['points']
        color = roi_data.get('color', None)
        # 이미 같은 이름이 있으면 업데이트
        if roi_manager.get_roi(name):
            roi_manager.update_roi(name, points=points, color=color)
            logger.info(f"ROI 업데이트: {name}")
        else:
            roi_manager.add_roi(name, points, color)

    has_roi = len(roi_manager.get_all_rois()) > 0
    return roi_manager, has_roi


# ==============================================================
#  CLI 파서
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='YouTube 영상 파이프라인 End-to-End 테스트'
    )
    parser.add_argument('url', help='YouTube URL')

    # 파이프라인
    parser.add_argument('--model', default='models/yolov8s.onnx',
                        help='YOLO 모델 경로 (default: models/yolov8s.onnx)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='검출 신뢰도 (default: 0.5)')
    parser.add_argument('--no-fp16', action='store_true',
                        help='FP16 비활성화')
    parser.add_argument('--start-roi', default=None,
                        help='대기 시작 ROI 이름')
    parser.add_argument('--end-roi', default=None,
                        help='대기 종료 ROI 이름')

    # ROI
    parser.add_argument('--roi-config', default='config/roi_config.json',
                        help='ROI 설정 파일 (default: config/roi_config.json)')
    parser.add_argument('--roi-json', default=None,
                        help='인라인 ROI 좌표 JSON')

    # DynamoDB
    parser.add_argument('--aws-config', default='config/aws_config.json',
                        help='AWS 설정 (default: config/aws_config.json)')
    parser.add_argument('--no-dynamodb', action='store_true',
                        help='DynamoDB 전송 비활성화')
    parser.add_argument('--send-interval', type=float, default=10.0,
                        help='전송 간격 - 영상 초 (default: 10.0)')

    # 기타
    parser.add_argument('--output-dir', default='data',
                        help='다운로드 경로 (default: data)')
    parser.add_argument('--resolution', default='720p',
                        choices=['480p', '720p', '1080p', 'best'],
                        help='영상 해상도 (default: 720p)')
    parser.add_argument('--skip-download', action='store_true',
                        help='다운로드 건너뛰기')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='최대 처리 프레임 수 (0=전체)')
    parser.add_argument('--min-dwell', type=int, default=30,
                        help='최소 체류 프레임 (default: 30)')

    return parser.parse_args()


# ==============================================================
#  메인
# ==============================================================

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("YouTube 파이프라인 테스트 시작")
    logger.info("=" * 60)

    # ── 1단계: 영상 다운로드 ──
    video_path = download_video(
        args.url, args.output_dir, args.resolution, args.skip_download
    )
    if video_path is None:
        return 1

    # ── 2단계: ROI 설정 ──
    roi_manager, has_roi = setup_roi(
        video_path, args.output_dir, args.roi_json, args.roi_config
    )
    if roi_manager is None:
        return 1

    if not has_roi:
        logger.info("")
        logger.info("ROI가 설정되지 않았습니다.")
        logger.info("첫 프레임을 확인한 후 ROI 좌표를 지정하세요:")
        logger.info(f"  확인: {args.output_dir}/youtube_first_frame.jpg")
        logger.info("")
        logger.info("예시:")
        logger.info('  python test_youtube_pipeline.py "URL" \\')
        logger.info("      --start-roi queue \\")
        logger.info("      --roi-json '{\"name\":\"queue\",\"points\":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'")
        return 0

    # ── 3단계: 파이프라인 초기화 ──
    logger.info("")
    logger.info("파이프라인 초기화 중...")

    # Detector
    detector = None
    if os.path.exists(args.model):
        detector = YOLOv8Detector(
            model_path=args.model,
            conf_threshold=args.conf_threshold,
            fp16=not args.no_fp16,
            target_classes=[0],
        )
        if not detector.initialize():
            logger.error("검출기 초기화 실패")
            return 1
        logger.info("검출기 초기화 완료")
    else:
        logger.error(f"모델 파일 없음: {args.model}")
        return 1

    # Tracker
    tracker = ByteTracker(max_age=30, min_hits=3, high_thresh=args.conf_threshold)
    logger.info("ByteTracker 초기화 완료")

    # Dwell filter
    dwell_filter = ROIDwellFilter(min_dwell_frames=args.min_dwell)
    logger.info(f"ROIDwellFilter 초기화 (min_dwell={args.min_dwell})")

    # WaitTimeEstimator
    wait_estimator = None
    if args.start_roi:
        wait_estimator = WaitTimeEstimator(
            start_roi=args.start_roi,
            end_roi=args.end_roi,
            predictor_type='hybrid',
        )
        logger.info(f"WaitTimeEstimator 초기화: start={args.start_roi}, end={args.end_roi}")

    # DynamoDB
    dynamodb_sender = None
    if not args.no_dynamodb and os.path.exists(args.aws_config):
        try:
            from src.cloud.dynamodb_sender import DynamoDBSender
            dynamodb_sender = DynamoDBSender(config_path=args.aws_config)
            dynamodb_sender.start()
            logger.info("DynamoDB 전송기 초기화 완료")
        except Exception as e:
            logger.warning(f"DynamoDB 전송기 초기화 실패 (전송 없이 계속): {e}")
    elif args.no_dynamodb:
        logger.info("DynamoDB 전송 비활성화")

    # ── 4단계: 프레임 처리 루프 ──
    logger.info("")
    logger.info("프레임 처리 시작...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"영상을 열 수 없습니다: {video_path}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"영상 정보: {total_frames} frames, {video_fps:.1f} fps")

    # DynamoDB 전송 타이밍 (video time 기준)
    base_real_time_ms = int(datetime.now(tz=KST).timestamp() * 1000)
    last_send_video_ms = 0.0
    send_interval_ms = args.send_interval * 1000.0

    # 통계
    frame_count = 0
    total_detections = 0
    total_tracked = 0
    dynamo_send_count = 0
    t_start = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if args.max_frames > 0 and frame_count > args.max_frames:
                break

            video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # 1. 검출
            detections = detector.detect(frame)
            total_detections += len(detections)

            # 2. 추적
            if detections:
                tracked = tracker.update(detections)
            else:
                tracked = tracker.update([])
            total_tracked += len(tracked)

            # 3. ROI 필터링
            roi_dets = {}
            roi_counts = {}
            if tracked:
                roi_dets = roi_manager.filter_detections_by_roi(tracked)
                roi_counts = dwell_filter.update(roi_dets)
            else:
                for roi in roi_manager.get_all_rois():
                    roi_counts[roi['name']] = 0

            # 4. 대기시간 추정
            wait_result = {}
            if wait_estimator is not None:
                if tracked:
                    roi_dets_for_wait = roi_manager.filter_detections_by_roi(tracked)
                else:
                    roi_dets_for_wait = {}
                wait_result = wait_estimator.update(roi_dets_for_wait)

            # 5. DynamoDB 전송 (video time 기준)
            if dynamodb_sender is not None and wait_estimator is not None:
                if video_time_ms - last_send_video_ms >= send_interval_ms:
                    predicted_min = int(wait_result.get('predicted_wait', 0) / 60)
                    queue_count = wait_result.get('current_queue', 0)
                    timestamp_ms = base_real_time_ms + int(video_time_ms)

                    dynamodb_sender.send({
                        'restaurant_id': dynamodb_sender._restaurant_id,
                        'corner_id': dynamodb_sender._corner_id,
                        'queue_count': queue_count,
                        'est_wait_time_min': predicted_min,
                        'timestamp': timestamp_ms,
                    })
                    dynamo_send_count += 1
                    last_send_video_ms = video_time_ms

            # 6. 진행 상황 출력 (100프레임마다)
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - t_start
                proc_fps = frame_count / elapsed if elapsed > 0 else 0
                video_sec = video_time_ms / 1000.0
                pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                logger.info(
                    f"[{frame_count}/{total_frames}] {pct:.0f}% | "
                    f"video={video_sec:.1f}s | "
                    f"fps={proc_fps:.1f} | "
                    f"det={len(detections)} trk={len(tracked)} | "
                    f"roi={roi_counts} | "
                    f"dynamo_sent={dynamo_send_count}"
                )

    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 중단됨")

    cap.release()
    elapsed_total = time.monotonic() - t_start

    # ── 5단계: 결과 출력 및 정리 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("처리 결과")
    logger.info("=" * 60)

    # 처리 통계
    proc_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
    logger.info(f"  처리 프레임: {frame_count}")
    logger.info(f"  처리 시간:   {elapsed_total:.1f}초")
    logger.info(f"  처리 FPS:    {proc_fps:.1f}")
    logger.info(f"  총 검출 수:  {total_detections}")
    logger.info(f"  총 추적 수:  {total_tracked}")
    avg_det = total_detections / frame_count if frame_count > 0 else 0
    avg_trk = total_tracked / frame_count if frame_count > 0 else 0
    logger.info(f"  프레임당 평균 검출: {avg_det:.1f}")
    logger.info(f"  프레임당 평균 추적: {avg_trk:.1f}")

    # 대기시간 통계
    if wait_estimator is not None:
        stats = wait_estimator.get_statistics()
        if stats:
            logger.info("")
            logger.info("대기시간 통계:")
            logger.info(f"  완료 건수:  {stats['total_completed']}")
            logger.info(f"  평균:       {stats['mean']:.1f}초")
            logger.info(f"  중앙값:     {stats['median']:.1f}초")
            logger.info(f"  최소/최대:  {stats['min']:.1f}초 / {stats['max']:.1f}초")
            logger.info(f"  이상치 제거: {stats['total_outliers_rejected']}건")
        else:
            logger.info("")
            logger.info("대기시간 통계: 완료된 대기 기록 없음")

        predicted = wait_estimator.get_predicted_wait()
        logger.info(f"  최종 예측 대기시간: {predicted:.1f}초 ({predicted/60:.1f}분)")

    # DynamoDB 통계
    if dynamodb_sender is not None:
        logger.info("")
        logger.info("DynamoDB 전송 대기 중 (3초)...")
        time.sleep(3)
        db_stats = dynamodb_sender.stats
        logger.info("DynamoDB 전송 통계:")
        logger.info(f"  전송 성공: {db_stats['sent']}건")
        logger.info(f"  전송 실패: {db_stats['errors']}건")
        logger.info(f"  대기 중:   {db_stats['pending']}건")
        logger.info(f"  스크립트 전송 요청: {dynamo_send_count}회")

    # 정리
    logger.info("")
    logger.info("정리 중...")
    if detector is not None:
        detector.destroy()
    if dynamodb_sender is not None:
        dynamodb_sender.stop()
    logger.info("완료")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
