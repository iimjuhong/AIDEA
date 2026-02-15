#!/usr/bin/env python3
"""로컬 비디오 파일 파이프라인 테스트

로컬 비디오 파일로 전체 파이프라인을 테스트합니다.

사용법:
    python test_local_video.py video.mp4 \
        --start-roi "대기줄" \
        --roi-json '{"name":"대기줄","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

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


def setup_roi(video_path, output_dir, roi_json_str, roi_config_path):
    """ROI 설정"""
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

    # ROI 매니저 초기화
    roi_manager = ROIManager(config_path=roi_config_path)
    roi_manager.load()

    # 인라인 ROI 추가
    if roi_json_str:
        roi_data = json.loads(roi_json_str)
        name = roi_data['name']
        points = roi_data['points']
        color = roi_data.get('color', None)
        if roi_manager.get_roi(name):
            roi_manager.update_roi(name, points=points, color=color)
            logger.info(f"ROI 업데이트: {name}")
        else:
            roi_manager.add_roi(name, points, color)
            logger.info(f"ROI 추가: {name}")

    has_roi = len(roi_manager.get_all_rois()) > 0
    return roi_manager, has_roi


def parse_args():
    parser = argparse.ArgumentParser(description='로컬 비디오 파이프라인 테스트')
    parser.add_argument('video', help='비디오 파일 경로')
    
    # 파이프라인
    parser.add_argument('--model', default='models/yolov8s.onnx',
                       help='YOLO 모델 (default: models/yolov8s.onnx)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='검출 신뢰도 (default: 0.5)')
    parser.add_argument('--no-fp16', action='store_true',
                       help='FP16 비활성화')
    parser.add_argument('--start-roi', default=None,
                       help='대기 시작 ROI')
    parser.add_argument('--end-roi', default=None,
                       help='대기 종료 ROI')
    
    # ROI
    parser.add_argument('--roi-config', default='config/roi_config.json',
                       help='ROI 설정 파일')
    parser.add_argument('--roi-json', default=None,
                       help='인라인 ROI JSON')
    
    # DynamoDB
    parser.add_argument('--aws-config', default='config/aws_config.json',
                       help='AWS 설정')
    parser.add_argument('--no-dynamodb', action='store_true',
                       help='DynamoDB 비활성화')
    parser.add_argument('--send-interval', type=float, default=10.0,
                       help='전송 간격 (초)')
    
    # 기타
    parser.add_argument('--max-frames', type=int, default=0,
                       help='최대 프레임 (0=전체)')
    parser.add_argument('--min-dwell', type=int, default=30,
                       help='최소 체류 프레임')
    parser.add_argument('--output-dir', default='data',
                       help='출력 디렉토리')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("로컬 비디오 파이프라인 테스트")
    logger.info("=" * 60)
    
    # 비디오 파일 확인
    if not os.path.exists(args.video):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
        return 1
    
    logger.info(f"비디오: {args.video}")
    
    # ROI 설정
    roi_manager, has_roi = setup_roi(
        args.video, args.output_dir, args.roi_json, args.roi_config
    )
    if roi_manager is None:
        return 1
    
    if not has_roi:
        logger.error("ROI가 설정되지 않았습니다")
        return 1
    
    # 파이프라인 초기화
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
        logger.info(f"WaitTimeEstimator: start={args.start_roi}, end={args.end_roi}")
    
    # DynamoDB
    dynamodb_sender = None
    if not args.no_dynamodb and os.path.exists(args.aws_config):
        try:
            from src.cloud.dynamodb_sender import DynamoDBSender
            dynamodb_sender = DynamoDBSender(config_path=args.aws_config)
            dynamodb_sender.start()
            logger.info("DynamoDB 전송기 초기화 완료")
        except Exception as e:
            logger.warning(f"DynamoDB 초기화 실패: {e}")
    
    # 프레임 처리
    logger.info("")
    logger.info("프레임 처리 시작...")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"영상을 열 수 없습니다: {args.video}")
        return 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"영상: {total_frames} frames, {video_fps:.1f} fps")
    
    # DynamoDB 전송 타이밍
    base_time_ms = int(datetime.now(tz=KST).timestamp() * 1000)
    last_send_time_ms = 0.0
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
            tracked = tracker.update(detections if detections else [])
            total_tracked += len(tracked)
            
            # 3. ROI 필터링
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
                roi_dets_for_wait = roi_manager.filter_detections_by_roi(tracked) if tracked else {}
                wait_result = wait_estimator.update(roi_dets_for_wait)
            
            # 5. DynamoDB 전송
            if dynamodb_sender is not None and wait_estimator is not None:
                if video_time_ms - last_send_time_ms >= send_interval_ms:
                    predicted_min = int(wait_result.get('predicted_wait', 0) / 60)
                    queue_count = wait_result.get('current_queue', 0)
                    timestamp_ms = base_time_ms + int(video_time_ms)
                    
                    dynamodb_sender.send({
                        'restaurant_id': dynamodb_sender._restaurant_id,
                        'corner_id': dynamodb_sender._corner_id,
                        'queue_count': queue_count,
                        'est_wait_time_min': predicted_min,
                        'timestamp': timestamp_ms,
                    })
                    dynamo_send_count += 1
                    last_send_time_ms = video_time_ms
            
            # 6. 진행 상황
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - t_start
                proc_fps = frame_count / elapsed if elapsed > 0 else 0
                pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                logger.info(
                    f"[{frame_count}/{total_frames}] {pct:.0f}% | "
                    f"fps={proc_fps:.1f} | "
                    f"det={len(detections)} trk={len(tracked)} | "
                    f"roi={roi_counts} | "
                    f"dyn={dynamo_send_count}"
                )
    
    except KeyboardInterrupt:
        logger.info("\n사용자 중단")
    
    cap.release()
    elapsed_total = time.monotonic() - t_start
    
    # 결과
    logger.info("")
    logger.info("=" * 60)
    logger.info("처리 결과")
    logger.info("=" * 60)
    
    proc_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
    logger.info(f"  프레임: {frame_count}")
    logger.info(f"  시간: {elapsed_total:.1f}초")
    logger.info(f"  FPS: {proc_fps:.1f}")
    logger.info(f"  검출: {total_detections}")
    logger.info(f"  추적: {total_tracked}")
    
    # 대기시간 통계
    if wait_estimator is not None:
        stats = wait_estimator.get_statistics()
        if stats:
            logger.info("")
            logger.info("대기시간 통계:")
            logger.info(f"  완료: {stats['total_completed']}건")
            logger.info(f"  평균: {stats['mean']:.1f}초")
            logger.info(f"  최소/최대: {stats['min']:.1f}초 / {stats['max']:.1f}초")
        
        predicted = wait_estimator.get_predicted_wait()
        logger.info(f"  예측: {predicted:.1f}초 ({predicted/60:.1f}분)")
    
    # DynamoDB 통계
    if dynamodb_sender is not None:
        logger.info("")
        logger.info("DynamoDB 대기 중 (3초)...")
        time.sleep(3)
        db_stats = dynamodb_sender.stats
        logger.info(f"  전송 성공: {db_stats['sent']}건")
        logger.info(f"  전송 실패: {db_stats['errors']}건")
        logger.info(f"  요청 횟수: {dynamo_send_count}회")
    
    # 정리
    logger.info("")
    logger.info("정리 중...")
    if detector is not None:
        detector.destroy()
    if dynamodb_sender is not None:
        dynamodb_sender.stop()
    logger.info("✅ 완료")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
