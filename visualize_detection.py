#!/usr/bin/env python3
"""검출 결과 시각화 도구

비디오에 검출/추적 결과를 그려서 저장합니다.

사용법:
    python visualize_detection.py input.mp4 output.mp4 \
        --roi-json '{"name":"대기줄","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'
"""

import argparse
import json
import logging
import os
import sys
import cv2
import numpy as np

from src.core.detector import YOLOv8Detector
from src.core.tracker import ByteTracker, ROIDwellFilter
from src.core.roi_manager import ROIManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """검출 결과 그리기"""
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        track_id = det.get('track_id', -1)
        
        # 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨
        if track_id >= 0:
            label = f"ID:{track_id} {conf:.2f}"
        else:
            label = f"{conf:.2f}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_rois(frame, roi_manager, roi_counts=None):
    """ROI 그리기"""
    for roi in roi_manager.get_all_rois():
        name = roi['name']
        points = np.array(roi['points'], dtype=np.int32)
        color = tuple(roi.get('color', [0, 255, 0]))
        
        # ROI 영역 (반투명)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # ROI 테두리
        cv2.polylines(frame, [points], True, color, 2)
        
        # ROI 이름 + 인원수
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        
        if roi_counts and name in roi_counts:
            count = roi_counts[name]
            label = f"{name}: {count}명"
        else:
            label = name
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, 
                     (center_x - label_size[0]//2 - 5, center_y - label_size[1] - 5),
                     (center_x + label_size[0]//2 + 5, center_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(frame, label, 
                   (center_x - label_size[0]//2, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def setup_roi(video_path, roi_json_str, roi_config_path):
    """ROI 설정"""
    roi_manager = ROIManager(config_path=roi_config_path)
    roi_manager.load()
    
    if roi_json_str:
        roi_data = json.loads(roi_json_str)
        name = roi_data['name']
        points = roi_data['points']
        color = roi_data.get('color', [0, 255, 0])
        
        if roi_manager.get_roi(name):
            roi_manager.update_roi(name, points=points, color=color)
        else:
            roi_manager.add_roi(name, points, color)
    
    return roi_manager


def parse_args():
    parser = argparse.ArgumentParser(description='검출 결과 시각화')
    parser.add_argument('input', help='입력 비디오')
    parser.add_argument('output', help='출력 비디오')
    
    parser.add_argument('--model', default='models/yolov8s.onnx',
                       help='YOLO 모델')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='검출 신뢰도')
    parser.add_argument('--no-fp16', action='store_true',
                       help='FP16 비활성화')
    
    parser.add_argument('--roi-config', default='config/roi_config.json',
                       help='ROI 설정')
    parser.add_argument('--roi-json', default=None,
                       help='인라인 ROI')
    
    parser.add_argument('--max-frames', type=int, default=0,
                       help='최대 프레임')
    parser.add_argument('--min-dwell', type=int, default=30,
                       help='최소 체류 프레임')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='N 프레임마다 처리 (속도 향상)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("검출 결과 시각화")
    logger.info("=" * 60)
    
    if not os.path.exists(args.input):
        logger.error(f"입력 파일 없음: {args.input}")
        return 1
    
    # ROI 설정
    roi_manager = setup_roi(args.input, args.roi_json, args.roi_config)
    logger.info(f"ROI {len(roi_manager.get_all_rois())}개 로드")
    
    # 검출기
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
    
    # 추적기
    tracker = ByteTracker(max_age=30, min_hits=3, high_thresh=args.conf_threshold)
    dwell_filter = ROIDwellFilter(min_dwell_frames=args.min_dwell)
    logger.info("추적기 초기화 완료")
    
    # 비디오
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"영상 열기 실패: {args.input}")
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"입력: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
    
    # 출력 비디오
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error(f"출력 파일 열기 실패: {args.output}")
        return 1
    
    logger.info(f"출력: {args.output}")
    logger.info("")
    logger.info("처리 시작...")
    
    frame_count = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if args.max_frames > 0 and frame_count > args.max_frames:
                break
            
            # N 프레임마다 처리
            if frame_count % args.skip_frames != 0:
                out.write(frame)
                continue
            
            processed_count += 1
            
            # 검출
            detections = detector.detect(frame)
            
            # 추적
            tracked = tracker.update(detections if detections else [])
            
            # ROI 필터링
            roi_counts = {}
            if tracked:
                roi_dets = roi_manager.filter_detections_by_roi(tracked)
                roi_counts = dwell_filter.update(roi_dets)
            else:
                for roi in roi_manager.get_all_rois():
                    roi_counts[roi['name']] = 0
            
            # 시각화
            # 1. 모든 검출 박스 (연한 초록)
            if detections:
                draw_detections(frame, detections, color=(100, 255, 100), thickness=1)
            
            # 2. 추적된 객체 (진한 초록, 굵게)
            if tracked:
                draw_detections(frame, tracked, color=(0, 255, 0), thickness=2)
            
            # 3. ROI 영역
            draw_rois(frame, roi_manager, roi_counts)
            
            # 4. 정보 표시
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(frame, f"Detected: {len(detections)}, Tracked: {len(tracked)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
            
            if processed_count % 100 == 0:
                pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"[{frame_count}/{total_frames}] {pct:.0f}% | "
                          f"det={len(detections)} trk={len(tracked)} | roi={roi_counts}")
    
    except KeyboardInterrupt:
        logger.info("\n사용자 중단")
    
    cap.release()
    out.release()
    detector.destroy()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ 완료: {args.output}")
    logger.info(f"   처리 프레임: {processed_count}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
