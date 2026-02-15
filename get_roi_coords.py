#!/usr/bin/env python3
"""ROI 좌표 추출 도구

이미지에서 마우스로 클릭해서 ROI 좌표를 얻습니다.

사용법:
    python get_roi_coords.py data/youtube_first_frame.jpg
    
    1. 이미지가 열립니다
    2. 마우스로 ROI 영역의 4개 꼭짓점을 순서대로 클릭
    3. 'q' 키를 누르면 좌표가 출력됩니다
    4. 출력된 명령어를 복사해서 사용하세요
"""

import sys
import cv2
import argparse

coords = []
img_display = None
original_img = None


def click_event(event, x, y, flags, params):
    """마우스 클릭 이벤트 핸들러"""
    global coords, img_display, original_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append([x, y])
        print(f"✓ 좌표 {len(coords)}: ({x}, {y})")
        
        # 이미지에 점 표시
        img_display = original_img.copy()
        
        # 클릭한 점들 그리기
        for i, coord in enumerate(coords):
            cv2.circle(img_display, tuple(coord), 5, (0, 0, 255), -1)
            cv2.putText(img_display, str(i+1), (coord[0]+10, coord[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 4개 이상이면 다각형 그리기
        if len(coords) >= 2:
            cv2.polylines(img_display, [np.array(coords)], 
                         len(coords) >= 4, (0, 255, 0), 2)
        
        cv2.imshow('ROI 좌표 선택', img_display)
        
        if len(coords) == 4:
            print("\n✅ 4개 좌표 선택 완료!")
            print("'q' 키를 눌러 종료하고 명령어를 확인하세요.")


def main():
    global img_display, original_img
    
    parser = argparse.ArgumentParser(description='ROI 좌표 추출 도구')
    parser.add_argument('image', help='이미지 파일 경로')
    parser.add_argument('--roi-name', default='대기줄', help='ROI 이름 (기본: 대기줄)')
    args = parser.parse_args()
    
    # 이미지 로드
    original_img = cv2.imread(args.image)
    if original_img is None:
        print(f"❌ 이미지를 열 수 없습니다: {args.image}")
        return 1
    
    img_display = original_img.copy()
    height, width = original_img.shape[:2]
    
    print("=" * 60)
    print("ROI 좌표 추출 도구")
    print("=" * 60)
    print(f"이미지: {args.image}")
    print(f"크기: {width}x{height}")
    print()
    print("📋 사용 방법:")
    print("  1. 마우스로 ROI 영역의 4개 꼭짓점을 순서대로 클릭")
    print("     (왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 권장)")
    print("  2. 'r'을 누르면 리셋")
    print("  3. 'q'를 누르면 완료")
    print("=" * 60)
    
    cv2.imshow('ROI 좌표 선택', img_display)
    cv2.setMouseCallback('ROI 좌표 선택', click_event)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' 키: 종료
        if key == ord('q'):
            break
        
        # 'r' 키: 리셋
        elif key == ord('r'):
            coords.clear()
            img_display = original_img.copy()
            cv2.imshow('ROI 좌표 선택', img_display)
            print("\n↻ 리셋됨 - 다시 클릭하세요")
    
    cv2.destroyAllWindows()
    
    # 결과 출력
    print()
    print("=" * 60)
    print("결과")
    print("=" * 60)
    
    if len(coords) < 3:
        print("❌ 최소 3개 이상의 좌표가 필요합니다")
        return 1
    
    print(f"✅ 선택한 좌표: {coords}")
    print()
    print("📋 복사해서 사용하세요:")
    print()
    
    # ROI JSON 생성
    roi_json = f'{{"name":"{args.roi_name}","points":{coords}}}'
    
    # 명령어 출력
    print("# YouTube 파이프라인 테스트:")
    print(f"python test_youtube_pipeline.py \"YOUR_URL\" \\")
    print(f"    --start-roi \"{args.roi_name}\" \\")
    print(f"    --roi-json '{roi_json}'")
    print()
    
    # config/roi_config.json에 추가하는 방법
    print("# 또는 config/roi_config.json에 직접 추가:")
    print(f'''{{
  "rois": [
    {{
      "name": "{args.roi_name}",
      "points": {coords},
      "color": [0, 255, 0]
    }}
  ]
}}''')
    
    print()
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    # numpy import 체크
    try:
        import numpy as np
    except ImportError:
        print("numpy가 필요합니다: pip install numpy")
        sys.exit(1)
    
    sys.exit(main())
