# 🧪 테스트 실행 가이드

모든 테스트는 `projects/aidea` 폴더에서 실행합니다.

## 0. YouTube 영상 파이프라인 테스트 (E2E)

YouTube 영상을 다운로드해서 검출→추적→ROI→대기시간→DynamoDB 전체 파이프라인을 테스트합니다.

```bash
# yt-dlp 설치 (최초 1회)
pip install yt-dlp

# 1단계: 첫 프레임 확인 (ROI 없이 → 프레임 저장 후 종료)
python test_youtube_pipeline.py "YOUTUBE_URL"
# → data/youtube_first_frame.jpg 확인

# 2단계: ROI 좌표 지정 후 전체 파이프라인
python test_youtube_pipeline.py "YOUTUBE_URL" \
    --start-roi queue \
    --roi-json '{"name":"queue","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'

# DynamoDB 없이 테스트
python test_youtube_pipeline.py "YOUTUBE_URL" \
    --start-roi queue --no-dynamodb \
    --roi-json '{"name":"queue","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'
```

---

## 1. 대기시간 추정 및 데이터 전송 테스트 (로그 확인용)
이 명령어는 DynamoDB로 데이터를 전송하고 콘솔에 로그를 출력합니다. 비디오 파일은 생성되지 않습니다.

```bash
# hybrid 모드 (대기줄 + 입구)
python test_local_video.py data/slow_0.3x.mp4 \
    --start-roi "queue" \
    --end-roi "entrance"
```

## 2. 검출 결과 시각화 비디오 생성 (눈으로 확인용)
이 명령어는 검출 박스와 ROI, 인원수를 영상에 그려서 저장합니다.

```bash
# 기본 모드 (30프레임 이상 머물러야 카운트)
python visualize_detection.py data/slow_0.3x.mp4 data/detection_result.mp4

# 빠른 검출 모드 (1프레임만 머물러도 카운트 - 테스트용 추천)
python visualize_detection.py data/slow_0.3x.mp4 data/detection_viz_fast.mp4 \
    --min-dwell 1
```

## 3. 웹 브라우저 재생을 위한 변환 (H.264)
생성된 비디오를 브라우저에서 볼 수 있게 변환합니다.

```bash
ffmpeg -i data/detection_viz_fast.mp4 -c:v libx264 -preset fast -crf 23 -c:a copy data/detection_viz_fast_h264.mp4 -y
```

## 4. 결과 비디오 웹 뷰어 실행
변환된 비디오를 브라우저에서 확인합니다.

```bash
# 기존 실행 중인 뷰어 종료 (필요시)
fuser -k 8000/tcp

# 뷰어 실행
python video_viewer.py data/detection_viz_fast_h264.mp4
```

브라우저 주소: `http://localhost:8000`
