#!/usr/bin/env python3
"""간단한 비디오 뷰어 웹 서버

사용법:
    python video_viewer.py data/detection_result.mp4
    
    브라우저에서 http://localhost:8000 접속
"""

import sys
import os
import argparse
from flask import Flask, render_template_string, send_file

app = Flask(__name__)

VIDEO_PATH = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>검출 결과 비디오</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        h1 { text-align: center; }
        .container {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        video {
            width: 100%;
            max-width: 100%;
            border-radius: 4px;
            background: black;
        }
        .info {
            background: #333;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }
        .controls {
            margin-top: 15px;
            text-align: center;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 검출 결과 비디오</h1>
        
        <video id="video" controls autoplay>
            <source src="/video" type="video/mp4">
            브라우저가 비디오를 지원하지 않습니다.
        </video>
        
        <div class="controls">
            <button onclick="video.currentTime = 0; video.play();">⏮️ 처음부터</button>
            <button onclick="video.playbackRate = 0.5;">🐌 0.5x</button>
            <button onclick="video.playbackRate = 1.0;">▶️ 1x</button>
            <button onclick="video.playbackRate = 2.0;">⏩ 2x</button>
        </div>
        
        <div class="info">
            <strong>📋 안내:</strong><br>
            - <strong>초록색 박스</strong>: 검출된 사람<br>
            - <strong>Track ID</strong>: 추적 ID (같은 사람 계속 추적)<br>
            - <strong>ROI 영역</strong>: 반투명 오버레이<br>
            - <strong>ROI별 인원수</strong>: ROI 중앙에 표시<br>
            <br>
            <strong>파일:</strong> {{ filename }}
        </div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        
        // 키보드 단축키
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                if (video.paused) video.play();
                else video.pause();
            } else if (e.code === 'ArrowLeft') {
                video.currentTime -= 5;
            } else if (e.code === 'ArrowRight') {
                video.currentTime += 5;
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    filename = os.path.basename(VIDEO_PATH)
    return render_template_string(HTML_TEMPLATE, filename=filename)


@app.route('/video')
def serve_video():
    return send_file(VIDEO_PATH, mimetype='video/mp4')


def main():
    global VIDEO_PATH
    
    parser = argparse.ArgumentParser(description='비디오 뷰어 웹 서버')
    parser.add_argument('video', help='비디오 파일 경로')
    parser.add_argument('--port', type=int, default=8000, help='포트 (기본: 8000)')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {args.video}")
        return 1
    
    VIDEO_PATH = os.path.abspath(args.video)
    
    print("=" * 60)
    print("🎥 비디오 뷰어 웹 서버")
    print("=" * 60)
    print(f"비디오: {VIDEO_PATH}")
    print()
    print(f"🌐 브라우저에서 접속:")
    print(f"   http://localhost:{args.port}")
    print()
    print("종료: Ctrl+C")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=False)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
