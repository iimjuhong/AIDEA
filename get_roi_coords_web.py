#!/usr/bin/env python3
"""웹 기반 ROI 좌표 선택 도구

GUI 없이 웹 브라우저에서 ROI 좌표를 선택합니다.

사용법:
    python get_roi_coords_web.py data/youtube_first_frame.jpg
    
    그러면 http://localhost:5001 에서 접속 가능
"""

import sys
import os
import argparse
from flask import Flask, render_template_string, jsonify, request, send_file

app = Flask(__name__)

IMAGE_PATH = None
ROI_NAME = "대기줄"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ROI 좌표 선택</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        canvas {
            border: 2px solid #ddd;
            cursor: crosshair;
            display: block;
            margin: 20px 0;
        }
        .coords {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
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
        button.reset { background: #f44336; }
        button.reset:hover { background: #da190b; }
        .info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .command {
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖱️ ROI 좌표 선택 도구</h1>
        
        <div class="info">
            <strong>📋 사용 방법:</strong><br>
            1. 이미지에서 ROI 영역의 4개 꼭짓점을 클릭하세요<br>
            2. 순서: 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 (권장)<br>
            3. 리셋 버튼으로 다시 시작 가능<br>
            4. 4개 클릭하면 아래에 명령어가 나타납니다
        </div>
        
        <canvas id="canvas"></canvas>
        
        <div>
            <button class="reset" onclick="resetCoords()">🔄 리셋</button>
            <button onclick="copyCommand()">📋 명령어 복사</button>
        </div>
        
        <div class="coords">
            <strong>선택한 좌표:</strong> <span id="coords-display">없음</span>
        </div>
        
        <div id="command-section" style="display: none;">
            <h3>✅ 명령어 복사:</h3>
            <div class="command" id="command-text"></div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const coordsDisplay = document.getElementById('coords-display');
        const commandSection = document.getElementById('command-section');
        const commandText = document.getElementById('command-text');
        
        let coords = [];
        let img = new Image();
        
        img.onload = function() {
            canvas.width = img.width;
            canvas.height = img.height;
            drawImage();
        };
        img.src = '/image';
        
        function drawImage() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            
            // 점 그리기
            coords.forEach((coord, i) => {
                ctx.fillStyle = 'red';
                ctx.beginPath();
                ctx.arc(coord[0], coord[1], 5, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.fillText(String(i + 1), coord[0] + 10, coord[1] - 10);
            });
            
            // 선 그리기
            if (coords.length >= 2) {
                ctx.strokeStyle = 'lime';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(coords[0][0], coords[0][1]);
                for (let i = 1; i < coords.length; i++) {
                    ctx.lineTo(coords[i][0], coords[i][1]);
                }
                if (coords.length === 4) {
                    ctx.closePath();
                }
                ctx.stroke();
            }
        }
        
        canvas.addEventListener('click', (e) => {
            if (coords.length >= 4) {
                alert('이미 4개 좌표를 선택했습니다. 리셋 후 다시 시도하세요.');
                return;
            }
            
            const rect = canvas.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);
            
            coords.push([x, y]);
            console.log(`좌표 ${coords.length}: (${x}, ${y})`);
            
            updateDisplay();
            drawImage();
        });
        
        function updateDisplay() {
            coordsDisplay.textContent = JSON.stringify(coords);
            
            if (coords.length === 4) {
                const roiJson = `{"name":"{{ roi_name }}","points":${JSON.stringify(coords)}}`;
                const command = `python test_youtube_pipeline.py "YOUR_URL" \\\\
    --start-roi "{{ roi_name }}" \\\\
    --roi-json '${roiJson}'`;
                
                commandText.textContent = command;
                commandSection.style.display = 'block';
            }
        }
        
        function resetCoords() {
            coords = [];
            coordsDisplay.textContent = '없음';
            commandSection.style.display = 'none';
            drawImage();
        }
        
        function copyCommand() {
            if (coords.length !== 4) {
                alert('4개 좌표를 먼저 선택하세요!');
                return;
            }
            
            const text = commandText.textContent;
            navigator.clipboard.writeText(text).then(() => {
                alert('✅ 명령어가 클립보드에 복사되었습니다!');
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, roi_name=ROI_NAME)


@app.route('/image')
def serve_image():
    return send_file(IMAGE_PATH, mimetype='image/jpeg')


def main():
    global IMAGE_PATH, ROI_NAME
    
    parser = argparse.ArgumentParser(description='웹 기반 ROI 좌표 선택')
    parser.add_argument('image', help='이미지 파일 경로')
    parser.add_argument('--roi-name', default='대기줄', help='ROI 이름')
    parser.add_argument('--port', type=int, default=5001, help='포트 번호')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        return 1
    
    IMAGE_PATH = os.path.abspath(args.image)
    ROI_NAME = args.roi_name
    
    print("=" * 60)
    print("웹 기반 ROI 좌표 선택 도구")
    print("=" * 60)
    print(f"이미지: {IMAGE_PATH}")
    print(f"ROI 이름: {ROI_NAME}")
    print()
    print(f"🌐 브라우저에서 접속하세요:")
    print(f"   http://localhost:{args.port}")
    print()
    print("종료하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=False)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
