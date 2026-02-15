# ⚡ 빠른 실행 가이드 (QUICKSTART)

---

## 🎯 시나리오별 명령어

### 1️⃣ 처음 시작할 때 (최초 1회만)

```bash
# 프로젝트 폴더로 이동
cd /home/iimjuhong/projects/aidea

# 의존성 설치
pip3 install -r requirements.txt

# YOLO 모델 다운로드
bash scripts/download_model.sh
```

---

### 2️⃣ 기본 실행 (카메라 + YOLO만)

```bash
# 프로젝트 폴더에서
cd /home/iimjuhong/projects/aidea

# 실행
python3 main.py

# 웹 UI 접속
# http://localhost:5000
```

**이 모드는**: 사람 검출만 하고, 대기시간 측정 안 함

---

### 3️⃣ 대기시간 측정 실행 (AWS 없이)

```bash
# ROI 설정이 필요합니다 (웹 UI에서 먼저 설정)
python3 main.py --start-roi "대기구역"

# 또는 플로우 모드 (시작→종료)
python3 main.py --start-roi "대기구역" --end-roi "카운터"
```

**이 모드는**: 대기시간 계산하지만 AWS 전송 안 함

---

### 4️⃣ AWS DynamoDB 전송까지 (풀 스택)

#### Step 1: AWS 자격증명 설정 (최초 1회)

```bash
# 환경 변수로 설정 (임시)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# 또는 ~/.bashrc에 영구 저장
echo 'export AWS_ACCESS_KEY_ID="your-access-key"' >> ~/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY="your-secret-key"' >> ~/.bashrc
source ~/.bashrc
```

#### Step 2: DynamoDB 설정 확인

```bash
# 설정 파일 열기
nano config/aws_config.json

# 내용 확인/수정:
# {
#   "region": "ap-northeast-2",
#   "table_name": "hyeat_YOLO_data",
#   "restaurant_id": "hanyang_plaza",
#   "corner_id": "western"
# }
```

#### Step 3: 실행

```bash
python3 main.py --start-roi "대기구역" --end-roi "카운터"
```

#### Step 4: 전송 확인

```bash
# DynamoDB 전송 통계 보기
curl http://localhost:5000/api/dynamodb/stats

# 응답 예시:
# {"sent": 152, "errors": 0, "pending": 0}
```

---

## 🔧 ROI (관심 영역) 설정하는 법

대기시간 측정을 하려면 ROI를 먼저 설정해야 합니다!

```bash
# 1. 프로그램 먼저 실행
python3 main.py

# 2. 웹 브라우저에서 접속
# http://localhost:5000

# 3. 웹 UI에서:
#    - ROI 이름 입력 (예: "대기구역")
#    - "그리기 시작" 클릭
#    - 마우스로 영역 지정:
#      * 좌클릭: 꼭짓점 추가
#      * 우클릭: 완성
#    - 자동 저장됨: config/roi_config.json

# 4. 설정 확인
cat config/roi_config.json
```

---

## 📊 주요 API 엔드포인트

프로그램 실행 중에 브라우저나 curl로 확인 가능:

```bash
# 실시간 통계 (FPS, 인원 수)
curl http://localhost:5000/api/stats

# 대기시간 정보
curl http://localhost:5000/api/wait_time

# ROI 목록 보기
curl http://localhost:5000/api/roi

# DynamoDB 전송 통계
curl http://localhost:5000/api/dynamodb/stats

# 시스템 헬스체크
curl http://localhost:5000/health
```

---

## 🚨 자주 하는 실수

### ❌ ROI 설정 안 하고 대기시간 측정 시도
```bash
python3 main.py --start-roi "대기구역"
# 에러: "대기구역" ROI가 없음!
```
**해결**: 웹 UI에서 ROI 먼저 그리기

### ❌ AWS 자격증명 없이 DynamoDB 전송
```bash
python3 main.py --start-roi "대기구역"
# 에러: Unable to locate credentials
```
**해결**: `export AWS_ACCESS_KEY_ID=...` 실행

### ❌ 카메라 접근 권한 없음
```bash
python3 main.py
# 에러: Failed to open camera
```
**해결**: 
```bash
sudo usermod -aG video $USER
# 로그아웃 후 재로그인
```

---

## 💡 팁

### 백그라운드 실행
```bash
# nohup으로 백그라운드 실행
nohup python3 main.py --start-roi "대기구역" > output.log 2>&1 &

# 프로세스 확인
ps aux | grep main.py

# 종료
pkill -f main.py
```

### 로그 보기
```bash
# 실시간 로그 확인
tail -f output.log
```

### 설정 파일 위치
```bash
config/
├── aws_config.json      # AWS DynamoDB 설정
└── roi_config.json      # ROI 영역 설정 (웹 UI에서 자동 생성)
```

---

## 📚 더 자세한 내용이 필요하면?

- **전체 문서**: [README.md](README.md)
- **대기시간 알고리즘**: [docs/Phase5_대기시간_알고리즘_가이드.md](docs/Phase5_대기시간_알고리즘_가이드.md)
- **프로젝트 구조**: README.md의 "프로젝트 구조" 섹션

---

## 🎬 가장 많이 쓰는 명령어 TOP 3

```bash
# 1위: 기본 실행
python3 main.py

# 2위: 대기시간 측정 + AWS 전송
python3 main.py --start-roi "대기구역" --end-roi "카운터"

# 3위: 통계 확인
curl http://localhost:5000/api/stats
```
