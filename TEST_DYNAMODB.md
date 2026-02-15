# 🧪 DynamoDB 테스트 가이드

> DynamoDB에 테스트 데이터를 전송하고 확인하는 방법

---

## 🚀 빠른 실행

### 1단계: AWS 자격증명 설정

```bash
# 환경 변수 설정 (임시)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# 확인
echo $AWS_ACCESS_KEY_ID
```

### 2단계: 테스트 스크립트 실행

```bash
# 프로젝트 폴더에서
cd /home/iimjuhong/projects/aidea

# 가상환경 활성화 (사용 시)
source venv/bin/activate

# 테스트 실행
python3 test_dynamodb_send.py
```

### 예상 출력:

```
============================================================
DynamoDB 테스트 전송 시작
============================================================
✅ DynamoDB 전송기 초기화 성공
✅ 백그라운드 워커 스레드 시작

📤 테스트 데이터 전송 중...
  [1/5] 전송: queue=5명, wait=3.5분
  [2/5] 전송: queue=6명, wait=4.0분
  [3/5] 전송: queue=7명, wait=4.5분
  [4/5] 전송: queue=8명, wait=5.0분
  [5/5] 전송: queue=9명, wait=5.5분

⏳ 전송 완료 대기 중 (5초)...

📊 전송 통계:
  ✅ 전송 성공: 5개
  ❌ 전송 실패: 0개
  ⏳ 대기 중: 0개

🎉 테스트 성공!
DynamoDB에 데이터가 정상적으로 전송되었습니다.

📋 DynamoDB에서 확인하는 방법:
  1. AWS 콘솔 접속
  2. DynamoDB → Tables → hyeat-waiting-data-dev
  3. 'Explore table items' 클릭
  4. 방금 전송한 데이터 확인

🧹 정리 중...
✅ 전송기 중지 완료
============================================================
```

---

## 📋 DynamoDB 콘솔에서 확인하기

### 방법 1: AWS 콘솔 (웹)

1. **AWS 콘솔 접속**: https://console.aws.amazon.com/
2. **DynamoDB 서비스 이동**: 검색창에 "DynamoDB" 입력
3. **테이블 선택**: `hyeat-waiting-data-dev` 클릭
4. **아이템 보기**: "Explore table items" 탭 클릭
5. **데이터 확인**: 방금 전송한 데이터 확인

**보이는 필드**:
```
pk: CORNER#hanyang_plaza#western
sk: 1770352800000
restaurantId: hanyang_plaza
cornerId: western
queueLen: 5
estWaitTimeMin: 3.5
dataType: observed
source: jetson_nano
timestampIso: 2026-02-15T21:00:00+09:00
createdAtIso: 2026-02-15T21:00:01+09:00
ttl: 1770612000
```

### 방법 2: AWS CLI

```bash
# 테이블 아이템 조회 (최근 10개)
aws dynamodb query \
  --table-name hyeat-waiting-data-dev \
  --key-condition-expression "pk = :pk" \
  --expression-attribute-values '{":pk":{"S":"CORNER#hanyang_plaza#western"}}' \
  --scan-index-forward false \
  --limit 10 \
  --region ap-northeast-2

# 전체 아이템 수 확인
aws dynamodb describe-table \
  --table-name hyeat-waiting-data-dev \
  --region ap-northeast-2 \
  --query 'Table.ItemCount'
```

---

## 🔧 테스트 스크립트 커스터마이징

### 전송 개수 변경

`test_dynamodb_send.py` 파일 수정:

```python
# 기본: 5개
test_data_list = generate_test_data(count=5)

# 변경: 10개
test_data_list = generate_test_data(count=10)
```

### 데이터 내용 변경

`test_dynamodb_send.py`의 `generate_test_data()` 함수 수정:

```python
data = {
    'restaurant_id': 'hanyang_plaza',  # 식당 ID
    'corner_id': 'korean',             # 코너 ID (변경 가능)
    'queue_count': 10,                 # 대기 인원
    'est_wait_time_min': 5.5,          # 예상 대기시간 (분)
    'timestamp': timestamp
}
```

---

## 🚨 문제 해결

### ❌ "Unable to locate credentials"

**원인**: AWS 자격증명 미설정

**해결**:
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

### ❌ "ResourceNotFoundException: Table not found"

**원인**: DynamoDB 테이블이 없음

**해결**:
```bash
# 테이블 생성
aws dynamodb create-table \
  --table-name hyeat-waiting-data-dev \
  --attribute-definitions \
    AttributeName=pk,AttributeType=S \
    AttributeName=sk,AttributeType=S \
  --key-schema \
    AttributeName=pk,KeyType=HASH \
    AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region ap-northeast-2
```

### ❌ "전송 성공: 0개"

**원인**: 네트워크 문제 또는 권한 부족

**확인 사항**:
1. 인터넷 연결 확인
2. AWS IAM 권한 확인 (DynamoDB 쓰기 권한 필요)
3. 로그 확인: `grep ERROR` 출력 확인

---

## 📊 실전 시뮬레이션 (연속 전송)

계속해서 데이터를 전송하는 스크립트:

```python
# test_dynamodb_continuous.py
import time
from test_dynamodb_send import DynamoDBSender, generate_test_data, logger

sender = DynamoDBSender(config_path='config/aws_config.json')
sender.start()

try:
    logger.info("10초마다 데이터 전송 시작... (Ctrl+C로 중지)")
    while True:
        data_list = generate_test_data(count=1)
        sender.send(data_list[0])
        logger.info(f"전송 완료: queue={data_list[0]['queue_count']}")
        time.sleep(10)
except KeyboardInterrupt:
    logger.info("중지됨")
finally:
    sender.stop()
```

**실행**:
```bash
python3 test_dynamodb_continuous.py
```

---

## 💡 팁

### 1. 전송 통계 실시간 모니터링

테스트 스크립트 실행 중 다른 터미널에서:

```bash
# 웹 서버가 실행 중이면
curl http://localhost:5000/api/dynamodb/stats

# 출력:
# {"sent": 5, "errors": 0, "pending": 0}
```

### 2. 로그 파일 확인

```bash
# 에러 로그만 필터링
python3 test_dynamodb_send.py 2>&1 | grep ERROR
```

### 3. DynamoDB 비용 절감

- 테스트 후 TTL이 자동으로 데이터 삭제 (3일 후)
- 필요 없으면 수동 삭제:

```bash
# 테이블 비우기 (주의!)
aws dynamodb scan \
  --table-name hyeat-waiting-data-dev \
  --region ap-northeast-2 \
  --attributes-to-get "pk" "sk" \
  --query "Items[*]" \
  | jq -c '.[]' \
  | while read item; do
      aws dynamodb delete-item \
        --table-name hyeat-waiting-data-dev \
        --key "$item" \
        --region ap-northeast-2
    done
```

---

## 📚 다음 단계

1. ✅ **테스트 완료**: DynamoDB 연동 확인
2. 🎯 **실전 배포**: 실제 카메라와 연동
3. 📊 **대시보드**: 웹에서 데이터 시각화 (Phase 7)

---

## 🔗 관련 문서

- **빠른 실행**: [QUICKSTART.md](QUICKSTART.md)
- **폴더 구조**: [FOLDER_GUIDE.md](FOLDER_GUIDE.md)
- **DynamoDB 송신 코드**: [src/cloud/dynamodb_sender.py](src/cloud/dynamodb_sender.py)
