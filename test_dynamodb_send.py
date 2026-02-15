#!/usr/bin/env python3
"""
DynamoDB 테스트 전송 스크립트

테스트 데이터를 DynamoDB로 전송해서 연동을 확인합니다.
"""

import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from src.cloud.dynamodb_sender import DynamoDBSender

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


def generate_test_data(count=5):
    """테스트 데이터 생성"""
    logger.info(f"테스트 데이터 {count}개 생성 중...")
    
    test_data = []
    now_ts = int(datetime.now(tz=KST).timestamp() * 1000)
    
    for i in range(count):
        # 각 데이터는 10초 간격
        timestamp = now_ts + (i * 10000)
        
        data = {
            'restaurant_id': 'hanyang_plaza',
            'corner_id': 'western',
            'queue_count': 5 + i,  # 5명부터 시작해서 증가
            'est_wait_time_min': 3.5 + (i * 0.5),  # 3.5분부터 증가
            'timestamp': timestamp
        }
        test_data.append(data)
    
    return test_data


def main():
    logger.info("=" * 60)
    logger.info("DynamoDB 테스트 전송 시작")
    logger.info("=" * 60)
    
    # 1. DynamoDB 전송기 초기화
    try:
        sender = DynamoDBSender(config_path='config/aws_config.json')
        logger.info("✅ DynamoDB 전송기 초기화 성공")
    except Exception as e:
        logger.error(f"❌ DynamoDB 전송기 초기화 실패: {e}")
        logger.error("AWS 자격증명을 확인하세요:")
        logger.error("  export AWS_ACCESS_KEY_ID='your-key'")
        logger.error("  export AWS_SECRET_ACCESS_KEY='your-secret'")
        return 1
    
    # 2. 워커 스레드 시작
    sender.start()
    logger.info("✅ 백그라운드 워커 스레드 시작")
    
    try:
        # 3. 테스트 데이터 생성
        test_data_list = generate_test_data(count=5)
        
        # 4. 데이터 전송
        logger.info("\n📤 테스트 데이터 전송 중...")
        for i, data in enumerate(test_data_list, 1):
            sender.send(data)
            logger.info(f"  [{i}/5] 전송: queue={data['queue_count']}명, "
                       f"wait={data['est_wait_time_min']:.1f}분")
        
        # 5. 전송 완료 대기 (워커 스레드가 처리할 시간 주기)
        logger.info("\n⏳ 전송 완료 대기 중 (5초)...")
        time.sleep(5)
        
        # 6. 전송 통계 확인
        stats = sender.stats  # Property, not method!
        logger.info("\n📊 전송 통계:")
        logger.info(f"  ✅ 전송 성공: {stats['sent']}개")
        logger.info(f"  ❌ 전송 실패: {stats['errors']}개")
        logger.info(f"  ⏳ 대기 중: {stats['pending']}개")
        
        # 7. 결과 판정
        if stats['sent'] >= 5:
            logger.info("\n🎉 테스트 성공!")
            logger.info("DynamoDB에 데이터가 정상적으로 전송되었습니다.")
            logger.info("\n📋 DynamoDB에서 확인하는 방법:")
            logger.info("  1. AWS 콘솔 접속")
            logger.info("  2. DynamoDB → Tables → hyeat-waiting-data-dev")
            logger.info("  3. 'Explore table items' 클릭")
            logger.info("  4. 방금 전송한 데이터 확인")
            result = 0
        else:
            logger.warning("\n⚠️ 일부 데이터 전송 실패")
            logger.warning("전송 통계를 확인하세요.")
            result = 1
        
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 중단됨")
        result = 1
    except Exception as e:
        logger.error(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        result = 1
    finally:
        # 8. 정리
        logger.info("\n🧹 정리 중...")
        sender.stop()
        logger.info("✅ 전송기 중지 완료")
    
    logger.info("=" * 60)
    return result


if __name__ == '__main__':
    sys.exit(main())
