#!/usr/bin/env python3
"""영상 속도 조절 도구

영상을 느리게 또는 빠르게 만듭니다.

사용법:
    # 0.3배속으로 느리게
    python slowmo_video.py input.mp4 output.mp4 --speed 0.3
    
    # 2배속으로 빠르게
    python slowmo_video.py input.mp4 output.mp4 --speed 2.0
"""

import sys
import argparse
import subprocess
import os


def check_ffmpeg():
    """ffmpeg 설치 확인"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def slow_video(input_path, output_path, speed=0.3):
    """ffmpeg로 영상 속도 조절
    
    Args:
        input_path: 입력 영상 경로
        output_path: 출력 영상 경로
        speed: 재생 속도 (0.3 = 0.3배속, 2.0 = 2배속)
    """
    if not os.path.exists(input_path):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        return False
    
    # setpts 필터 값 계산
    # speed 0.3 → setpts=1/0.3=3.33 (느려짐)
    # speed 2.0 → setpts=1/2.0=0.5 (빨라짐)
    pts_value = 1.0 / speed
    
    # ffmpeg 명령어
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-filter:v', f'setpts={pts_value}*PTS',  # 비디오 속도 조절
        '-an',  # 오디오 제거 (속도 변경 시 오디오 싱크 문제 방지)
        '-y',  # 덮어쓰기
        output_path
    ]
    
    print(f"🎬 영상 속도 조절 중: {speed}x")
    print(f"   입력: {input_path}")
    print(f"   출력: {output_path}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 완료: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 변환 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='영상 속도 조절 도구')
    parser.add_argument('input', help='입력 영상 파일')
    parser.add_argument('output', help='출력 영상 파일')
    parser.add_argument('--speed', type=float, default=0.3,
                       help='재생 속도 (0.3 = 0.3배속, 기본: 0.3)')
    args = parser.parse_args()
    
    # ffmpeg 확인
    if not check_ffmpeg():
        print("❌ ffmpeg가 설치되어 있지 않습니다")
        print("   설치: sudo apt install ffmpeg")
        return 1
    
    # 속도 조절
    if slow_video(args.input, args.output, args.speed):
        print()
        print("📋 다음 단계:")
        print(f"   python test_youtube_pipeline.py \"{args.output}\" \\")
        print(f"       --start-roi \"대기줄\" \\")
        print(f"       --roi-json '{{\"name\":\"대기줄\",\"points\":[[137,109],[443,63],[564,182],[247,325]]}}'")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
