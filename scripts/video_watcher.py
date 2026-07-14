# -*- coding: utf-8 -*-
"""
video_watcher.py
=================
1단계: 영상 입력 -> watchdog 감시 -> 지정 프레임 간격마다 이미지로 추출.

지정 폴더(field_config.VIDEO_INPUT_DIR)에 영상 파일(.mp4 등)이 새로 생성되면
watchdog이 감지하여, field_config.FRAME_EXTRACT_INTERVAL 프레임마다 1장씩
이미지를 추출해 field_config.FRAME_OUTPUT_DIR에 저장합니다.
추출이 끝나면 바로 이어서(2~6단계) MissionPipeline을 실행해 8개 JSON까지 생성합니다
(--no-auto-run 옵션으로 추출만 하고 파이프라인 실행은 생략할 수 있음).

실전 운용 시나리오: 1분 준비시간에 --watch --send로 실행해두면, 3분 비행 동안
드론 영상이 여러 개 파일로 나뉘어 반복 전송되어도(대시보드로 진행도를 보다가
일찍 착륙시켜 끝내는 것도 가능) PC를 만지지 않아도 됩니다. 이때 영상 하나만의
결과가 아니라, 그 세션에서 지금까지 들어온 "모든" 영상의 프레임을 누적해서
매번 전체를 다시 계산 + 전송합니다 (한 영상 결과가 다음 영상 처리 때 덮어써져
유실되는 것을 방지). 파이프라인 자체가 매우 빨라서(프레임 몇 장 기준 0.1초대)
매번 전체 재계산해도 성능 문제는 없습니다.

사용법:
  1) 폴더를 계속 감시하며 새 영상이 들어올 때마다 누적 처리:
     python scripts/video_watcher.py --watch --send

  2) 폴더에 이미 있는 영상들을 전부 누적해서 한 번 처리하고 종료:
     python scripts/video_watcher.py --once

ultralytics와 마찬가지로, watchdog은 이 스크립트를 실제로 실행할 때만 필요합니다
(`pip install watchdog`).
"""
import os
import sys
import time
import argparse

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import field_config as fc
from pipeline import MissionPipeline
from img_io import imread_safe, imwrite_safe
from transmitter import summarize_failures


def _wait_until_stable(path: str, poll_sec: float = 0.5, stable_checks: int = 2):
    """영상 파일 쓰기가 끝났다고 판단될 때까지(크기 변화가 멈출 때까지) 대기"""
    last_size = -1
    stable_count = 0
    while stable_count < stable_checks:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size == last_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
        last_size = size
        time.sleep(poll_sec)


def extract_frames(video_path: str, output_dir: str, interval: int = None) -> list:
    """
    video_path의 영상에서 N프레임마다 1장씩 추출해 output_dir에 저장.
    반환: 저장된 프레임 파일 경로 리스트
    """
    interval = interval or fc.FRAME_EXTRACT_INTERVAL
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []
    frame_idx = 0
    saved_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:
                out_path = os.path.join(output_dir, f"{base_name}_frame_{saved_idx:04d}.png")
                imwrite_safe(out_path, frame)
                saved_paths.append(out_path)
                saved_idx += 1
            frame_idx += 1
    finally:
        cap.release()

    return saved_paths


class MissionState:
    """한 번의 임무 세션(--watch 또는 --once 실행 1회) 동안 유지되는 누적 상태.
    영상이 여러 개로 나뉘어 들어와도 프레임을 계속 누적하고, MissionPipeline 인스턴스도
    재사용해 YOLO 모델을 매번 다시 로드하지 않게 합니다."""

    def __init__(self, args):
        self.frames = []
        self.pipeline = MissionPipeline(
            mission_code=fc.MISSION_CODE, use_llm=not args.no_llm,
            output_dir=args.output,
            detector_backend=args.detector_backend, facility_backend=args.facility_backend,
            object_weights=args.weights, facility_weights=args.facility_weights,
        )
        # start.json은 특정 영상의 프레임 추출 완료 시점이 아니라, video_watcher.py 세션이
        # 시작되는 지금(첫 영상이 들어오기 전) 딱 1번만 생성/전송합니다.
        start_result = self.pipeline.send_start(send_to_dashboard=args.send)
        for line in summarize_failures(start_result.get("transmit_result")):
            print(f"[video_watcher] 전송 실패: {line}")


def process_video(video_path: str, args, state: "MissionState"):
    print(f"[video_watcher] 새 영상 감지: {video_path}")
    _wait_until_stable(video_path, poll_sec=min(1.0, fc.VIDEO_STABLE_WAIT_SEC), stable_checks=2)

    frame_paths = extract_frames(video_path, fc.FRAME_OUTPUT_DIR, args.interval)
    print(f"[video_watcher] 프레임 {len(frame_paths)}장 추출 완료 -> {fc.FRAME_OUTPUT_DIR}")

    if args.no_auto_run or not frame_paths:
        return

    new_frames = [imread_safe(p) for p in frame_paths]
    new_frames = [f for f in new_frames if f is not None]
    if not new_frames:
        print("[video_watcher] 추출된 프레임을 읽지 못해 파이프라인을 건너뜁니다.")
        return

    # 이번 영상만이 아니라, 이 세션에서 지금까지 들어온 모든 영상의 프레임을 누적해서
    # 매번 전체를 다시 계산합니다 (한 영상 결과가 다음 영상 처리 때 덮어써져 유실되는 것 방지).
    state.frames.extend(new_frames)
    result = state.pipeline.run(state.frames, send_to_dashboard=args.send)
    print(f"[video_watcher] 누적 프레임 {len(state.frames)}장 기준 파이프라인 실행 완료. 검증: "
          f"{'통과' if result['validation']['ok'] else '실패'} / 소요시간(초): {result['timing'].get('total')}")
    if not result["validation"]["ok"]:
        print("  오류:", result["validation"]["errors"])
    for line in summarize_failures(result.get("transmit_result")):
        print(f"[video_watcher] 전송 실패: {line}")


class _VideoCreatedHandler:
    """watchdog import는 --watch 실행 시에만 필요하므로, 클래스 생성도 그때 지연시킴"""

    def __init__(self, args, state, FileSystemEventHandlerBase):
        self.args = args
        self.state = state

        class Handler(FileSystemEventHandlerBase):
            def _handle(handler_self, path):
                if not path.lower().endswith(fc.VIDEO_FILE_EXTENSIONS):
                    return
                try:
                    process_video(path, self.args, self.state)
                except Exception as e:
                    print(f"[video_watcher] 처리 실패({path}): {e}")

            def on_created(handler_self, event):
                # 영상이 바로 이 이름으로 생성되는 경우(단순 복사 등)
                if event.is_directory:
                    return
                handler_self._handle(event.src_path)

            def on_moved(handler_self, event):
                # filerecvsender처럼 임시 파일명으로 받은 뒤 최종 파일명으로 rename하는 경우.
                # Windows에서 rename은 on_created가 아니라 on_moved로 감지되므로 반드시 같이 처리해야 함.
                if event.is_directory:
                    return
                handler_self._handle(event.dest_path)

        self.handler_instance = Handler()


def run_once(args):
    """폴더에 이미 존재하는 영상 파일들을 전부 누적해서 한 번 처리하고 종료"""
    video_dir = args.video_dir
    if not os.path.isdir(video_dir):
        print(f"영상 입력 폴더가 없습니다: {video_dir}")
        return
    files = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(fc.VIDEO_FILE_EXTENSIONS))
    if not files:
        print(f"처리할 영상이 없습니다: {video_dir}")
        return
    state = MissionState(args)
    for f in files:
        process_video(os.path.join(video_dir, f), args, state)


def run_watch(args):
    """watchdog으로 폴더를 계속 감시하며 새 영상이 생길 때마다 누적 처리"""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("watchdog이 설치되어 있지 않습니다. `pip install watchdog` 후 다시 실행하세요.")
        sys.exit(1)

    os.makedirs(args.video_dir, exist_ok=True)
    state = MissionState(args)
    handler = _VideoCreatedHandler(args, state, FileSystemEventHandler).handler_instance

    observer = Observer()
    observer.schedule(handler, args.video_dir, recursive=False)
    observer.start()
    print(f"[video_watcher] 감시 시작: {args.video_dir} (Ctrl+C로 종료)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, default=fc.VIDEO_INPUT_DIR, help="감시할 영상 입력 폴더")
    parser.add_argument("--video-file", type=str, default=None,
                         help="video-dir 전체를 스캔하지 않고 지정한 영상 파일 하나만 처리")
    parser.add_argument("--output", type=str, default="output", help="결과 JSON 저장 폴더")
    parser.add_argument("--interval", type=int, default=None, help="N프레임마다 추출 (생략 시 field_config.FRAME_EXTRACT_INTERVAL)")
    parser.add_argument("--watch", action="store_true", help="폴더를 계속 감시(watchdog)")
    parser.add_argument("--once", action="store_true", help="폴더에 있는 영상을 한 번만 처리하고 종료")
    parser.add_argument("--no-auto-run", action="store_true", help="프레임 추출만 하고 파이프라인은 실행하지 않음")
    parser.add_argument("--no-llm", action="store_true", help="로컬 LLM 시도 없이 템플릿 보고서만 사용")
    parser.add_argument("--send", action="store_true", help="파이프라인 완료 후 대시보드 전송(6단계)까지 실행")
    parser.add_argument("--detector-backend", choices=["classical", "yolo"], default=None)
    parser.add_argument("--facility-backend", choices=["classical", "yolo"], default=None)
    parser.add_argument("--weights", type=str, default=None,
                         help="yolo 백엔드일 때 field_config.YOLO_OBJECT_WEIGHTS 대신 쓸 폭파구/불발탄 탐지 "
                              "가중치(.pt) 경로 (학습 중인 체크포인트를 바로 테스트할 때 등)")
    parser.add_argument("--facility-weights", type=str, default=None,
                         help="yolo 백엔드일 때 field_config.YOLO_FACILITY_WEIGHTS 대신 쓸 시설물 상태 분류 "
                              "가중치(.pt) 경로")
    args = parser.parse_args()

    if args.video_file:
        process_video(args.video_file, args, MissionState(args))
        return

    if not args.watch and not args.once:
        print("`--video-file <파일>`, `--watch`(계속 감시), `--once`(폴더에 있는 영상만 1회 처리) 중 하나를 지정하세요.")
        sys.exit(1)

    if args.once:
        run_once(args)
    else:
        run_watch(args)


if __name__ == "__main__":
    main()
