# -*- coding: utf-8 -*-
"""
train_yolo.py
=============
대회 테스트 기간 중 촬영한 이미지로 YOLO 모델을 학습시키는 진입점입니다.
학습이 끝나면 나온 가중치(.pt) 경로를 config/field_config.py의
YOLO_OBJECT_WEIGHTS / YOLO_FACILITY_WEIGHTS에 넣고 DETECTOR_BACKEND(또는
FACILITY_BACKEND)를 "yolo"로 바꾸면 src/pipeline.py 수정 없이 바로 전환됩니다.

사용법:
  1) 폭파구/불발탄 통합 탐지 모델 (object detection, 클래스는
     field_config.YOLO_OBJECT_CLASS_MAP 순서와 반드시 일치시킬 것: big/medium/small + missile/dumb/cluster)
     python scripts/train_yolo.py --task detect --data data/object.yaml \
         --model yolo11n.pt --epochs 100 --out models/yolo11n_object

  1-b) zone 타일 crop 학습 (권장): 추론(src/tiling.crop_zone_tiles)과 동일하게
     탑뷰 이미지를 RW/TW zone 구간별로 잘라 학습 데이터를 만든 뒤 학습합니다.
     추론 시 모델이 보는 입력(zone 타일)과 학습 입력을 일치시켜 정확도를 높입니다.
     python scripts/train_yolo.py --task detect --data data/object.yaml --tile \
         --model yolo11n.pt --epochs 100 --out models/yolo11n_object
     (원본 --data는 '이미 탑뷰로 정렬된' 이미지 + YOLO txt 라벨이어야 함. crop된
      새 데이터셋은 --tile-out 위치에 생성되고, 그 data.yaml로 학습됩니다.)

  2) 시설물 상태 분류 모델 (classification: normal/destroy/fire 폴더로 이미지 정리)
     python scripts/train_yolo.py --task classify --data data/facility_cls \
         --model yolo11n-cls.pt --epochs 50 --out models/yolo11n_facility_cls

ultralytics는 이 스크립트를 실제로 실행할 때만 필요합니다
(requirements.txt의 주석 처리된 줄을 해제하고 `pip install ultralytics`).

data.yaml/폴더 구성 방법은 ultralytics 공식 문서를 따르면 됩니다:
  - detect: images/, labels/ (YOLO txt 포맷) + data.yaml
  - classify: <root>/train/<클래스명>/*.jpg, <root>/val/<클래스명>/*.jpg
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import field_config as fc


# =====================================================================
# zone 타일 crop 전처리 (추론 src/tiling.crop_zone_tiles 와 동일한 구간 분할)
#   - 입력: '이미 탑뷰로 정렬된' 이미지 + YOLO 라벨 (data.yaml)
#   - 처리: 경기장 전체를 덮는 탑뷰라고 가정하고 SEGMENTS 를 비율로 환산해 zone 별로 crop
#           (해상도에 무관 - 픽셀 슬라이싱 대신 정규화 좌표로 계산). 라벨 bbox 도 각 타일
#           로컬 정규화 좌표로 재매핑하고, 타일 경계 밖으로 나간 부분은 잘라냄.
#   - 출력: 새 YOLO detect 데이터셋(images/labels/train,val + data.yaml)
# =====================================================================
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _zone_fractions(zone_order):
    """zone_name -> (fx0,fy0,fx1,fy1) : 경기장 전체를 1로 본 정규화 구간 좌표"""
    fracs = {}
    for z in zone_order:
        b = fc.SEGMENTS[z]
        fracs[z] = (
            b["x_min"] / fc.FIELD_WIDTH_CM,
            b["y_min"] / fc.FIELD_HEIGHT_CM,
            b["x_max"] / fc.FIELD_WIDTH_CM,
            b["y_max"] / fc.FIELD_HEIGHT_CM,
        )
    return fracs


def _read_yolo_labels(label_path):
    """YOLO txt 라벨 읽기 -> [(cls,cx,cy,w,h), ...] (전부 정규화 좌표).
    두 포맷을 모두 지원함:
      - bbox: cls cx cy w h (5컬럼)
      - segmentation polygon: cls x1 y1 x2 y2 ... xn yn (7컬럼 이상, 홀수) - Roboflow에서
        폴리곤으로 라벨링해 내보낸 경우 이 포맷으로 옴. parts[1:5]만 잘라 읽으면 폴리곤의
        앞쪽 두 꼭짓점을 cx,cy,w,h로 잘못 해석하게 되므로(인접 꼭짓점끼리 좌표가 비슷해
        w≈cx처럼 보이는 깨진 값이 나옴), 점들의 min/max로 axis-aligned bbox를 만들어 사용함.
    """
    labels = []
    if not os.path.exists(label_path):
        return labels  # 라벨 없는 배경 이미지 -> 빈 라벨(정탐 억제용)로 취급
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            if len(parts) == 5:
                cx, cy, w, h = coords
            else:
                xs, ys = coords[0::2], coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                w, h = x_max - x_min, y_max - y_min
            labels.append((cls, cx, cy, w, h))
    return labels


def _remap_labels_to_zone(labels, frac, min_visibility):
    """
    전체 이미지 정규화 라벨 -> 특정 zone 타일 로컬 정규화 라벨.
    타일과 겹치는 부분만 남기고, (겹친 면적/원본 면적) < min_visibility 이면 버림.
    """
    fx0, fy0, fx1, fy1 = frac
    zw, zh = fx1 - fx0, fy1 - fy0
    if zw <= 0 or zh <= 0:
        return []
    out = []
    for cls, cx, cy, w, h in labels:
        bx0, by0, bx1, by1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        ix0, iy0 = max(bx0, fx0), max(by0, fy0)
        ix1, iy1 = min(bx1, fx1), min(by1, fy1)
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            continue  # 이 타일과 겹치지 않음
        orig_area = (bx1 - bx0) * (by1 - by0)
        visibility = (iw * ih) / orig_area if orig_area > 0 else 0.0
        if visibility < min_visibility:
            continue  # 살짝만 걸친 잘린 객체 -> 학습 노이즈이므로 제외
        ncx = ((ix0 + ix1) / 2 - fx0) / zw
        ncy = ((iy0 + iy1) / 2 - fy0) / zh
        nw, nh = iw / zw, ih / zh
        out.append((cls, ncx, ncy, nw, nh))
    return out


def _find_label_path(image_path):
    """이미지 경로 -> 대응하는 라벨 txt 경로 (…/images/… -> …/labels/…, 확장자 .txt)"""
    base = os.path.splitext(image_path)[0]
    # 경로 중 마지막 'images' 세그먼트만 'labels'로 치환 (ultralytics 관례)
    parts = base.replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return os.path.normpath("/".join(parts) + ".txt")


def _list_split_images(data_dir, root, split_value):
    """data.yaml 의 train/val 값(이미지 폴더 경로)을 실제 이미지 파일 리스트로 해석"""
    if os.path.isabs(split_value):
        images_dir = split_value
    elif root:
        images_dir = os.path.join(root, split_value)
    else:
        images_dir = os.path.join(data_dir, split_value)
    images_dir = os.path.normpath(images_dir)
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, "*" + ext)))
        files.extend(glob.glob(os.path.join(images_dir, "*" + ext.upper())))
    return sorted(set(files))


def prepare_tiled_dataset(data_yaml, out_dir, zone_order, min_visibility):
    """
    탑뷰 detect 데이터셋을 zone 타일 단위로 잘라 새 데이터셋을 생성.
    반환: 새로 만든 data.yaml 경로.
    """
    import cv2
    import yaml

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = os.path.dirname(os.path.abspath(data_yaml))
    root = cfg.get("path")
    if root and not os.path.isabs(root):
        root = os.path.normpath(os.path.join(data_dir, root))
    names = cfg.get("names")

    fracs = _zone_fractions(zone_order)
    out_dir = os.path.abspath(out_dir)

    splits = [s for s in ("train", "val", "test") if cfg.get(s)]
    if not splits:
        raise RuntimeError(f"{data_yaml} 에 train/val 경로가 없습니다.")

    counts = {}
    for split in splits:
        img_out = os.path.join(out_dir, "images", split)
        lbl_out = os.path.join(out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        images = _list_split_images(data_dir, root, cfg[split])
        n_tiles = 0
        for img_path in images:
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [경고] 이미지를 읽지 못함, 건너뜀: {img_path}")
                continue
            h, w = image.shape[:2]
            labels = _read_yolo_labels(_find_label_path(img_path))
            stem = os.path.splitext(os.path.basename(img_path))[0]

            for zone_name, frac in fracs.items():
                fx0, fy0, fx1, fy1 = frac
                x0, y0 = int(round(fx0 * w)), int(round(fy0 * h))
                x1, y1 = int(round(fx1 * w)), int(round(fy1 * h))
                x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
                y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
                if x1 <= x0 or y1 <= y0:
                    continue
                tile = image[y0:y1, x0:x1]
                tile_labels = _remap_labels_to_zone(labels, frac, min_visibility)

                safe_zone = zone_name.replace("-", "")
                out_name = f"{stem}__{safe_zone}"
                cv2.imwrite(os.path.join(img_out, out_name + ".png"), tile)
                with open(os.path.join(lbl_out, out_name + ".txt"), "w", encoding="utf-8") as f:
                    for cls, cx, cy, bw, bh in tile_labels:
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                n_tiles += 1
        counts[split] = (len(images), n_tiles)
        print(f"  [{split}] 원본 {len(images)}장 -> zone 타일 {n_tiles}장")

    out_yaml = os.path.join(out_dir, "data.yaml")
    new_cfg = {"path": out_dir, "names": names}
    for split in splits:
        new_cfg[split] = os.path.join("images", split).replace("\\", "/")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    total = sum(t for _, t in counts.values())
    print(f"zone 타일 데이터셋 생성 완료: {out_yaml} (타일 총 {total}장, zone {len(zone_order)}구간)")
    return out_yaml


def _print_classify_class_order(data_root):
    """
    분류 학습 폴더(train/<클래스명>/)를 훑어, ultralytics가 부여할 클래스 인덱스를 미리 보여준다.
    ultralytics(=torchvision ImageFolder)는 폴더명을 '알파벳순'으로 정렬해 인덱스를 매기므로,
    이 순서가 config의 YOLO_FACILITY_CLASS_MAP과 다르면 경고를 띄운다.
    (추론은 model.names를 우선 사용하므로 실제로는 폴더명만 맞으면 되지만, 폴백 맵/혼동 방지용 안내)
    """
    train_dir = os.path.join(data_root, "train")
    scan_dir = train_dir if os.path.isdir(train_dir) else data_root
    classes = sorted(
        d for d in os.listdir(scan_dir)
        if os.path.isdir(os.path.join(scan_dir, d))
    )
    if not classes:
        print(f"  [경고] {scan_dir} 에서 클래스 폴더를 찾지 못했습니다.")
        return
    print("[train_yolo] 분류 클래스 인덱스(알파벳순 자동 부여):")
    for idx, name in enumerate(classes):
        print(f"    {idx}: {name}")
    expected = set(fc.FACILITY_STATUS_OPTIONS) - {"unconfirmed"}
    unknown = set(classes) - set(fc.FACILITY_STATUS_OPTIONS)
    if unknown:
        print(f"  [경고] 상태 코드가 아닌 폴더명이 있습니다: {sorted(unknown)} "
              f"(허용: {sorted(expected)}). 폴더명을 normal/destroy/fire로 맞추세요.")
    cfg_order = [fc.YOLO_FACILITY_CLASS_MAP.get(i) for i in range(len(classes))]
    if cfg_order != classes:
        print(f"  [참고] config YOLO_FACILITY_CLASS_MAP 순서({cfg_order})와 폴더 순서({classes})가 다릅니다. "
              f"추론은 model.names를 우선 쓰므로 폴더명만 맞으면 정상 동작하지만, 폴백 맵을 맞추려면 "
              f"config를 이 순서로 갱신하세요.")


# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detect", "classify"], required=True,
                         help="detect=폭파구/불발탄 통합 탐지, classify=시설물 상태 분류")
    parser.add_argument("--data", required=True,
                         help="detect: data.yaml 경로 / classify: train·val 폴더가 있는 루트 경로")
    parser.add_argument("--model", default=None,
                         help="시작 가중치 (기본: detect=yolo11n.pt, classify=yolo11n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=None,
                         help="입력 이미지 크기 (기본: detect=640, classify=224)")
    parser.add_argument("--out", default=None,
                         help="학습된 가중치를 복사해둘 경로 (확장자 없이, .pt 자동 추가)")
    # --- zone 타일 crop 전처리 옵션 (detect 전용, 입력은 탑뷰 정렬 이미지 가정) ---
    parser.add_argument("--tile", action="store_true",
                         help="학습 전에 탑뷰 이미지를 RW/TW zone 구간별로 crop (추론 crop_zone_tiles와 동일). detect 전용")
    parser.add_argument("--tile-out", default=None,
                         help="crop된 zone 타일 데이터셋 저장 경로 (기본: <data.yaml 폴더>/_tiled)")
    parser.add_argument("--tile-min-visibility", type=float, default=0.3,
                         help="타일 경계에 걸린 객체를 남기는 최소 노출 비율(0~1). 이보다 적게 보이면 라벨 제외")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics가 설치되어 있지 않습니다. `pip install ultralytics` 후 다시 실행하세요.")
        sys.exit(1)

    data_path = args.data
    if args.tile:
        if args.task != "detect":
            print("--tile 은 detect 태스크에서만 사용할 수 있습니다(시설물 분류는 폴더=라벨 구조).")
            sys.exit(1)
        tile_out = args.tile_out or os.path.join(os.path.dirname(os.path.abspath(args.data)), "_tiled")
        print(f"[train_yolo] zone 타일 crop 전처리: {args.data} -> {tile_out}")
        data_path = prepare_tiled_dataset(
            args.data, tile_out, fc.ZONE_TILE_ORDER, args.tile_min_visibility,
        )

    if args.task == "classify":
        _print_classify_class_order(data_path)

    default_model = "yolo11n.pt" if args.task == "detect" else "yolo11n-cls.pt"
    model = YOLO(args.model or default_model)

    imgsz = args.imgsz if args.imgsz is not None else (640 if args.task == "detect" else 224)
    print(f"[train_yolo] task={args.task} data={data_path} model={args.model or default_model} "
          f"epochs={args.epochs} imgsz={imgsz}")
    results = model.train(data=data_path, epochs=args.epochs, imgsz=imgsz)

    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"학습 완료. 가중치: {best_weights}")

    if args.out:
        import shutil
        out_path = args.out if args.out.endswith(".pt") else args.out + ".pt"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        shutil.copyfile(best_weights, out_path)
        print(f"가중치를 {out_path} 로 복사했습니다.")
        cfg_key = "YOLO_OBJECT_WEIGHTS" if args.task == "detect" else "YOLO_FACILITY_WEIGHTS"
        print(f"config/field_config.py의 {cfg_key} = {out_path!r} 로 갱신하고, "
              f"{'DETECTOR_BACKEND' if args.task == 'detect' else 'FACILITY_BACKEND'} = \"yolo\" 로 바꾸세요.")


if __name__ == "__main__":
    main()
