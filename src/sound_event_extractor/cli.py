"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys

from .audio import SAMPLE_RATE, extract_waveform
from .detect import detect_segments, format_timestamp, write_csv, write_debug_scores
from .labels import match_classes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound-event-extractor",
        description="動画からラベル指定した音響イベントを検出し、区間を CSV 出力します",
    )
    parser.add_argument("video", nargs="?", help="入力動画/音声ファイル")
    parser.add_argument("--label", "-l", help="検出する音のラベル(例: 犬の鳴き声, dog)")
    parser.add_argument("--output", "-o", help="出力 CSV パス(既定: <動画名>_events.csv)")
    parser.add_argument("--threshold", type=float, default=0.3, help="スコアしきい値 0〜1(既定: 0.3)")
    parser.add_argument("--merge-gap", type=float, default=1.0, help="この秒数以内の区間を結合(既定: 1.0)")
    parser.add_argument("--min-duration", type=float, default=0.5, help="この秒数未満の区間を除外(既定: 0.5)")
    parser.add_argument("--list-labels", action="store_true", help="AudioSet クラス名の一覧を表示して終了")
    parser.add_argument(
        "--debug-scores",
        nargs="?",
        const="",
        metavar="PATH",
        help="フレームごとのスコア内訳 CSV を出力(パス省略時: <動画名>_scores.csv)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    print("YAMNet モデルを読み込み中(初回は約 20MB をダウンロード)...", flush=True)
    from .model import YamNet  # defer heavy tensorflow import

    model = YamNet()

    if args.list_labels:
        for name in model.class_names:
            print(name)
        return

    if not args.video or not args.label:
        print("エラー: 動画ファイルと --label を指定してください", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.video):
        print(f"エラー: ファイルが見つかりません: {args.video}", file=sys.stderr)
        sys.exit(2)

    indices = match_classes(model.class_names, args.label)
    if not indices:
        print(
            f"エラー: ラベル '{args.label}' に一致するクラスがありません。"
            "--list-labels で一覧を確認してください",
            file=sys.stderr,
        )
        sys.exit(2)
    matched = ", ".join(model.class_names[i] for i in indices)
    print(f"対象クラス ({len(indices)} 件): {matched}")

    print("音声を抽出中...", flush=True)
    waveform = extract_waveform(args.video)
    duration = waveform.size / SAMPLE_RATE
    print(f"音声長: {format_timestamp(duration)}")

    def show_progress(ratio: float) -> None:
        print(f"\r解析中... {ratio * 100:5.1f}%", end="", flush=True)

    scores = model.scores(waveform, progress=show_progress)
    print()

    segments = detect_segments(
        scores,
        indices,
        threshold=args.threshold,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    output = args.output or f"{os.path.splitext(args.video)[0]}_events.csv"
    write_csv(output, args.label, segments)

    if args.debug_scores is not None:
        debug_path = args.debug_scores or f"{os.path.splitext(args.video)[0]}_scores.csv"
        write_debug_scores(debug_path, model.class_names, scores, indices, args.label)
        print(f"スコア内訳 CSV: {debug_path}")

    print(f"検出区間: {len(segments)} 件 -> {output}")
    for seg in segments:
        print(
            f"  {format_timestamp(seg.start)} - {format_timestamp(seg.end)}"
            f"  (長さ {seg.duration:.2f}s, 最大スコア {seg.max_score:.3f})"
        )


if __name__ == "__main__":
    main()
