"""PySide6 desktop GUI: analysis, video playback, spectrogram, score timeline."""

from __future__ import annotations

import sys
import threading

import numpy as np
from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from .audio import SAMPLE_RATE, extract_waveform
from .detect import combined_frame_scores, detect_segments, format_timestamp, write_csv
from .gui_instant import InstantScoreView, InstantSpectrumView
from .gui_widgets import ScoreTimeline, SpectrogramView
from .labels import SUGGESTED_LABELS, match_classes
from .model import FRAME_HOP_SEC
from .spectrogram import compute_spectrogram

MEDIA_FILTER = "動画/音声 (*.mp4 *.mov *.mkv *.avi *.webm *.m4a *.mp3 *.wav);;すべて (*)"
TABLE_HEADERS = ["開始", "終了", "長さ(秒)", "最大スコア", "平均スコア"]
STREAM_FRAMES = 25  # ~12 s per inference chunk: quick first feedback during playback


class ExtractWorker(QObject):
    """Decodes the audio track right after a file is loaded (no model needed)."""

    done = Signal(object)  # (waveform, normalized: bool)
    failed = Signal(str)

    def __init__(self, path: str, normalize: bool) -> None:
        super().__init__()
        self._path = path
        self._normalize = normalize

    def run(self) -> None:
        try:
            waveform = extract_waveform(self._path, normalize=self._normalize)
            self.done.emit((waveform, self._normalize))
        except Exception as exc:
            self.failed.emit(f"音声抽出に失敗: {exc}")


class AnalysisWorker(QObject):
    """Runs extraction + inference on a plain thread; reports via signals."""

    status = Signal(str)
    progress = Signal(float)
    failed = Signal(str)
    waveform_ready = Signal(object)  # decoded waveform, right after extraction
    spectrogram_ready = Signal(object)  # (image, duration), right after extraction
    stream_started = Signal(object)  # {"threshold", "duration"}
    scores_partial = Signal(object)  # 1-D matched-score chunk on the 0.48 s grid
    finished = Signal(object)  # dict with segments/frame_scores/spec_image/...

    _model = None  # cached across runs

    def __init__(
        self,
        path: str,
        label: str,
        threshold: float,
        waveform: np.ndarray | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self._path = path
        self._label = label
        self._threshold = threshold
        self._waveform = waveform  # reuse the already-decoded audio when available
        self._normalize = normalize

    def run(self) -> None:
        try:
            if AnalysisWorker._model is None:
                self.status.emit("モデル読み込み中(初回はダウンロードあり)...")
                from .model import YamNet  # defer heavy tensorflow import

                AnalysisWorker._model = YamNet()
            model = AnalysisWorker._model
            indices = match_classes(model.class_names, self._label)
            if not indices:
                raise RuntimeError(f"ラベル '{self._label}' に一致するクラスがありません")
            waveform = self._waveform
            if waveform is None:
                self.status.emit("音声を抽出中...")
                waveform = extract_waveform(self._path, normalize=self._normalize)
            self.waveform_ready.emit((waveform, self._normalize))
            spec_image, duration = compute_spectrogram(waveform)
            self.spectrogram_ready.emit((spec_image, duration))
            self.stream_started.emit(
                {"threshold": self._threshold, "duration": duration, "label": self._label}
            )
            self.status.emit(f"解析中... (対象クラス {len(indices)} 件)")
            parts: list[np.ndarray] = []
            done_samples = 0
            chunk_samples = STREAM_FRAMES * FRAME_HOP_SEC * SAMPLE_RATE
            for chunk_scores in model.iter_scores(waveform, frames_per_chunk=STREAM_FRAMES):
                parts.append(chunk_scores)
                top_idx = np.argsort(chunk_scores, axis=1)[:, :-6:-1]  # top-5 per frame
                self.scores_partial.emit(
                    {
                        "combined": combined_frame_scores(chunk_scores, indices),
                        "top": [
                            [(model.class_names[j], float(row[j])) for j in idxs]
                            for row, idxs in zip(chunk_scores, top_idx)
                        ],
                    }
                )
                done_samples += chunk_samples
                self.progress.emit(min(done_samples / waveform.size, 1.0))
            scores = np.concatenate(parts, axis=0)
            self.finished.emit(
                {
                    "segments": detect_segments(scores, indices, threshold=self._threshold),
                    "frame_scores": combined_frame_scores(scores, indices),
                    "spec_image": spec_image,
                    "duration": duration,
                    "threshold": self._threshold,
                    "n_classes": len(indices),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sound Event Extractor")
        self.resize(940, 960)
        self.segments = []
        self.current_label = ""
        self._worker: AnalysisWorker | None = None
        self._analysis_running = False
        self._analysis_done = False
        self._waveform: np.ndarray | None = None
        self._extractor: ExtractWorker | None = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self._build_ui()
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_playback_state)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        row = QHBoxLayout()
        row.addWidget(QLabel("動画ファイル:"))
        self.path_edit = QLineEdit()
        row.addWidget(self.path_edit, stretch=1)
        browse_btn = QPushButton("参照...")
        browse_btn.clicked.connect(self._browse)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("ラベル:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.addItems(SUGGESTED_LABELS)
        row.addWidget(self.label_combo)
        row.addWidget(QLabel("しきい値:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.95)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.05)
        row.addWidget(self.threshold_spin)
        self.normalize_check = QCheckBox("小さい音を増幅")
        self.normalize_check.setToolTip(
            "音量を動的に正規化し、遠く・小音量の音を持ち上げてから解析します\n"
            "(暗騒音も増幅されるため誤検出が増える場合はしきい値を上げてください)"
        )
        self.normalize_check.toggled.connect(self._on_normalize_toggled)
        row.addWidget(self.normalize_check)
        self.run_btn = QPushButton("解析開始")
        self.run_btn.clicked.connect(self._start_analysis)
        row.addWidget(self.run_btn)
        self.save_btn = QPushButton("CSV 保存")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_csv)
        row.addWidget(self.save_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMaximumWidth(140)
        row.addWidget(self.progress_bar)
        self.status_label = QLabel("動画とラベルを選び、再生または「解析開始」を押してください")
        row.addWidget(self.status_label, stretch=1)
        layout.addLayout(row)

        # --- playback area ---
        self.video = QVideoWidget()
        self.video.setMinimumHeight(240)
        layout.addWidget(self.video, stretch=1)

        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ 再生")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.sliderMoved.connect(self.player.setPosition)
        row.addWidget(self.position_slider, stretch=1)
        self.time_label = QLabel("0:00:00.000 / 0:00:00.000")
        row.addWidget(self.time_label)
        self.score_label = QLabel("スコア -")
        self.score_label.setMinimumWidth(90)
        row.addWidget(self.score_label)
        row.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        row.addWidget(self.volume_slider)
        self.volume_label = QLabel("100%")
        self.volume_label.setMinimumWidth(40)
        row.addWidget(self.volume_label)
        layout.addLayout(row)

        # instantaneous view of the moment being played
        row = QHBoxLayout()
        self.instant_spectrum = InstantSpectrumView()
        row.addWidget(self.instant_spectrum, stretch=3)
        self.instant_score = InstantScoreView()
        row.addWidget(self.instant_score, stretch=2)
        layout.addLayout(row)

        self.spectrogram = SpectrogramView()
        self.spectrogram.seekRequested.connect(self._seek)
        layout.addWidget(self.spectrogram)
        self.timeline = ScoreTimeline()
        self.timeline.seekRequested.connect(self._seek)
        layout.addWidget(self.timeline)

        self.table = QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_row_activated)
        self.table.setMinimumHeight(140)
        layout.addWidget(self.table)

        self.setCentralWidget(central)

    # --- playback ---

    def _load_media(self, path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play_btn.setEnabled(True)
        # decode the audio up front so the instant spectrum works without analysis
        self._start_extraction(path)

    def _start_extraction(self, path: str) -> None:
        self._waveform = None
        self.instant_spectrum.clear()
        extractor = ExtractWorker(path, self.normalize_check.isChecked())
        extractor.done.connect(self._on_waveform_ready)
        extractor.failed.connect(self.status_label.setText)
        self._extractor = extractor  # keep a reference while the thread runs
        threading.Thread(target=extractor.run, daemon=True).start()

    def _on_normalize_toggled(self, _checked: bool) -> None:
        path = self.path_edit.text().strip()
        if not path:
            return
        self._analysis_done = False
        self._start_extraction(path)
        self.status_label.setText(
            "音量増幅の設定を変更しました。再生または「解析開始」で再解析されます"
        )

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return
        if not self._analysis_running and not self._analysis_done:
            self._start_analysis(auto=True)  # live-fill the strips during playback
        self.player.play()

    def _on_volume_changed(self, value: int) -> None:
        # quadratic curve: finer control at low volumes, closer to perceived loudness
        self.audio_output.setVolume((value / 100) ** 2)
        self.volume_label.setText(f"{value}%")

    def _on_playback_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setText("⏸ 一時停止" if playing else "▶ 再生")

    def _on_duration(self, duration_ms: int) -> None:
        self.position_slider.setRange(0, duration_ms)
        self.spectrogram.set_duration(duration_ms / 1000)
        self.timeline.set_duration(duration_ms / 1000)

    def _on_position(self, position_ms: int) -> None:
        if not self.position_slider.isSliderDown():
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(position_ms)
            self.position_slider.blockSignals(False)
        seconds = position_ms / 1000
        self.spectrogram.set_position(seconds)
        self.timeline.set_position(seconds)
        self.instant_spectrum.set_position(seconds)
        self.instant_score.set_position(seconds)
        total = max(self.player.duration(), 0) / 1000
        self.time_label.setText(f"{format_timestamp(seconds)} / {format_timestamp(total)}")
        score = self.timeline.score_at(seconds)
        self.score_label.setText("スコア -" if score is None else f"スコア {score:.3f}")

    def _seek(self, seconds: float) -> None:
        self.player.setPosition(max(0, int(seconds * 1000)))

    def _on_row_activated(self, row: int, _column: int) -> None:
        if 0 <= row < len(self.segments):
            self._seek(max(self.segments[row].start - 0.5, 0.0))

    # --- analysis ---

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "動画ファイルを選択", "", MEDIA_FILTER)
        if path:
            self.path_edit.setText(path)
            self._load_media(path)
            self._analysis_done = False
            self.segments = []
            self.table.setRowCount(0)
            self.save_btn.setEnabled(False)
            self.spectrogram.clear()
            self.timeline.clear()
            self.instant_score.clear()
            self.status_label.setText("再生すると自動で解析が始まります(「解析開始」でも可)")

    def _start_analysis(self, auto: bool = False) -> None:
        if self._analysis_running:
            return
        path = self.path_edit.text().strip()
        label = self.label_combo.currentText().strip()
        if not path or not label:
            if not auto:
                QMessageBox.warning(self, "入力不足", "動画ファイルとラベルを指定してください")
            return
        if self.player.source().isEmpty():
            self._load_media(path)
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)
        self.timeline.clear()
        self.instant_score.clear()
        self.current_label = label
        self._analysis_running = True
        self._analysis_done = False
        worker = AnalysisWorker(
            path,
            label,
            self.threshold_spin.value(),
            waveform=self._waveform,
            normalize=self.normalize_check.isChecked(),
        )
        worker.status.connect(self.status_label.setText)
        worker.progress.connect(lambda ratio: self.progress_bar.setValue(int(ratio * 100)))
        worker.waveform_ready.connect(self._on_waveform_ready)
        worker.spectrogram_ready.connect(self._on_spectrogram_ready)
        worker.stream_started.connect(self._on_stream_started)
        worker.scores_partial.connect(self._on_scores_partial)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker  # keep a reference while the thread runs
        threading.Thread(target=worker.run, daemon=True).start()

    def _on_failed(self, message: str) -> None:
        self._analysis_running = False
        self.run_btn.setEnabled(True)
        self.status_label.setText("エラーが発生しました")
        QMessageBox.critical(self, "エラー", message)

    def _on_waveform_ready(self, payload) -> None:
        waveform, normalized = payload
        if normalized != self.normalize_check.isChecked():
            return  # stale extraction from a previous setting
        self._waveform = waveform
        self.instant_spectrum.set_waveform(waveform)

    def _on_spectrogram_ready(self, payload) -> None:
        image, duration = payload
        self.spectrogram.set_data(image, duration)

    def _on_stream_started(self, info: dict) -> None:
        self.timeline.begin_stream(FRAME_HOP_SEC, info["threshold"], info["duration"])
        self.instant_score.begin_stream(FRAME_HOP_SEC, info["label"], info["threshold"])

    def _on_scores_partial(self, payload: dict) -> None:
        self.timeline.append_scores(payload["combined"])
        self.instant_score.append(payload["combined"], payload["top"])

    def _on_finished(self, result: dict) -> None:
        self._analysis_running = False
        self._analysis_done = True
        self.segments = result["segments"]
        self.spectrogram.set_data(result["spec_image"], result["duration"])
        self.timeline.set_data(
            result["frame_scores"], FRAME_HOP_SEC, result["threshold"], result["duration"]
        )
        self.table.setRowCount(len(self.segments))
        for i, seg in enumerate(self.segments):
            values = [
                format_timestamp(seg.start),
                format_timestamp(seg.end),
                f"{seg.duration:.2f}",
                f"{seg.max_score:.3f}",
                f"{seg.mean_score:.3f}",
            ]
            for col, value in enumerate(values):
                self.table.setItem(i, col, QTableWidgetItem(value))
        self.progress_bar.setValue(100)
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(bool(self.segments))
        self.status_label.setText(
            f"検出区間: {len(self.segments)} 件(対象クラス {result['n_classes']} 件)"
            " — 行をダブルクリックでその位置へジャンプ"
        )

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "CSV を保存", "events.csv", "CSV (*.csv)")
        if path:
            write_csv(path, self.current_label, self.segments)
            self.status_label.setText(f"保存しました: {path}")


def _run_smoke_test(app: QApplication) -> int:
    """Verify a frozen bundle: Qt is up, ffmpeg and TensorFlow are included."""
    import os

    from .audio import find_ffmpeg

    try:
        assert os.path.exists(find_ffmpeg()), "bundled ffmpeg missing"
        import tensorflow  # noqa: F401  # heavy import: proves the bundle is complete
        import tensorflow_hub  # noqa: F401
    except Exception as exc:
        print(f"SMOKE NG: {exc}")
        return 1
    QTimer.singleShot(500, app.quit)
    app.exec()
    print("SMOKE OK")
    return 0


def main() -> None:
    smoke = "--smoke-test" in sys.argv
    app = QApplication([a for a in sys.argv if a != "--smoke-test"])
    window = MainWindow()
    window.show()
    if smoke:
        raise SystemExit(_run_smoke_test(app))
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
