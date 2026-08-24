"""Tkinter desktop GUI."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .audio import extract_waveform
from .detect import Segment, detect_segments, format_timestamp, write_csv
from .labels import SUGGESTED_LABELS, match_classes

MEDIA_FILETYPES = [
    ("動画/音声", "*.mp4 *.mov *.mkv *.avi *.webm *.m4a *.mp3 *.wav"),
    ("すべてのファイル", "*.*"),
]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Sound Event Extractor")
        root.geometry("760x500")
        self.model = None  # loaded lazily in the worker thread
        self.queue: queue.Queue = queue.Queue()
        self.segments: list[Segment] = []
        self.current_label = ""
        self._build_widgets()
        root.after(100, self._poll_queue)

    def _build_widgets(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="動画ファイル:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(row, text="参照...", command=self._browse).pack(side=tk.LEFT)

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="ラベル:").pack(side=tk.LEFT)
        self.label_var = tk.StringVar(value=SUGGESTED_LABELS[0])
        ttk.Combobox(
            row, textvariable=self.label_var, values=SUGGESTED_LABELS, width=18
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="しきい値:").pack(side=tk.LEFT, padx=(12, 0))
        self.threshold_var = tk.DoubleVar(value=0.3)
        ttk.Scale(
            row,
            from_=0.05,
            to=0.95,
            variable=self.threshold_var,
            command=self._on_threshold,
            length=160,
        ).pack(side=tk.LEFT, padx=4)
        self.threshold_label = ttk.Label(row, text="0.30", width=5)
        self.threshold_label.pack(side=tk.LEFT)

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=4)
        self.run_btn = ttk.Button(row, text="解析開始", command=self._start)
        self.run_btn.pack(side=tk.LEFT)
        self.save_btn = ttk.Button(
            row, text="CSV 保存", command=self._save_csv, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=6)
        self.status_var = tk.StringVar(value="動画とラベルを選んで「解析開始」を押してください")
        ttk.Label(row, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)

        self.progress = ttk.Progressbar(frm, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=4)

        columns = ("start", "end", "duration", "max", "mean")
        headings = ["開始", "終了", "長さ(秒)", "最大スコア", "平均スコア"]
        self.tree = ttk.Treeview(frm, columns=columns, show="headings")
        for key, text in zip(columns, headings):
            self.tree.heading(key, text=text)
            self.tree.column(key, width=130, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)

    def _on_threshold(self, _value: str) -> None:
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="動画ファイルを選択", filetypes=MEDIA_FILETYPES
        )
        if path:
            self.path_var.set(path)

    def _start(self) -> None:
        path = self.path_var.get().strip()
        label = self.label_var.get().strip()
        if not path or not label:
            messagebox.showwarning("入力不足", "動画ファイルとラベルを指定してください")
            return
        self.run_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0.0
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.current_label = label
        threading.Thread(
            target=self._worker,
            args=(path, label, self.threshold_var.get()),
            daemon=True,
        ).start()

    # --- worker thread ---

    def _worker(self, path: str, label: str, threshold: float) -> None:
        try:
            if self.model is None:
                self.queue.put(("status", "モデル読み込み中(初回はダウンロードあり)..."))
                from .model import YamNet  # defer heavy tensorflow import

                self.model = YamNet()
            indices = match_classes(self.model.class_names, label)
            if not indices:
                raise RuntimeError(f"ラベル '{label}' に一致するクラスがありません")
            self.queue.put(("status", "音声を抽出中..."))
            waveform = extract_waveform(path)
            self.queue.put(("status", f"解析中... (対象クラス {len(indices)} 件)"))
            scores = self.model.scores(
                waveform, progress=lambda r: self.queue.put(("progress", r))
            )
            segments = detect_segments(scores, indices, threshold=threshold)
            self.queue.put(("done", segments))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    # --- UI thread ---

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "done":
                    self._show_results(payload)
                elif kind == "error":
                    self.run_btn.config(state=tk.NORMAL)
                    self.status_var.set("エラーが発生しました")
                    messagebox.showerror("エラー", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_results(self, segments: list[Segment]) -> None:
        self.segments = segments
        for seg in segments:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    format_timestamp(seg.start),
                    format_timestamp(seg.end),
                    f"{seg.duration:.2f}",
                    f"{seg.max_score:.3f}",
                    f"{seg.mean_score:.3f}",
                ),
            )
        self.progress["value"] = 1.0
        self.run_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL if segments else tk.DISABLED)
        self.status_var.set(f"検出区間: {len(segments)} 件")

    def _save_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="events.csv",
        )
        if not path:
            return
        write_csv(path, self.current_label, self.segments)
        self.status_var.set(f"保存しました: {path}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
