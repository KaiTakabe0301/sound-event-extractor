# sound-event-extractor

動画ファイルからラベル指定した音響イベント(例:「犬の鳴き声」)を検出し、
開始・終了時刻を時系列の CSV として出力するデスクトップツールです。

Google の学習済み音響イベント分類モデル
[YAMNet](https://tfhub.dev/google/yamnet/1)(AudioSet 521 クラス)を使い、
0.48 秒刻みのフレームごとにクラススコアを計算 → しきい値処理 → 区間マージ、
という流れで検出します。すべて CPU 上でローカル実行されます。

## 必要環境

- macOS(Apple Silicon で動作確認)
- [mise](https://mise.jdx.dev/) … Python / uv / Gitleaks の導入に使用
- ネットワーク接続(初回のみ。YAMNet モデル約 20MB をダウンロードし
  `~/.cache/sound-event-extractor/tfhub` にキャッシュ)

ffmpeg はシステムに無くても、依存パッケージ `imageio-ffmpeg` 同梱の
バイナリを自動で使うため、追加インストール不要です。

## セットアップ

```sh
mise trust        # 初回のみ: このリポジトリの mise.toml を信頼
mise install      # Python 3.12 / uv / gitleaks を導入
mise run setup    # uv sync + gitleaks pre-commit フックの有効化
```

## 使い方

### GUI(デスクトップアプリ)

```sh
mise run gui
```

1. 「参照…」から動画ファイルを選択
2. ラベルをドロップダウンから選択(自由入力も可)
3. 必要ならしきい値を調整して「解析開始」
4. 結果テーブルを確認し「CSV 保存」で出力

### CLI

```sh
# 基本形(結果は <動画名>_events.csv に出力)
mise run cli -- input.mp4 --label 犬の鳴き声

# オプション指定
mise run cli -- input.mp4 --label dog \
  --output result.csv --threshold 0.3 --merge-gap 1.0 --min-duration 0.5

# 指定可能な AudioSet クラス名(英語 521 種)の一覧
mise run cli -- --list-labels
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--label` / `-l` | (必須) | 検出したい音のラベル。日本語エイリアス(下記)または AudioSet クラス名の部分一致 |
| `--output` / `-o` | `<動画名>_events.csv` | 出力 CSV パス |
| `--threshold` | `0.3` | 検出スコアのしきい値(0〜1)。下げると検出が増え、誤検出も増える |
| `--merge-gap` | `1.0` | この秒数以内に隣接する区間を 1 つにまとめる |
| `--min-duration` | `0.5` | この秒数未満の区間を捨てる |

### ラベルの指定方法

以下の日本語エイリアスは、関連する AudioSet クラス群に自動展開されます:

犬の鳴き声 / 猫の鳴き声 / 鳥の鳴き声 / サイレン / 赤ちゃんの泣き声 /
話し声 / 笑い声 / クラクション / 銃声 / 音楽

それ以外の文字列は AudioSet クラス名(英語)への部分一致として扱われます
(例: `--label thunder` → "Thunder", "Thunderstorm")。
解析開始時に、実際にマッチしたクラス一覧が表示されます。

## CSV フォーマット

```csv
label,start_seconds,end_seconds,duration_seconds,start_time,end_time,max_score,mean_score
犬の鳴き声,3.360,5.280,1.920,0:00:03.360,0:00:05.280,0.912,0.774
```

- `start_seconds` / `end_seconds` … 動画先頭からの秒数
- `start_time` / `end_time` … `時:分:秒.ミリ秒` 表記
- `max_score` / `mean_score` … 区間内フレームスコアの最大値・平均値(0〜1)

## シークレット流出防止(Gitleaks)

mise 経由で [Gitleaks](https://github.com/gitleaks/gitleaks) を導入し、
`mise run setup` で pre-commit フック(`.githooks/pre-commit`)を有効化します。
コミット時にステージ済み差分をスキャンし、API キー等の混入を検出すると
コミットを中断します。履歴全体の手動スキャンは:

```sh
mise run scan
```

誤検知は `.gitleaks.toml` の allowlist に追記して除外できます。

## 開発

```sh
mise run test     # ユニットテスト(pytest)
```

主な構成:

```
src/sound_event_extractor/
  audio.py    # ffmpeg による音声抽出(16kHz mono float32)
  model.py    # YAMNet のロードとチャンク推論
  labels.py   # 日本語ラベル → AudioSet クラスのマッピング
  detect.py   # スコア列 → 区間化・マージ・CSV 出力
  cli.py      # コマンドラインインターフェース
  gui.py      # Tkinter デスクトップ GUI
```

## 制約・既知の注意点

- 検出できるのは AudioSet の 521 クラスに関連する音のみ(任意の自由文
  ラベルを意味解釈するわけではない)
- しきい値は音源・録音状況に依存するため、まず既定値 0.3 で試し、
  過検出なら上げ、取りこぼしがあれば下げて調整する
- 長時間動画も処理可能(30 分動画で音声全体をチャンク分割して推論)
