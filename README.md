# sound-event-extractor

動画ファイルからラベル指定した音響イベント(例:「犬の鳴き声」)を検出し、
開始・終了時刻を時系列の CSV として出力するデスクトップツールです。

Google の学習済み音響イベント分類モデル
[YAMNet](https://tfhub.dev/google/yamnet/1)(AudioSet 521 クラス)を使い、
0.48 秒刻みのフレームごとにクラススコアを計算 → しきい値処理 → 区間マージ、
という流れで検出します。すべて CPU 上でローカル実行されます。

## 必要環境

- macOS(Apple Silicon で動作確認)
- [mise](https://mise.jdx.dev/) … Python / uv / BetterLeaks / lefthook の導入に使用
- ネットワーク接続(初回のみ。YAMNet モデル約 20MB をダウンロードし
  `~/.cache/sound-event-extractor/tfhub` にキャッシュ)

ffmpeg はシステムに無くても、依存パッケージ `imageio-ffmpeg` 同梱の
バイナリを自動で使うため、追加インストール不要です。

## セットアップ

```sh
mise trust        # 初回のみ: このリポジトリの mise.toml を信頼
mise install      # Python 3.12 / uv / gitleaks を導入
mise run setup    # uv sync + lefthook の git フック有効化(シークレット検査)
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

GUI(PySide6 製)には検証用の機能があります:

- **再生位置の瞬間ビュー**: 動画の現在時刻に合わせて 2 つのパネルが常時更新
  - *周波数スペクトル*: 再生位置を中心とした約 0.26 秒窓の FFT(0〜8kHz、
    ピーク周波数の数値付き)。動画を読み込むと音声を先に抽出するため、
    解析を実行しなくても再生・シークに追従する
  - *瞬間スコア*: そのフレームのラベルスコア(しきい値マーク付きバー)と、
    その瞬間の上位 5 クラスの内訳(解析実行後に表示)

- **動画再生**: 音声付きで再生・一時停止・シーク・音量調節が可能。
  結果テーブルの行をダブルクリックすると、その区間の 0.5 秒前にジャンプ
- **スペクトログラム**: 0〜8kHz の周波数成分を時系列表示。再生位置に
  赤いカーソルが追従し、クリックでその時刻へシーク
- **スコアタイムライン**: 対象ラベルのフレームごとの検出スコアを
  しきい値ライン付きで表示。再生中は現在時刻のスコア値も数値表示
- **再生連動のその場解析**: 未解析の状態で再生を始めると自動で解析が走り、
  スペクトログラムは数秒で表示、スコア曲線は先頭から順に伸びていく。
  解析は再生より速く進むため、通常は再生中に全体が追いつく(未解析位置へ
  シークした場合は、追いつくまで「スコア -」表示。灰色の縦線が解析済みの
  先端位置)

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
| `--threshold` | `0.05` | 検出スコアのしきい値(0〜1)。下げると検出が増え、誤検出も増える |
| `--merge-gap` | `1.0` | この秒数以内に隣接する区間を 1 つにまとめる |
| `--min-duration` | `0.5` | この秒数未満の区間を捨てる |
| `--normalize` | (無効) | 音量を動的に正規化(dynaudnorm)し、遠く・小音量の音を増幅してから解析。GUI では「小さい音を増幅」チェックボックス |
| `--debug-scores [PATH]` | (無効) | フレームごとのスコア内訳 CSV を出力(パス省略時: `<動画名>_scores.csv`) |

### 精度チューニング(スコア内訳の見方)

取りこぼしや誤検出がある場合は `--debug-scores` を付けて実行し、
問題の時刻のフレームを確認します:

- 対象ラベルのスコアが 0.15〜0.25 に留まっている → `--threshold` を下げる
- 対象外クラス(例: Squeak, Animal)に点が付いている →
  `labels.py` のキーワードに追加するか、そのクラス名を直接ラベル指定する
- 「Silence」に高スコアが付いている(スコア 0.000)→ その瞬間の録音レベルが
  低すぎる。`--normalize`(GUI: 「小さい音を増幅」)を有効にする。
  暗騒音も増幅されるので、誤検出が増えたらしきい値を上げて調整する
- どのクラスにもほぼ点が付いていない → モデルの検出限界の可能性が高い

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

## シークレット流出防止(BetterLeaks + lefthook)

mise 経由で [BetterLeaks](https://github.com/betterleaks/betterleaks) と
[lefthook](https://lefthook.dev/) を導入し、`mise run setup` で
git フック(`lefthook.yml` の pre-commit)を有効化します。コミット時に
ステージ済み差分をスキャンし、API キー等の混入を検出するとコミットを
中断します。履歴全体の手動スキャンは:

```sh
mise run scan
```

誤検知は `.betterleaks.toml` の allowlist に追記して除外できます。

## Windows 対応と配布用ビルド

### Windows でソースから動かす

コードは全てクロスプラットフォーム(Python / PySide6 / TensorFlow CPU /
imageio-ffmpeg)なので、Windows でも動作します(macOS でのみ動作確認済み。
Windows は未検証)。mise の Windows 対応は発展途上のため、Windows では
[uv](https://docs.astral.sh/uv/) を直接使うのが簡単です:

```powershell
uv sync
uv run sound-event-extractor-gui
```

Windows の TensorFlow は CPU 版のみですが、本ツールはもともと CPU 推論の
ため問題ありません。

### スタンドアロンアプリのビルド(PyInstaller)

Python 環境なしで起動できる配布用アプリを生成します:

```sh
mise run build    # 生成物: dist/SoundEventExtractor(.app)
```

- **クロスビルドは不可**: Windows 用 .exe は Windows 上で、macOS 用 .app は
  macOS 上でビルドする必要がある
- GitHub にリポジトリを push すれば、GitHub Actions
  (`.github/workflows/build.yml`)が Windows / macOS 両方の成果物を自動
  ビルドする(`v*` タグの push、または手動実行)
- TensorFlow を同梱するため生成物は大きい(1GB 超)
- ビルド済みアプリも初回起動時の解析で YAMNet(約 20MB)をダウンロード
  するため、初回のみネットワーク接続が必要
- 動作確認: ビルド後に `dist/SoundEventExtractor/SoundEventExtractor
  --smoke-test` で同梱物(Qt / ffmpeg / TensorFlow)を自己検査できる

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
  cli.py         # コマンドラインインターフェース
  gui.py         # PySide6 デスクトップ GUI(再生・解析のメイン画面)
  gui_widgets.py # スペクトログラム/スコアタイムラインのカスタムウィジェット
  gui_instant.py # 再生位置の瞬間スペクトル/瞬間スコアのパネル
  spectrogram.py # 表示用 STFT スペクトログラム計算(numpy のみ)
```

## ライセンス

本プロジェクトのソースコードは [MIT License](LICENSE) です。
配布用アプリ(Release の zip)には Qt / PySide6(LGPL-3.0)、
ffmpeg(GPL ビルド)、TensorFlow(Apache-2.0)などのサードパーティ
コンポーネントが同梱されます。各コンポーネントのライセンスと入手元は
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください
(同ファイルは Release zip にも同梱されます)。

## 制約・既知の注意点

- 検出できるのは AudioSet の 521 クラスに関連する音のみ(任意の自由文
  ラベルを意味解釈するわけではない)
- しきい値は音源・録音状況に依存するため、まず既定値 0.05 で試し、
  過検出なら上げて調整する(0.05 は取りこぼし優先の設定なので、
  誤検出が多い環境では 0.2〜0.4 程度まで上げる)
- 長時間動画も処理可能(30 分動画で音声全体をチャンク分割して推論)
