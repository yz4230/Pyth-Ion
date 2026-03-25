# PythIon コードベース キャッチアップ資料

## 📋 プロジェクト概要

**PythIon** は、ナノポア電流データの解析・可視化を行うデスクトップGUIアプリケーションです。Wanunu Lab で内部利用されています。

### 主な目的

- ナノポア実験から取得した電流トレースデータの読み込み・表示
- イベント（電流変化）の自動検出
- サブイベント状態（CUSUM法による）の検出
- 解析結果の可視化とエクスポート

### 技術スタック

| 分野 | 使用技術 |
|------|----------|
| GUI フレームワーク | PyQt5 |
| プロット・可視化 | pyqtgraph, matplotlib |
| 数値計算 | NumPy, SciPy |
| 並列処理 | multiprocessing |
| その他 | tqdm, colorcet, pyyaml |

---

## 🏗️ アーキテクチャ概要

```
┌────────────────────────────────────────────────────────────────┐
│                      Pythion.py                                │
│               (ExtAppMainWindow - メインウィンドウ)              │
│                      ↓ 継承                                    │
│              BaseApp.py (BaseAppMainWindow)                   │
│                GUI基盤・プロット要素の初期化                      │
└────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐          ┌─────────┐
   │  IO.py  │          │Analysis.py│          │Painting.py│
   │ファイル入出力│        │ 解析処理  │          │ 描画処理  │
   └─────────┘          └──────────┘          └─────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────────────────────────────────────────────────┐
   │              DataTypes.py                           │
   │         (TraceData, FileData クラス)                │
   │              データ構造定義                          │
   └─────────────────────────────────────────────────────┘
```

---

## 📁 ファイル構成と責務

### コアモジュール

| ファイル | 責務 | 主要クラス/関数 |
|---------|------|----------------|
| `Pythion.py` | エントリーポイント、メインウィンドウ | `ExtAppMainWindow`, `start()` |
| `BaseApp.py` | GUI基盤、プロット要素初期化 | `BaseAppMainWindow` |
| `DataTypes.py` | データ構造定義 | `TraceData`, `FileData` |
| `Analysis.py` | イベント検出・解析 | `Config`, `AnalysisResults`, `computeAnalysis()` |
| `IO.py` | ファイル読み込み・保存 | `loadFile()`, `saveAnalysis()` |
| `Painting.py` | プロット描画処理 | `paintCurrentTrace()`, `plotAnalysis()` |
| `Selections.py` | 選択領域管理 | `autoFindCutLRs()`, `measureSelections()` |
| `Edits.py` | データ編集操作 | `doCut()`, `doBaseline()`, `invertData()` |
| `CUSUMV3.py` | CUSUMアルゴリズム | `detect_cusum()` |

### UIファイル (ui/ ディレクトリ)

| ファイル | 用途 |
|---------|------|
| `maingui.py/.ui` | メインウィンドウのレイアウト |
| `loadfile.py/.ui` | ファイル読み込みダイアログ |
| `analysissettings.py/.ui` | 解析設定ダイアログ |
| `subeventstatesettings.py/.ui` | サブイベント状態設定ダイアログ |
| `exportevents.py/.ui` | イベントエクスポートダイアログ |
| `exporttraceselection.py/.ui` | 選択領域エクスポートダイアログ |
| `ivselection.py/.ui` | I-V選択ダイアログ |

---

## 🔑 主要クラス詳細

### 1. TraceData (DataTypes.py)

電流トレースデータを管理するクラス。データのセグメント分割に対応。

```python
class TraceData:
    _raw: list[np.ndarray]   # 生データ（セグメントごと）
    _filt: list[np.ndarray]  # フィルタ後データ
    srange: list[tuple]      # 各セグメントの(開始, 終了)インデックス
    original_length: int     # 元データの長さ
    source_file_name: str    # 元ファイル名
```

**主要メソッド:**

- `setOriginalData(raw, filt, source)` - 初期データ設定
- `trim(trange)` - 指定範囲をカット
- `invert()` - 電流符号を反転
- `getConcatDataPoints(grange)` - 範囲指定でデータ取得

### 2. FileData (DataTypes.py)

ファイルごとの解析状態を保持するクラス。

```python
class FileData:
    data: TraceData              # トレースデータ
    analysis_results: AnalysisResults  # 解析結果
    baseline: float              # ベースライン電流値
    baseline_std: float          # ベースライン標準偏差
    ADC_samplerate_Hz: float     # サンプリングレート
    LRs: list[pg.LinearRegionItem]  # 選択領域リスト
    t_V_record: np.ndarray       # 電圧記録（XML由来）
```

### 3. Config (Analysis.py)

解析パラメータを管理するクラス。

```python
class Config:
    baseline_A: float              # ベースライン [A]
    baseline_std_A: float          # ベースライン標準偏差 [A]
    threshold_A: float             # 検出閾値 [A] (default: 0.3nA)
    enable_subevent_state_detection: bool  # サブイベント検出有効化
    maxNsta: int                   # 最大状態数 (default: 16)
    cusum_stepsize: int            # CUSUMステップサイズ
    cusum_threshhold: int          # CUSUM閾値
    merge_delta_blockade: float    # 状態マージ閾値
    prefilt_window_us: float       # 前処理フィルタ窓 [μs]
    state_min_duration_us: float   # 最小状態持続時間 [μs]
```

### 4. AnalysisResults (Analysis.py)

解析結果を格納するクラス。

```python
class AnalysisResults:
    analysis_config: Config     # 使用した設定
    result_dtype: np.dtype      # 結果テーブルの型定義
    tables: dict[str, np.ndarray]  # 結果テーブル
        # 'Event' - 検出されたイベント一覧
        # 'CUSUMState' - サブイベント状態一覧
```

**結果テーブルのカラム:**

| カラム | 型 | 説明 |
|--------|-----|------|
| id | int | 一意識別子 |
| N_child | int | 子状態数 |
| parent_id | int | 親イベントID |
| category | str | 'Event' or 'CUSUMState' |
| index | int | イベント/状態インデックス |
| seg | int | セグメント番号 |
| local_startpt | int | セグメント内開始点 |
| local_endpt | int | セグメント内終了点 |
| global_startpt | int | 全体での開始点 |
| global_endpt | int | 全体での終了点 |
| deli | float | ΔI (電流変化量) |
| frac | float | 分数電流ブロック |
| dwell | float | 滞留時間 [μs] |
| dt | float | イベント間隔 [s] |
| mean | float | 平均電流 |
| stdev | float | 標準偏差 |
| skewness | float | 歪度 |
| kurtosis | float | 尖度 |

---

## 🔄 主要処理フロー

### ファイル読み込みフロー

```
1. UI: File → Load でダイアログ表示
2. LoadFileDialog: パス・サンプルレート・フィルタ設定
3. IO.loadFile():
   ├── .opt/.bin ファイル: np.fromfile() で読み込み
   │   └── Bessel フィルタ適用 (scipy.signal)
   ├── .tracedata ファイル: pickle.load()
   └── .xml ファイル: 電圧記録・ユーザーノート読み込み
4. TraceData.setOriginalData() でデータ格納
5. Painting.paintCurrentTrace() で描画
```

### 解析フロー

```
1. UI: Analysis → Analyze でダイアログ表示
2. AnalyzeDialog: 閾値・サブイベント設定
3. Analysis.computeAnalysis():
   ├── 閾値以下の点を検出
   ├── イベント開始/終了点を特定
   ├── ベースラインまでトラックバック
   ├── 最小値位置で終点調整
   ├── (オプション) CUSUM でサブイベント検出
   │   └── multiprocessing で並列処理
   └── AnalysisResults に結果格納
4. Painting.plotAnalysis() で可視化
5. IO.saveAnalysis() で保存
```

---

## 📤 CSVエクスポート（Event Points）

### 目的

イベント（`tables["Event"]`）ごとに、

- **開始点（start）** と **終了点（end）** の座標
- 解析で既に計算済みのイベント特徴量（dwell, frac, 統計量など）

を **1イベント=1行** のCSVとして出力します。機械学習入力（特徴量テーブル）として使う想定です。

### UI導線

- 解析実行後に `Analysis → Export Event Points (CSV)...`

### 出力元（実装）

- CSV書き出し: `PythIon/IO.py` の `exportEventPointsCSV(app)`
- メニュー追加: `PythIon/Pythion.py`

### start/end点の定義（重要）

このCSVの start/end は、メイン波形上の注釈点に揃えています（描画は `Painting.plotAnalysis()` / `Painting.inspectEvent_()` を参照）。

- `start_*` は `local_startpt/global_startpt` の点
- `end_*` は `local_endpt/global_endpt` の点

注意：`local_endpt` は実装上、しばしば **「baselineに戻った点」ではなく「イベント内の最後の局所最小（谷）」**に調整されます。
そのため `end_current_A_filt` は「復帰点の電流」ではなく、イベントの終盤の谷の電流として解釈する方が安全です。

### 列の意味（概要）

#### メタデータ（再現性）

- `source_file_name`: 元データファイル名（`TraceData.source_file_name`）
- `ADC_samplerate_Hz`: サンプルレート（時間換算に使用）
- `LPFilter_cutoff_Hz`: フィルタ条件
- `baseline_A`, `baseline_std_A`, `threshold_A`: 解析条件

#### 位置（インデックス/時間）

- `local_startpt`, `local_endpt`: セグメント内インデックス
- `global_startpt`, `global_endpt`: 全体インデックス
- `t_start_s`, `t_end_s`: 秒（`global_* / ADC_samplerate_Hz`）

#### 注釈点の電流（filtered）

- `start_current_A_filt`: start点の電流（A）
- `end_current_A_filt`: end点の電流（A）

#### 代表的なイベント特徴量

- `deli_A`: ブロッケードの大きさ（概ね baseline からの落ち込み）
- `frac`: `deli_A / baseline_A` に相当する比（無次元）
- `dwell_us`: イベント滞在時間（μs）
- `dt_s`: イベント間隔（秒）
- `mean_A`, `stdev_A`, `skewness`, `kurtosis`: イベント区間の統計量

※本CSVは「イベント（Event）を1行に要約する」用途に絞っているため、trough-to-trough（`*_tt_*`）やサブイベント（CUSUMState）に関する列は含めません。

---

## 🔻 サブイベント（CUSUMState）とプロット記号

散布図は概ね次の対応です（`Painting.plotAnalysis()`参照）。

- イベント（Event）: `symbol="o"`（丸）
- サブイベント状態（CUSUMState）: `symbol="t"`（三角形。UI上で下向き矢印に見えることがあります）

「サブイベントを持つイベント」は、矢印ではなく `N_child` に応じた色/サイズで区別されます。

### 列の意味（詳細・一覧）

| カラム | 単位 | 説明 |
|---|---|---|
| source_file_name | 文字列 | 元データ（トレース）ファイル名。後で元波形と突合する際の手がかり。 |
| ADC_samplerate_Hz | Hz | サンプルレート。`t_start_s/t_end_s` の換算に使用。 |
| LPFilter_cutoff_Hz | Hz | フィルタのカットオフ。フィルタ条件が違うと特徴量が変化するため記録。 |
| baseline_A | A | baseline電流（解析条件）。 |
| baseline_std_A | A | baselineノイズの標準偏差（解析条件）。 |
| threshold_A | A | イベント検出の閾値（解析条件）。 |
| event_id | なし | イベントの一意ID。 |
| event_index | なし | イベントの連番インデックス。 |
| seg | なし | セグメント番号（分割トレースの場合）。 |
| local_startpt | サンプル | セグメント内の開始点インデックス。 |
| local_endpt | サンプル | セグメント内の終了点インデックス（実装上、最後の谷に調整されることがあります）。 |
| global_startpt | サンプル | 全体トレースの開始点インデックス。 |
| global_endpt | サンプル | 全体トレースの終了点インデックス。 |
| t_start_s | s | start時刻。`global_startpt / ADC_samplerate_Hz`。 |
| t_end_s | s | end時刻。`global_endpt / ADC_samplerate_Hz`。 |
| start_current_A_filt | A | start点でのfiltered電流値。 |
| end_current_A_filt | A | end点でのfiltered電流値（baseline復帰点ではなく谷寄りの意味になる場合あり）。 |
| deli_A | A | ブロッケードの大きさ（概ね baseline からの落ち込み）。 |
| frac | なし | ブロッケード比（概ね `deli_A / baseline_A`）。 |
| dwell_us | μs | イベント滞在時間。 |
| dt_s | s | イベント間隔（開始点同士の時間差）。 |
| mean_A | A | イベント区間の平均電流。 |
| stdev_A | A | イベント区間の標準偏差。 |
| skewness | なし | イベント区間分布の歪度。 |
| kurtosis | なし | イベント区間分布の尖度。 |

### CUSUM (Cumulative Sum) アルゴリズム

```
目的: イベント内のサブ状態（電流レベルの変化点）を検出

1. 移動平均・メディアンフィルタで前処理
2. 正/負方向ジャンプの対数尤度を計算
3. 累積和が閾値を超えたらジャンプ検出
4. 過分割された状態をマージ (merge_oversegmentation)
```

---

## 🎨 GUIレイアウト

### メインウィンドウ構成

```
┌─────────────────────────────────────────────────────────────┐
│ Menu Bar: File | Edit | Selection | Analysis                │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                    Signal Plot (p1)                     │ │
│ │                   電流トレース表示                        │ │
│ └─────────────────────────────────────────────────────────┘ │
├──────────────────────────┬──────────────────────────────────┤
│ ┌──────────────────────┐ │ ┌──────────────────────────────┐ │
│ │   Scatter Plot (w1)  │ │ │    Histograms (w2-w5)       │ │
│ │  Dwell vs Blockade   │ │ │  Frac/Deli/Dwell/dt Hist    │ │
│ └──────────────────────┘ │ └──────────────────────────────┘ │
├──────────────────────────┼──────────────────────────────────┤
│ ┌──────────────────────┐ │ ┌──────────────────────────────┐ │
│ │   Event Plot (p3)    │ │ │        Log Text             │ │
│ │  個別イベント表示      │ │ │      操作ログ表示            │ │
│ └──────────────────────┘ │ └──────────────────────────────┘ │
└──────────────────────────┴──────────────────────────────────┘
```

### プロットオブジェクト一覧

| 変数名 | 親Widget | 用途 |
|--------|----------|------|
| `p1` | signalplot | メイン電流トレース |
| `w1` | scatterplot | Dwell-Blockage 散布図 |
| `w1std/skew/kurt` | stdev/skew/kurtosisplot | 統計量散布図 |
| `w2` | frachistplot | Fractional blockage ヒストグラム |
| `w3` | delihistplot | ΔI ヒストグラム |
| `w4` | dwellhistplot | Dwell time ヒストグラム |
| `w5` | dthistplot | イベント間隔ヒストグラム |
| `p3` | eventplot | 個別イベント詳細 |

---

## 📝 主要メニューアクション

### File メニュー

| アクション | 関数 | 説明 |
|-----------|------|------|
| Load | `IO.LoadFileDialog` | ファイル読み込み |
| Reload | `IO.loadFile()` | 再読み込み |
| Save Traces | `IO.saveTrace()` | トレースデータ保存 (.tracedata) |
| Save Segment Info | `IO.saveSegInfo()` | セグメント情報保存 |

### Edit メニュー

| アクション | 関数 | 説明 |
|-----------|------|------|
| Cut | `Edits.doCut()` | 選択範囲をカット |
| Set Baseline | `Edits.doBaseline()` | 選択範囲、または次のトレース左クリック位置からベースライン設定 |
| Invert Current Sign | `Edits.invertData()` | 電流符号反転 |

### Selection メニュー

| アクション | ショートカット | 関数 |
|-----------|--------------|------|
| Auto Detect Clears | Alt+A | `Selections.autoFindCutLRs()` |
| Add One Cut Region | Alt+D | `Selections.addOneManualCutLR()` |
| Delete Last Cut Region | - | `Selections.deleteOneCutLR()` |
| Clear Selections | - | `Selections.clearLRs()` |
| Export Data | - | `IO.exportSelection()` |
| Export Measurement | - | `Selections.measureSelections()` |
| Inspect Selection | - | `Painting.inspectSelection()` |

### Analysis メニュー

| アクション | 関数 | 説明 |
|-----------|------|------|
| Analyze | `Analysis.AnalyzeDialog` | イベント検出実行 |
| Subevent State Settings | `Analysis.SubEventStateSettingsDialog` | CUSUM設定 |
| Repaint Analysis | `Painting.plotAnalysis()` | 解析結果再描画 |

---

## 🔧 開発のヒント

### 新機能追加時のパターン

1. **新しいダイアログを追加する場合:**

   ```python
   # 1. ui/ に .ui ファイルを作成 (Qt Designer)
   # 2. pyuic5 で .py に変換
   # 3. 新しいDialog クラスを作成
   class MyDialog(QtWidgets.QDialog):
       def __init__(self, parent:BaseAppMainWindow):
           self.parent = parent
           super().__init__(parent)
           self.ui = Ui_MyDialog()
           self.ui.setupUi(self)
           self.accepted.connect(self.dialogAccept)
   ```

2. **新しい解析処理を追加する場合:**

   ```python
   # Analysis.py に処理関数を追加
   def myNewAnalysis(app: BaseAppMainWindow):
       # app.perfiledata.data から入力データ取得
       # 処理実行
       # app.perfiledata.analysis_results.tables['MyResult'] = result
   ```

3. **新しいプロットを追加する場合:**

   ```python
   # BaseApp.py の __init__ でプロット要素初期化
   self.my_plot = self.ui.myplotwidget.addPlot()
   
   # Painting.py で描画処理
   def plotMyData(app: BaseAppMainWindow):
       app.my_plot.clear()
       app.my_plot.plot(x, y, pen='b')
   ```

### データアクセスパターン

```python
# アプリケーション状態へのアクセス
app.perfiledata.data           # TraceData オブジェクト
app.perfiledata.data.filt[k]   # k番目セグメントのフィルタデータ
app.perfiledata.data.srange    # セグメント範囲リスト
app.perfiledata.ADC_samplerate_Hz  # サンプリングレート

# ベースライン
app.ui_baseline        # ベースライン値
app.ui_baseline_std    # ベースライン標準偏差

# 解析設定
app.analysis_config.threshold_A  # 検出閾値
app.analysis_config.enable_subevent_state_detection  # サブイベント有効

# 解析結果
app.perfiledata.analysis_results.tables['Event']      # イベント結果
app.perfiledata.analysis_results.tables['CUSUMState'] # 状態結果
```

### pyqtgraph プロット操作

```python
# データプロット
handle = app.p1.plot(x, y, pen='b')  # ラインプロット
handle.setDownsampling(ds=4096, auto=False, method='peak')

# 散布図
scatter = pg.ScatterPlotItem()
scatter.addPoints(x=x, y=y, symbol='o', brush='r', size=5)
app.w1.addItem(scatter)

# 線追加
app.p1.addLine(y=baseline, pen='g')

# 選択領域
lr = pg.LinearRegionItem()
lr.setRegion((start, end))
app.p1.addItem(lr)
```

---

## 📚 用語集

| 用語 | 説明 |
|-----|------|
| Event | 電流がベースラインから閾値以下に低下した期間 |
| Subevent State | イベント内の電流レベル変化で区切られたサブ区間 |
| Dwell Time | イベントの持続時間 [μs] |
| ΔI (deli) | 電流変化量 = baseline - event_mean |
| Fractional Blockage (frac) | deli / baseline |
| Baseline | 定常状態の電流値 |
| Segment | カット操作で分割されたデータ区間 |
| CUSUM | Cumulative Sum - 変化点検出アルゴリズム |
| LinearRegionItem (LR) | pyqtgraph の選択領域オブジェクト |

---

## 🚀 起動方法

```bash
# インストール
pip install -e .

# 起動
PythIon

# または
python -m PythIon
```

---

## 📄 ファイルフォーマット

### 入力ファイル

| 拡張子 | 形式 | エンディアン |
|--------|------|------------|
| `.opt` | バイナリ (float64) | Big endian |
| `.bin` | バイナリ (float64) | Little endian |
| `.tracedata` | pickle (TraceData) | - |
| `.xml` | XMLメタデータ | - |

### 出力ファイル

| 拡張子 | 内容 |
|--------|------|
| `.tracedata` | 処理済みトレースデータ |
| `_analysis/` | 解析結果ディレクトリ |
| `*_segments.txt` | セグメント情報 |
| `*_measurement.txt` | 測定結果 |
| `*_log.txt` | 操作ログ |
