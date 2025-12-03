# PythIon リファクタリング TODO

## リファクタリング候補一覧

| # | 対象箇所 | 問題点 | 難易度 | リスク | 可読性向上 | 推奨度 |
|---|----------|--------|:------:|:------:|:----------:|:------:|
| **1** | `Analysis.py` の `computeAnalysis()` | 関数が500行超で責務が多すぎる（イベント検出、CUSUM処理、結果構築を1つで実行） | 中 | 低 | **高** | ⭐⭐⭐ |
| **2** | `BaseApp.py` の散布図初期化 | 4つの散布図（w1, w1std, w1skew, w1kurt）のコードがほぼ同一でコピペ | 低 | **低** | **高** | ⭐⭐⭐ |
| **3** | `IO.py` の `loadFile()` | .opt/.bin ファイル読み込み処理が重複している | 低 | 低 | 中 | ⭐⭐⭐ |
| **4** | `IO.py` の `saveAnalysis()` | `result_tables.has_key()` は Python3 で廃止されたメソッド（**バグ**） | 低 | **低** | - | ⭐⭐⭐ |
| **5** | `Painting.py` の `inspectEvent_()` | 250行超の長い関数、アノテーション処理が重複 | 中 | 低 | **高** | ⭐⭐ |
| **6** | 型ヒントの欠如 | 関数のパラメータや戻り値に型ヒントがほぼない | 低 | **低** | **高** | ⭐⭐ |
| **7** | `Config` / `LoadConfig` クラス | `dataclass` を使うとボイラープレートを削減できる | 低 | 低 | 中 | ⭐⭐ |
| **8** | マジックナンバー | `1e9`, `1e-9`, `1e6` などの単位変換定数が散在 | 低 | 低 | 中 | ⭐ |
| **9** | `Painting.py` のアノテーション追加 | 同じパターンが4回繰り返されている | 低 | **低** | 中 | ⭐⭐ |

---

## 詳細分析と推奨アクション

### 1. 🔴 [緊急] `IO.py` のバグ修正

```python
# 現在のコード (line ~304)
if app.perfiledata.analysis_results.result_tables.has_key('Event'):
```

`.has_key()` は Python 2 のメソッドで、Python 3 では削除されています。これは実行時エラーになります。

**修正**: `'Event' in app.perfiledata.analysis_results.tables` に変更

---

### 2. ⭐⭐⭐ `BaseApp.py` の散布図初期化の重複排除

現在、w1, w1std, w1skew, w1kurt の初期化コードが完全に重複しています：

```python
# このパターンが4回繰り返されている
self.w1std = self.ui.stdevplot.addPlot()
axis = LogExponentAxisItem(orientation='bottom')
self.w1std.setAxisItems({'bottom':axis})
self.p2std = dict()
for entry in self.scatter_entries:
    p = pg.ScatterPlotItem()
    self.w1std.addItem(p)
    self.p2std[entry] = p
# ... ラベル設定、グリッド設定など
```

**推奨**: ヘルパー関数で統一

```python
def _setupScatterPlot(self, widget, x_label, y_label, y_units=None):
    w = widget.addPlot()
    axis = LogExponentAxisItem(orientation='bottom')
    w.setAxisItems({'bottom': axis})
    p = {entry: pg.ScatterPlotItem() for entry in self.scatter_entries}
    for item in p.values():
        w.addItem(item)
    # ... 設定
    return w, p
```

---

### 3. ⭐⭐⭐ `computeAnalysis()` の分割

500行超の関数を3つに分割することを推奨：

- `_detect_events()` - イベント検出ロジック
- `_detect_cusum_states()` - CUSUMサブイベント検出
- `_build_result_tables()` - 結果テーブル構築

---

### 4. ⭐⭐ 型ヒントの追加

現在:
```python
def computeAnalysis(app:BaseAppMainWindow):
```

推奨:
```python
def computeAnalysis(app: BaseAppMainWindow) -> None:
```

主要な関数に戻り値と引数の型ヒントを追加することで、IDE補完とコードの自己文書化が向上します。

---

## 優先順位付きアクションプラン

| 優先度 | アクション | 工数 |
|:------:|------------|------|
| 1 | `IO.py` の `.has_key()` バグ修正 | 5分 |
| 2 | `BaseApp.py` 散布図初期化の重複排除 | 30分 |
| 3 | `Analysis.py` の `computeAnalysis()` 分割 | 1-2時間 |
| 4 | 型ヒント追加（主要関数のみ） | 1時間 |
