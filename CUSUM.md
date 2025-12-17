# CUSUM (累積和) アルゴリズムの説明

このプログラムは、時系列データにおける**変化点検出（Change Point Detection）**を行うための**CUSUM（Cumulative Sum）アルゴリズム**を実装しています。主にナノポアセンシングやイオンチャネル記録などの電気生理学的データの解析に使用されます。

## 主要な処理フロー

### 1. 前処理（ノイズ除去）

```plain
data → 移動平均 → 移動中央値 → 平滑化されたデータ
```

- **`central_moving_average`**: 中央移動平均でデータを平滑化
- **`central_moving_median`**: 中央移動中央値でさらにノイズを除去

### 2. CUSUM 変化点検出アルゴリズム

CUSUMは、データが正方向または負方向にジャンプしたかを統計的に検出します。

#### 核心となる計算式

**瞬間対数尤度（Log-Likelihood）**:
$$\text{logp}_k = \frac{\text{stepsize} \cdot \sigma_{\text{base}}}{\sigma^2} \left( x_k - \mu - \frac{\text{stepsize} \cdot \sigma_{\text{base}}}{2} \right)$$

$$\text{logn}_k = -\frac{\text{stepsize} \cdot \sigma_{\text{base}}}{\sigma^2} \left( x_k - \mu + \frac{\text{stepsize} \cdot \sigma_{\text{base}}}{2} \right)$$

- $x_k$: 現在のデータ点
- $\mu$: 局所平均
- $\sigma^2$: 局所分散（オンライン計算）
- $\sigma_{\text{base}}$: ベースラインの標準偏差

**累積対数尤度**:
$$\text{cpos}_k = \text{cpos}_{k-1} + \text{logp}_k$$
$$\text{cneg}_k = \text{cneg}_{k-1} + \text{logn}_k$$

**決定関数**:
$$\text{gpos}_k = \max(\text{gpos}_{k-1} + \text{logp}_k, 0)$$
$$\text{gneg}_k = \max(\text{gneg}_{k-1} + \text{logn}_k, 0)$$

### 3. 変化点の検出条件

決定関数が閾値を超えたときにジャンプを検出:
$$\text{gpos}_k > \text{threshold} \quad \text{または} \quad \text{gneg}_k > \text{threshold}$$

ジャンプの位置は累積対数尤度の**最小点**で決定:
$$\text{jump} = \text{anchor} + \arg\min_{i \in [\text{anchor}, k]} \text{cpos}_i$$

## パラメータの意味

| パラメータ | 説明 |
|-----------|------|
| `basesd` | ベースラインノイズの標準偏差 |
| `threshold` | 変化検出の感度（高いほど鈍感） |
| `stepsize` | 検出したいジャンプの大きさ（$\sigma_{\text{base}}$の倍数） |
| `minlength` | 最小状態持続時間（サンプル数） |
| `maxstates` | 最大状態数（超えると自動で閾値を緩和） |
| `moving_oneside_window` | 前処理のウィンドウサイズ |

## 出力

```python
cusum = {
    'nStates': 検出された状態の数,
    'starts': 各状態の開始インデックス配列,
    'threshold': 使用された閾値,
    'stepsize': 使用されたステップサイズ
}
```

## 用途

このアルゴリズムは、イオンチャネルやナノポアを通過する分子が引き起こす**電流変化（サブイベント）**を検出するために設計されています。検出された各「状態」は、異なる電流レベルを表します。
