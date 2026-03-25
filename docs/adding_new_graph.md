# Adding a New Graph

This document describes the procedure for adding a new event feature graph to PythIon. The FFT graph implementation is used as an example.

## Overview

To add a new graph, you need to modify the following files:

| File | Changes |
|------|---------|
| `Analysis.py` | Field definition + calculation logic (legacy) |
| `calc/analysis.py` | Field definition + calculation logic (refactored) |
| `calc/compat.py` | Data conversion mapping |
| `ui/maingui.ui` | UI plot widget |
| `ui/maingui.py` | UI regeneration |
| `BaseApp.py` | Plot initialization |
| `Painting.py` | Drawing + click highlighting |

## Steps

### 1. Add field to Analysis.py

Add the new field to the `result_spec` list in the `AnalysisResults` class.

```python
# Analysis.py (line ~85)
self.result_spec = [
    # ... existing fields ...
    ("kurtosis_tt", float, r"%.18e"),
    ("fft_mean", float, r"%.18e"),  # ← new field
]
```

### 2. Add calculation logic to Analysis.py

Compute the new feature within the event loop.

```python
# Analysis.py - inside event calculation loop
seg_fft_mean = np.full(seg_numberofevents, np.nan)

for kx, x in enumerate(startpoints):
    # ... existing calculations ...
    
    # FFT computation (use rfft for real input)
    event_signal = seg_filt[x : endpoints[kx]]
    if len(event_signal) > 1:
        fft_magnitude = np.abs(np.fft.rfft(event_signal))
        if len(fft_magnitude) > 1:
            seg_fft_mean[kx] = np.mean(fft_magnitude[1:])  # exclude DC

# Add to result table
seg_result_table["fft_mean"] = seg_fft_mean
```

### 3. Add field to calc/analysis.py (refactored version)

Update `EVENT_HEADERS` list and calculation logic.

```python
# calc/analysis.py
EVENT_HEADERS = [
    # ... existing headers ...
    "kurtosis_tt",
    "fft_mean",  # ← new field
]

# Inside analyze_tables() function
fft_mean = np.full(number_of_events, np.nan)

for event_index, (start_point, end_point) in enumerate(...):
    segment = data[start_point:end_point]
    # ... existing calculations ...
    
    if segment.size > 1:
        fft_magnitude = np.abs(np.fft.rfft(segment))
        if fft_magnitude.size > 1:
            fft_mean[event_index] = np.mean(fft_magnitude[1:])

# Add to DataFrame
events = pd.DataFrame({
    # ... existing columns ...
    "fft_mean": fft_mean,
}, columns=EVENT_HEADERS)
```

### 4. Add mapping to calc/compat.py

Add conversion mapping to the legacy format.

```python
# calc/compat.py - inside _populate_event_table() function
def _populate_event_table(...):
    # ... existing mappings ...
    seg_result["kurtosis_tt"] = ev["kurtosis_tt"].to_numpy()
    seg_result["fft_mean"] = ev["fft_mean"].to_numpy()  # ← new mapping
```

### 5. Add plot widget to ui/maingui.ui

Use Qt Designer or edit XML directly.

```xml
<!-- ui/maingui.ui - add after existing tabs -->
<widget class="QWidget" name="ffttab">
  <attribute name="title">
    <string>FFT</string>
  </attribute>
  <layout class="QHBoxLayout" name="horizontalLayout_fft">
    <item>
      <widget class="GraphicsLayoutWidget" name="fftplot"/>
    </item>
  </layout>
</widget>
```

Regenerate the UI file:
```bash
pyuic5 PythIon/ui/maingui.ui -o PythIon/ui/maingui.py
```

### 6. Add plot initialization to BaseApp.py

Add plot widget initialization and background color setting.

```python
# BaseApp.py - background color setting
for p in (self.ui.stdevplot, self.ui.skewnessplot, self.ui.kurtosisplot, self.ui.fftplot):
    p.setBackground("w")

# Plot initialization (follow existing pattern)
self.w1fft = self.ui.fftplot.addPlot()
axis = LogExponentAxisItem(orientation="bottom")
self.w1fft.setAxisItems({"bottom": axis})
self.p2fft = dict()
for entry in self.scatter_entries:
    p = pg.ScatterPlotItem()
    self.w1fft.addItem(p)
    self.p2fft[entry] = p
self.w1fft.setLabel("bottom", text="Log Dwell Time", units="Log10(μs)")
self.w1fft.setLabel("left", text="Mean FFT Magnitude", units="A")
self.w1fft.setLogMode(x=True, y=False)
self.w1fft.showGrid(x=True, y=True)
self.w1fft.getAxis("bottom").enableAutoSIPrefix(False)
setAxisFont(self.w1fft.getAxis("bottom"))
setAxisFont(self.w1fft.getAxis("left"))

# Add to p2s tuple (for click handling)
self.p2s = (self.p2, self.p2std, self.p2skew, self.p2kurt, self.p2fft)
```

### 7. Add drawing logic to Painting.py

#### 7a. Add scatter plot drawing to plotAnalysis()

```python
# Painting.py - inside plotAnalysis() function
app.p2fft["events"].addPoints(
    x=scatter_pts_x,
    y=getValid(event_result_table["fft_mean"]),
    symbol="o",
    brush=event_colors,
    pen=None,
    size=event_sizes,
)
```

#### 7b. Add highlight display to inspectEvent()

```python
# Painting.py - inside inspectEvent() function

# Get value
event_fft_mean = [event_result_table["fft_mean"][event_row_number]]

# Add annotations (3-layer structure: yellow outline → black outline → colored center)
app.p2fft["annotations"].addPoints(
    x=log_event_dwell,
    y=event_fft_mean,
    symbol="o",
    brush=None,
    pen=pg.mkPen("y", width=2),
    size=12,
)
app.p2fft["annotations"].addPoints(
    x=log_event_dwell,
    y=event_fft_mean,
    symbol="o",
    brush=None,
    pen=pg.mkPen("k", width=2),
    size=8,
)
app.p2fft["annotations"].addPoints(
    x=log_event_dwell,
    y=event_fft_mean,
    symbol="o",
    brush=event_color,
    size=6,
)
```

## Verification

1. Import test:
   ```bash
   uv run python -c "from PythIon import Analysis, BaseApp, Painting; print('OK')"
   ```

2. Launch application:
   ```bash
   uv run PythIon
   ```

3. Checklist:
   - [ ] New tab is displayed
   - [ ] Data is plotted after analysis
   - [ ] Points are highlighted on click
   - [ ] Works with refactored engine

## Performance Tips

- Use `np.fft.rfft()` for real input FFT (~2x faster than `fft()`)
- Avoid copying large arrays
- Vectorize calculations outside loops when possible

## Related Documentation

- `cusum.md`: Details of the CUSUM algorithm
- `codebase_overview.md`: Overview of the entire codebase
