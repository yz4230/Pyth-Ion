# -*- coding: utf-8 -*-

from .BaseApp import *
# import cProfile


FFT_SPECTRUM_BIN_WIDTH_HZ = 20_000
FFT_SPECTRUM_REFERENCE_COLOR = (30, 130, 200, 100)
FFT_SPECTRUM_LEGEND_COLORS = {
    "red": (220, 40, 40),
    "green": (90, 110, 85),
    "blue": (30, 130, 200),
    "purple": (173, 27, 183),
}


def _getFftSpectrumBinWidthHz(app: BaseAppMainWindow):
    try:
        bin_width_khz = float(app.ui.fftspectrumbinwidth.text())
    except ValueError:
        bin_width_khz = np.nan

    if not np.isfinite(bin_width_khz) or bin_width_khz <= 0:
        default_bin_width_khz = FFT_SPECTRUM_BIN_WIDTH_HZ / 1e3
        app.ui.fftspectrumbinwidth.setText(f"{default_bin_width_khz:g}")
        app.printlog("Invalid FFT spectrum bin width; reset to 20 kHz")
        return FFT_SPECTRUM_BIN_WIDTH_HZ

    return bin_width_khz * 1e3


def _computeFftSpectrumBins(
    event_signal, samplerate_hz, bin_width_hz=FFT_SPECTRUM_BIN_WIDTH_HZ
):
    event_signal = np.asarray(event_signal)
    if (
        event_signal.size <= 1
        or samplerate_hz is None
        or not np.isfinite(samplerate_hz)
        or samplerate_hz <= 0
        or bin_width_hz <= 0
    ):
        return np.array([]), np.array([]), np.array([])

    fft_frequency_hz = np.fft.rfftfreq(event_signal.size, d=1 / samplerate_hz)
    fft_magnitude = np.abs(np.fft.rfft(event_signal))

    fft_frequency_hz = fft_frequency_hz[1:]
    fft_magnitude = fft_magnitude[1:]
    valid = np.isfinite(fft_frequency_hz) & np.isfinite(fft_magnitude)
    fft_frequency_hz = fft_frequency_hz[valid]
    fft_magnitude = fft_magnitude[valid]
    if fft_frequency_hz.size == 0:
        return np.array([]), np.array([]), np.array([])

    max_frequency_hz = fft_frequency_hz[-1]
    bin_edges_hz = np.arange(0, max_frequency_hz + bin_width_hz, bin_width_hz)
    bin_starts_hz = bin_edges_hz[:-1]
    bin_ends_hz = np.minimum(bin_starts_hz + bin_width_hz, max_frequency_hz)
    bin_means = np.full(bin_starts_hz.size, np.nan)

    for index, (bin_start_hz, bin_end_hz) in enumerate(
        zip(bin_starts_hz, bin_ends_hz)
    ):
        if index == bin_starts_hz.size - 1:
            in_bin = (fft_frequency_hz >= bin_start_hz) & (
                fft_frequency_hz <= bin_end_hz
            )
        else:
            in_bin = (fft_frequency_hz >= bin_start_hz) & (
                fft_frequency_hz < bin_end_hz
            )
        if np.any(in_bin):
            bin_means[index] = np.mean(fft_magnitude[in_bin])

    has_data = np.isfinite(bin_means)
    return (
        bin_starts_hz[has_data] / 1e3,
        bin_ends_hz[has_data] / 1e3,
        bin_means[has_data],
    )


def _computeFftSpectrumDifferenceBins(
    event_signal,
    reference_signal,
    samplerate_hz,
    bin_width_hz=FFT_SPECTRUM_BIN_WIDTH_HZ,
):
    event_starts, event_ends, event_means = _computeFftSpectrumBins(
        event_signal, samplerate_hz, bin_width_hz
    )
    reference_starts, reference_ends, reference_means = _computeFftSpectrumBins(
        reference_signal, samplerate_hz, bin_width_hz
    )
    if event_means.size == 0 or reference_means.size == 0:
        return np.array([]), np.array([]), np.array([])

    reference_bins = {
        (round(float(start), 9), round(float(end), 9)): mean
        for start, end, mean in zip(reference_starts, reference_ends, reference_means)
    }
    shared_starts = []
    shared_ends = []
    diff_means = []
    for start, end, event_mean in zip(event_starts, event_ends, event_means):
        key = (round(float(start), 9), round(float(end), 9))
        reference_mean = reference_bins.get(key)
        if reference_mean is None:
            continue
        shared_starts.append(start)
        shared_ends.append(end)
        diff_means.append(event_mean - reference_mean)

    return np.array(shared_starts), np.array(shared_ends), np.array(diff_means)


def _getFftSpectrumReferenceSignal(app: BaseAppMainWindow):
    reference_region = getattr(app.perfiledata, "fft_spectrum_reference_region", None)
    if reference_region is None:
        return np.array([])
    reference_region = np.asarray(reference_region, dtype=int)
    if reference_region.size != 2 or reference_region[1] <= reference_region[0]:
        return np.array([])
    return app.perfiledata.data.getConcatDataPoints(reference_region, rawdata=False)


def _getFftSpectrumLegendColorName(color):
    rgb = np.array(pg.mkColor(color).getRgb()[:3])
    return min(
        FFT_SPECTRUM_LEGEND_COLORS,
        key=lambda name: np.sum(
            (rgb - np.array(FFT_SPECTRUM_LEGEND_COLORS[name])) ** 2
        ),
    )


def _addFftSpectrumLegendItem(app: BaseAppMainWindow, label, color):
    legend_color = pg.mkColor(color)
    legend_color.setAlpha(255)
    color_name = _getFftSpectrumLegendColorName(color)
    app.w1fftspectrumlegend.addItem(
        pg.PlotDataItem(
            pen=pg.mkPen(legend_color, width=3),
            symbol="s",
            symbolSize=10,
            symbolBrush=pg.mkBrush(legend_color),
            symbolPen=pg.mkPen(legend_color, width=2),
        ),
        f"{label} ({color_name})",
    )


def _plotEventFftSpectrum(app: BaseAppMainWindow, event_signal, event_color):
    app.w1fftspectrum.clear()
    app.w1fftspectrumlegend.clear()
    bin_width_hz = _getFftSpectrumBinWidthHz(app)
    reference_signal = _getFftSpectrumReferenceSignal(app)
    show_difference = (
        hasattr(app.ui, "checkBoxFftSpectrumDifference")
        and app.ui.checkBoxFftSpectrumDifference.isChecked()
    )
    if show_difference:
        if reference_signal.size <= 1:
            app.ui.checkBoxFftSpectrumDifference.setChecked(False)
            app.printlog(
                "No FFT spectrum reference region available; showing event spectrum."
            )
            show_difference = False

    if show_difference:
        bin_starts_khz, bin_ends_khz, bin_means = _computeFftSpectrumDifferenceBins(
            event_signal,
            reference_signal,
            app.perfiledata.ADC_samplerate_Hz,
            bin_width_hz,
        )
        app.w1fftspectrum.setLabel(
            "left", text="FFT Magnitude Difference", units="A"
        )
        reference_starts_khz = np.array([])
        reference_ends_khz = np.array([])
        reference_means = np.array([])
    else:
        bin_starts_khz, bin_ends_khz, bin_means = _computeFftSpectrumBins(
            event_signal, app.perfiledata.ADC_samplerate_Hz, bin_width_hz
        )
        reference_starts_khz, reference_ends_khz, reference_means = (
            _computeFftSpectrumBins(
                reference_signal, app.perfiledata.ADC_samplerate_Hz, bin_width_hz
            )
        )
        app.w1fftspectrum.setLabel("left", text="Mean FFT Magnitude", units="A")

    if bin_means.size == 0:
        if show_difference:
            app.printlog("No shared FFT spectrum bins available for difference display")
        else:
            app.printlog("No FFT spectrum available for selected event")
        return

    if reference_means.size > 0:
        reference_bars = pg.BarGraphItem(
            x0=reference_starts_khz,
            x1=reference_ends_khz,
            height=reference_means,
            brush=pg.mkBrush(FFT_SPECTRUM_REFERENCE_COLOR),
            pen=pg.mkPen(FFT_SPECTRUM_REFERENCE_COLOR[:3], width=2),
        )
        app.w1fftspectrum.addItem(reference_bars)
        _addFftSpectrumLegendItem(
            app, "Reference region", FFT_SPECTRUM_REFERENCE_COLOR
        )

    spectrum_bars = pg.BarGraphItem(
        x0=bin_starts_khz,
        x1=bin_ends_khz,
        height=bin_means,
        brush=event_color,
        pen=pg.mkPen("k", width=1),
    )
    spectrum_name = "Event - Reference" if show_difference else "Event"
    app.w1fftspectrum.addItem(spectrum_bars)
    _addFftSpectrumLegendItem(app, spectrum_name, event_color)
    if show_difference:
        app.w1fftspectrum.addLine(y=0, pen=pg.mkPen("k", width=1))
    max_frequency_khz = bin_ends_khz[-1]
    if reference_ends_khz.size > 0:
        max_frequency_khz = max(max_frequency_khz, reference_ends_khz[-1])
    app.w1fftspectrum.setRange(xRange=[0, max(max_frequency_khz, 20)])


def refreshEventFftSpectrum(app: BaseAppMainWindow):
    analysis_results = getattr(app.perfiledata, "analysis_results", None)
    if analysis_results is None or "Event" not in analysis_results.tables:
        app.w1fftspectrum.clear()
        app.w1fftspectrumlegend.clear()
        return

    event_result_table = analysis_results.tables["Event"]
    if len(event_result_table) == 0:
        app.w1fftspectrum.clear()
        app.w1fftspectrumlegend.clear()
        return

    try:
        event_row_number = int(app.ui.eventnumberentry.text())
    except ValueError:
        event_row_number = 0
    event_row_number = min(max(event_row_number, 0), len(event_result_table) - 1)
    app.ui.eventnumberentry.setText(str(event_row_number))

    event_res = event_result_table[event_row_number]
    event_seg_filt = app.perfiledata.data.filt[event_res["seg"]]
    event_signal = event_seg_filt[
        event_res["local_startpt"] : event_res["local_endpt"]
    ]
    if app.perfiledata.event_colors is not None:
        event_color = app.perfiledata.event_colors[event_row_number]
    else:
        event_color = app.inspect_event_fit_color_singlestate
    _plotEventFftSpectrum(app, event_signal, event_color)


def paintCurrentTrace(app: BaseAppMainWindow):
    print("Painting Current Trace")
    app.p1.clear()
    # FIXME
    # skips plotting first and last two points, there was a weird spike issue
    # app.p1.plot(app.t[2:][:-2],app.data.filt[2:][:-2],pen='b')
    # app.p1RawTraceHandle = app.p1.plot(app.t,app.data.raw,pen='gray', antialiasing=False)
    dscl = app.current_display_scale_factor
    app.perfiledata.p1RawTraceHandles = []
    app.perfiledata.p1FiltTraceHandles = []
    for k in range(app.perfiledata.data.Nseg):
        x = app.perfiledata.data.getSegCoord(k)
        handle = pg.PlotDataItem(
            x, (app.perfiledata.data.raw[k] * dscl), pen="gray", antialiasing=False
        )
        app.p1.addItem(handle)
        handle.setDownsampling(ds=app.DSratio, auto=False, method="peak")
        # FIXME As opposed to the documentation, setting cliptoview to true decreased the
        # performance drastically on x-axis dragging.
        # app.p1RawTraceHandle.setClipToView(True)
        # app.p1RawTraceHandle.setSkipFiniteCheck(True)
        handle.setDynamicRangeLimit(10.0)
        app.perfiledata.p1RawTraceHandles.append(handle)

        handle2 = pg.PlotDataItem(
            x, (app.perfiledata.data.filt[k] * dscl), pen="blue", antialiasing=False
        )
        app.p1.addItem(handle2)
        handle2.setDownsampling(ds=app.DSratio, auto=False, method="peak")
        handle2.setDynamicRangeLimit(10.0)
        app.perfiledata.p1FiltTraceHandles.append(handle2)

        region_handle = pg.LinearRegionItem(
            values=app.perfiledata.data.srange[k],
            brush=pg.mkBrush(0, 255, 0, 127),
            pen=pg.mkPen(color="orange", width=3),
            span=(0, 0.1),
            movable=False,
        )
        app.p1.addItem(region_handle)

    app.p1.getAxis("bottom").setScale(1 / app.perfiledata.ADC_samplerate_Hz)

    # Create draggable baseline line
    from . import Edits
    baseline_line = pg.InfiniteLine(
        pos=app.ui_baseline * dscl,
        angle=0,
        pen=pg.mkPen("g", width=2),
        movable=True,
        hoverPen=pg.mkPen("lime", width=3),
    )
    baseline_line.sigPositionChangeFinished.connect(
        lambda: Edits.handleBaselineDrag(app, baseline_line)
    )
    app.p1.addItem(baseline_line)
    app.perfiledata.baselineHandle = baseline_line

    # Baseline std deviation lines (static, dashed)
    std_line_upper = app.p1.addLine(
        y=(app.ui_baseline + app.ui_baseline_std) * dscl,
        pen=pg.mkPen("g", style=QtCore.Qt.DashLine),
    )
    std_line_lower = app.p1.addLine(
        y=(app.ui_baseline - app.ui_baseline_std) * dscl,
        pen=pg.mkPen("g", style=QtCore.Qt.DashLine),
    )
    app.perfiledata.baselineStdHandles = [std_line_upper, std_line_lower]

    app.updateThresholdLine()

    # if app.perfiledata.isFullTrace:
    if True:
        if (
            app.perfiledata.t_V_record is not None
            and len(app.perfiledata.t_V_record) > 0
        ):
            t_V_record = app.perfiledata.t_V_record
            t_V_curve = np.full((2 * len(t_V_record), 2), np.nan)
            t_V_curve[::2, 1] = t_V_record["mV"]
            t_V_curve[1::2, 1] = t_V_record["mV"]
            t_V_curve[0:-1:2, 0] = (
                t_V_record["msec"] * app.perfiledata.ADC_samplerate_Hz * 1e-3
            )
            t_V_curve[1:-1:2, 0] = (
                t_V_record["msec"][1:] * app.perfiledata.ADC_samplerate_Hz * 1e-3
            )
            t_V_curve[-1, 0] = app.perfiledata.data.original_length
            pV = pg.PlotDataItem(
                t_V_curve[:, 0], t_V_curve[:, 1] * 1e-12 * dscl, pen="magenta"
            )
            app.p1.addItem(pV)

    app.p1.showGrid(x=True, y=True)
    app.updateRawFiltVisibility()


def plotAnalysis(app: BaseAppMainWindow):
    app.printlog("Plotting analysis")
    for p in app.p2s:
        for entry in app.scatter_entries:
            p[entry].clear()
            p[entry].update()

    app.ui.scatterplot.update()

    app.w2.clear()
    app.w3.clear()
    app.w4.clear()
    app.w5.clear()
    app.w1fftspectrum.clear()
    app.w1fftspectrumlegend.clear()
    paintCurrentTrace(app)

    event_result_table = app.perfiledata.analysis_results.tables["Event"]

    print("started annotating main trace plot")
    dscl = app.current_display_scale_factor

    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for event in event_result_table:
        k = event["seg"]
        seg_filt = app.perfiledata.data.filt[k]
        x0.append(event["global_startpt"])
        x1.append(event["global_endpt"])
        y0.append(seg_filt[event["local_startpt"]] * dscl)
        y1.append(seg_filt[event["local_endpt"]] * dscl)

    app.p1.plot(x0, y0, pen=None, symbol="o", symbolBrush="g", symbolSize=10)
    app.p1.plot(x1, y1, pen=None, symbol="o", symbolBrush="r", symbolSize=10)

    print("finished annotating main trace plot")

    print("started plotting scatterplots")
    # event_number_of_states = np.array([len(state_result_table[state_result_table['parent_id']==event['id']]) for event in event_result_table])
    event_number_of_states = event_result_table["N_child"]
    event_sizes = np.where(event_number_of_states > 1, 1, 3)
    app.perfiledata.event_sizes = event_sizes
    event_colors = [
        app.inspect_event_fit_color_multistate
        if Nsta > 1
        else app.inspect_event_fit_color_singlestate
        for Nsta in event_number_of_states
    ]
    app.perfiledata.event_colors = event_colors
    scatter_pts_x = np.log10(event_result_table["dwell"])
    app.p2["events"].addPoints(
        x=scatter_pts_x,
        y=event_result_table["frac"],
        symbol="o",
        brush=event_colors,
        pen=None,
        size=event_sizes,
    )

    # def getValid(column):
    #     pts_valid = column > 0
    #     column_masked = np.full_like(column, np.nan)
    #     column_masked[pts_valid] = column[pts_valid]
    #     return column_masked
    def getValid(column):
        return column

    app.p2std["events"].addPoints(
        x=scatter_pts_x,
        y=getValid(event_result_table["stdev_tt"]),
        symbol="o",
        brush=event_colors,
        pen=None,
        size=event_sizes,
    )
    app.p2skew["events"].addPoints(
        x=scatter_pts_x,
        y=getValid(event_result_table["skewness_tt"]),
        symbol="o",
        brush=event_colors,
        pen=None,
        size=event_sizes,
    )
    app.p2kurt["events"].addPoints(
        x=scatter_pts_x,
        y=getValid(event_result_table["kurtosis_tt"]),
        symbol="o",
        brush=event_colors,
        pen=None,
        size=event_sizes,
    )
    app.p2fft["events"].addPoints(
        x=scatter_pts_x,
        y=getValid(event_result_table["fft_mean"]),
        symbol="o",
        brush=event_colors,
        pen=None,
        size=event_sizes,
    )

    if app.perfiledata.analysis_results.analysis_config.enable_subevent_state_detection:
        print("start plotting state scatterplots")
        state_result_table = app.perfiledata.analysis_results.tables["CUSUMState"]
        state_scatter_x = np.log10(state_result_table["dwell"])
        state_scatter_color = [
            app.state_colors[ind_state] for ind_state in state_result_table["index"]
        ]
        app.p2["cusum_states"].addPoints(
            x=state_scatter_x,
            y=state_result_table["frac"],
            symbol="t",
            brush=state_scatter_color,
            pen=None,
            size=5,
        )
        app.p2std["cusum_states"].addPoints(
            x=state_scatter_x,
            y=state_result_table["stdev"],
            symbol="t",
            brush=state_scatter_color,
            pen=None,
            size=5,
        )
        app.p2skew["cusum_states"].addPoints(
            x=state_scatter_x,
            y=state_result_table["skewness"],
            symbol="t",
            brush=state_scatter_color,
            pen=None,
            size=5,
        )
        app.p2kurt["cusum_states"].addPoints(
            x=state_scatter_x,
            y=state_result_table["kurtosis"],
            symbol="t",
            brush=state_scatter_color,
            pen=None,
            size=5,
        )

    # app.w1.addItem(app.p2)
    app.w1.setLogMode(x=True, y=False)
    app.w1.autoRange()
    app.ui.scatterplot.update()
    app.w1.setRange(yRange=[0, 1])
    print("finished plotting scatterplots")

    print("started plotting histograms")

    color = app.inspect_event_fit_color_singlestate
    fracy, fracx = np.histogram(
        event_result_table["frac"], bins=np.linspace(0, 1, int(app.ui.fracbins.text()))
    )
    deliy, delix = np.histogram(
        event_result_table["deli"],
        bins=np.linspace(
            float(app.ui.delirange0.text()) * 10**-9,
            float(app.ui.delirange1.text()) * 10**-9,
            int(app.ui.delibins.text()),
        ),
    )
    dwelly, dwellx = np.histogram(
        np.log10(event_result_table["dwell"]),
        bins=np.linspace(
            float(app.ui.dwellrange0.text()),
            float(app.ui.dwellrange1.text()),
            int(app.ui.dwellbins.text()),
        ),
    )
    dty, dtx = np.histogram(
        event_result_table["dt"],
        bins=np.linspace(
            float(app.ui.dtrange0.text()),
            float(app.ui.dtrange1.text()),
            int(app.ui.dtbins.text()),
        ),
    )

    hist = pg.BarGraphItem(height=fracy, x0=fracx[:-1], x1=fracx[1:], brush=color)
    app.w2.addItem(hist)
    hist = pg.BarGraphItem(height=deliy, x0=delix[:-1], x1=delix[1:], brush=color)
    app.w3.addItem(hist)
    app.w3.setRange(
        xRange=[
            float(app.ui.delirange0.text()) * 10**-9,
            float(app.ui.delirange1.text()) * 10**-9,
        ]
    )
    hist = pg.BarGraphItem(height=dwelly, x0=dwellx[:-1], x1=dwellx[1:], brush=color)
    app.w4.addItem(hist)
    hist = pg.BarGraphItem(height=dty, x0=dtx[:-1], x1=dtx[1:], brush=color)
    app.w5.addItem(hist)
    print("finished plotting histograms")
    app.setSubeventStateVisibility(app.ui.checkBoxShowSubeventStates.isChecked())


def inspectEvent(app: BaseAppMainWindow, clickedentry=None, clicked=None):
    # cProfile.runctx('inspectEvent_(app, clickedentry, clicked)', globals(), locals())
    inspectEvent_(app, clickedentry, clicked)


def inspectEvent_(app: BaseAppMainWindow, clickedentry=None, clicked=None):
    if clickedentry is None:
        clickedentry = "events"
    if clickedentry == "annotations":
        return

    analysis_results = app.perfiledata.analysis_results
    event_result_table = analysis_results.tables["Event"]
    if len(event_result_table) == 0:
        app.w1fftspectrum.clear()
        app.w1fftspectrumlegend.clear()
        app.perfiledata.selected_event_id = None
        app.printlog("No event available for inspection")
        return

    # Reset plot
    app.p3.setLabel("bottom", text="Time", units="s")
    app.p3.setLabel("left", text="Current", units="A")
    app.p3.clear()
    for p in app.p2s:
        p["annotations"].clear()

    # Correct for user error if non-extistent number is entered
    eventbuffer = int(app.ui.eventbufferentry.text())

    # Get row number of event
    if clicked is None:
        event_row_number = int(app.ui.eventnumberentry.text())
    else:
        if clickedentry == "events":
            event_row_number = clicked
        elif clickedentry == "cusum_states":
            if not app.perfiledata.analysis_results.analysis_config.enable_subevent_state_detection:
                return
            state_result_table = analysis_results.tables["CUSUMState"]
            state_parent_id = state_result_table[clicked]["parent_id"]
            # print(event_id, state['id'])
            event_row_number = np.nonzero(event_result_table["id"] == state_parent_id)[
                0
            ][0]

    if event_row_number >= len(analysis_results.tables["Event"]):
        event_row_number = len(analysis_results.tables["Event"]) - 1
    if event_row_number < 0:
        event_row_number = 0
    app.ui.eventnumberentry.setText(str(event_row_number))

    event_res = event_result_table[event_row_number]
    app.perfiledata.selected_event_id = event_res["id"]

    event_color = app.perfiledata.event_colors[event_row_number]

    # plot event trace
    k_seg = event_res["seg"]
    # print(k_seg)
    seg_range = app.perfiledata.data.srange[k_seg]
    event_seg_filt = app.perfiledata.data.filt[k_seg]
    flank_local_start = event_res["local_startpt"] - eventbuffer
    flank_local_start = max(0, flank_local_start)
    flank_global_start = flank_local_start + seg_range[0]
    flank_local_end = event_res["local_endpt"] + eventbuffer
    flank_local_end = min(len(event_seg_filt), flank_local_end)
    flank_global_end = flank_local_end + seg_range[0]
    event_signal = event_seg_filt[
        event_res["local_startpt"] : event_res["local_endpt"]
    ]
    _plotEventFftSpectrum(app, event_signal, event_color)

    app.p3.plot(
        app.perfiledata.getT(range(flank_global_start, flank_global_end)),
        event_seg_filt[flank_local_start:flank_local_end],
        pen="b",
    )

    # plot event fit
    x = (
        flank_global_start,
        event_res["global_startpt"],
        event_res["global_startpt"],
        event_res["global_endpt"],
        event_res["global_endpt"],
        flank_global_end,
    )

    y = (
        app.ui_baseline,
        app.ui_baseline,
        app.ui_baseline - event_result_table["deli"][event_row_number],
        app.ui_baseline - event_result_table["deli"][event_row_number],
        app.ui_baseline,
        app.ui_baseline,
    )
    app.p3.plot(app.perfiledata.getT(x), y, pen=pg.mkPen(color=event_color, width=2))
    app.p3.autoRange()

    # Mark event start and end points
    app.p3.plot(
        [app.perfiledata.getT(event_res["global_startpt"])],
        [event_seg_filt[event_res["local_startpt"]]],
        symbol="o",
        symbolBrush="g",
        symbolSize=12,
    )
    app.p3.plot(
        [
            app.perfiledata.getT(
                event_res["global_startpt"] + event_res["offset_first_min"]
            )
        ],
        [event_seg_filt[event_res["local_startpt"] + event_res["offset_first_min"]]],
        symbol="d",
        symbolBrush="g",
        symbolSize=12,
    )
    app.p3.plot(
        [app.perfiledata.getT(event_res["global_endpt"])],
        [event_seg_filt[event_res["local_endpt"]]],
        symbol="o",
        symbolBrush="r",
        symbolSize=12,
    )

    # Annotate event parameters
    app.ui.eventinfolabel.setText(
        "Dwell Time="
        + str(round(event_result_table[event_row_number]["dwell"], 2))
        + " μs,   Deli="
        + str(round(event_result_table[event_row_number]["deli"] * 1e9, 2))
        + " nA"
    )

    # Annotate event plot
    log_event_dwell = [np.log10(event_result_table["dwell"][event_row_number])]
    event_frac = [event_result_table["frac"][event_row_number]]
    event_stdev = [event_result_table["stdev_tt"][event_row_number]]
    event_skew = [event_result_table["skewness_tt"][event_row_number]]
    event_kurt = [event_result_table["kurtosis_tt"][event_row_number]]
    event_fft_mean = [event_result_table["fft_mean"][event_row_number]]

    app.p2["annotations"].addPoints(
        x=log_event_dwell,
        y=event_frac,
        symbol="o",
        brush=None,
        pen=pg.mkPen("y", width=2),
        size=12,
    )
    app.p2["annotations"].addPoints(
        x=log_event_dwell,
        y=event_frac,
        symbol="o",
        brush=None,
        pen=pg.mkPen("k", width=2),
        size=8,
    )
    app.p2["annotations"].addPoints(
        x=log_event_dwell, y=event_frac, symbol="o", brush=event_color, size=6
    )
    app.p2std["annotations"].addPoints(
        x=log_event_dwell,
        y=event_stdev,
        symbol="o",
        brush=None,
        pen=pg.mkPen("y", width=2),
        size=12,
    )
    app.p2std["annotations"].addPoints(
        x=log_event_dwell,
        y=event_stdev,
        symbol="o",
        brush=None,
        pen=pg.mkPen("k", width=2),
        size=8,
    )
    app.p2std["annotations"].addPoints(
        x=log_event_dwell, y=event_stdev, symbol="o", brush=event_color, size=6
    )
    app.p2skew["annotations"].addPoints(
        x=log_event_dwell,
        y=event_skew,
        symbol="o",
        brush=None,
        pen=pg.mkPen("y", width=2),
        size=12,
    )
    app.p2skew["annotations"].addPoints(
        x=log_event_dwell,
        y=event_skew,
        symbol="o",
        brush=None,
        pen=pg.mkPen("k", width=2),
        size=8,
    )
    app.p2skew["annotations"].addPoints(
        x=log_event_dwell, y=event_skew, symbol="o", brush=event_color, size=6
    )
    app.p2kurt["annotations"].addPoints(
        x=log_event_dwell,
        y=event_kurt,
        symbol="o",
        brush=None,
        pen=pg.mkPen("y", width=2),
        size=12,
    )
    app.p2kurt["annotations"].addPoints(
        x=log_event_dwell,
        y=event_kurt,
        symbol="o",
        brush=None,
        pen=pg.mkPen("k", width=2),
        size=8,
    )
    app.p2kurt["annotations"].addPoints(
        x=log_event_dwell, y=event_kurt, symbol="o", brush=event_color, size=6
    )
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
        x=log_event_dwell, y=event_fft_mean, symbol="o", brush=event_color, size=6
    )

    if app.perfiledata.analysis_results.analysis_config.enable_subevent_state_detection:
        # Plot subevent state fits
        state_result_table = analysis_results.tables["CUSUMState"]
        state_row_numbers = np.nonzero(
            state_result_table["parent_id"]
            == event_result_table["id"][event_row_number]
        )[0]
        state_colors = [
            app.state_colors[state["index"]]
            for state in state_result_table[state_row_numbers]
        ]
        for state_row_number in state_row_numbers:
            state = state_result_table[state_row_number]
            app.p3.plot(
                [app.perfiledata.getT(state["global_startpt"])],
                [event_seg_filt[state["local_startpt"]]],
                symbol="t",
                symbolBrush="m",
                symbolSize=10,
            )
            app.p3.plot(
                [app.perfiledata.getT(state["global_endpt"])],
                [event_seg_filt[state["local_endpt"]]],
                symbol="t",
                symbolBrush="m",
                symbolSize=10,
            )
            x = (
                app.perfiledata.getT(state["global_startpt"]),
                app.perfiledata.getT(state["global_endpt"]),
            )
            y = (state["mean"], state["mean"])
            app.p3.plot(
                x, y, pen=pg.mkPen(color=app.state_colors[state["index"]], width=4)
            )

        # Annotate subevent state plot
        state_frac_list = state_result_table["frac"][state_row_numbers]
        state_stdev_list = state_result_table["stdev"][state_row_numbers]
        state_skew_list = state_result_table["skewness"][state_row_numbers]
        state_kurt_list = state_result_table["kurtosis"][state_row_numbers]
        log_state_dwell_list = np.log10(state_result_table["dwell"][state_row_numbers])
        app.p2["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_frac_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("y", width=2),
            size=12,
        )
        app.p2["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_frac_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("k", width=2),
            size=8,
        )
        app.p2["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_frac_list,
            symbol="t",
            brush=state_colors,
            size=6,
        )
        app.p2std["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_stdev_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("y", width=2),
            size=12,
        )
        app.p2std["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_stdev_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("k", width=2),
            size=8,
        )
        app.p2std["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_stdev_list,
            symbol="t",
            brush=state_colors,
            size=6,
        )
        app.p2skew["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_skew_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("y", width=2),
            size=12,
        )
        app.p2skew["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_skew_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("k", width=2),
            size=8,
        )
        app.p2skew["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_skew_list,
            symbol="t",
            brush=state_colors,
            size=6,
        )
        app.p2kurt["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_kurt_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("y", width=2),
            size=12,
        )
        app.p2kurt["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_kurt_list,
            symbol="t",
            brush=None,
            pen=pg.mkPen("k", width=2),
            size=8,
        )
        app.p2kurt["annotations"].addPoints(
            x=log_state_dwell_list,
            y=state_kurt_list,
            symbol="t",
            brush=state_colors,
            size=6,
        )


def scatterClicked(app: BaseAppMainWindow, plot, points):
    for entry in app.scatter_entries:
        for p in app.p2s:
            if plot is p[entry]:
                clickedentry = entry
                break
    clickedindex = points[0].index()

    # Track selected event id for deletion
    if clickedentry == "events":
        event_table = app.perfiledata.analysis_results.tables.get("Event")
        if event_table is not None and clickedindex < len(event_table):
            app.perfiledata.selected_event_id = event_table[clickedindex]["id"]
            app.printlog(f"Selected event id: {app.perfiledata.selected_event_id}")
    elif clickedentry == "cusum_states":
        # For CUSUM states, select the parent event
        state_table = app.perfiledata.analysis_results.tables.get("CUSUMState")
        if state_table is not None and clickedindex < len(state_table):
            app.perfiledata.selected_event_id = state_table[clickedindex]["parent_id"]
            app.printlog(f"Selected parent event id: {app.perfiledata.selected_event_id}")

    inspectEvent(app, clickedentry, clickedindex)


def inspectRange(app: BaseAppMainWindow, grange: tuple[int]):
    range_data = app.perfiledata.data.getConcatDataPoints(
        grange, rawdata=False, gap_filler=np.nan
    )
    app.p3.clear()
    app.p3.setLabel("bottom", text="Time", units="s")
    app.p3.setLabel("left", text="Current", units="A")
    app.p3.plot(app.perfiledata.getT(range(grange[0], grange[1])), range_data, pen="b")
    app.p3.autoRange()


def inspectSelection(app: BaseAppMainWindow):
    """
    Inspect the selected linear region.
    """
    if len(app.perfiledata.LRs) > 0:
        selected_region = app.perfiledata.LRs[-1]
        region = selected_region.getRegion()
        app.printlog(f"Inspecting selected region: {region}")
        start, end = int(region[0]), int(region[1])
        grange = (start, end)
        inspectRange(app, grange)
