from .BaseApp import *


def doCut(app: BaseAppMainWindow):
    with app.awaitresponse:
        if len(app.perfiledata.LRs) > 0:
            app.printlog("Executing batch cutting...")
            app.perfiledata.isFullTrace = False
            cutString = ""
            # cutMask = np.zeros_like(app.perfiledata.data.filt, bool)
            # dummy_cutregion = None
            for cutLR in app.perfiledata.LRs:
                cutRegion = np.round(cutLR.getRegion()).astype(int)
                cutIndexL, cutIndexR = cutRegion
                # dummy_cutregion = cutRegion
                # cutIndexL = max(0, int(cutRegion[0]))
                # cutIndexR = min(len(app.perfiledata.data.filt), int(cutRegion[1]))
                cutString += f" {cutRegion!s}"
                # cutMask[cutIndexL:cutIndexR] = True

                app.perfiledata.data.trim(cutRegion)
                # app.perfiledata.data.trim(np.nonzero(cutMask)[0])
            app.printlog(f"Regions: {cutString:s}")
            app.perfiledata.LRs.clear()

            if app.perfiledata.hasbaselinebeenset == 0:
                app.ui_baseline = np.median(app.perfiledata.data.filt[0])
                app.ui_baseline_std = np.std(app.perfiledata.data.filt[0])

            app.paintCurrentTrace()
            app.p3.clear()
            aphy, aphx = np.histogram(app.perfiledata.data.filt[0], bins=1000)
            aphhist = pg.BarGraphItem(
                height=aphy, x0=aphx[:-1], x1=aphx[1:], brush="b", pen=None
            )
            app.p3.addItem(aphhist)
            app.p3.setXRange(
                np.min(app.perfiledata.data.filt[0]),
                np.max(app.perfiledata.data.filt[0]),
            )

            app.printlog("Cut done.")


def doBaseline(app: BaseAppMainWindow):
    with app.awaitresponse:
        trace_data = app.perfiledata.data
        original_length = trace_data.original_length
        if original_length is None or original_length <= 0:
            app.printlog("No trace loaded. Load data before setting baseline.")
            return

        if len(app.perfiledata.LRs) > 0:
            baseline_lr = app.perfiledata.LRs[-1]
            calcregion = np.sort(np.round(baseline_lr.getRegion()).astype(int))
            calcregion = np.clip(calcregion, 0, original_length)
            start_ndx = int(calcregion[0])
            end_ndx = int(calcregion[1])
            if start_ndx >= original_length:
                start_ndx = max(original_length - 1, 0)
            if end_ndx <= start_ndx:
                end_ndx = min(start_ndx + 1, original_length)
            calcregion = np.array([start_ndx, end_ndx], dtype=int)

            filt_data = trace_data.getConcatDataPoints(calcregion)
            if filt_data.size == 0:
                app.printlog(
                    "Selected baseline region contains no data. Adjust the region and try again."
                )
                return
            app.ui_baseline = np.median(filt_data)
            app.ui_baseline_std = np.std(filt_data)
            # app.baseline=np.median(app.perfiledata.data.filt[np.arange(int(calcregion[0]),int(calcregion[1]))])
            # app.var=np.std(app.perfiledata.data.filt[np.arange(int(calcregion[0]),int(calcregion[1]))])
            app.clearSelections()
            app.paintCurrentTrace()
            app.perfiledata.hasbaselinebeenset = 1
            app.awaiting_baseline_click = False
            app.printlog(
                f"Baseline measured on {calcregion!s}. Baseline is {app.ui_baseline * 1e9:.4f} nA. Stdev is {app.ui_baseline_std * 1e9:.5f} nA"
            )
            return

        app.awaiting_baseline_click = True
        app.printlog(
            "No selection found. Left-click on the current trace to set baseline."
        )


def _estimateBaselineStdNearClick(
    app: BaseAppMainWindow, center_ndx: int, trace_length: int
) -> float:
    samplerate = app.perfiledata.ADC_samplerate_Hz
    if samplerate is not None and np.isfinite(samplerate) and samplerate > 0:
        half_window = max(int(np.round(0.5e-3 * samplerate)), 1)
    else:
        half_window = 32
    local_region = np.array(
        [center_ndx - half_window, center_ndx + half_window + 1], dtype=int
    )
    local_region = np.clip(local_region, 0, trace_length)
    local_filt_data = app.perfiledata.data.getConcatDataPoints(local_region)
    if local_filt_data.size > 1:
        return float(np.std(local_filt_data))

    first_seg = app.perfiledata.data.filt[0]
    if len(first_seg) > 1:
        return float(np.std(first_seg))
    return 0.0


def handleSignalPlotClickForBaseline(app: BaseAppMainWindow, click_event):
    # Check for Ctrl+Click (immediate baseline set without awaiting_baseline_click)
    is_ctrl_click = click_event.modifiers() & QtCore.Qt.ControlModifier

    if not app.awaiting_baseline_click and not is_ctrl_click:
        return
    if click_event.button() != QtCore.Qt.LeftButton:
        return

    trace_data = app.perfiledata.data
    original_length = trace_data.original_length
    if original_length is None or original_length <= 0:
        app.awaiting_baseline_click = False
        app.printlog("No trace loaded. Load data before setting baseline.")
        return

    view_box = app.p1.vb
    if not view_box.sceneBoundingRect().contains(click_event.scenePos()):
        return

    mouse_point = view_box.mapSceneToView(click_event.scenePos())
    clicked_ndx = int(np.round(mouse_point.x()))
    clicked_ndx = int(np.clip(clicked_ndx, 0, original_length - 1))

    baseline = float(mouse_point.y() / app.current_display_scale_factor)
    baseline_std = _estimateBaselineStdNearClick(app, clicked_ndx, original_length)
    app.ui_baseline = baseline
    app.ui_baseline_std = baseline_std
    app.perfiledata.hasbaselinebeenset = 1
    app.awaiting_baseline_click = False
    app.paintCurrentTrace()

    samplerate = app.perfiledata.ADC_samplerate_Hz
    click_method = "Ctrl+click" if is_ctrl_click else "click"
    if samplerate is not None and np.isfinite(samplerate) and samplerate > 0:
        clicked_time_ms = clicked_ndx * 1e3 / samplerate
        app.printlog(
            f"Baseline set from {click_method} at {clicked_time_ms:.3f} ms. Baseline is {baseline * 1e9:.4f} nA. Stdev is {baseline_std * 1e9:.5f} nA"
        )
    else:
        app.printlog(
            f"Baseline set from {click_method}. Baseline is {baseline * 1e9:.4f} nA. Stdev is {baseline_std * 1e9:.5f} nA"
        )
    click_event.accept()


def handleBaselineDrag(app: BaseAppMainWindow, baseline_line):
    """Handle baseline line drag completion."""
    dscl = app.current_display_scale_factor
    new_baseline = baseline_line.value() / dscl
    app.ui_baseline = new_baseline
    app.perfiledata.hasbaselinebeenset = 1

    # Update std deviation lines position
    if app.perfiledata.baselineStdHandles:
        std_upper, std_lower = app.perfiledata.baselineStdHandles
        std_upper.setValue((new_baseline + app.ui_baseline_std) * dscl)
        std_lower.setValue((new_baseline - app.ui_baseline_std) * dscl)

    app.printlog(f"Baseline set from drag. Baseline is {new_baseline * 1e9:.4f} nA")


def invertData(app: BaseAppMainWindow):
    app.perfiledata.data.invert()
    if app.perfiledata.hasbaselinebeenset == 0:
        app.ui_baseline = np.median(app.perfiledata.data.filt[0])
        app.ui_baseline_std = np.std(app.perfiledata.data.filt[0])
    app.paintCurrentTrace()
    app.printlog("Data inverted")


def deleteSelectedEvent(app: BaseAppMainWindow):
    """Delete the currently selected event from scatter plot."""
    from . import Painting

    event_id = app.perfiledata.selected_event_id
    if event_id is None:
        app.printlog("No event selected for deletion")
        return

    if app.perfiledata.analysis_results is None:
        app.printlog("No analysis results available")
        return

    event_table = app.perfiledata.analysis_results.tables.get("Event")
    if event_table is None or len(event_table) == 0:
        app.printlog("No events in analysis results")
        return

    # Find and remove the event with the matching id
    event_mask = event_table["id"] != event_id
    if np.all(event_mask):
        app.printlog(f"Event id {event_id} not found")
        return

    # Remove the event
    app.perfiledata.analysis_results.tables["Event"] = event_table[event_mask]

    # Also remove related CUSUM states if they exist
    state_table = app.perfiledata.analysis_results.tables.get("CUSUMState")
    if state_table is not None and len(state_table) > 0:
        state_mask = state_table["parent_id"] != event_id
        app.perfiledata.analysis_results.tables["CUSUMState"] = state_table[state_mask]

    app.printlog(f"Deleted event id {event_id}")

    # Clear selection
    app.perfiledata.selected_event_id = None

    # Repaint analysis plots
    Painting.plotAnalysis(app)
