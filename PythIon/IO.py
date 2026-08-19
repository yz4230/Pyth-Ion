# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
import yaml
import csv
import json

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


from .__version__ import __version__
from .BaseApp import *
from .ui.loadfile import *
from .ui.exportevents import *
from .ui.exporttraceselection import *


class LoadConfig:
    def __init__(self):
        self.datafilepath = ""
        self.ADC_samplerate_kHz = 250
        self.LPFilter_cutoff_kHz = 100


class LoadFileDialog(QtWidgets.QDialog):
    def __init__(self, parent: BaseAppMainWindow = None):
        self.parent = parent
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.ui = Ui_LoadFileDialog()
        self.ui.setupUi(self)
        self.accepted.connect(self.dialogAccept)
        self.rejected.connect(self.dialogReject)
        self.ui.pushButton_Browse.clicked.connect(self.browseFile)

        load_config: LoadConfig = parent.load_config
        self.ui.plainTextEdit_DataFilePath.setPlainText(load_config.datafilepath)
        self.ui.lineEdit_ADC_Samplerate.setText(str(load_config.ADC_samplerate_kHz))
        self.ui.lineEdit_LPFilter.setText(str(load_config.LPFilter_cutoff_kHz))

    def browseFile(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", "", "Data Files (*.opt *.bin *.txt *.tracedata)"
        )
        if file[0]:
            self.ui.plainTextEdit_DataFilePath.setPlainText(file[0])

    def dialogAccept(self):
        load_config: LoadConfig = self.parent.load_config
        load_config.datafilepath = self.ui.plainTextEdit_DataFilePath.toPlainText()
        load_config.ADC_samplerate_kHz = float(self.ui.lineEdit_ADC_Samplerate.text())
        load_config.LPFilter_cutoff_kHz = float(self.ui.lineEdit_LPFilter.text())
        if os.path.exists(load_config.datafilepath):
            loadFile(self.parent)
        self.close()

    def dialogReject(self):
        self.close()


def _load_text_trace(path):
    data = np.loadtxt(path, ndmin=1)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Text traces must contain one finite pA value per line")
    return data * 1e-12


def loadFile(app: BaseAppMainWindow, loadandplot=True):
    def tryLoadXml(xml_file_path):
        if os.path.isfile(xml_file_path):
            app.printlog(f"Found xml auxiliary file {xml_file_path!s}")
            app.perfiledata.xmltree = ET.parse(xml_file_path)
            app.perfiledata.xmlroot = app.perfiledata.xmltree.getroot()

            voltage_timestamps = app.perfiledata.xmlroot.findall("timestamp/voltage/..")
            t_V_record = np.full(
                len(voltage_timestamps),
                -1,
                dtype=[("msec", "int64"), ("mV", "float64")],
            )
            for k, elem in enumerate(voltage_timestamps):
                msec = int(elem.get("msec"))
                mV = float(elem.find("voltage").get("volt"))
                t_V_record[k] = (msec, mV)
            app.perfiledata.t_V_record = t_V_record
            app.printlog(f"read {len(voltage_timestamps):d} voltage records")

            usernote_timestamps = app.perfiledata.xmlroot.findall(
                "timestamp/usernote/.."
            )
            usernote_record = []
            for k, elem in enumerate(usernote_timestamps):
                msec = int(elem.get("msec"))
                usernote_text = elem.find("usernote").text
                usernote_record.append((msec, usernote_text))
            app.perfiledata.usernote_record = usernote_record
            app.printlog(f"read {len(usernote_record):d} user notes")

    with app.awaitresponse:
        load_config: LoadConfig = app.load_config
        app.perfiledata = FileData()
        app.perfiledata.datafilename = load_config.datafilepath
        app.clearPerFileDisplays()
        app.ui.filelabel.setText(app.perfiledata.datafilename)
        app.printlog(app.perfiledata.datafilename)

        datafilebase, datafileext = os.path.splitext(app.perfiledata.datafilename)
        app.perfiledata.matfilename = datafilebase
        datafilename_head, datafilename_tail = os.path.split(
            app.perfiledata.datafilename
        )

        app.perfiledata.LPFilter_cutoff_Hz = load_config.LPFilter_cutoff_kHz * 1e3
        app.perfiledata.ADC_samplerate_Hz = (
            load_config.ADC_samplerate_kHz * 1e3
        )  # use integer multiples of 4166.67 ie 2083.33 or 1041.67

        rawdata = None
        if datafileext == ".opt":
            rawdata = np.fromfile(app.perfiledata.datafilename, dtype=np.dtype(">d"))
            app.perfiledata.isFullTrace = True
            app.printlog("opt loaded")
        elif datafileext == ".bin":
            rawdata = np.fromfile(app.perfiledata.datafilename, dtype=np.dtype("<d"))
            app.printlog("bin loaded")
        elif datafileext == ".txt":
            rawdata = _load_text_trace(app.perfiledata.datafilename)
            app.printlog("txt loaded (pA)")

        if rawdata is not None:
            if np.isfinite(app.perfiledata.LPFilter_cutoff_Hz):
                Wn = round(
                    app.perfiledata.LPFilter_cutoff_Hz
                    / (app.perfiledata.ADC_samplerate_Hz / 2),
                    4,
                )
                b, a = signal.bessel(4, Wn, btype="low")
                filtdata = signal.filtfilt(b, a, rawdata)
                app.printlog(
                    f"Data filtered at {app.perfiledata.LPFilter_cutoff_Hz:.0f} Hz"
                )
            else:
                filtdata = rawdata
                app.printlog(
                    "Filter value specified as no-filtering, data not filtered"
                )
            app.printlog(f"Read data size: {rawdata.shape!s}")
            app.perfiledata.data.setOriginalData(rawdata, filtdata, datafilename_tail)

            if datafileext == ".opt":
                tryLoadXml(datafilebase + ".xml")

        elif datafileext == ".tracedata":
            with open(app.perfiledata.datafilename, "rb") as dataf:
                tracedata: TraceData = pickle.load(dataf)
            app.perfiledata.data = tracedata
            app.printlog(".tracedata loaded")
            app.printlog(
                f"trace data created by PythIon version {tracedata.pythion_version:s} of {tracedata.Nseg:d} segments loaded. The original data source was {tracedata.source_file_name:s}"
            )
            source_data_file_base = os.path.splitext(tracedata.source_file_name)[0]
            xml_file_path = os.path.join(
                datafilename_head, source_data_file_base + ".xml"
            )
            tryLoadXml(xml_file_path)

        if app.perfiledata.hasbaselinebeenset == 0:
            app.ui_baseline = np.median(app.perfiledata.data.filt[0])
            app.ui_baseline_std = np.std(app.perfiledata.data.filt[0])

        if loadandplot == True:
            app.paintCurrentTrace()
            app.p1.autoRange()
            app.p3.clear()
            # FIXME
            aphy, aphx = np.histogram(app.perfiledata.data.filt[0], bins=1000)
            aphhist = pg.PlotCurveItem(
                aphx, aphy, stepMode=True, fillLevel=0, brush="b"
            )
            app.p3.addItem(aphhist)
            app.p3.autoRange()
            app.p3.setXRange(
                np.min(app.perfiledata.data.filt[0]),
                np.max(app.perfiledata.data.filt[0]),
            )


class ExportEventsDialog(QtWidgets.QDialog):
    def __init__(self, parent: BaseAppMainWindow = None):
        self.parent = parent
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.ui = Ui_ExportEventsDialog()
        self.ui.setupUi(self)
        self.accepted.connect(self.dialogAccept)
        self.rejected.connect(self.dialogReject)
        self.event_result_table = parent.perfiledata.analysis_results.result_tables[
            "Event"
        ]

        self.totalNpoints = sum(
            [e["local_endpt"] - e["local_startpt"] for e in self.event_result_table]
        )

        self.csv_fmt = "+.18e"
        self.bin_fmt = np.dtype("<d")

    def updateSizeEstimate(self):
        x = 0
        csv_byte_per_point = len((f"{x:{self.csv_fmt}}\n").encode("utf-8"))
        csv_MiB = int(csv_byte_per_point * self.totalNpoints) >> 20
        self.ui.label_csv_description.setText(f"{csv_MiB:d} MiB per trace")

        bin_byte_per_point = self.bin_fmt.itemsize
        bin_MiB = int(bin_byte_per_point * self.totalNpoints) >> 20
        self.ui.label_bin_description.setText(f"{bin_MiB:d} MiB per trace")


def exportEvents(app: BaseAppMainWindow):
    pass


def exportEventPointsCSV(app: BaseAppMainWindow):
    """Export per-event start/end points (option C) and event features to a CSV.

    Start/end points correspond to the green/red markers in Painting.plotAnalysis:
    - start: (global_startpt, seg_filt[local_startpt])
    - end:   (global_endpt,   seg_filt[local_endpt])
    """

    analysis_results = getattr(app.perfiledata, "analysis_results", None)
    if analysis_results is None:
        app.printlog("No analysis results. Run Analyze first.")
        return

    event_table = analysis_results.tables.get("Event")
    if event_table is None or len(event_table) == 0:
        app.printlog('No "Event" table to export. Run Analyze first.')
        return

    samplerate = app.perfiledata.ADC_samplerate_Hz
    if samplerate is None or not np.isfinite(samplerate) or samplerate <= 0:
        app.printlog("Invalid samplerate; cannot compute event times.")
        return

    timestamp = app.getSaveTimeStamp()
    default_path = app.perfiledata.matfilename + f"_{timestamp}_event_points.csv"
    save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        app,
        "Export Event Points (CSV)",
        default_path,
        "CSV (*.csv)",
    )
    if not save_path:
        return

    analysis_config = analysis_results.analysis_config

    def fmt_float(x: float) -> str:
        try:
            if x is None or not np.isfinite(x):
                return ""
            return f"{float(x):.18e}"
        except Exception:
            return ""

    def fmt_int(x: int) -> str:
        try:
            return str(int(x))
        except Exception:
            return ""

    header = [
        "source_file_name",
        "ADC_samplerate_Hz",
        "LPFilter_cutoff_Hz",
        "baseline_A",
        "baseline_std_A",
        "threshold_A",
        "event_id",
        "event_index",
        "seg",
        "local_startpt",
        "local_endpt",
        "global_startpt",
        "global_endpt",
        "t_start_s",
        "t_end_s",
        "start_current_A_filt",
        "end_current_A_filt",
        "deli_A",
        "frac",
        "dwell_us",
        "dt_s",
        "mean_A",
        "stdev_A",
        "skewness",
        "kurtosis",
    ]

    source_file_name = getattr(app.perfiledata.data, "source_file_name", "") or ""

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ev in event_table:
            seg = int(ev["seg"])
            local_start = int(ev["local_startpt"])
            local_end = int(ev["local_endpt"])
            global_start = int(ev["global_startpt"])
            global_end = int(ev["global_endpt"])

            start_t_s = global_start / samplerate
            end_t_s = global_end / samplerate

            start_current = np.nan
            end_current = np.nan
            try:
                seg_filt = app.perfiledata.data.filt[seg]
                if 0 <= local_start < len(seg_filt):
                    start_current = float(seg_filt[local_start])
                if 0 <= local_end < len(seg_filt):
                    end_current = float(seg_filt[local_end])
            except Exception:
                pass

            row = [
                source_file_name,
                fmt_float(app.perfiledata.ADC_samplerate_Hz),
                fmt_float(app.perfiledata.LPFilter_cutoff_Hz),
                fmt_float(analysis_config.baseline_A),
                fmt_float(analysis_config.baseline_std_A),
                fmt_float(analysis_config.threshold_A),
                fmt_int(ev["id"]),
                fmt_int(ev["index"]),
                fmt_int(seg),
                fmt_int(local_start),
                fmt_int(local_end),
                fmt_int(global_start),
                fmt_int(global_end),
                fmt_float(start_t_s),
                fmt_float(end_t_s),
                fmt_float(start_current),
                fmt_float(end_current),
                fmt_float(ev["deli"]),
                fmt_float(ev["frac"]),
                fmt_float(ev["dwell"]),
                fmt_float(ev["dt"]),
                fmt_float(ev["mean"]),
                fmt_float(ev["stdev"]),
                fmt_float(ev["skewness"]),
                fmt_float(ev["kurtosis"]),
            ]
            writer.writerow(row)

    app.printlog(f"Exported {len(event_table):d} events to CSV: {save_path:s}")


def _json_number(value):
    try:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _json_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _csv_float(value):
    try:
        if value is None or not np.isfinite(value):
            return ""
        return f"{float(value):.18e}"
    except Exception:
        return ""


def _write_trace_csv(
    app: BaseAppMainWindow,
    path: str,
    *,
    global_start: int,
    global_end: int,
    samplerate_hz: float,
    event_global_start: int,
    subevent_global_start: int | None = None,
):
    raw_data = app.perfiledata.data.getConcatDataPoints(
        (global_start, global_end), rawdata=True, gap_filler=np.nan
    )
    filt_data = app.perfiledata.data.getConcatDataPoints(
        (global_start, global_end), rawdata=False, gap_filler=np.nan
    )
    sample_indexes = np.arange(global_start, global_end, dtype=np.int64)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if subevent_global_start is None:
            writer.writerow(["sample_index", "t_abs_s", "t_event_s", "raw_A", "filt_A"])
            for sample_index, raw, filt in zip(sample_indexes, raw_data, filt_data):
                writer.writerow(
                    [
                        int(sample_index),
                        _csv_float(sample_index / samplerate_hz),
                        _csv_float((sample_index - event_global_start) / samplerate_hz),
                        _csv_float(raw),
                        _csv_float(filt),
                    ]
                )
        else:
            writer.writerow(
                [
                    "sample_index",
                    "t_abs_s",
                    "t_parent_event_s",
                    "t_subevent_s",
                    "raw_A",
                    "filt_A",
                ]
            )
            for sample_index, raw, filt in zip(sample_indexes, raw_data, filt_data):
                writer.writerow(
                    [
                        int(sample_index),
                        _csv_float(sample_index / samplerate_hz),
                        _csv_float((sample_index - event_global_start) / samplerate_hz),
                        _csv_float(
                            (sample_index - subevent_global_start) / samplerate_hz
                        ),
                        _csv_float(raw),
                        _csv_float(filt),
                    ]
                )

    return filt_data


def _write_fft_csv(path: str, event_signal, samplerate_hz: float):
    event_signal = np.asarray(event_signal)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frequency_Hz", "real_A", "imag_A", "magnitude_A", "phase_rad"]
        )
        if event_signal.size == 0:
            return
        fft_values = np.fft.rfft(event_signal)
        frequencies = np.fft.rfftfreq(event_signal.size, d=1 / samplerate_hz)
        for frequency, fft_value in zip(frequencies, fft_values):
            writer.writerow(
                [
                    _csv_float(frequency),
                    _csv_float(np.real(fft_value)),
                    _csv_float(np.imag(fft_value)),
                    _csv_float(np.abs(fft_value)),
                    _csv_float(np.angle(fft_value)),
                ]
            )


def _export_dir_name(row_number: int) -> str:
    return f"{row_number + 1:05d}"


def _get_state_rows_for_event(state_table, event_id: int):
    if state_table is None or len(state_table) == 0:
        return []
    try:
        return list(state_table[state_table["parent_id"] == event_id])
    except Exception:
        return []


class _EventExportProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent, event_count: int):
        super().__init__(parent)
        self._canceled = False
        self.setWindowTitle("Export Event Package")
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("Exporting event package...", self)
        layout.addWidget(self.label)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, event_count)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def cancel(self):
        self._canceled = True
        self.cancel_button.setEnabled(False)
        self.label.setText("Canceling after current event...")

    def set_progress(
        self,
        completed_count: int,
        event_count: int,
        current_event_number: int | None = None,
    ):
        if current_event_number is not None:
            self.label.setText(
                f"Exporting event {current_event_number:d} of {event_count:d}..."
            )
        self.progress_bar.setMaximum(event_count)
        self.progress_bar.setValue(completed_count)

    def wasCanceled(self):
        return self._canceled


def _make_export_progress_dialog(app: BaseAppMainWindow, event_count: int):
    if QtWidgets.QApplication.instance() is None:
        return None
    progress = _EventExportProgressDialog(app, event_count)
    progress.show()
    progress.raise_()
    progress.activateWindow()
    QtWidgets.QApplication.instance().processEvents()
    return progress


def _update_export_progress(
    progress,
    completed_count: int,
    event_count: int,
    current_event_number: int | None = None,
):
    if progress is None:
        return False
    if current_event_number is not None:
        progress.set_progress(
            completed_count,
            event_count,
            current_event_number=current_event_number,
        )
    else:
        progress.set_progress(completed_count, event_count)
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()
    return progress.wasCanceled()


def exportEventPackage(app: BaseAppMainWindow, export_dir_path: str | None = None):
    """Export events as a directory tree for external analysis apps."""

    analysis_results = getattr(app.perfiledata, "analysis_results", None)
    if analysis_results is None:
        app.printlog("No analysis results. Run Analyze first.")
        return

    event_table = analysis_results.tables.get("Event")
    if event_table is None or len(event_table) == 0:
        app.printlog('No "Event" table to export. Run Analyze first.')
        return

    samplerate = app.perfiledata.ADC_samplerate_Hz
    if samplerate is None or not np.isfinite(samplerate) or samplerate <= 0:
        app.printlog("Invalid samplerate; cannot export event package.")
        return

    if export_dir_path is None:
        timestamp = app.getSaveTimeStamp()
        default_path = app.perfiledata.matfilename + f"_{timestamp}_pythion_export"
        export_dir_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            app,
            "Export Event Package",
            default_path,
            "Pyth-Ion Export Directory (*)",
        )
        if not export_dir_path:
            return

    if os.path.exists(export_dir_path):
        if not os.path.isdir(export_dir_path):
            app.printlog(
                f"Export path exists and is not a directory: {export_dir_path:s}"
            )
            return
        if os.listdir(export_dir_path):
            app.printlog(f"Export directory is not empty: {export_dir_path:s}")
            return
    else:
        os.makedirs(export_dir_path)

    events_dir = os.path.join(export_dir_path, "events")
    os.makedirs(events_dir, exist_ok=True)

    analysis_config = analysis_results.analysis_config
    metadata = {
        "source_file_name": getattr(app.perfiledata.data, "source_file_name", "") or "",
        "ADC_samplerate_Hz": _json_number(app.perfiledata.ADC_samplerate_Hz),
        "LPFilter_cutoff_Hz": _json_number(app.perfiledata.LPFilter_cutoff_Hz),
        "baseline_A": _json_number(analysis_config.baseline_A),
        "baseline_std_A": _json_number(analysis_config.baseline_std_A),
        "threshold_A": _json_number(analysis_config.threshold_A),
    }
    _write_json(os.path.join(export_dir_path, "metadata.json"), metadata)

    segments = []
    for seg_index, seg_range in enumerate(app.perfiledata.data.srange):
        segments.append(
            {
                "seg": int(seg_index),
                "start": _json_int(seg_range[0]),
                "end": _json_int(seg_range[1]),
            }
        )
    _write_json(os.path.join(export_dir_path, "segments.json"), segments)

    state_table = analysis_results.tables.get("CUSUMState")
    events_index = []
    event_count = len(event_table)
    progress = _make_export_progress_dialog(app, event_count)

    for event_row_number, event in enumerate(event_table):
        if _update_export_progress(
            progress,
            event_row_number,
            event_count,
            current_event_number=event_row_number + 1,
        ):
            _write_json(os.path.join(events_dir, "index.json"), events_index)
            progress.close()
            app.printlog(
                "Event package export canceled. "
                f"Partial export remains at: {export_dir_path:s}"
            )
            return

        internal_event_id = int(event["id"])
        event_index = int(event["index"])
        event_dir_name = _export_dir_name(event_row_number)
        event_dir = os.path.join(events_dir, event_dir_name)
        subevents_dir = os.path.join(event_dir, "subevents")
        os.makedirs(subevents_dir, exist_ok=True)

        local_start = int(event["local_startpt"])
        local_end = int(event["local_endpt"])
        global_start = int(event["global_startpt"])
        global_end = int(event["global_endpt"])

        event_meta = {
            "event_index": event_index,
            "seg": int(event["seg"]),
            "local_startpt": local_start,
            "local_endpt": local_end,
            "global_startpt": global_start,
            "global_endpt": global_end,
            "t_start_s": _json_number(global_start / samplerate),
            "t_end_s": _json_number(global_end / samplerate),
            "dwell_us": _json_number(event["dwell"]),
            "deli_A": _json_number(event["deli"]),
            "frac": _json_number(event["frac"]),
        }
        _write_json(os.path.join(event_dir, "meta.json"), event_meta)

        event_signal = _write_trace_csv(
            app,
            os.path.join(event_dir, "trace.csv"),
            global_start=global_start,
            global_end=global_end,
            samplerate_hz=samplerate,
            event_global_start=global_start,
        )
        _write_fft_csv(os.path.join(event_dir, "fft.csv"), event_signal, samplerate)

        events_index.append(
            {
                "event_index": event_index,
                "path": event_dir_name,
            }
        )

        subevents_index = []
        for state_row_number, state in enumerate(
            _get_state_rows_for_event(state_table, internal_event_id)
        ):
            state_id = int(state["id"])
            state_dir_name = _export_dir_name(state_row_number)
            state_dir = os.path.join(subevents_dir, state_dir_name)
            os.makedirs(state_dir, exist_ok=True)

            state_global_start = int(state["global_startpt"])
            state_global_end = int(state["global_endpt"])

            state_meta = {
                "state_id": state_id,
                "state_index": int(state["index"]),
                "seg": int(state["seg"]),
                "local_startpt": int(state["local_startpt"]),
                "local_endpt": int(state["local_endpt"]),
                "global_startpt": state_global_start,
                "global_endpt": state_global_end,
                "t_start_s": _json_number(state_global_start / samplerate),
                "t_end_s": _json_number(state_global_end / samplerate),
                "dwell_us": _json_number(state["dwell"]),
                "deli_A": _json_number(state["deli"]),
                "frac": _json_number(state["frac"]),
            }
            _write_json(os.path.join(state_dir, "meta.json"), state_meta)

            _write_trace_csv(
                app,
                os.path.join(state_dir, "trace.csv"),
                global_start=state_global_start,
                global_end=state_global_end,
                samplerate_hz=samplerate,
                event_global_start=global_start,
                subevent_global_start=state_global_start,
            )

            subevents_index.append(
                {
                    "state_id": state_id,
                    "state_index": int(state["index"]),
                    "path": state_dir_name,
                }
            )

        _write_json(os.path.join(subevents_dir, "index.json"), subevents_index)
        if _update_export_progress(progress, event_row_number + 1, event_count):
            _write_json(os.path.join(events_dir, "index.json"), events_index)
            progress.close()
            app.printlog(
                "Event package export canceled. "
                f"Partial export remains at: {export_dir_path:s}"
            )
            return

    _write_json(os.path.join(events_dir, "index.json"), events_index)
    if progress is not None:
        progress.set_progress(event_count, event_count)
        QtWidgets.QApplication.instance().processEvents()
        progress.close()
    app.printlog(
        f"Exported {len(event_table):d} events to package: {export_dir_path:s}"
    )


class ExportTraceSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        self.parent: BaseAppMainWindow = parent
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.ui = Ui_ExportTraceSelectionDialog()
        self.ui.setupUi(self)
        self.accepted.connect(self.dialogAccept)
        self.rejected.connect(self.dialogReject)
        print("ExportTraceSelectionDialog init")
        self.csv_fmt = "+.18e"
        self.bin_fmt = np.dtype("<d")
        self.selection_N_points = [
            ((lambda x: x[1] - x[0])(lr.getRegion()))
            for lr in self.parent.perfiledata.LRs
        ]
        self.afterDS_points = None
        self.DSRatio = 1
        self.afterDS_samplerate = 0
        self.updateDSRatio()
        self.ui.horizontalSlider_ratio.valueChanged.connect(self.updateDSRatio)

    def updateDSRatio(self):
        self.DSRatio = 1 << int(self.ui.horizontalSlider_ratio.value())
        self.ui.label_cur_ratio.setText(f"{self.DSRatio:d}X")
        self.afterDS_samplerate = (
            self.parent.perfiledata.ADC_samplerate_Hz / self.DSRatio
        )
        self.afterDS_points = sum(
            int(np.ceil(n / self.DSRatio)) for n in self.selection_N_points
        )
        self.ui.label_cur_samplerate.setText(f"{self.afterDS_samplerate:.4f} Hz")
        self.ui.label_cur_Npoints.setText(f"{self.afterDS_points:d}")

        x = 0
        csv_byte_per_point = len((f"{x:{self.csv_fmt}}\n").encode("utf-8"))
        csv_MiB = int(csv_byte_per_point * self.afterDS_points) >> 20
        self.ui.label_csv_description.setText(f"{csv_MiB:d} MiB per trace")

        bin_byte_per_point = self.bin_fmt.itemsize
        bin_MiB = int(bin_byte_per_point * self.afterDS_points) >> 20
        self.ui.label_bin_description.setText(f"{bin_MiB:d} MiB per trace")

    def peakDS(self, data):
        DSRatio = self.DSRatio
        pack_size = DSRatio * 2
        n = len(data)
        N_padded = int(np.ceil(n / pack_size) * pack_size)
        N_to_pad = N_padded - n
        data = np.pad(data, pad_width=(0, N_to_pad), constant_values=np.nan)
        data = np.reshape(data, (-1, pack_size))
        mins = np.nanmin(data, axis=1)
        maxs = np.nanmax(data, axis=1)
        return np.ravel(np.transpose([mins, maxs]))

    def meanDS(self, data):
        DSRatio = self.DSRatio
        pack_size = DSRatio
        n = len(data)
        N_padded = int(np.ceil(n / pack_size) * pack_size)
        N_to_pad = N_padded - n
        data = np.pad(data, pad_width=(0, N_to_pad), constant_values=np.nan)
        data = np.reshape(data, (-1, pack_size))
        return np.nanmean(data, axis=1)

    def subsamplingDS(self, data):
        DSRatio = self.DSRatio
        pack_size = DSRatio
        return data[::pack_size]

    def dialogAccept(self):
        if_save_raw = self.ui.checkBox_raw.isChecked()
        if_save_filt = self.ui.checkBox_filt.isChecked()
        if_exp_bin = self.ui.checkBox_exp_bin.isChecked()
        if_exp_csv = self.ui.checkBox_exp_csv.isChecked()
        if_exp_png = self.ui.checkBox_exp_png.isChecked()
        if any((if_save_filt, if_save_raw)) and any(
            (if_exp_bin, if_exp_csv, if_exp_png)
        ):
            timestamp = self.parent.getSaveTimeStamp()
            export_dir_path = (
                self.parent.perfiledata.matfilename + "_export_" + timestamp
            )
            try:
                os.mkdir(export_dir_path)
            except FileExistsError:
                pass
            for k, lr in enumerate(self.parent.perfiledata.LRs):
                sel_region = np.round(lr.getRegion()).astype(int)
                filt_data = self.parent.perfiledata.data.getConcatDataPoints(
                    sel_region, rawdata=False, gap_filler=np.nan
                )
                raw_data = self.parent.perfiledata.data.getConcatDataPoints(
                    sel_region, rawdata=True, gap_filler=np.nan
                )
                if self.ui.radioButton_peak.isChecked():
                    DS = self.peakDS
                    DS_string = "peak"
                elif self.ui.radioButton_mean.isChecked():
                    DS = self.meanDS
                    DS_string = "mean"
                elif self.ui.radioButton_subsampling.isChecked():
                    DS = self.subsamplingDS
                    DS_string = "subsampling"
                if if_save_filt:
                    DS_filt_data = DS(filt_data)
                if if_save_raw:
                    DS_raw_data = DS(raw_data)

                export_prefix = os.path.join(export_dir_path, f"selection_{k:d}")

                export_info = {
                    "time": timestamp,
                    "filename": self.parent.perfiledata.datafilename,
                    "source_data_filename": self.parent.perfiledata.data.source_file_name,
                    "selection_range": str(sel_region),
                    "export_filtered": if_save_filt,
                    "export_raw": if_save_raw,
                    "downsampling_ratio": self.DSRatio,
                    "downsampling_method": DS_string,
                    "original_samplerate": float(
                        self.parent.perfiledata.ADC_samplerate_Hz
                    ),
                    "downsampled_samplerate(Hz)": float(self.afterDS_samplerate),
                }

                export_info_string = yaml.dump(export_info, sort_keys=False)
                with open(export_prefix + "_info.yaml.txt", "w") as expf:
                    expf.write(export_info_string)

                if if_exp_bin:
                    if if_save_filt:
                        DS_filt_data.astype("<d").tofile(export_prefix + "_filt.bin")
                    if if_save_raw:
                        DS_raw_data.astype("<d").tofile(export_prefix + "_raw.bin")
                if if_exp_csv:
                    if if_save_filt:
                        np.savetxt(
                            export_prefix + "_filt.csv",
                            DS_filt_data,
                            fmt="%" + self.csv_fmt,
                        )
                    if if_save_raw:
                        np.savetxt(
                            export_prefix + "_raw.csv",
                            DS_raw_data,
                            fmt="%" + self.csv_fmt,
                        )
                if if_exp_png:

                    def plot_data(data):
                        fig = plt.figure(figsize=(5, 2), dpi=600)
                        ax = fig.add_subplot(111)
                        ax.plot(
                            np.arange(len(data)) / self.afterDS_samplerate,
                            1e12 * data,
                            "k-",
                        )
                        ax.set_xlabel("t(s)")
                        ax.set_ylabel("I(pA)")
                        fig.tight_layout()
                        return fig

                    if if_save_filt:
                        plot_data(DS_filt_data).savefig(export_prefix + "_filt.png")
                    if if_save_raw:
                        plot_data(DS_raw_data).savefig(export_prefix + "_raw.png")
            self.parent.printlog(
                f"Data in the selections exported to directory {export_dir_path:s}"
            )
        self.close()

    def dialogReject(self):
        self.close()


def exportSelection(app: BaseAppMainWindow):
    if len(app.perfiledata.LRs) > 0:
        export_dialog = ExportTraceSelectionDialog(parent=app)
        export_dialog.exec()


def saveSegInfo(app: BaseAppMainWindow):
    save_dtype = np.dtype(
        [
            ("start", int),
            ("end", int),
            ("start_sec", float),
            ("end_sec", float),
            ("duration_sec", float),
        ]
    )
    Nseg = app.perfiledata.data.Nseg
    srange = app.perfiledata.data.srange
    save_table = np.full(Nseg, -1, dtype=save_dtype)
    for kseg in range(Nseg):
        seg = srange[kseg]
        save_table["start"][kseg] = seg[0]
        save_table["end"][kseg] = seg[1]
        save_table["start_sec"][kseg] = seg[0] / app.perfiledata.ADC_samplerate_Hz
        save_table["end_sec"][kseg] = seg[1] / app.perfiledata.ADC_samplerate_Hz
        save_table["duration_sec"][kseg] = (
            seg[1] - seg[0]
        ) / app.perfiledata.ADC_samplerate_Hz
    header = f"""file: "{app.perfiledata.datafilename:s}"
    source_data_file: "{app.perfiledata.data.source_file_name:s}" 
    """
    header += "\t".join(save_table.dtype.names)
    timestamp = app.getSaveTimeStamp()
    save_path = app.perfiledata.matfilename + "_" + timestamp + "_segments.txt"
    np.savetxt(save_path, save_table, delimiter="\t", header=header)
    app.printlog(f"Segment information saved to {save_path:s}")


def saveTrace(app: BaseAppMainWindow):
    timestamp = app.getSaveTimeStamp()
    tracedata_savename = app.perfiledata.matfilename + "_" + timestamp + ".tracedata"
    with open(tracedata_savename, "wb") as outf:
        pickle.dump(app.perfiledata.data, outf)

    app.printlog(f"Trace data saved to...\n{tracedata_savename!s}\n")
    savelog(app)
    # TODO
    # save processing information etc.


def savelog(app: BaseAppMainWindow, logfilepath=None):
    timestamp = app.getSaveTimeStamp()
    if logfilepath is None:
        logfilepath = app.perfiledata.matfilename + "_" + timestamp + "_log.txt"
    app.printlog(f"saving PythIon log to {logfilepath!s}")
    app.printlog(f"... PythIon version {__version__!s} ...")
    with open(logfilepath, "w") as logfile:
        logfile.write(app.perfiledata.logtext)


def saveAnalysis(app: BaseAppMainWindow):
    timestamp = app.getSaveTimeStamp()
    savedir = app.perfiledata.matfilename + "_" + timestamp + "_analysis"
    app.printlog(f"trying to save analysis results to folder {savedir:s}")
    try:
        os.mkdir(savedir)
        save_prefix = os.path.join(
            savedir, os.path.basename(app.perfiledata.matfilename) + "_" + timestamp
        )

        table_dir = os.path.join(savedir, "tables")
        os.mkdir(table_dir)
        for table_key in app.perfiledata.analysis_results.tables.keys():
            table = app.perfiledata.analysis_results.tables[table_key]
            savename_table = os.path.join(table_dir, f"{table_key:s}.txt")
            save_format = [
                spec[2] for spec in app.perfiledata.analysis_results.result_spec
            ]
            np.savetxt(
                savename_table,
                table,
                fmt=save_format,
                delimiter="\t",
                header="\t".join(table.dtype.names),
            )
            app.printlog(f'Table "{table_key:s}" saved to {savename_table!s}')
        if app.perfiledata.analysis_results.result_tables.has_key("Event"):
            event_table = app.perfiledata.analysis_results.result_tables["Event"]
            legacy_columns = ["deli", "frac", "dwell", "dt"]
            legacy_event_table = event_table[legacy_columns]
            savename_legacy_event_table = os.path.join(table_dir, "Event_LegacyDB.txt")
            save_format = "%.18e"
            np.savetxt(
                savename_legacy_event_table,
                legacy_event_table,
                fmt=save_format,
                delimiter="\t",
                header="\t".join(legacy_event_table.dtype.names),
            )
            app.printlog(f"Legacy event table saved to {savename_legacy_event_table!s}")

        def exportFig(fig, save_name):
            exporter = pg.exporters.ImageExporter(fig)
            exporter.parameters()["width"] = 4000
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()
            exporter.export(save_name)

        fig_save_names = [
            save_prefix + "_" + nm + ".png" for nm in ("w2", "w3", "w4", "w5")
        ]
        figs_to_save = [app.w2, app.w3, app.w4, app.w5]
        fig_tab_pages = [
            app.ui.frachisttab,
            app.ui.delitab,
            app.ui.dwelltab,
            app.ui.dttab,
        ]
        for fig, save_name, tab_page in zip(
            figs_to_save, fig_save_names, fig_tab_pages
        ):
            app.ui.tabWidget.setCurrentWidget(tab_page)
            exportFig(fig, save_name)

        app.ui.tabWidget.setCurrentWidget(app.ui.scattertab)
        fig_save_names = [
            save_prefix + "_" + nm + ".png"
            for nm in ("w1", "w1std", "w1skew", "w1kurt")
        ]
        figs_to_save = [app.w1, app.w1std, app.w1skew, app.w1kurt]
        fig_tab_pages = [
            app.ui.blockadetab,
            app.ui.stdevtab,
            app.ui.skewnesstab,
            app.ui.kurtosistab,
        ]
        for fig, save_name, tab_page in zip(
            figs_to_save, fig_save_names, fig_tab_pages
        ):
            app.ui.tabWidget_2.setCurrentWidget(tab_page)
            exportFig(fig, save_name)
        app.ui.tabWidget_2.setCurrentWidget(app.ui.blockadetab)

    except Exception as e:
        app.printlog(str(e))

    savelog(app, logfilepath=os.path.join(savedir, "log.txt"))
    savelog(app)
