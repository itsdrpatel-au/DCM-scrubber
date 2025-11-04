from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .controller import run_batch
from .logging_setup import setup_logging


class LogView(QtWidgets.QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)

    @QtCore.Slot(str, str)
    def append_log(self, level: str, message: str) -> None:
        color = {"error": "#d33", "warn": "#c90"}.get(level, "#333")
        self.append(f'<span style="color:{color}">{message}</span>')


class FolderList(QtWidgets.QListWidget):
    def add_folder(self, path: Path) -> None:
        if str(path) not in [self.item(i).text() for i in range(self.count())]:
            self.addItem(str(path))

    def selected_paths(self) -> list[Path]:
        return [Path(i.text()) for i in self.selectedItems()]

    def all_paths(self) -> list[Path]:
        return [Path(self.item(i).text()) for i in range(self.count())]


class OptionsPanel(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QFormLayout(self)

        self.recurse = QtWidgets.QCheckBox()
        self.recurse.setChecked(True)

        self.rows = QtWidgets.QSpinBox()
        self.rows.setRange(0, 5000)
        self.rows.setValue(75)

        self.workers = QtWidgets.QSpinBox()
        self.workers.setRange(0, 256)
        self.workers.setValue(os.cpu_count() or 4)

        self.dry_run = QtWidgets.QCheckBox()
        self.dry_run.setChecked(False)

        self.export_png = QtWidgets.QCheckBox()
        self.export_png.setChecked(True)

        self.pii_ocr = QtWidgets.QCheckBox()
        self.pii_ocr.setChecked(True)

        self.skip_first = QtWidgets.QCheckBox()
        self.skip_first.setChecked(True)

        self.output = QtWidgets.QLineEdit()
        self.output.setReadOnly(True)
        self.pick_output_btn = QtWidgets.QPushButton("Choose…")

        layout.addRow("Recurse into subfolders", self.recurse)
        layout.addRow("Rows to mask (top)", self.rows)
        layout.addRow("Concurrency (workers)", self.workers)
        layout.addRow("Dry run (don’t write files)", self.dry_run)
        layout.addRow("Export PNG previews", self.export_png)
        layout.addRow("OCR-based PII mask (no LLM)", self.pii_ocr)
        layout.addRow("Delete first image per study", self.skip_first)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.output)
        out_row.addWidget(self.pick_output_btn)
        layout.addRow("Output folder", out_row)


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(int, int, float, float)
    log = QtCore.Signal(str, str)
    finished = QtCore.Signal(object)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM De-identification")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # top area
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        left = QtWidgets.QVBoxLayout()
        top.addLayout(left, 1)
        self.folder_list = FolderList()
        btns = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add Folder…")
        self.remove_btn = QtWidgets.QPushButton("Remove Selected")
        btns.addWidget(self.add_btn)
        btns.addWidget(self.remove_btn)
        left.addWidget(self.folder_list)
        left.addLayout(btns)

        self.options = OptionsPanel()
        top.addWidget(self.options, 1)

        # bottom area
        bottom = QtWidgets.QVBoxLayout()
        root.addLayout(bottom)
        self.log_view = LogView()
        bottom.addWidget(self.log_view)
        self.progress = QtWidgets.QProgressBar()
        bottom.addWidget(self.progress)

        controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.open_report_btn = QtWidgets.QPushButton("Open Report")
        self.open_output_btn = QtWidgets.QPushButton("Open Output Folder")
        controls.addWidget(self.start_btn)
        controls.addWidget(self.cancel_btn)
        controls.addStretch(1)
        controls.addWidget(self.open_report_btn)
        controls.addWidget(self.open_output_btn)
        bottom.addLayout(controls)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # signals
        self.signals = _Signals()
        self.signals.progress.connect(self._on_progress)
        self.signals.log.connect(self._on_log)
        self.signals.finished.connect(self._on_finished)

        # wiring
        self.options.pick_output_btn.clicked.connect(self.on_pick_output)
        self.add_btn.clicked.connect(self.on_add_folder)
        self.remove_btn.clicked.connect(self.on_remove_selected)
        self.start_btn.clicked.connect(self.on_start)
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.open_output_btn.clicked.connect(self.on_open_output)
        self.open_report_btn.clicked.connect(self.on_open_report)

        self.cancel_event: threading.Event | None = None
        self.report_path: Path | None = None

        # logging - route GUI logs via signal to ensure thread-safety
        log_dir = Path.home() / ".dicom_deid_gui" / "logs"
        self.logger = setup_logging(log_dir, self.signals.log.emit)

    @QtCore.Slot()
    def on_pick_output(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.options.output.setText(d)

    @QtCore.Slot()
    def on_add_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add input folder")
        if d:
            self.folder_list.add_folder(Path(d))

    @QtCore.Slot()
    def on_remove_selected(self) -> None:
        for item in self.folder_list.selectedItems():
            row = self.folder_list.row(item)
            self.folder_list.takeItem(row)

    def _collect_inputs(self) -> list[Path]:
        return self.folder_list.all_paths()

    @QtCore.Slot(int, int, float, float)
    def _on_progress(self, done: int, total: int, rate: float, eta_s: float) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.showMessage(f"{done}/{total} processed, ETA: {int(eta_s)}s")

    @QtCore.Slot(str, str)
    def _on_log(self, level: str, msg: str) -> None:
        self.log_view.append_log(level, msg)

    @QtCore.Slot(object)
    def _on_finished(self, report_path: object) -> None:
        if isinstance(report_path, Path):
            self.report_path = report_path
        self.status.showMessage("Done")

    @QtCore.Slot()
    def on_start(self) -> None:
        inputs = self._collect_inputs()
        if not inputs:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Add at least one input folder")
            return
        out = self.options.output.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(self, "Missing output", "Choose an output folder")
            return
        output_dir = Path(out)
        rows = int(self.options.rows.value())
        recurse = bool(self.options.recurse.isChecked())
        workers_val = int(self.options.workers.value())
        workers: int | None = workers_val if workers_val > 0 else None
        dry_run = bool(self.options.dry_run.isChecked())
        export_pngs = bool(self.options.export_png.isChecked())
        pii_ocr = bool(self.options.pii_ocr.isChecked())
        skip_first = bool(self.options.skip_first.isChecked())

        self.cancel_event = threading.Event()
        self.progress.setValue(0)
        self.status.showMessage("Starting…")

        def run() -> None:
            report = run_batch(
                inputs=inputs,
                output_dir=output_dir,
                rows=rows,
                recurse=recurse,
                workers=workers,
                dry_run=dry_run,
                on_progress=self.signals.progress.emit,
                on_log=self.signals.log.emit,
                cancelled_flag=self.cancel_event,
                export_pngs=export_pngs,
                pii_ocr=pii_ocr,
                skip_first_of_study=skip_first,
            )
            # Notify GUI thread on completion
            self.signals.finished.emit(report)

        t = threading.Thread(target=run, daemon=True)
        t.start()

    @QtCore.Slot()
    def on_cancel(self) -> None:
        if self.cancel_event:
            self.cancel_event.set()

    @QtCore.Slot()
    def on_open_output(self) -> None:
        out = self.options.output.text().strip()
        if not out:
            return
        # Use OS default opener
        opener = (
            "open"
            if sys.platform == "darwin"
            else ("explorer" if sys.platform == "win32" else "xdg-open")
        )
        QtCore.QProcess.startDetached(opener, [out])

    @QtCore.Slot()
    def on_open_report(self) -> None:
        if self.report_path and self.report_path.exists():
            opener = (
                "open"
                if sys.platform == "darwin"
                else ("explorer" if sys.platform == "win32" else "xdg-open")
            )
            QtCore.QProcess.startDetached(opener, [str(self.report_path)])


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
