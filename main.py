import sys
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton, QDialog, QVBoxLayout, QTextEdit, QLabel, QInputDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QHBoxLayout
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread, Signal, Qt, QObject
import os
import subprocess
import webbrowser
from yolo.detector import YOLODetector
import threading
import requests
import cv2
from PySide6.QtGui import QImage, QPixmap
import time
import numpy as np

YOLO_MODELS = [
    ("yolov8n.pt", "Nano (fastest, smallest)"),
    ("yolov8s.pt", "Small (fast, good accuracy)"),
    ("yolov8m.pt", "Medium (balanced)"),
    ("yolov8l.pt", "Large (high accuracy, slow)"),
    ("yolov8x.pt", "X-Large (highest accuracy, slowest)")
]
YOLO_DOWNLOAD_URLS = {
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
}

selected_model_path = None
selected_classes = set()

def check_python_installed():
    for cmd in ["python", "python3"]:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0 and "Python" in result.stdout + result.stderr:
                return True
        except Exception:
            continue
    return False

def show_python_required_dialog():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Python Required")
    msg.setText(
        "Python is required to run this application.\n\nPlease install Python from the official website and then restart the app.\n\nDownload: https://www.python.org/downloads/"
    )
    download_btn = QPushButton("Download Python")
    msg.addButton(download_btn, QMessageBox.ActionRole)
    msg.addButton(QMessageBox.Ok)
    def open_python_download():
        webbrowser.open("https://www.python.org/downloads/")
    download_btn.clicked.connect(open_python_download)
    msg.exec()

def check_required_packages():
    required = ["PySide6", "cv2", "ultralytics", "numpy"]
    missing = []
    for pkg in required:
        try:
            if pkg == "cv2":
                import cv2
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing

def prompt_install_requirements():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Missing Requirements")
    msg.setText(
        "Some required Python packages are missing.\nWould you like to install them now?"
    )
    yes_btn = QPushButton("Yes, install now")
    no_btn = QPushButton("No, exit")
    msg.addButton(yes_btn, QMessageBox.YesRole)
    msg.addButton(no_btn, QMessageBox.NoRole)
    result = msg.exec()
    return result == 0  # 0 is YesRole

class PipInstallThread(QThread):
    log_signal = Signal(str)
    done_signal = Signal(bool, str)

    def run(self):
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "-r", os.path.join(os.path.dirname(__file__), "requirements.txt")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in process.stdout:
                self.log_signal.emit(line)
            process.wait()
            if process.returncode == 0:
                self.done_signal.emit(True, "All required packages have been installed. Please restart the app.")
            else:
                self.done_signal.emit(False, f"pip exited with code {process.returncode}. Please install requirements manually.")
        except Exception as e:
            self.done_signal.emit(False, f"Failed to install requirements. Error: {e}")

def install_requirements():
    dialog = QDialog()
    dialog.setWindowTitle("Installing Requirements")
    layout = QVBoxLayout(dialog)
    label = QLabel("Installing required packages. Please wait...")
    log_view = QTextEdit()
    log_view.setReadOnly(True)
    layout.addWidget(label)
    layout.addWidget(log_view)
    dialog.setLayout(layout)
    thread = PipInstallThread()
    thread.log_signal.connect(lambda text: log_view.append(text.rstrip()))
    def on_done(success, message):
        if success:
            QMessageBox.information(dialog, "Success", message)
        else:
            QMessageBox.critical(dialog, "Error", message)
        dialog.accept()
        sys.exit(0)
    thread.done_signal.connect(on_done)
    thread.start()
    dialog.exec()

class DownloadThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    def __init__(self, url, dest):
        super().__init__()
        self.url = url
        self.dest = dest
    def run(self):
        import requests
        try:
            r = requests.get(self.url, stream=True)
            total = int(r.headers.get('content-length', 0))
            with open(self.dest, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded * 100 // total if total else 0
                        self.log_signal.emit(f"Downloaded {percent}%...")
            self.log_signal.emit("Download complete.")
            self.finished_signal.emit(True, "Download complete.")
        except Exception as e:
            self.log_signal.emit(f"Error: {e}")
            self.finished_signal.emit(False, str(e))

def download_yolo_weight(model_name, parent=None):
    url = YOLO_DOWNLOAD_URLS[model_name]
    dest = os.path.join(os.path.dirname(__file__), "yolo", model_name)
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"Downloading {model_name}")
    layout = QVBoxLayout(dlg)
    label = QLabel(f"Downloading {model_name}... Please wait.")
    log = QTextEdit()
    log.setReadOnly(True)
    layout.addWidget(label)
    layout.addWidget(log)
    dlg.setLayout(layout)
    thread = DownloadThread(url, dest)
    thread.log_signal.connect(log.append)
    def on_finished(success, msg):
        if success:
            log.append("Download finished successfully.")
        else:
            log.append(f"Download failed: {msg}")
        dlg.accept()
    thread.finished_signal.connect(on_finished)
    thread.start()
    dlg.exec()

def select_classes_dialog(class_names, parent=None):
    dlg = QDialog(parent)
    dlg.setWindowTitle("Select Classes to Count")
    layout = QVBoxLayout(dlg)
    label = QLabel("Check the classes you want to count and display:")
    layout.addWidget(label)
    list_widget = QListWidget()
    for cname in class_names:
        item = QListWidgetItem(cname)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)  # Start unchecked
        list_widget.addItem(item)
    layout.addWidget(list_widget)
    # Add Check All / Uncheck All buttons
    btn_layout = QHBoxLayout()
    btn_check_all = QPushButton("Check All")
    btn_uncheck_all = QPushButton("Uncheck All")
    btn_layout.addWidget(btn_check_all)
    btn_layout.addWidget(btn_uncheck_all)
    layout.addLayout(btn_layout)
    def check_all():
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(Qt.Checked)
    def uncheck_all():
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(Qt.Unchecked)
    btn_check_all.clicked.connect(check_all)
    btn_uncheck_all.clicked.connect(uncheck_all)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(buttons)
    def accept():
        dlg.accept()
    def reject():
        dlg.reject()
    buttons.accepted.connect(accept)
    buttons.rejected.connect(reject)
    dlg.setLayout(layout)
    if dlg.exec() == QDialog.Accepted:
        checked = set()
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.Checked:
                checked.add(item.text())
        return checked
    else:
        return None

def select_yolo_model(parent=None):
    yolo_dir = os.path.join(os.path.dirname(__file__), "yolo")
    available = [fname for fname, _ in YOLO_MODELS if os.path.exists(os.path.join(yolo_dir, fname))]
    items = []
    for fname, desc in YOLO_MODELS:
        status = "(downloaded)" if fname in available else "(will download)" if fname in YOLO_DOWNLOAD_URLS else "(not available)"
        items.append(f"{fname} {status} - {desc}")
    item, ok = QInputDialog.getItem(parent, "Select YOLO Model", "Choose a YOLOv8 model:", items, 0, False)
    if ok and item:
        fname = item.split()[0]
        model_path = os.path.join(yolo_dir, fname)
        if not os.path.exists(model_path):
            if fname in YOLO_DOWNLOAD_URLS:
                download_yolo_weight(fname, parent)
            else:
                QMessageBox.critical(parent, "Model Not Available", f"Model {fname} is not available.")
                return select_yolo_model(parent)
        return model_path
    else:
        sys.exit(0)

class VideoThread(QThread):
    frame_signal = Signal(np.ndarray)
    count_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, video_path, yolo_detector):
        super().__init__()
        self.video_path = video_path
        self.yolo_detector = yolo_detector
        self._running = True
        self._paused = False
        self._stopped = False
        self.cap = None

    def run(self):
        print(f"[DEBUG] Starting video thread for: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        while self.cap.isOpened() and self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                print(f"[DEBUG] End of video or failed to read frame at index {frame_idx}")
                break
            print(f"[DEBUG] Read frame {frame_idx}")
            detections = self.yolo_detector.detect(frame)
            # Filter detections by selected_classes
            filtered = [det for det in detections if det['class_name'] in selected_classes]
            print(f"[DEBUG] Detections for frame {frame_idx}: {len(filtered)} objects (filtered)")
            # Draw boxes and labels
            for det in filtered:
                x1, y1, x2, y2 = map(int, det['box'])
                label = f"{det['class_name']} {det['conf']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # Count objects
            counts = {}
            for det in filtered:
                cname = det['class_name']
                counts[cname] = counts.get(cname, 0) + 1
            print(f"[DEBUG] Emitting frame {frame_idx} to UI")
            self.frame_signal.emit(frame)
            self.count_signal.emit(counts)
            frame_idx += 1
            # Wait for next frame (simulate real-time)
            time.sleep(1 / max(self.cap.get(cv2.CAP_PROP_FPS), 1))
            if self._stopped:
                print(f"[DEBUG] Video thread stopped by user at frame {frame_idx}")
                break
        self.cap.release()
        print(f"[DEBUG] Video thread finished.")
        self.finished_signal.emit()

    def pause(self):
        self._paused = True
    def resume(self):
        self._paused = False
    def stop(self):
        self._running = False
        self._stopped = True

class VideoLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._zoom_callback = None
        self._pan_callback = None
        self._dragging = False
        self._last_pos = None
        self._resize_callback = None
    def set_zoom_callback(self, callback):
        self._zoom_callback = callback
    def set_pan_callback(self, callback):
        self._pan_callback = callback
    def set_resize_callback(self, callback):
        self._resize_callback = callback
    def wheelEvent(self, event):
        if self._zoom_callback:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_callback(1.1)
            elif delta < 0:
                self._zoom_callback(0.9)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._zoom_callback and self._pan_callback:
            self._dragging = True
            self._last_pos = event.pos()
    def mouseMoveEvent(self, event):
        if self._dragging and self._pan_callback and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self._pan_callback(delta.x(), delta.y())
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._last_pos = None
    def resizeEvent(self, event):
        if self._resize_callback:
            self._resize_callback()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not check_python_installed():
        show_python_required_dialog()
        sys.exit(1)

    missing = check_required_packages()
    if missing:
        if prompt_install_requirements():
            install_requirements()
        else:
            sys.exit(0)

    def after_model_selected():
        global selected_classes
        class_names = list(yolo_detector.model.names.values()) if hasattr(yolo_detector.model, 'names') else []
        checked = select_classes_dialog(class_names, window)
        if checked is not None:
            selected_classes.clear()
            selected_classes.update(checked)
        else:
            selected_classes.clear()
            selected_classes.update(class_names)

    # Model selection dialog
    selected_model_path = select_yolo_model()
    yolo_detector = YOLODetector(selected_model_path)

    loader = QUiLoader()
    ui_file = QFile(os.path.join(os.path.dirname(__file__), 'ui', 'main_window.ui'))
    ui_file.open(QFile.ReadOnly)
    window = loader.load(ui_file, None)
    ui_file.close()

    after_model_selected()  # Only call here, after window is defined

    # Replace label_video with VideoLabel instance
    video_label = VideoLabel()
    video_label.setObjectName("label_video")
    video_label.setText("[Video will appear here]")
    video_label.setMinimumSize(400, 240)
    video_label.setAlignment(Qt.AlignCenter)
    video_label.setFrameShape(QLabel.Box)
    # Replace the old label_video in the UI
    layout = window.centralwidget.layout()
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() and item.widget().objectName() == "label_video":
            layout.replaceWidget(item.widget(), video_label)
            item.widget().deleteLater()
            break
    window.label_video = video_label

    # Store the last displayed frame (QImage)
    window._last_qimage = None
    pan_offset = {'x': 0, 'y': 0}
    def render_last_frame():
        if window._last_qimage:
            orig_w = window._last_qimage.width()
            orig_h = window._last_qimage.height()
            scale = zoom_factor['value']
            scaled_w = int(orig_w * scale)
            scaled_h = int(orig_h * scale)
            view_w = window.label_video.width()
            view_h = window.label_video.height()
            max_x = max(0, scaled_w - view_w)
            max_y = max(0, scaled_h - view_h)
            # Clamp pan offset
            if scale <= 1.0:
                pan_offset['x'] = 0
                pan_offset['y'] = 0
            else:
                pan_offset['x'] = min(max(pan_offset['x'], -max_x), max_x)
                pan_offset['y'] = min(max(pan_offset['y'], -max_y), max_y)
            scaled_pixmap = QPixmap.fromImage(window._last_qimage).scaled(
                scaled_w, scaled_h, aspectMode=Qt.KeepAspectRatio)
            # Center if not zoomed, else use pan offset
            if scale <= 1.0:
                x = max(0, (scaled_w - view_w) // 2)
                y = max(0, (scaled_h - view_h) // 2)
            else:
                x = max(0, pan_offset['x'])
                y = max(0, pan_offset['y'])
            cropped = scaled_pixmap.copy(x, y, view_w, view_h)
            window.label_video.setPixmap(cropped)
    zoom_factor = {'value': 1.0}
    def set_zoom(factor):
        prev_zoom = zoom_factor['value']
        zoom_factor['value'] = max(1.0, min(zoom_factor['value'] * factor, 5.0))
        print(f"[DEBUG] Zoom factor set to {zoom_factor['value']}")
        if zoom_factor['value'] == 1.0:
            pan_offset['x'] = 0
            pan_offset['y'] = 0
        render_last_frame()
    video_label.set_zoom_callback(set_zoom)
    window.button_zoom_in.clicked.connect(lambda: set_zoom(1.1))
    window.button_zoom_out.clicked.connect(lambda: set_zoom(0.9))

    # Screenshot functionality
    def take_screenshot():
        if window._last_qimage:
            file_name, _ = QFileDialog.getSaveFileName(window, "Save Screenshot", "screenshot.jpg", "Images (*.png *.jpg *.bmp)")
            if file_name:
                window._last_qimage.save(file_name)
    window.button_screenshot.clicked.connect(take_screenshot)

    # Helper to enable/disable video controls
    video_thread = {'thread': None}
    def set_video_controls_enabled(enabled):
        window.button_play.setEnabled(enabled)
        window.button_pause.setEnabled(enabled)
        window.button_stop.setEnabled(enabled)
        # Always keep Screenshot and Change Engine enabled and visible
        window.button_screenshot.setEnabled(True)
        window.button_zoom_in.setEnabled(enabled)
        window.button_zoom_out.setEnabled(enabled)
        window.button_change_engine.setEnabled(True)
        window.button_play.setVisible(enabled)
        window.button_pause.setVisible(enabled)
        window.button_stop.setVisible(enabled)
        window.button_screenshot.setVisible(True)
        window.button_zoom_in.setVisible(enabled)
        window.button_zoom_out.setVisible(enabled)
        window.button_change_engine.setVisible(True)

    def stop_current_video_thread():
        if video_thread['thread'] is not None:
            try:
                video_thread['thread'].frame_signal.disconnect()
            except Exception:
                pass
            try:
                video_thread['thread'].count_signal.disconnect()
            except Exception:
                pass
            try:
                video_thread['thread'].finished_signal.disconnect()
            except Exception:
                pass
            video_thread['thread'].stop()
            video_thread['thread'].wait()
            video_thread['thread'] = None

    set_video_controls_enabled(False)

    def use_webcam():
        stop_current_video_thread()
        window.label_status.setText("Webcam selected. (Not yet implemented)")
        set_video_controls_enabled(False)
    def open_mp4():
        stop_current_video_thread()
        file_name, _ = QFileDialog.getOpenFileName(
            window,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.mpg *.mpeg *.webm);;All Files (*)"
        )
        if file_name:
            window.label_status.setText(f"Selected Video: {file_name}")
            set_video_controls_enabled(True)
            # Start video thread
            video_thread['thread'] = VideoThread(file_name, yolo_detector)
            window._last_qimage = None  # Store QImage to keep buffer alive
            def update_frame(frame):
                print(f"[DEBUG] update_frame called")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                window._last_qimage = qimg  # Keep reference
                render_last_frame()
            def update_count(counts):
                if counts:
                    text = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                else:
                    text = "Object Count: 0"
                window.label_count.setText(text)
            video_thread['thread'].frame_signal.connect(update_frame, Qt.QueuedConnection)
            video_thread['thread'].count_signal.connect(update_count, Qt.QueuedConnection)
            video_thread['thread'].finished_signal.connect(lambda: set_video_controls_enabled(False))
            video_thread['thread'].start()
        else:
            set_video_controls_enabled(False)
    def enter_camera_link():
        stop_current_video_thread()
        link, ok = QInputDialog.getText(window, "Enter Camera Link", "Enter the camera/network stream URL (e.g. http://... or rtsp://...):")
        if ok and link:
            window.label_status.setText(f"Camera link: {link}")
            set_video_controls_enabled(False)  # Hide controls for live stream
            video_thread['thread'] = VideoThread(link, yolo_detector)
            window._last_qimage = None
            def update_frame(frame):
                print(f"[DEBUG] update_frame called (camera link)")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                window._last_qimage = qimg
                render_last_frame()
            def update_count(counts):
                if counts:
                    text = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                else:
                    text = "Object Count: 0"
                window.label_count.setText(text)
            def on_finished():
                QMessageBox.warning(window, "Stream Ended", "The camera stream ended or could not be opened.")
                set_video_controls_enabled(False)
            video_thread['thread'].frame_signal.connect(update_frame, Qt.QueuedConnection)
            video_thread['thread'].count_signal.connect(update_count, Qt.QueuedConnection)
            video_thread['thread'].finished_signal.connect(on_finished)
            video_thread['thread'].start()
        else:
            set_video_controls_enabled(False)

    def change_engine():
        stop_current_video_thread()
        # Reset UI state
        window.label_video.clear()
        window.label_video.setText("[Video will appear here]")
        window.label_count.setText("Object Count: 0")
        window.label_status.setText("Select a video source to start counting objects:")
        zoom_factor['value'] = 1.0
        pan_offset['x'] = 0
        pan_offset['y'] = 0
        window._last_qimage = None
        set_video_controls_enabled(False)
        # Prompt for new model and classes
        global selected_model_path, yolo_detector
        selected_model_path = select_yolo_model(window)
        yolo_detector = YOLODetector(selected_model_path)
        after_model_selected()

    window.button_mp4.clicked.connect(open_mp4)
    window.button_camera_link.clicked.connect(enter_camera_link)
    window.button_change_engine.clicked.connect(change_engine)

    # Playback controls
    def play_video():
        if video_thread['thread']:
            video_thread['thread'].resume()
    def pause_video():
        if video_thread['thread']:
            video_thread['thread'].pause()
    def stop_video():
        if video_thread['thread']:
            video_thread['thread'].stop()
            set_video_controls_enabled(False)
    window.button_play.clicked.connect(play_video)
    window.button_pause.clicked.connect(pause_video)
    window.button_stop.clicked.connect(stop_video)

    # Pan functionality
    pan_offset = {'x': 0, 'y': 0}
    def set_pan(dx, dy):
        # Only allow panning if zoomed in
        if zoom_factor['value'] > 1.0 and window._last_qimage:
            orig_w = window._last_qimage.width()
            orig_h = window._last_qimage.height()
            scale = zoom_factor['value']
            scaled_w = int(orig_w * scale)
            scaled_h = int(orig_h * scale)
            view_w = window.label_video.width()
            view_h = window.label_video.height()
            max_x = max(0, scaled_w - view_w)
            max_y = max(0, scaled_h - view_h)
            pan_offset['x'] = min(max(pan_offset['x'] - dx, -max_x), max_x)
            pan_offset['y'] = min(max(pan_offset['y'] - dy, -max_y), max_y)
            render_last_frame()
    video_label.set_pan_callback(set_pan)

    video_label.set_resize_callback(render_last_frame)

    window.show()
    sys.exit(app.exec()) 