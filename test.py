import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QHBoxLayout, \
    QMainWindow, QSlider, QTableWidget, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import defaultdict
import sqlite3
from datetime import datetime
import pandas as pd

# Tải mô hình YOLO trước
try:
    model = YOLO("D:/pt/appPc/train7/runs/detect/train7/weights/best.pt")
    print("Mô hình YOLO đã tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    sys.exit(1)

class_names = ['xe_oto', 'xe_may', 'xe_tai', 'xe_buyt']


class ChartWindow(QMainWindow):
    def __init__(self, vehicle_counts):
        super().__init__()
        self.setWindowTitle("Thống kê phương tiện")
        self.setGeometry(200, 200, 600, 400)
        self.vehicle_counts = vehicle_counts
        self.figure = None
        self.ax = None
        self.canvas = None
        self.init_chart()

    def init_chart(self):
        try:
            self.figure, self.ax = plt.subplots()
            self.canvas = FigureCanvas(self.figure)
            self.setCentralWidget(self.canvas)
            self.update_chart()
        except Exception as e:
            print(f"Lỗi khi khởi tạo biểu đồ: {e}")

    def update_chart(self):
        try:
            if self.ax and self.canvas:
                self.ax.clear()
                labels = ['xe_oto', 'xe_may', 'xe_tai', 'xe_buyt']
                values = [self.vehicle_counts[label] for label in labels]
                self.ax.bar(labels, values, color=['green', 'orange', 'blue', 'red'])
                self.ax.set_xlabel("Loại phương tiện")
                self.ax.set_ylabel("Số lượng")
                self.ax.set_title("Thống kê phương tiện")
                self.canvas.draw()
        except Exception as e:
            print(f"Lỗi khi cập nhật biểu đồ: {e}")

    def closeEvent(self, event):
        try:
            if self.figure:
                plt.close(self.figure)
            event.accept()
        except Exception as e:
            print(f"Lỗi khi đóng cửa sổ biểu đồ: {e}")


class DataWindow(QMainWindow):
    def __init__(self, conn):
        super().__init__()
        self.setWindowTitle("Dữ liệu phương tiện")
        self.setGeometry(200, 200, 800, 400)
        self.conn = conn
        self.init_table()

    def init_table(self):
        try:
            self.table = QTableWidget()
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(['ID', 'Thời gian', 'Xe ô tô', 'Xe máy', 'Xe tải', 'Xe buýt'])
            self.setCentralWidget(self.table)
            self.update_table()
        except Exception as e:
            print(f"Lỗi khi khởi tạo bảng dữ liệu: {e}")

    def update_table(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM vehicle_counts ORDER BY id DESC")
            rows = cursor.fetchall()
            self.table.setRowCount(len(rows))
            for row_idx, row_data in enumerate(rows):
                for col_idx, col_data in enumerate(row_data):
                    self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))
            self.table.resizeColumnsToContents()
        except Exception as e:
            print(f"Lỗi khi cập nhật bảng dữ liệu: {e}")


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.label = None
        self.total_count_label = None
        self.count_labels = None
        self.init_db()
        self.init_ui()
        self.vehicle_counts = defaultdict(int)
        self.tracked_objects = {}
        self.detection_zone = None
        self.detection_thickness = 50  # Giá trị mặc định
        self.object_id = 0
        self.tracking_threshold = 60
        self.detected_ids = set()
        self.is_paused = False
        self.chart_window = None
        self.data_window = None
        self.threshold = 50
        self.has_warned = False
        self.warning_timer = QTimer(self)
        self.warning_timer.timeout.connect(self.check_warning)
        self.timer = None

    def init_db(self):
        try:
            self.conn = sqlite3.connect('vehicle_counts.db')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    xe_oto INTEGER,
                    xe_may INTEGER,
                    xe_tai INTEGER,
                    xe_buyt INTEGER
                )
            ''')
            self.conn.commit()
            print("Kết nối database thành công.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo database: {e}")

    def init_ui(self):
        try:
            self.label = QLabel("Chọn video để nhận diện:")
            self.button = QPushButton("Chọn file")
            self.chart_button = QPushButton("Biểu đồ")
            self.pause_button = QPushButton("Tạm dừng")
            self.db_button = QPushButton("Lưu dữ liệu")
            self.export_button = QPushButton("Xuất Excel")
            self.reset_db_button = QPushButton("Reset Database")
            self.video_label = QLabel()
            self.line_slider = QSlider(Qt.Horizontal)
            self.line_slider.setMinimum(50)
            self.line_slider.setMaximum(600)
            self.line_slider.setValue(400)
            self.thickness_slider = QSlider(Qt.Horizontal)
            self.thickness_slider.setMinimum(10)
            self.thickness_slider.setMaximum(100)
            self.thickness_slider.setValue(50)
            self.count_labels = {label: QLabel(f"{label}: 0") for label in class_names}
            self.total_count_label = QLabel("Tổng số xe: 0")

            main_layout = QVBoxLayout()
            control_layout = QHBoxLayout()
            slider_layout = QHBoxLayout()
            thickness_layout = QHBoxLayout()
            count_layout = QHBoxLayout()

            control_layout.addWidget(self.button)
            control_layout.addWidget(self.chart_button)
            control_layout.addWidget(self.pause_button)
            control_layout.addWidget(self.db_button)
            control_layout.addWidget(self.export_button)
            control_layout.addWidget(self.reset_db_button)
            control_layout.addStretch(1)

            slider_layout.addWidget(QLabel("Đường đếm:"))
            slider_layout.addWidget(self.line_slider)

            thickness_layout.addWidget(QLabel("Kích thước vùng:"))
            thickness_layout.addWidget(self.thickness_slider)

            for lbl in self.count_labels.values():
                count_layout.addWidget(lbl)
            count_layout.addWidget(self.total_count_label)

            main_layout.addWidget(self.label)
            main_layout.addLayout(control_layout)
            main_layout.addLayout(slider_layout)
            main_layout.addLayout(thickness_layout)
            main_layout.addLayout(count_layout)
            main_layout.addWidget(self.video_label, stretch=1)

            self.setLayout(main_layout)
            self.setWindowTitle("Nhận diện Phương tiện")
            self.setGeometry(100, 100, 800, 700)

            self.button.clicked.connect(self.open_file)
            self.chart_button.clicked.connect(self.show_chart)
            self.pause_button.clicked.connect(self.toggle_pause)
            self.db_button.clicked.connect(self.show_data_window)
            self.export_button.clicked.connect(self.export_to_excel)
            self.reset_db_button.clicked.connect(self.reset_database)
            self.line_slider.valueChanged.connect(self.update_line_position)
            self.thickness_slider.valueChanged.connect(self.update_thickness)
            print("Giao diện đã khởi tạo thành công.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo giao diện: {e}")
            if self.label:
                self.label.setText(f"Lỗi khởi tạo giao diện: {str(e)}")

    def update_line_position(self):
        try:
            self.detection_zone = self.line_slider.value()
        except Exception as e:
            print(f"Lỗi khi cập nhật vị trí đường đếm: {e}")

    def update_thickness(self):
        try:
            self.detection_thickness = self.thickness_slider.value()
            print(f"Cập nhật kích thước vùng: {self.detection_thickness}")
        except Exception as e:
            print(f"Lỗi khi cập nhật kích thước vùng: {e}")

    def save_to_db(self):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data = (
                timestamp,
                self.vehicle_counts['xe_oto'],
                self.vehicle_counts['xe_may'],
                self.vehicle_counts['xe_tai'],
                self.vehicle_counts['xe_buyt']
            )
            self.cursor.execute('''
                INSERT INTO vehicle_counts (timestamp, xe_oto, xe_may, xe_tai, xe_buyt)
                VALUES (?, ?, ?, ?, ?)
            ''', data)
            self.conn.commit()
            print(f"Dữ liệu đã được lưu tại: {timestamp}")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu vào database: {e}")

    def export_to_excel(self):
        try:
            print("Bắt đầu xuất file Excel...")
            df = pd.read_sql_query("SELECT * FROM vehicle_counts ORDER BY id DESC", self.conn)
            if df.empty:
                if self.label:
                    self.label.setText("Không có dữ liệu để xuất!")
                print("Không có dữ liệu trong database.")
                return

            file_path, _ = QFileDialog.getSaveFileName(self, "Lưu file Excel", "", "Excel Files (*.xlsx)")
            if not file_path:
                if self.label:
                    self.label.setText("Chưa chọn vị trí lưu file!")
                print("Không chọn vị trí lưu file.")
                return

            df.to_excel(file_path, index=False)
            print(f"Dữ liệu đã được xuất ra file: {file_path}")
            if self.label:
                self.label.setText(f"Đã xuất dữ liệu ra {file_path}")
        except Exception as e:
            print(f"Lỗi khi xuất file Excel: {e}")
            if self.label:
                self.label.setText(f"Lỗi xuất Excel: {str(e)}")

    def reset_database(self):
        try:
            reply = QMessageBox.question(
                self,
                "Xác nhận Reset",
                "Bạn có chắc chắn muốn xóa toàn bộ dữ liệu trong database không? Hành động này không thể hoàn tác!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.cursor.execute("DROP TABLE IF EXISTS vehicle_counts")
                self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS vehicle_counts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        xe_oto INTEGER,
                        xe_may INTEGER,
                        xe_tai INTEGER,
                        xe_buyt INTEGER
                    )
                ''')
                self.conn.commit()

                if self.data_window:
                    self.data_window.update_table()

                if self.label:
                    self.label.setText("Database đã được reset thành công!")
                print("Database đã được reset.")
            else:
                print("Reset database bị hủy.")
        except Exception as e:
            print(f"Lỗi khi reset database: {e}")
            if self.label:
                self.label.setText(f"Lỗi reset database: {str(e)}")

    def show_data_window(self):
        try:
            self.save_to_db()
            if self.data_window is not None:
                self.data_window.close()
            self.data_window = DataWindow(self.conn)
            self.data_window.show()
        except Exception as e:
            print(f"Lỗi khi hiển thị cửa sổ dữ liệu: {e}")
            if self.label:
                self.label.setText(f"Lỗi hiển thị dữ liệu: {str(e)}")

    def open_file(self):
        try:
            print("Bắt đầu mở file...")
            if self.label is None:
                raise ValueError("self.label không được khởi tạo.")
            print("self.label tồn tại.")
            self.label.setText("Đang nhận diện...")
            print("Đã đặt text 'Đang nhận diện...'")
            file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file", "", "Videos (*.mp4 *.avi)")
            print(f"File path: {file_path}")
            if not file_path:
                self.label.setText("Chưa chọn video!")
                print("Không chọn file, thoát hàm.")
                return
            self.vehicle_counts.clear()
            for label in self.count_labels:
                if self.count_labels[label] is None:
                    print(f"Warning: count_labels[{label}] là None")
                    self.count_labels[label] = QLabel(f"{label}: 0")
                self.count_labels[label].setText(f"{label}: 0")
                self.count_labels[label].setStyleSheet("")
            if self.total_count_label is None:
                print("Warning: total_count_label là None, khởi tạo lại.")
                self.total_count_label = QLabel("Tổng số xe: 0")
            self.total_count_label.setText("Tổng số xe: 0")
            self.total_count_label.setStyleSheet("")
            self.tracked_objects.clear()
            self.detected_ids.clear()
            self.has_warned = False
            self.warning_timer.start(1000)
            self.detect_video(file_path)
        except Exception as e:
            print(f"Lỗi khi mở file: {e}")
            if self.label:
                self.label.setText(f"Lỗi: {str(e)}")

    def detect_video(self, video_path):
        try:
            self.cap = cv2.VideoCapture(video_path)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Không thể mở video.")
            frame_height = frame.shape[0]
            self.detection_zone = int(0.4 * frame_height)
            self.line_slider.setMaximum(frame_height)
            self.line_slider.setValue(self.detection_zone)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_frame)
            self.timer.start(50)
            self.is_paused = False
            self.pause_button.setText("Tạm dừng")
            print(f"Đã mở video: {video_path}")
            if self.label:
                self.label.setText("Đang nhận diện video...")
        except Exception as e:
            print(f"Lỗi khi khởi tạo video: {e}")
            if self.label:
                self.label.setText(f"Lỗi khi mở video: {str(e)}")

    def toggle_pause(self):
        try:
            if self.is_paused:
                if self.timer:
                    self.timer.start(50)
                self.pause_button.setText("Tạm dừng")
                self.is_paused = False
                if self.label:
                    self.label.setText("Đang nhận diện...")
            else:
                if self.timer:
                    self.timer.stop()
                self.pause_button.setText("Tiếp tục")
                self.is_paused = True
                if self.label:
                    self.label.setText("Đã tạm dừng!")
        except Exception as e:
            print(f"Lỗi khi tạm dừng/tiếp tục: {e}")
            if self.label:
                self.label.setText(f"Lỗi: {str(e)}")

    def next_frame(self):
        try:
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                return
            ret, frame = self.cap.read()
            if not ret:
                if self.timer:
                    self.timer.stop()
                self.warning_timer.stop()
                self.cap.release()
                self.is_paused = False
                self.pause_button.setText("Tạm dừng")
                if self.label:
                    self.label.setText("Nhận diện video hoàn tất!")
                return

            line_y = self.detection_zone
            frame_height, frame_width = frame.shape[:2]
            # Vẽ vùng phát hiện
            cv2.rectangle(frame, (0, line_y - self.detection_thickness),
                         (frame_width, line_y + self.detection_thickness), (0, 0, 255), 2)
            # Vẽ đường trung tâm để dễ hình dung
            cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 1)
            results = model.predict(frame, imgsz=320, conf=0.5)

            print(f"Khung hình - Số đối tượng phát hiện: {len(results[0].boxes)}")
            new_tracked_objects = {}
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    obj_id = None
                    min_distance = float('inf')
                    for prev_id, (prev_x, prev_y, _) in self.tracked_objects.items():
                        distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                        if distance < self.tracking_threshold and distance < min_distance:
                            obj_id = prev_id
                            min_distance = distance

                    if obj_id is None:
                        obj_id = self.object_id
                        self.object_id += 1
                        print(f"Tạo ID mới: {obj_id} cho {class_name} tại ({center_x}, {center_y})")

                    if obj_id in self.tracked_objects:
                        prev_y = self.tracked_objects[obj_id][2]
                        print(f"ID {obj_id} - Trước: {prev_y}, Sau: {center_y}, Đường trung tâm: {line_y}")
                        if obj_id not in self.detected_ids:
                            # Đếm khi xe đi qua đường trung tâm từ trên xuống dưới hoặc ngược lại
                            if (prev_y < line_y and center_y >= line_y) or (prev_y > line_y and center_y <= line_y):
                                self.vehicle_counts[class_name] += 1
                                if self.count_labels[class_name]:
                                    self.count_labels[class_name].setText(
                                        f"{class_name}: {self.vehicle_counts[class_name]}")
                                self.detected_ids.add(obj_id)
                                total_count = sum(self.vehicle_counts.values())
                                if self.total_count_label:
                                    self.total_count_label.setText(f"Tổng số xe: {total_count}")
                                print(f"Đếm {class_name} - ID: {obj_id} tại ({center_x}, {center_y})")
                            else:
                                print(f"ID {obj_id} chưa đi qua đường trung tâm")

                    new_tracked_objects[obj_id] = (center_x, center_y, center_y)

            self.tracked_objects = new_tracked_objects
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                          frame_rgb.strides[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
            del frame
            del frame_rgb
        except Exception as e:
            print(f"Lỗi trong next_frame: {e}")
            if self.timer:
                self.timer.stop()
            if self.label:
                self.label.setText(f"Lỗi nhận diện: {str(e)}")

    def check_warning(self):
        try:
            total_count = sum(self.vehicle_counts.values())
            if total_count > self.threshold and not self.has_warned:
                if self.total_count_label:
                    self.total_count_label.setStyleSheet("color: red; font-weight: bold;")
                    QMessageBox.warning(self, "Cảnh báo", f"Tổng số lượng phương tiện vượt ngưỡng: {total_count}")
                    self.has_warned = True
        except Exception as e:
            print(f"Lỗi trong check_warning: {e}")

    def show_chart(self):
        try:
            if self.chart_window is not None:
                self.chart_window.close()
            self.chart_window = ChartWindow(self.vehicle_counts)
            self.chart_window.show()
        except Exception as e:
            print(f"Lỗi khi hiển thị biểu đồ: {e}")
            if self.label:
                self.label.setText(f"Lỗi biểu đồ: {str(e)}")

    def closeEvent(self, event):
        try:
            self.conn.close()
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if self.chart_window:
                self.chart_window.close()
            if self.data_window:
                self.data_window.close()
            self.warning_timer.stop()
            if self.timer:
                self.timer.stop()
            event.accept()
            print("Đã đóng ứng dụng.")
        except Exception as e:
            print(f"Lỗi khi đóng ứng dụng: {e}")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MyApp()
        window.show()
        print("Đã hiển thị cửa sổ chính.")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Lỗi trong main: {e}")