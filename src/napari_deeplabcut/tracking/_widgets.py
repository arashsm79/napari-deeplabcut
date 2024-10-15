from magicgui import magic_factory
import debugpy
from functools import partial
from magicgui.widgets import CheckBox, Container, create_widget, ComboBox, Slider, SpinBox
from napari.types import PointsData, ImageData
from napari.layers import Points, Image
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QProgressBar
from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QSlider, QLabel, QComboBox, QSizePolicy, QGridLayout
from qtpy.QtCore import Qt, Slot
from skimage.util import img_as_float
from napari_deeplabcut.tracking.tracking_worker import TrackingWorker, TrackingWorkerData
import napari
from napari.viewer import Viewer
from napari.utils.events.event import Event


class TrackingControls(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer: Viewer = viewer

        # Layout
        self._tracking_method_combo = QComboBox()
        self._keypoint_layer_combo: ComboBox = create_widget(annotation=Points)
        self._video_layer_combo: ComboBox = create_widget(annotation=Image)
        self._video_layer_combo.changed.connect(self._video_layer_changed)
        self._backward_slider = QSlider(Qt.Horizontal)
        self._backward_spinbox_absolute = QSpinBox()
        self._backward_spinbox_relative = QSpinBox()
        self._reference_spinbox = QSpinBox()
        self._set_ref_button = QPushButton()
        self._forward_slider = QSlider(Qt.Horizontal)
        self._forward_spinbox_absolute = QSpinBox()
        self._forward_spinbox_relative = QSpinBox()
        self._tracking_stop_button = QPushButton()
        self._tracking_forward_button = QPushButton()
        self._tracking_forward_button.clicked.connect(self.track_forward)
        self._tracking_backward_button = QPushButton()
        self._tracking_bothway_button = QPushButton()
        self._tracking_progress_bar = QProgressBar()

        # Controls
        self._forward_slider.valueChanged.connect(self._forward_spinbox_relative.setValue)
        self._forward_spinbox_relative.valueChanged.connect(lambda x : (self._forward_slider.setValue(x), self._forward_spinbox_absolute.setValue(self._reference_spinbox.value() + x)))
        self._forward_spinbox_absolute.valueChanged.connect(lambda x : self._forward_spinbox_relative.setValue(x - self._reference_spinbox.value() if x - self._reference_spinbox.value() > 0 else 0))

        self._viewer.dims.events.current_step.connect(lambda e: self._reference_spinbox.setValue(e.value[0]))
        self._reference_spinbox.valueChanged.connect(self._update_controls)

        # Worker
        self.is_tracking = False
        self.worker: TrackingWorker = TrackingWorker()
        self.worker.started.connect(self.tracking_started)
        self.worker.progress.connect(self._tracking_progress_bar.setValue)
        self.worker.finished.connect(self.tracking_finished)

        self._build_layout()

    @Slot(int)
    def _update_controls(self, new_val: int):
        if self.video_layer is None:
            return
        self._forward_slider.setRange(0, self.video_layer.data.shape[0] - 1 - new_val)
        self._forward_spinbox_relative.setRange(0, self.video_layer.data.shape[0] - 1 - new_val)
        self._forward_spinbox_absolute.setRange(new_val, self.video_layer.data.shape[0] - 1)
        self._forward_spinbox_absolute.setValue(new_val+self._forward_spinbox_relative.value())

        self._viewer.dims.current_step = (new_val, *self._viewer.dims.current_step[1:])
        

    @property
    def keypoint_layer(self) -> Points:
        return self._keypoint_layer_combo.value

    @property
    def video_layer(self) -> Image | None:
        return self._video_layer_combo.value

    @Slot()
    def _video_layer_changed(self):
        self._update_controls(0)

    @Slot()
    def tracking_started(self):
        self.is_tracking = True
        self._tracking_progress_bar.setValue(0)

    @Slot()
    def tracking_finished(self):
        self.is_tracking = False
        self._tracking_progress_bar.setValue(100)

    @Slot()
    def track_forward(self):
        if self.is_tracking:
            return

        tracking_data = TrackingWorkerData()
        

    def _build_layout(self):
        self.setLayout(QVBoxLayout())
        self._tracking_method_combo.addItems(["Cotracker", "PIP"])
        self._tracking_method_combo.setCurrentText("Cotracker")
        _tracking_method_layout = QHBoxLayout()
        _tracking_method_layout.addWidget(QLabel("Tracker"))
        _tracking_method_layout.addWidget(self._tracking_method_combo)
        self._viewer.layers.events.inserted.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._keypoint_layer_combo.reset_choices)
        _keypoint_layer_method_layout = QHBoxLayout()
        _keypoint_layer_method_layout.addWidget(QLabel("Keypoints"))
        _keypoint_layer_method_layout.addWidget(self._keypoint_layer_combo.native)
        self._viewer.layers.events.inserted.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._video_layer_combo.reset_choices)
        _video_layer_method_layout = QHBoxLayout()
        _video_layer_method_layout.addWidget(QLabel("Video"))
        _video_layer_method_layout.addWidget(self._video_layer_combo.native)

        self.layout().addLayout(_tracking_method_layout)
        self.layout().addLayout(_keypoint_layer_method_layout)
        self.layout().addLayout(_video_layer_method_layout)
        range_controls_layout = QGridLayout() # 3 by 5
        self._backward_slider.setRange(0, 100)
        self._backward_slider.setInvertedAppearance(True)
        self._backward_slider.setInvertedControls(True)
        range_controls_layout.addWidget(self._backward_slider, 0, 0, 1, 2)
        self._backward_spinbox_absolute.setRange(0, 100)
        self._backward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_absolute.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(self._backward_spinbox_absolute, 1, 1)
        range_controls_layout.addWidget(QLabel("<< Abs"), 1, 0)
        self._backward_spinbox_relative.setRange(-100, 0)
        self._backward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_relative.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(QLabel("<< Rel"), 2, 0)
        range_controls_layout.addWidget(self._backward_spinbox_relative, 2, 1)
        _ref_label = QLabel("Ref")
        self._reference_spinbox.setRange(0, 100)
        self._reference_spinbox.setAlignment(Qt.AlignCenter)
        self._reference_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(self._reference_spinbox, 1, 2)
        _ref_label.setAlignment(Qt.AlignCenter)
        range_controls_layout.addWidget(_ref_label, 0, 2)
        self._set_ref_button.setText("Set")
        range_controls_layout.addWidget(self._set_ref_button, 2, 2)
        self._forward_slider.setRange(0, 100)
        range_controls_layout.addWidget(self._forward_slider, 0, 3, 1, 2)
        self._forward_spinbox_absolute.setRange(0, 100)
        self._forward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_absolute.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(QLabel("Abs >>"), 1, 4)
        range_controls_layout.addWidget(self._forward_spinbox_absolute, 1, 3)
        self._forward_spinbox_relative.setRange(0, 100)
        self._forward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_relative.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(QLabel("Rel >>"), 2, 4)
        range_controls_layout.addWidget(self._forward_spinbox_relative, 2, 3)

        self.layout().addLayout(range_controls_layout)
        tracking_controls_layout = QGridLayout() # 2 by 5
        self._tracking_stop_button.setText("□")
        tracking_controls_layout.addWidget(self._tracking_stop_button, 0, 2)
        self._tracking_forward_button.setText("⇥")
        tracking_controls_layout.addWidget(self._tracking_forward_button, 0, 3)
        self._tracking_bothway_button.setText("↹")
        self._tracking_backward_button.setText("⇤")
        tracking_controls_layout.addWidget(self._tracking_backward_button, 0, 1)
        tracking_controls_layout.addWidget(self._tracking_bothway_button, 1, 2)

        self._tracking_progress_bar.setRange(0, 100)
        self.layout().addLayout(tracking_controls_layout)
        self.layout().addWidget(self._tracking_progress_bar)
