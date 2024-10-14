from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget, ComboBox, Slider, SpinBox
from napari.types import PointsData, ImageData
from napari.layers import Points, Image
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QSlider, QLabel, QComboBox, QSizePolicy, QGridLayout
from qtpy.QtCore import Qt
from skimage.util import img_as_float
import napari


class TrackingControls(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._tracking_method_combo = QComboBox()
        self._keypoint_layer_combo: QComboBox = create_widget(annotation=PointsData)
        self._video_layer_combo: QComboBox = create_widget(annotation=ImageData)
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
        self._tracking_backward_button = QPushButton()
        self._tracking_bothway_button = QPushButton()

        self._build_layout()

    @property
    def keypoint_layer(self) -> Points:
        return self._viewer.layers[self._keypoint_layer_combo.currentText()]

    @property
    def video_layer(self) -> Image:
        return self._viewer.layers[self._video_layer_combo.currentText()]


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
        self.layout().addLayout(tracking_controls_layout)
