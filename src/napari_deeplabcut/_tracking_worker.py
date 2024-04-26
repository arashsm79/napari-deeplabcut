# TODO :
# - Implement the tracking worker with multithreading
# - Implement a Log widget to display the tracking progress + a progress bar
# - Prepare I/O with the actual tracking backend
from functools import partial
from pathlib import Path

import napari
import numpy as np
import torch
from cotracker.predictor import CoTrackerOnlinePredictor
from napari._qt.qthreading import GeneratorWorker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.utils._qthreading import GeneratorWorkerSignals, WorkerBaseSignals

from napari_deeplabcut._tracking_utils import (
    ContainerWidget,
    LayerSelecter,
    Log,
    QWidgetSingleton,
    add_widgets,
    get_time,
)


class TrackingModule(QWidget, metaclass=QWidgetSingleton):
    """Plugin for tracking."""

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        """Creates a widget with links to documentation and about page."""
        super().__init__()
        self._viewer = napari_viewer
        self._worker = None
        self._keypoint_layer = None
        ### Widgets ###
        self.video_layer_dropdown = LayerSelecter(
            self._viewer,
            name="Video layer",
            layer_type=napari.layers.Image,
            parent=self,
        )
        self.keypoint_layer_dropdown = LayerSelecter(
            self._viewer,
            name="Keypoint layer",
            layer_type=napari.layers.Points,
            parent=self,
        )
        self.start_button = QPushButton("Start tracking")
        self.start_button.clicked.connect(self._start)
        #############################
        # status report docked widget
        self.container_docked = False  # check if already docked

        self.report_container = ContainerWidget(l=10, t=5, r=5, b=5)

        self.report_container.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Minimum
        )
        self.progress = QProgressBar(self.report_container)
        self.progress.setVisible(False)
        """Widget for the progress bar"""

        self.log = Log(self.report_container)
        self.log.setVisible(False)
        """Read-only display for process-related info. Use only for info destined to user."""
        self._build()

    # Use @property to get/set the keypoint layer
    @property
    def keypoint_layer(self):
        """Get the keypoint layer."""
        return self._keypoint_layer

    @keypoint_layer.setter
    def keypoint_layer(self, layer_name):
        """Set the keypoint layer from the viewer."""
        for l in self._viewer.layers:
            if l.name == layer_name:
                self._keypoint_layer = l
                break

    def _build(self):
        """Create a TrackingModule plugin with :

        - A dropdown menu to select the keypoint layer
        - A set of keypoints to track
        - A button to start tracking
        - A Log that shows when starting. providing feedback on the tracking process
        """
        layout = QVBoxLayout()

        widgets = [
            self.video_layer_dropdown,
            self.keypoint_layer_dropdown,
            self.start_button,
        ]
        add_widgets(layout, widgets)
        self.setLayout(layout)

    def check_ready(self):
        """Check if the inputs are ready for tracking."""
        if self.video_layer_dropdown.layer is None:
            return False
        if self.keypoint_layer_dropdown.layer is None:
            return False
        return True

    def _start(self):
        """Start the tracking process."""
        # TODO : implement the tracking process
        print("Started tracking")
        print(f"Is ready : {self.check_ready()}")
        # TODO : setup worker
        ### Below is code to start the worker and update the button for the use to start/stop the tracking process
        if not self.check_ready():
            err = "Aborting, please choose valid inputs"
            self.log.print_and_log(err)
            raise ValueError(err)

        if self._worker is not None:
            if self._worker.is_running:
                pass
            else:
                self._worker.start()
        else:
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)
            self._setup_worker()

        if self._worker.is_running:  # if worker is running, tries to stop
            self.log.print_and_log(
                "Stop request, waiting for next inference..."
            )
            self._worker.quit()
        else:  # once worker is started, update buttons
            self._worker.start()

    def _setup_worker(self):

        self._worker = TrackingWorker()
        self._worker.started.connect(self._on_start)

        self._worker.log_signal.connect(self.log.print_and_log)
        self._worker.log_w_replace_signal.connect(self.log.replace_last_line)
        self._worker.warn_signal.connect(self.log.warn)
        self._worker.error_signal.connect(self.log.error)

        self._worker.yielded.connect(partial(self._on_yield))
        self._worker.errored.connect(partial(self._on_error))
        self._worker.finished.connect(self._on_finish)

        keypoint_cord = self.keypoint_layer_dropdown.layer_data()
        frames = self.video_layer_dropdown.layer_data()

        self.log.print_and_log(f"keypoint started at {keypoint_cord}")
        self.log.print_and_log(f"frames started at {frames}")


        

    def _on_yield(self, results):
        # TODO : display the results in the viewer
        pass

    def _on_start(self):
        """Catches start signal from worker to call :py:func:`~display_status_report`."""
        # self.display_status_report()
        # self._set_self_config()
        self.log.print_and_log(f"Worker started at {get_time()}")
        #self.log.print_and_log(f"Saving results to : {self.results_path}")
        self.log.print_and_log("Worker is running...")

    def _on_error(self, error):
        """Catches errors and tries to clean up."""
        self.log.print_and_log("!" * 20)
        self.log.print_and_log("Worker errored...")
        self.log.error(error)
        self._worker.quit()
        self.on_finish()

    def _on_finish(self):
        """Catches finished signal from worker, resets workspace for next run."""
        self.log.print_and_log(f"\nWorker finished at {get_time()}")
        self.log.print_and_log("*" * 20)

        self._worker = None
        self._worker_config = None
        # self.empty_cuda_cache()
        return True  # signal clean exit


### -------- Tracking worker -------- ###


class LogSignal(WorkerBaseSignals):
    """Signal to send messages to be logged from another thread.

    Separate from Worker instances as indicated `on this post`_

    .. _on this post: https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect
    """  # TODO link ?

    log_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged"""
    log_w_replace_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged, replacing the last line"""
    warn_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some warning should be emitted in main thread"""
    error_signal = Signal(Exception, str)
    """qtpy.QtCore.Signal: signal to be sent when some error should be emitted in main thread"""

    # Should not be an instance variable but a class variable, not defined in __init__, see
    # https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect

    def __init__(self, parent=None):
        """Creates a LogSignal."""
        super().__init__(parent=parent)


class TrackingWorker(GeneratorWorker):
    """A custom worker to run tracking in."""

    def __init__(self, config=None):
        """Creates a TrackingWorker."""
        super().__init__(self.run_tracking)
        self._signals = LogSignal()
        self.log_signal = self._signals.log_signal
        self.log_w_replace_signal = self._signals.log_w_replace_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal

        self.config = config  # use if needed

    def log(self, msg):
        """Log a message."""
        self.log_signal.emit(msg)

    def log_w_replace(self, msg):
        """Log a message, replacing the last line. For us with progress bars mainly."""
        self.log_w_replace_signal.emit(msg)

    def warn(self, msg):
        """Log a warning."""
        self.warn_signal.emit(msg)

    def run_tracking(
        self,
        video: np.ndarray,
        keypoints: np.ndarray,
    ):
        """Run the tracking."""
        self.log("Started tracking")
        # tracks = track_mock(video, keypoints)
        tracks = cotrack_online(video, keypoints)
        self.log("Finished tracking")
        yield tracks


def track_mock(
    video: np.ndarray,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Mocks what a tracker would do.

    This method's signature should be re-used by all trackers (PIPS++ and CoTracker).

    Args:
        video: The path to a video in which the points should be tracked.
        keypoints: The position of keypoints to track in the video. This array should
            have shape (n_animals, n_keypoints, 2), where
                n_animals: the number of animals to track
                n_keypoints: the number of keypoints to track for each individual
                2: as each point is defined by its (x, y) coordinates

    Returns:
        an array of shape (num_frames, n_animals, n_keypoints, 2) corresponding to the
        position of each keypoint in each frame of the video
    """
    return np.repeat(keypoints, (len(video), 1, 1, 1))


# TODO: REQUIRES TO RUN pip install src/co-tracker
def cotrack_online(
    video: np.ndarray,
    keypoints: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    def _process_step(window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2:]), device=device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(video_chunk, is_first_step=is_first_step, queries=queries[None])

    # model = CoTrackerOnlinePredictor(
    #     checkpoint=Path(
    #       "/home/lucas/Projects/deeplabcut-tracking/models/cotracker2.pth"
    #     )
    # )
    n_frames = len(video)
    n_animals, n_keypoints = keypoints.shape[:2]

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(device)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()
    window_frames = []

    queries = np.zeros((n_animals * n_keypoints, 3))
    queries[:, 1:] = keypoints.reshape((-1, 2))
    queries = torch.from_numpy(queries).to(device).float()

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    i = 0
    for i, frame in enumerate(video[0]):
        frame = frame.permute(1, 2, 0)
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries=queries,
            )
            is_first_step = False
        window_frames.append(frame)

    # Processing final frames in case video length is not a multiple of model.step
    # TODO: Use visibility
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1:],
        is_first_step,
        queries=queries,
    )
    print()
    tracks = pred_tracks.squeeze().cpu().numpy()
    return tracks.reshape((n_frames, n_animals, n_keypoints, 2))


def track_cotracker(
    video: Path,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Tracks keypoints in a video using CoTracker.

    Args:
        video: The path to a video in which the points should be tracked.
        keypoints: The position of keypoints to track in the video. This array should
            have shape (n_animals, n_keypoints, 2), where
                n_animals: the number of animals to track
                n_keypoints: the number of keypoints to track for each individual
                2: as each point is defined by its (x, y) coordinates

    Returns:
        an array of shape (num_frames, n_animals, n_keypoints, 2) corresponding to the
        position of each keypoint in each frame of the video
    """
    # TODO: Implement your code here!


def track_pips(
    video: Path,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Tracks keypoints in a video using PIPS++.

    Args:
        video: The path to a video in which the points should be tracked.
        keypoints: The position of keypoints to track in the video. This array should
            have shape (n_animals, n_keypoints, 2), where
                n_animals: the number of animals to track
                n_keypoints: the number of keypoints to track for each individual
                2: as each point is defined by its (x, y) coordinates

    Returns:
        an array of shape (num_frames, n_animals, n_keypoints, 2) corresponding to the
        position of each keypoint in each frame of the video
    """
    # TODO: Implement your code here!


if __name__ == "__main__":
    import glob
    import json
    import imageio
    import os

    def load_video_from_frames_folder(video_path):
        if os.path.isdir(video_path):
            # list the frames in video_path
            # get rgb frames
            filenames = glob.glob(f'{video_path}/*.jpg')
            # sort the filenames by filename number
            video_frames = sorted(
                filenames, key=lambda x: int(x.split('/')[-1].split('.')[0])
                )

            frames = []
            for im in video_frames:
                im = imageio.v2.imread(im)
                frames.append(im)  # H, W, C
            rgbs = np.stack(frames, axis=0)
            rgbs = rgbs[:, :, :, ::-1].copy()
            return rgbs
        else:
            raise ValueError(f'video_path {video_path} is not a directory')


    def load_init_pose(data_path):

        with open(data_path, "r") as f:
            data = json.load(f)

        img_h = data['imageHeight']
        img_w = data['imageWidth']

        labels = []
        for shape in data['shapes']:
            labels.append(int(shape["group_id"]))
        init_frame_kpts = np.zeros((max(labels), 17, 2))

        for shape in data["shapes"]:
            p = np.array(shape["points"])
            if shape["shape_type"] == "point" and shape["label"] != "snow":
                init_frame_kpts[
                    int(shape["group_id"]) - 1, int(shape["label"]) - 1, 0] = p[:, 0]
                init_frame_kpts[
                    int(shape["group_id"]) - 1, int(shape["label"]) - 1, 1] = p[:, 1]

        return init_frame_kpts, img_h, img_w

    video = load_video_from_frames_folder(
        "/Users/niels/Documents/upamathis/events/2024_04_lemanic_sv_hackathon/deeplabcut-tracking/data 2/7chimps/v2c5"
    )
    keypoints, _, _ = load_init_pose(
        "/Users/niels/Documents/upamathis/events/2024_04_lemanic_sv_hackathon/deeplabcut-tracking/data 2/7chimps/v2c5/0000.json"
    )
    print(keypoints.shape)
    tracks = cotrack_online(
        video,
        keypoints,
        device="cpu",
    )
    print(tracks.shape)
