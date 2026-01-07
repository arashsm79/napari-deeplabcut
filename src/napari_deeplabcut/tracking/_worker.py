import enum
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qtpy.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DEBUG = True
if DEBUG:
    import debugpy


@dataclass
class TorchHubModel:
    org: str
    model: str


@dataclass(frozen=True)
class TrackerInfo:
    name: str
    torchhub: TorchHubModel
    info_text: str = ""

    def load_model(self, device: str):
        import torch

        model = torch.hub.load(
            self.torchhub.org,
            self.torchhub.model,
        ).to(device)
        return model

    @staticmethod
    def get_info_from_name(name: str) -> "TrackerInfo | None":
        """Get the TrackerInfo dataclass from its name."""
        if name is not None:
            for tracker in TrackerType:
                if tracker.value.name == name:
                    return tracker.value
        return None


class TrackerType(enum.Enum):
    COTRACKER = TrackerInfo(
        name="Cotracker 3",
        torchhub=TorchHubModel(
            org="facebookresearch/co-tracker",
            model="cotracker3_online",
        ),
        info_text="Cotracker 3 model from Facebook Research.\n"
        "See https://cotracker3.github.io/ and CoTracker3: "
        "Simpler and Better Point Tracking by "
        "Pseudo-Labelling Real Videos by Karaev et al., 2024.",
    )

    def get_all_names() -> list[str]:
        """Get a list of all implemented tracker names."""
        return [tracker.value.name for tracker in TrackerType]

    def get_from_name(name: str) -> "TrackerType | None":
        """Get the TrackerType enum member from its name."""
        for tracker in TrackerType:
            if tracker.value.name == name:
                return tracker
        return None


@dataclass
class TrackingWorkerData:
    tracker: TrackerType
    video: np.ndarray
    keypoints: np.ndarray  # (num_keypoint, 3)
    # [0]: frame number in `video` [1]: x, [2]: y
    keypoint_features: dict
    keypoint_range: tuple[int, int]
    backward_tracking: bool


class TrackingWorker(QObject):
    started = Signal()
    finished = Signal()
    progress = Signal(tuple)
    trackingStarted = Signal()
    trackingFinished = Signal(TrackingWorkerData)
    trackingStopped = Signal()

    def __init__(self):
        super().__init__()
        import torch

        self.is_stopped = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO implement MPS support
        self.model: object | None = None

    @Slot(TrackingWorkerData)
    def track(self, cfg: TrackingWorkerData):
        if DEBUG:
            debugpy.debug_this_thread()
        import torch

        if self.model is None:
            self.model = cfg.tracker.value.load_model(self.device)
            self.model.eval()

        def _process_step(window_frames, is_first_step, queries):
            # NOTE ideally each model should have its own processing
            # function implemented separately and return the same format
            video_chunk = (
                torch.tensor(np.stack(window_frames[-self.model.step * 2 :]), device=self.device)
                .float()
                .permute(0, 3, 1, 2)[None]
            )  # (1, T, 3, H, W)
            logger.debug(f"Video chunk shape: {video_chunk.shape},Queries shape: {queries.shape}")
            return self.model(
                video_chunk,
                is_first_step=is_first_step,
                queries=queries[None],
                add_support_grid=True,
            )

        # video is originally of shape (num_frames, height, width, channels)
        video = np.array(cfg.video)
        window_frames = []

        # We need to swap x, y so that it matches what cotracker expects
        cfg.keypoints[:, [1, 2]] = cfg.keypoints[:, [2, 1]]

        queries = torch.from_numpy(cfg.keypoints).to(self.device).float()

        # Iterating over video frames, processing one window at a time:
        is_first_step = True
        for i, frame in enumerate(video):
            if i % self.model.step == 0 and i != 0:
                pred_tracks, _pred_visibility = _process_step(window_frames, is_first_step, queries=queries)
                is_first_step = False
            window_frames.append(frame)
            self.progress.emit((i, len(video)))
            if self._should_stop():
                return

        # Processing final frames in case video length is not a multiple of model.step
        # TODO: Use visibility
        logger.debug(f"Window frames shape before final processing:{np.array(window_frames).shape}")
        pred_tracks, _pred_visibility = _process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1 :],
            is_first_step,
            queries=queries,
        )
        if DEBUG and pred_tracks is None:
            debugpy.breakpoint()
        logger.debug(f"Predicted tracks : {pred_tracks}")
        logger.debug(f"Predicted visibility: {_pred_visibility}")
        self.progress.emit((len(video), len(video)))

        tracks = pred_tracks.squeeze().cpu().numpy()
        # drop the support grid (necessary only for cotracker version < 3)
        # as we are using ct3, we skip dropping the support grid
        # tracks = tracks[:, :cfg.keypoints.shape[0], :]
        tracks = tracks.reshape(-1, 2)
        if cfg.backward_tracking:
            tracks = tracks[::-1]
        frame_ids = np.repeat(
            np.arange(cfg.keypoint_range[0], cfg.keypoint_range[1]),
            cfg.keypoints.shape[0],
        )
        tracks = np.column_stack((frame_ids, tracks))
        cfg.keypoint_features = pd.concat([cfg.keypoint_features] * len(np.unique(tracks[:, 0])), ignore_index=True)
        cfg.keypoints = tracks
        cfg.keypoints[:, [1, 2]] = cfg.keypoints[:, [2, 1]]
        self.trackingFinished.emit(cfg)

        # self.model = None  # free up memory ?

    def run(self):
        self.started.emit()

    def start(self):
        self.thread = QThread()
        self.moveToThread(self.thread)

        self.finished.connect(self.thread.quit)
        self.thread.started.connect(self.run)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot()
    def stop_tracking(self):
        self.is_stopped = True

    def _should_stop(self) -> bool:
        QCoreApplication.processEvents()
        if self.is_stopped:
            self.trackingStopped.emit()
            self.is_stopped = False
            return True
        return False
