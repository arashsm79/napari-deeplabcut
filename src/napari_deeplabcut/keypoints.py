from collections import namedtuple
from enum import auto
from typing import List, Sequence

import numpy as np
from napari.layers import Points

from napari_deeplabcut.misc import CycleEnum


class LabelMode(CycleEnum):
    """
    Labeling modes.
    SEQUENTIAL: points are placed in sequence, then frame after frame;
        clicking to add an already annotated point has no effect.
    QUICK: similar to SEQUENTIAL, but trying to add an already
        annotated point actually moves it to the cursor location.
    LOOP: the first point is placed frame by frame, then it wraps
        to the next label at the end and restart from frame 1, etc.
    """

    SEQUENTIAL = auto()
    QUICK = auto()
    LOOP = auto()

    @classmethod
    def default(cls):
        return cls.SEQUENTIAL


Keypoint = namedtuple("Keypoint", ["label", "id"])


class KeypointStore:
    def __init__(self, viewer, layer: Points):
        self.viewer = viewer
        self.layer = layer
        all_pairs = self.layer.metadata["header"].form_individual_bodypart_pairs()
        self._keypoints = [
            Keypoint(label, id_) for id_, label in all_pairs
        ]  # Ordered references to all possible keypoints
        self.viewer.dims.set_current_step(0, 0)

    @property
    def current_step(self):
        return self.viewer.dims.current_step[0]

    @property
    def n_steps(self):
        return self.viewer.dims.nsteps[0]

    @property
    def annotated_keypoints(self) -> List[Keypoint]:
        mask = self.current_mask
        labels = self.layer.properties["label"][mask]
        ids = self.layer.properties["id"][mask]
        return [Keypoint(label, id_) for label, id_ in zip(labels, ids)]

    @property
    def current_mask(self) -> Sequence[bool]:
        # return self.layer.data[:, 0] == self.current_step
        return np.asarray(self.layer.data[:, 0] == self.layer._slice_indices[0])

    @property
    def current_keypoint(self) -> Keypoint:
        props = self.layer.current_properties
        return Keypoint(label=props["label"][0], id=props["id"][0])

    @current_keypoint.setter
    def current_keypoint(self, keypoint: Keypoint):
        # Avoid changing the properties of a selected point
        if not len(self.layer.selected_data):
            current_properties = self.layer.current_properties
            current_properties["label"] = np.asarray([keypoint.label])
            current_properties["id"] = np.asarray([keypoint.id])
            self.layer.current_properties = current_properties

    def smart_reset(self, event):
        """Set current keypoint to the first unlabeled one."""
        unannotated = ""
        already_annotated = self.annotated_keypoints
        for keypoint in self._keypoints:
            if keypoint not in already_annotated:
                unannotated = keypoint
                break
        self.current_keypoint = unannotated if unannotated else self._keypoints[0]

    def next_keypoint(self, *args):
        ind = self._keypoints.index(self.current_keypoint) + 1
        if ind <= len(self._keypoints) - 1:
            self.current_keypoint = self._keypoints[ind]

    def prev_keypoint(self, *args):
        ind = self._keypoints.index(self.current_keypoint) - 1
        if ind >= 0:
            self.current_keypoint = self._keypoints[ind]

    @property
    def labels(self) -> List[str]:
        return self.layer.metadata["header"].bodyparts

    @property
    def current_label(self) -> str:
        return self.layer.current_properties["label"][0]

    @current_label.setter
    def current_label(self, label: str):
        if not len(self.layer.selected_data):
            current_properties = self.layer.current_properties
            current_properties["label"] = np.asarray([label])
            self.layer.current_properties = current_properties

    @property
    def ids(self) -> List[str]:
        return self.layer.metadata["header"].individuals

    @property
    def current_id(self) -> str:
        return self.layer.current_properties["id"][0]

    @current_id.setter
    def current_id(self, id_: str):
        if not len(self.layer.selected_data):
            current_properties = self.layer.current_properties
            current_properties["id"] = np.asarray([id_])
            self.layer.current_properties = current_properties

    def _advance_step(self, event):
        ind = (self.current_step + 1) % self.n_steps
        self.viewer.dims.set_current_step(0, ind)


def _add(store, coord):
    if store.current_keypoint not in store.annotated_keypoints:
        store.layer.data = np.append(
            store.layer.data,
            np.atleast_2d(coord),
            axis=0,
        )
    elif store.layer.metadata["controls"]._label_mode is LabelMode.QUICK:
        ind = store.annotated_keypoints.index(store.current_keypoint)
        data = store.layer.data
        data[np.flatnonzero(store.current_mask)[ind]] = coord
        store.layer.data = data
    store.layer.selected_data = set()
    if store.layer.metadata["controls"]._label_mode is LabelMode.LOOP:
        store.layer.events.query_next_frame()
    else:
        store.next_keypoint()