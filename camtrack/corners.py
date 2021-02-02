#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    N = 5000
    block_size = 7
    quality = 0.01
    min_distance = 10
    win_size = (15, 15)
    max_level = 2
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    feature_params = dict(maxCorners=N,
                          qualityLevel=quality,
                          minDistance=min_distance,
                          blockSize=block_size)
    lk_params = dict(winSize=win_size,
                     maxLevel=max_level,
                     criteria=criteria)

    image_0 = frame_sequence[0]
    old_frame_corners = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params)
    old_corners = FrameCorners(
        np.array([i for i in range(len(old_frame_corners))]),
        old_frame_corners,
        np.array([block_size] * len(old_frame_corners))
    )
    last_id = len(old_frame_corners) - 1
    image_0 = (image_0 * 256).astype(np.uint8)

    builder.set_corners_at_frame(0, old_corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        new_frame_corners = cv2.goodFeaturesToTrack(image_1.astype(np.float32), mask=None, **feature_params)
        new_frame_corners = np.array([[x[0][0], x[0][1]] for x in new_frame_corners])

        image_1 = (image_1 * 256).astype(np.uint8)
        new_corners, status, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, old_frame_corners, None, **lk_params)

        good_new = new_corners[status == 1]

        good_new = np.concatenate((good_new, new_frame_corners))
        size = min(N, len(good_new))
        good_new = good_new[:size]

        ids = np.array([old_corners.ids[i][0] for i in range(len(status)) if status[i] == 1], dtype=np.int32)
        old_len = len(ids)
        new_ids = np.array([i for i in range(last_id + 1, last_id + 1 + size - old_len)], dtype=np.int32)
        last_id = last_id + size - old_len
        ids = np.concatenate((ids, new_ids))

        corners = FrameCorners(
            ids,
            good_new,
            np.array([block_size] * size)
        )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1
        old_frame_corners = good_new.reshape(-1, 1, 2)
        old_corners = corners


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
