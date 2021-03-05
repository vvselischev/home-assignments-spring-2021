#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import sys
from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4
)


def solve_homogeneous(U):
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))
    return e_vecs[:, np.argmin(e_vals)]


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    max_reprojection_error = 5
    min_angle = 5
    start_triangulation_parameters = TriangulationParameters(7, 0, 0)
    triangulation_parameters = TriangulationParameters(max_reprojection_error, min_angle, 0)

    first_view = pose_to_view_mat3x4(known_view_1[1])
    second_view = pose_to_view_mat3x4(known_view_2[1])

    first_frame = known_view_1[0]
    second_frame = known_view_2[0]

    frame_corrs = build_correspondences(corner_storage[first_frame], corner_storage[second_frame])
    (points_3d, ids, median_cos) = triangulate_correspondences(
        frame_corrs, first_view, second_view, intrinsic_mat, start_triangulation_parameters)

    builder = PointCloudBuilder()
    builder.add_points(ids, points_3d)

    frame_count = len(corner_storage)
    unknown_frames = set([i for i in range(frame_count)])
    unknown_frames.remove(first_frame)
    unknown_frames.remove(second_frame)
    known_frames = {first_frame, second_frame}

    failed_frames = set()

    view_mats = [eye3x4()] * frame_count
    view_mats[first_frame] = first_view
    view_mats[second_frame] = second_view

    outliers = set()

    while len(unknown_frames) > 0:
        # choose the unknown frame as the closest to the first known frame
        current_known_frame = min(known_frames)
        best_unknown_frame = current_known_frame - 1
        best_known_frame = current_known_frame
        while best_unknown_frame >= 0 and best_unknown_frame not in unknown_frames:
            best_unknown_frame -= 1
            best_known_frame = best_unknown_frame + 1
        if best_unknown_frame == -1:
            best_unknown_frame = current_known_frame + 1
            while best_unknown_frame < frame_count and best_unknown_frame not in unknown_frames:
                best_unknown_frame += 1
                best_known_frame = best_unknown_frame - 1

        if best_unknown_frame < 0 or best_unknown_frame >= frame_count or best_unknown_frame not in unknown_frames:
            break

        # build correspondences
        frame_corrs = build_correspondences(corner_storage[best_known_frame], corner_storage[best_unknown_frame])
        best_corrs = frame_corrs

        if best_corrs is None:
            break

        # retrive known 3d points and filter outliers
        known_ids_on_best_frame = snp.intersect(best_corrs.ids, builder.ids.flatten())
        known_ids_on_best_frame = [x for x in known_ids_on_best_frame if x not in outliers]

        best_ids_indices = np.searchsorted(best_corrs.ids, known_ids_on_best_frame)
        best_3d_ids_indices = np.searchsorted(builder.ids.flatten(), known_ids_on_best_frame)

        points_2d = corner_storage[best_unknown_frame].points[best_ids_indices]
        points_3d = builder.points[best_3d_ids_indices]

        if len(points_2d) < 6 or len(points_3d) < 6:
            break

        # sometimes I'm getting a strange exception about count(points) == 5 (expected 6)
        # while the actual number of passed points is large
        # however, that happens very rarely, so can ignore such frames
        try:
            retval, R, t, inliers = cv2.solvePnPRansac(points_3d,
                                                       points_2d,
                                                       intrinsic_mat,
                                                       None,
                                                       reprojectionError=max_reprojection_error)
        except:
            unknown_frames.remove(best_unknown_frame)
            continue

        if not retval:
            unknown_frames.remove(best_unknown_frame)
            continue

        print(f'Processing frame {frame_count - len(unknown_frames)}/{frame_count}: '
              f'inliers count={len(inliers)}, '
              f'cloud size={len(builder.ids)},'
              f'used correspondences={len(points_2d)}', file=sys.stdout)

        # add the current view
        current_view = rodrigues_and_translation_to_view_mat3x4(R, t)
        view_mats[best_unknown_frame] = current_view

        # update outliers
        current_outliers = np.delete(np.arange(best_ids_indices.size, dtype=np.int), inliers.astype(np.int))
        outliers_ids = corner_storage[best_unknown_frame].ids[best_ids_indices[current_outliers]].flatten()
        outliers.update(outliers_ids)

        # retriangulate known frames with the current one
        for known_frame in known_frames:
            to_delete = np.array(list(outliers)) if len(outliers) > 0 else None
            frame_corrs = build_correspondences(corner_storage[known_frame],
                                                corner_storage[best_unknown_frame],
                                                to_delete)
            first_view = view_mats[known_frame]
            second_view = view_mats[best_unknown_frame]
            (points_3d, ids, median_cos) = triangulate_correspondences(
                frame_corrs, first_view, second_view, intrinsic_mat, triangulation_parameters)

            builder.add_points(ids, points_3d)

        unknown_frames.remove(best_unknown_frame)
        known_frames.add(best_unknown_frame)

    # set view matrices of unknown frames equal to the closest known frame
    # (actually, this part can be improved)
    start_frame = first_frame - 1
    while start_frame >= 0:
        if start_frame not in known_frames:
            view_mats[start_frame] = view_mats[start_frame + 1]
        start_frame -= 1
    start_frame = first_frame + 1
    while start_frame < frame_count:
        if start_frame not in known_frames:
            view_mats[start_frame] = view_mats[start_frame - 1]
        start_frame += 1

    calc_point_cloud_colors(
        builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
