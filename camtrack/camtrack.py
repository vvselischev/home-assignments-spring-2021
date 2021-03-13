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
    eye3x4,
    compute_reprojection_errors
)


def init_camera(intrinsic_mat, corner_storage: CornerStorage):
    threshold = 1
    confidence = 0.99
    reprojection_error = 2
    min_angle = 1
    max_iters = 300
    essential_validation_threshold = 0.9
    min_inliers = 10

    frame_count = len(corner_storage)

    best_view = None
    best_global_error = 10000
    best_first_frame = 0
    best_second_frame = 1
    init_view = np.hstack((np.eye(3), np.zeros((3, 1))))
    steps = [i for i in range(4, 30)]


    while True:
        reprojection_error = min(8, reprojection_error + 1)
        triangulation_parameters = TriangulationParameters(reprojection_error, min_angle, 0)

        for first_frame in range(min(int(frame_count * 0.7), 150)):
            for step in steps:
                second_frame = first_frame - step
                if second_frame < 0:
                    break
                print(f'{first_frame} / {frame_count}')
                corrs = build_correspondences(corner_storage[first_frame], corner_storage[second_frame])
                first_points = corrs.points_1
                second_points = corrs.points_2
                corr_ids = corrs.ids

                essential_mat, essential_mask = cv2.findEssentialMat(first_points, second_points, intrinsic_mat,
                                                                     method=cv2.RANSAC,
                                                                     threshold=threshold,
                                                                     prob=confidence)

                homography, homography_mask = cv2.findHomography(first_points, second_points,
                                                                 method=cv2.RANSAC,
                                                                 ransacReprojThreshold=reprojection_error,
                                                                 confidence=confidence,
                                                                 maxIters=max_iters)

                essential_mask = essential_mask.flatten()
                homography_mask = homography_mask.flatten()

                essential_first_inliers = first_points[essential_mask == 1]
                essential_second_inliers = second_points[essential_mask == 1]

                outlier_ids = corr_ids[essential_mask == 0]

                homography_first_inliers = first_points[homography_mask == 1]

                inlier_fraction = len(homography_first_inliers) / len(essential_first_inliers)

                if inlier_fraction < essential_validation_threshold or len(essential_first_inliers) < min_inliers:
                    continue

                R1, R2, t = cv2.decomposeEssentialMat(essential_mat)
                candidates = [np.hstack((R1, t)),
                              np.hstack((R1, -t)),
                              np.hstack((R2, t)),
                              np.hstack((R2, -t))]

                corrs = build_correspondences(corner_storage[first_frame], corner_storage[second_frame], outlier_ids)

                best_candidate = None
                best_points_count = 0

                for candidate_view in candidates:
                    points_3d, ids, _ = triangulate_correspondences(
                        corrs, init_view, candidate_view, intrinsic_mat, triangulation_parameters)
                    if len(points_3d) > best_points_count and len(points_3d) > min_inliers:
                        best_candidate = candidate_view
                        best_points_count = len(points_3d)

                if best_candidate is None:
                    continue

                points_3d, ids, _ = triangulate_correspondences(
                    corrs, init_view, best_candidate, intrinsic_mat, triangulation_parameters)
                ids = set([i for i in ids])
                first_points_indices = [i for i in range(len(corner_storage[first_frame].ids))
                                        if corner_storage[first_frame].ids[i][0] in ids]
                second_points_indices = [i for i in range(len(corner_storage[second_frame].ids))
                                        if corner_storage[second_frame].ids[i][0] in ids]
                sum_error = compute_reprojection_errors(points_3d, corner_storage[first_frame].points[first_points_indices],
                                                        intrinsic_mat @ init_view) + \
                            compute_reprojection_errors(points_3d, corner_storage[second_frame].points[second_points_indices],
                                                        intrinsic_mat @ best_candidate)
                current_error = np.mean(sum_error)

                if current_error > 1 or np.linalg.norm(best_candidate[:, 3]) < 0.9:
                    continue

                if best_candidate is not None and current_error < best_global_error:
                    best_global_error = current_error
                    best_view = best_candidate
                    best_first_frame = first_frame
                    best_second_frame = second_frame

        if best_view is not None:
            return best_first_frame, best_second_frame, init_view, best_view, reprojection_error, min_angle


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        first_frame, second_frame, first_view, second_view, max_reprojection_error, min_angle = \
            init_camera(intrinsic_mat, corner_storage)
        print(f'Using frames: {first_frame}, {second_frame}')
    else:
        first_view = pose_to_view_mat3x4(known_view_1[1])
        second_view = pose_to_view_mat3x4(known_view_2[1])
        first_frame = known_view_1[0]
        second_frame = known_view_2[0]
        max_reprojection_error = 5
        min_angle = 2.2

    start_triangulation_parameters = TriangulationParameters(4, 0, 0)
    triangulation_parameters = TriangulationParameters(3, 0.8, 0)

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

        if len(points_2d) < 4 or len(points_3d) < 4:
            break

        retval, R, t, inliers = cv2.solvePnPRansac(points_3d,
                                                   points_2d,
                                                   intrinsic_mat,
                                                   None,
                                                   confidence=0.995,
                                                   flags=cv2.SOLVEPNP_EPNP,
                                                   reprojectionError=2)
        if retval:
            retval, R, t = cv2.solvePnP(points_3d[inliers.flatten()],
                                        points_2d[inliers.flatten()],
                                        intrinsic_mat,
                                        None,
                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                        useExtrinsicGuess=True,
                                        rvec=R,
                                        tvec=t)

        if not retval:
            unknown_frames.remove(best_unknown_frame)
            continue

        print(f'Processing frame {frame_count - len(unknown_frames)}/{frame_count}: '
              f'inliers count={len(inliers)}, '
              f'cloud size={len(builder.ids)}, '
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
            try:
                (points_3d, ids, median_cos) = triangulate_correspondences(
                    frame_corrs, first_view, second_view, intrinsic_mat, triangulation_parameters)
            except:
                continue

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
