from __future__ import annotations

import numpy as np


def _bool_calculate_kernel(img_cal: np.ndarray, length: int, width: int, height: int) -> int:
    i = length
    j = width
    k = height
    kernel_data = [
        img_cal[i - 1, j - 1, k - 1],
        img_cal[i - 1, j - 1, k],
        img_cal[i - 1, j - 1, k + 1],
        img_cal[i - 1, j, k - 1],
        img_cal[i - 1, j, k],
        img_cal[i - 1, j, k + 1],
        img_cal[i - 1, j + 1, k - 1],
        img_cal[i - 1, j + 1, k],
        img_cal[i - 1, j + 1, k + 1],
        img_cal[i, j - 1, k - 1],
        img_cal[i, j - 1, k],
        img_cal[i, j - 1, k + 1],
        img_cal[i, j, k - 1],
        img_cal[i, j, k + 1],
        img_cal[i, j + 1, k - 1],
        img_cal[i, j + 1, k],
        img_cal[i, j + 1, k + 1],
        img_cal[i + 1, j - 1, k - 1],
        img_cal[i + 1, j - 1, k],
        img_cal[i + 1, j - 1, k + 1],
        img_cal[i + 1, j, k - 1],
        img_cal[i + 1, j, k],
        img_cal[i + 1, j, k + 1],
        img_cal[i + 1, j + 1, k - 1],
        img_cal[i + 1, j + 1, k],
        img_cal[i + 1, j + 1, k + 1],
    ]
    arr_kernel_data = np.asarray(kernel_data)
    return 1 if (arr_kernel_data > 1).any() else 0


def calculate_edge(seg_label: np.ndarray) -> np.ndarray:
    img_cal = seg_label
    img_shape = seg_label.shape
    edge_label_arr = np.zeros(img_shape, dtype=np.float32)

    img_length, img_width, img_height = img_shape
    for i in range(1, img_length - 1):
        for j in range(1, img_width - 1):
            for k in range(1, img_height - 1):
                if img_cal[i, j, k] == 1 and _bool_calculate_kernel(img_cal, i, j, k) == 1:
                    edge_label_arr[i, j, k] = 1
    return edge_label_arr


def calculate_voxels_distance(edge_indexes_arr: np.ndarray, length: int, width: int, height: int) -> float:
    voxel_set = np.asarray(edge_indexes_arr)
    current_set = np.asarray([length, width, height])
    dis_cur = voxel_set - current_set
    distance_matrix = (
        np.power(dis_cur[:, 0], 2)
        + np.power(dis_cur[:, 1], 2)
        + np.power(dis_cur[:, 2], 2)
    )
    return float(np.sqrt(np.min(distance_matrix)))


def distance_map(seg_label: np.ndarray, edge_label: np.ndarray) -> np.ndarray:
    background_param = -10
    seg_label_arr = seg_label
    edge_label_arr = edge_label
    distance_map_arr = np.ones(seg_label_arr.shape, dtype=np.float32) * background_param
    edge_indexes_arr = np.argwhere(edge_label_arr == 1)

    if edge_indexes_arr.shape[0] <= 50:
        return np.ones_like(seg_label_arr, dtype=np.float32)

    img_length, img_width, img_height = edge_label_arr.shape
    for i in range(1, img_length - 1):
        for j in range(1, img_width - 1):
            for k in range(1, img_height - 1):
                if seg_label_arr[i, j, k] != -1:
                    distance_map_arr[i, j, k] = calculate_voxels_distance(edge_indexes_arr, i, j, k)
    return distance_map_arr


def dis_map_weight_relu(distance_map_matrix: np.ndarray) -> np.ndarray:
    matrix_relu = np.zeros_like(distance_map_matrix, dtype=np.float32)
    max_value = float(np.max(distance_map_matrix))
    if max_value <= 0:
        matrix_relu[distance_map_matrix == -10] = 100
        matrix_relu[distance_map_matrix != -10] = 10
    else:
        matrix_relu[distance_map_matrix == -10] = 100
        matrix_relu[distance_map_matrix != -10] = distance_map_matrix[distance_map_matrix != -10] / max_value * 10
    # Background voxels intentionally use a large value and should saturate
    # back to 0.2. Clip the exponent input to avoid harmless float32 overflow
    # warnings while preserving the FracSegNet logistic shaping.
    exp_input = np.clip(matrix_relu.astype(np.float64, copy=False) - 5.0, -60.0, 60.0)
    matrix_relu = np.reciprocal(1.0 + np.exp(exp_input)) * 0.8 + 0.2
    return matrix_relu.astype(np.float32, copy=False)


def calculate_dis_map(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    if np.unique(label).max() > 1:
        seg_label = label[0]
        edge_label = calculate_edge(seg_label)
        distance_map_matrix = distance_map(seg_label, edge_label)
        distance_map_matrix = dis_map_weight_relu(distance_map_matrix)
        return distance_map_matrix[None].astype(np.float32, copy=False)
    return (np.ones_like(label, dtype=np.float32) * 0.2).astype(np.float32, copy=False)


calculate_disMap = calculate_dis_map
