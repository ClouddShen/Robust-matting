import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
from tqdm import tqdm
import os
import pickle
import argparse

# c_color: current_color
# f_color: foreground_color
# b_color: background_color
def calculate_alpha(c_color, f_color, b_color):
    nominator = (c_color - b_color) @ (f_color - b_color)
    tmp = f_color - b_color
    denominator = tmp @ tmp + epsilon
    return nominator / denominator


def calculate_distance_ratio_sq(c_color, f_color, b_color, alpha):
    tmp = c_color - (alpha * f_color + (1 - alpha) * b_color)
    nominator = tmp @ tmp
    tmp = f_color - b_color
    denominator = tmp @ tmp + epsilon
    return nominator / denominator


def calculate_f_b_weight(c_color, fb_color, min_dist_fb_c_sq):
    tmp = fb_color - c_color
    nominator = tmp @ tmp
    tmp = nominator / (min_dist_fb_c_sq + epsilon)
    return np.exp(-tmp)


def calculate_confidence(distance_ratio_sq, f_weight, b_weight):
    ro = 0.1
    nominator = distance_ratio_sq * f_weight * b_weight
    denominator = ro * ro
    return np.exp(-(nominator / denominator))


class Sample:
    def __init__(self, row, col, idx, color):
        self.row = row
        self.col = col
        self.idx = idx
        self.color = color

    def __str__(self):
        info = f"row: {self.row}\n" \
               + f"col: {self.col}\n" \
               + f"idx: {self.idx}\n" \
               + f"color: {self.color}"
        return info


def select_boundary_samples(x, y, store_edge, map_edge, n_sample=20):
    num_pixels = store_edge.shape[0]
    dt = np.dtype([('idx', 'i4'), ('dis_sq', 'f8')])
    structured_idx_dis_sq_pairs = np.zeros(num_pixels, dtype=dt)
    n_col = map_edge.shape[1]

    for k in range(num_pixels):
        i = store_edge[k][0]
        j = store_edge[k][1]
        idx = i * n_col + j
        dis_sq = (x - i) ** 2 + (y - j) ** 2
        structured_idx_dis_sq_pairs[k][0] = idx
        structured_idx_dis_sq_pairs[k][1] = dis_sq

    structured_idx_dis_sq_pairs = np.sort(structured_idx_dis_sq_pairs, order="dis_sq")
    sample_list = []
    gap = 1
    pick = 0
    for i in range(n_sample):
        idx = structured_idx_dis_sq_pairs[pick][0]
        row = int(idx // n_col)
        col = int(idx % n_col)
        sample_list.append(Sample(row, col, idx, image[row][col]))
        pick += gap
    return sample_list


def color_dist(c1, c2):
    tmp = c1 - c2
    return tmp @ tmp


def calculate_min_dist_fb_c_sq(c_color, sample_list):
    dists = np.zeros(len(sample_list))
    for idx, sample in enumerate(sample_list):
        dists[idx] = color_dist(c_color, sample.color)
    return min(dists)


def calculate_alpha_confidence(c_color, f_samples, b_samples):
    n_f_samples = len(f_samples)
    n_b_samples = len(b_samples)
    n_pairs = n_f_samples * n_b_samples
    alpha_confidence_pairs = np.zeros((n_pairs, 2))  # [alpha, confidence]
    min_dist_f_c_sq = calculate_min_dist_fb_c_sq(c_color, f_samples)
    min_dist_b_c_sq = calculate_min_dist_fb_c_sq(c_color, b_samples)
    k = 0
    for i in range(n_f_samples):
        f_color = f_samples[i].color
        for j in range(n_b_samples):
            b_color = b_samples[j].color
            alpha = calculate_alpha(c_color, f_color, b_color)
            distance_ratio_sq = calculate_distance_ratio_sq(c_color, f_color, b_color, alpha)
            f_weight = calculate_f_b_weight(c_color, f_color, min_dist_f_c_sq)
            b_weight = calculate_f_b_weight(c_color, b_color, min_dist_b_c_sq)
            confidence = calculate_confidence(distance_ratio_sq, f_weight, b_weight)

            # for alphas < 0.05 or alpha > 1.05, assign its confidence to 0
            # because we don't want the weird alphas have high confidence
            if alpha < -0.05:
                confidence = 0
                alpha = 0
            elif alpha > 1.05:
                confidence = 0
                alpha = 1
            elif -0.05 < alpha < 0:
                alpha = 0
            elif 1 < alpha < 1.05:
                alpha = 1

            alpha_confidence_pairs[k][0] = alpha
            alpha_confidence_pairs[k][1] = confidence
            k += 1
    return alpha_confidence_pairs


# take 3 pair of [alpha, confidence] with the highest the confidence
# average over alpha and confidence to get [mean_alpha, mean_confidence]
def calculate_avg_alpha_confidence(alpha_confidence_pairs, num_pairs=3):
    dt = np.dtype([('alpha', 'f8'), ('confidence', 'f8')])
    l = alpha_confidence_pairs.shape[0]
    structured_alpha_confidence_pairs = np.zeros(l, dtype=dt)
    structured_alpha_confidence_pairs['alpha'] = np.ravel(alpha_confidence_pairs[:, 0])
    structured_alpha_confidence_pairs['confidence'] = np.ravel(alpha_confidence_pairs[:, 1])
    structured_alpha_confidence_pairs = np.sort(structured_alpha_confidence_pairs, order="confidence")[::-1]

    sum_alpha = 0
    sum_confidence = 0
    for i in range(num_pairs):
        sum_alpha += structured_alpha_confidence_pairs[i]["alpha"]
        sum_confidence += structured_alpha_confidence_pairs[i]["confidence"]
    avg_alpha = sum_alpha / num_pairs
    avg_confidence = sum_confidence / num_pairs
    return np.array((avg_alpha, avg_confidence))


def calculate_all_alpha_using_boundary_sampling():
    initial_alpha = trimap / 255 * 1.0
    initial_confidence = np.ones(initial_alpha.shape)

    height = image.shape[0]
    width = image.shape[1]
    sample_data_path = os.path.join(sample_data_dir, args.sample_name)
    if not os.path.exists(sample_data_path):
        dist_map = calculate_boundary_sampling()
    else:
        with open(sample_data_path, "rb") as f:
            dist_map = pickle.load(f)

    for i in tqdm(range(height)):
        for j in range(width):
            if un_map[i][j] != 0:
                f_samples = dist_map[i][j][0:n_sample]
                b_samples = dist_map[i][j][n_sample:]

                alpha_confidence_pairs = calculate_alpha_confidence(image[i][j], f_samples, b_samples)
                avg_alpha_confidence_pair = calculate_avg_alpha_confidence(alpha_confidence_pairs)
                initial_alpha[i][j] = avg_alpha_confidence_pair[0]
                initial_confidence[i][j] = avg_alpha_confidence_pair[1]
    return initial_alpha, initial_confidence


def calculate_boundary_sampling():
    height = image.shape[0]
    width = image.shape[1]
    dist_map = [[[Sample(0,0,0,np.array([0,0,0])) for d in range(2 * n_sample)] for j in range(width)] for i in range(height)]
    for i in tqdm(range(height)):
        for j in range(width):
            if un_map[i][j] != 0:
                f_samples = select_boundary_samples(i, j, f_store_edge, f_map_edge)
                b_samples = select_boundary_samples(i, j, b_store_edge, b_map_edge)
                dist_map[i][j][0:n_sample] = f_samples
                dist_map[i][j][n_sample:] = b_samples
    sample_data_path = os.path.join(sample_data_dir, args.sample_name)
    with open(sample_data_path, "wb") as f:
        pickle.dump(dist_map, f)
    return dist_map


def longitudinal_arrange(win_img):
    rearranged_matrix = np.zeros((9, 3)) # [3,3,3] -> [9,3]
    k = 0
    for i in range(3):
        for j in range(3):
            rearranged_matrix[k][0] = win_img[i][j][0]
            rearranged_matrix[k][1] = win_img[i][j][1]
            rearranged_matrix[k][2] = win_img[i][j][2]
            k += 1
    return rearranged_matrix


def calculate_edge_weight(height, width, fb_map_erode, Lu, Rt0):
    for m in tqdm(range(1, height - 1)):
        for n in range(1, width - 1):
            if fb_map_erode[m][n] != 0:
                continue
            win_flag_idx = idx_map[m - 1:m + 2, n - 1:n + 2]
            win_img = image[m - 1:m + 2, n - 1:n + 2]
            rearranged_matrix = longitudinal_arrange(win_img)
            each_channel_mean = np.mean(rearranged_matrix, axis=0)[None, :]
            mean9x3 = np.tile(each_channel_mean, (9, 1))
            win_covar_matrix = rearranged_matrix.T @ rearranged_matrix * (1 / 9) - each_channel_mean.T @ each_channel_mean + epsilon / 9 * np.eye(3)
            inv_win_covar_matrix = np.linalg.inv(win_covar_matrix)
            tmp = (rearranged_matrix - mean9x3) @ inv_win_covar_matrix @ (rearranged_matrix - mean9x3).T
            Wij = (tmp + 1) / 9
            for i in range(1, 9):
                for j in range(i):
                    row_i = i // 3
                    col_i = i % 3
                    row_j = j // 3
                    col_j = j % 3
                    idx_i = win_flag_idx[row_i][col_i]
                    idx_j = win_flag_idx[row_j][col_j]
                    w = Wij[i][j]
                    if idx_i > 0 and idx_j > 0:
                        continue
                    elif idx_i < 0 and idx_j < 0:
                        real_idx_i = -idx_i - 1
                        real_idx_j = -idx_j - 1
                        Lu[real_idx_i, real_idx_j] += -w # construct laplacian matrix
                        Lu[real_idx_j, real_idx_i] += -w
                        Lu[real_idx_i, real_idx_i] += w
                        Lu[real_idx_j, real_idx_j] += w
                    elif idx_i < 0 and idx_j > 0:
                        real_idx_i = -idx_i - 1
                        real_idx_j = idx_j - 1
                        Rt0[real_idx_i, real_idx_j] += -w
                        Lu[real_idx_i, real_idx_i] += w
                    else:
                        real_idx_i = idx_i - 1
                        real_idx_j = -idx_j - 1
                        Rt0[real_idx_j, real_idx_i] += -w
                        Lu[real_idx_j, real_idx_j] += w


def calculate_data_weight(Au, Ak, height, width, Lu, gamma, Rt1):
    k1 = k2 = 0
    for i in range(height):
        for j in range(width):
            idx = idx_map[i][j]
            if idx > 0:
                if f_map[i][j] == 0:
                    Ak[k1] = 0
                else:
                    Ak[k1] = 1
                k1 += 1
                continue
            alpha = initial_alpha[i][j]
            confidence = initial_confidence[i][j]
            real_idx = -idx - 1
            Lu[real_idx, real_idx] += gamma
            x1 = -gamma * (confidence * alpha + 1 - confidence)
            x2 = -gamma * confidence * alpha
            x3 = -gamma * confidence * (1 - alpha)
            x4 = -gamma * (confidence * (1 - alpha) + (1 - confidence))
            if alpha > 0.5:
                Rt1[real_idx, 0] = x1
                Rt1[real_idx, 1] = x3
            else:
                Rt1[real_idx, 0] = x2
                Rt1[real_idx, 1] = x4
            Au[k2] = alpha
            k2 += 1


def build_laplacian_matrix(un_map, fb_map):
    n_un_map = sum(np.ravel(un_map != 0))
    n_fb_map = sum(np.ravel(fb_map != 0))
    Lu = lil_matrix((n_un_map, n_un_map))
    Rt0 = lil_matrix((n_un_map, n_fb_map))
    Rt1 = lil_matrix((n_un_map, 2))

    height = image.shape[0]
    width = image.shape[1]
    fb_map_erode = cv2.erode(fb_map, np.ones(3, np.uint8))
    gamma = 0.005  # instead of 0.1 in the paper
    calculate_edge_weight(height, width, fb_map_erode, Lu, Rt0)

    Ak = np.zeros(n_fb_map)
    Au = np.zeros(n_un_map)
    calculate_data_weight(Au, Ak, height, width, Lu, gamma, Rt1)
    return Ak, Lu, Rt0, Rt1


def solve_refined_alpha_matting(un_map, fb_map):
    Ak, Lu, Rt0, Rt1 = build_laplacian_matrix(un_map, fb_map)
    n_un_map = sum(np.ravel(un_map != 0))
    onezero = np.array([1.0, 0.0])
    rhs = -(Rt0 * Ak + Rt1 * onezero)
    Au, _ = cg(Lu, rhs)
    for i in range(n_un_map):
        if Au[i] < 0.02:
            Au[i] = 0
        elif Au[i] > 0.98:
            Au[i] = 1

    result = initial_alpha.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            idx = idx_map[i][j]
            if idx < 0:
                result[i][j] = Au[-idx - 1]
    result *= 255
    return result

def store_edge_info(map_edge, num_pixels):
    store_edge = np.zeros((num_pixels, 2))
    k = 0
    height = map_edge.shape[0]
    width = map_edge.shape[1]
    for i in range(height):
        for j in range(width):
            if map_edge[i][j] != 0:
                store_edge[k][0] = i
                store_edge[k][1] = j
                k += 1
    return store_edge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="robust matting")
    parser.add_argument("-d", "--image_dir_name")
    parser.add_argument("-s", "--sample_name")
    parser.add_argument("-i", "--initial_result_name")
    parser.add_argument("-r", "--refined_result_name")
    args = parser.parse_args()

    epsilon = 1e-7
    n_sample = 20
    np.random.seed(0)

    trimap_path = os.path.join("image_data", args.image_dir_name, "trimap.png")
    image_path = os.path.join("image_data", args.image_dir_name, "image.png")

    trimap = cv2.imread(trimap_path, 0)  # numpy ndarray
    image = cv2.imread(image_path, 1) * 1.0 / 255.0
    f_map = (trimap == 255).astype("uint8")  # otherwise the dtype is bool, not supported by cv2.
    b_map = (trimap == 0).astype("uint8")
    fb_map = f_map + b_map
    un_map = 1 - fb_map

    se = np.ones([3,3], np.uint8)
    f_map_erode = cv2.erode(f_map, se)  # range: 0 or 1
    b_map_erode = cv2.erode(b_map, se)  # range: 0 or 1
    un_map_erode = cv2.erode(un_map, se) # range: 0 or 1

    f_map_edge = f_map - f_map_erode
    b_map_edge = b_map - b_map_erode
    un_map_edge = un_map - un_map_erode

    idx_map = np.zeros(image.shape[:2], dtype=np.int32)
    un_idx = -1
    fb_idx = 1
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if un_map[row][col] != 0:
                idx_map[row][col] = un_idx
                un_idx -= 1
            else:
                idx_map[row][col] = fb_idx
                fb_idx += 1


    f_num_pixels = np.sum(np.ravel(f_map_edge) != 0)
    b_num_pixels = np.sum(np.ravel(b_map_edge) != 0)
    f_store_edge = store_edge_info(f_map_edge, f_num_pixels)
    b_store_edge = store_edge_info(b_map_edge, b_num_pixels)

    sample_data_dir = os.path.join("sample_data", args.image_dir_name)
    os.makedirs(sample_data_dir, exist_ok=True)
    initial_result_dir = os.path.join("initial_result", args.image_dir_name)
    os.makedirs(initial_result_dir, exist_ok=True)
    refined_result_dir = os.path.join("refined_result", args.image_dir_name)
    os.makedirs(refined_result_dir, exist_ok=True)

    initial_result_file_name = f"{args.initial_result_name}.png"  # initial result file name
    if not os.path.exists(os.path.join(initial_result_dir, "initial_confidence")):
        initial_alpha, initial_confidence = calculate_all_alpha_using_boundary_sampling()
        cv2.imwrite(os.path.join(initial_result_dir, initial_result_file_name), initial_alpha * 255)
        with open(os.path.join(initial_result_dir, "initial_confidence"), "wb") as f:
            pickle.dump(initial_confidence, f)
    else:
        initial_alpha = cv2.imread(os.path.join(initial_result_dir, initial_result_file_name), 0) / 255.0
        with open(os.path.join(initial_result_dir, "initial_confidence"), "rb") as f:
            initial_confidence = pickle.load(f)
    refined_result_file_name = f"{args.refined_result_name}.png"  # refined result1 file name
    refined_alpha = solve_refined_alpha_matting(un_map, fb_map)
    cv2.imwrite(os.path.join(refined_result_dir, refined_result_file_name), refined_alpha)