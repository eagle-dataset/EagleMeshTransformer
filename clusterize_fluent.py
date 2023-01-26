import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
import os
import numba

parser = argparse.ArgumentParser()
parser.add_argument('--max_cluster_size', type=int, default=20)
parser.add_argument('--geom', type=str, default="Cre")
parser.add_argument('--traj', type=int, default=1)
args = parser.parse_args()

PATH = ""


class FluentDataset:
    def __init__(self):
        super(FluentDataset, self).__init__()
        self.fn = PATH
        assert os.path.exists(self.fn), "Path does not exist"

        self.dataloc = []
        geom_type = args.geom
        path = os.path.join(self.fn, geom_type)
        for example in os.listdir(path):
            self.dataloc.append(os.path.join(path, example, str(args.traj)))

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        path = os.path.join(self.dataloc[item], 'sim.npz')
        data = np.load(path, mmap_mode='r')
        mesh_pos = data["pointcloud"].copy().astype(np.float32)

        node_type = data['mask'].copy().astype(np.int32)
        shape = node_type.shape
        node_type = node_type.reshape(-1)
        node_type = np.eye(9)[node_type.astype(np.int64)]
        node_type = node_type.reshape(([shape[0], shape[1], 9]))

        output = {'mesh_pos': mesh_pos,
                  'node_type': node_type}
        return output


@numba.njit(fastmath=True)
def kmeans(x, K=10, Niter=300, centers=None):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    x = x
    centers = x[:K, :].copy() if centers is None else centers  # Simplistic initialization for the centroids
    x_i = x.reshape((N, 1, D))

    c_j = centers.reshape((1, K, D))
    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
    clusters_index = np.zeros((N), dtype=np.float32)
    for n in range(N):
        clusters_index[n] = D_ij[n].argmin()

    for i in range(Niter):
        c_j = centers.reshape((1, K, D))
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        new_cluster_index = np.zeros((N), dtype=np.float32)
        for n in range(N):
            new_cluster_index[n] = D_ij[n].argmin()

        new_centers = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            average = np.zeros((D), dtype=np.float32)
            y = x[new_cluster_index == k]
            for n in range(y.shape[0]):
                average = average + y[n]
            average = average / y.shape[0]
            new_centers[k] = average
        if np.all(new_cluster_index == clusters_index):
            break
        elif np.all(((new_centers - centers) ** 2).sum(-1) < 1e-6):
            break
        centers = new_centers
        clusters_index = new_cluster_index
    return centers


@numba.njit(fastmath=True)
def assignement(pointcloud, cluster_centers, max_cluster_size, n_points, n_cluster):
    clusters = np.zeros((n_points), dtype=np.int64)

    temp1 = pointcloud.reshape((n_points, 1, pointcloud.shape[-1]))
    temp2 = cluster_centers.reshape((1, n_cluster, pointcloud.shape[-1]))
    distances = ((temp1 - temp2) ** 2).sum(axis=-1)

    scores = np.zeros((n_points), dtype=np.float32)
    for i in range(n_points):
        scores[i] = np.min(distances[i]) - np.max(distances[i])
    cluster_sizes = np.zeros((n_cluster), dtype=np.int64)
    sorted_index = scores.argsort()

    ordered_clusters = np.zeros((n_points, n_cluster), dtype=np.int64)
    for i in range(n_points):
        ordered_clusters[i] = distances[i].argsort()

    for i in sorted_index:
        idx = 0
        closest_cluster = ordered_clusters[i, idx]
        while cluster_sizes[closest_cluster] >= max_cluster_size:
            idx += 1
            closest_cluster = ordered_clusters[i, idx]
        cluster_sizes[closest_cluster] += 1
        clusters[i] = closest_cluster

    return clusters


@numba.njit(fastmath=True)
def swap(pointcloud, clusters, n_cluster, n_points, max_cluster_size):
    # 1. Compute current cluster means
    cluster_centers = np.zeros((n_cluster, pointcloud.shape[-1]), dtype=np.float32)
    for n in range(pointcloud.shape[0]):
        cluster_centers[clusters[n]] = cluster_centers[clusters[n]] + pointcloud[n]

    cluster_centers = cluster_centers / np.bincount(clusters, minlength=1).reshape((-1, 1)).astype(np.float32)
    clusters_size = np.bincount(clusters, minlength=1).astype(np.int64)

    # 2. For each object, compute the distances to the cluster means
    temp1 = pointcloud.reshape((n_points, 1, pointcloud.shape[-1]))
    temp2 = cluster_centers.reshape((1, n_cluster, pointcloud.shape[-1]))
    distances = ((temp1 - temp2) ** 2).sum(axis=-1)

    # 3. Sort elements based on the delta of the current assignment and the best possible alternate assignment.
    delta = np.zeros((n_points), dtype=np.float32)
    for i in range(n_points):
        delta[i] = distances[i, clusters[i]] - np.min(distances[i])

    sorted_points_index = delta.argsort()
    wanting_to_leave = np.zeros((n_cluster, n_points), dtype=np.int64)
    number_of_swaps = 0
    for t, i in enumerate(sorted_points_index[::-1]):
        has_changed = False
        best_cluster = distances[i].argmin()
        if clusters[i] == best_cluster:
            break

        for j in distances[i].argsort():
            if j == clusters[i]:
                break
            if distances[i, clusters[i]] > distances[i, j] and clusters_size[j] < max_cluster_size:
                # plt.plot([pointcloud[i, 0], cluster_centers[j, 0]], [pointcloud[i, 1], cluster_centers[j, 1]],
                #          'r--')
                # plt.plot([pointcloud[i, 0], cluster_centers[clusters[i], 0]],
                #          [pointcloud[i, 1], cluster_centers[clusters[i], 1]], 'b--')
                clusters_size[clusters[i]] -= 1
                clusters_size[j] += 1
                clusters[i] = j
                has_changed = True
                number_of_swaps += 1
                break

            candidate = np.where(wanting_to_leave[j] == 1)[0]
            if len(candidate) > 0:
                scores = np.zeros((len(candidate)), dtype=np.float32)
                for ik, k in enumerate(candidate):
                    scores[ik] = -distances[i, clusters[i]] - distances[k, clusters[k]] + distances[i, clusters[k]] + \
                                 distances[k, clusters[i]]
                if np.min(scores) < 0:
                    candidate = candidate[scores.argmin()]
                    clusters[candidate] = clusters[i]
                    clusters[i] = j
                    wanting_to_leave[j, candidate] = 0
                    has_changed = True
                    number_of_swaps += 1
                    # plt.plot([pointcloud[i][0], pointcloud[candidate][0]],
                    #          [pointcloud[i][1], pointcloud[candidate][1]], 'k--')
                    break
        if not has_changed:
            wanting_to_leave[clusters[i], i] = 1
    # plt.scatter(pointcloud[:, 0], pointcloud[:, 1], c=clusters, cmap='jet')
    # plt.pause(0.001)

    cluster_centers = np.zeros((n_cluster, pointcloud.shape[-1]), dtype=np.float32)
    for n in range(pointcloud.shape[0]):
        cluster_centers[clusters[n]] = cluster_centers[clusters[n]] + pointcloud[n]

    cluster_centers = cluster_centers / np.bincount(clusters, minlength=1).reshape((-1, 1)).astype(np.float32)

    return clusters, number_of_swaps, cluster_centers


def constrained_clustering_numpy(pointcloud, init_clusters, n_cluster, max_cluster_size):
    cluster_centers = kmeans(pointcloud, n_cluster, centers=init_clusters)
    clusters = assignement(pointcloud, cluster_centers, max_cluster_size, pointcloud.shape[0], n_cluster)

    for _ in range(1000):
        clusters, number_of_swaps, cluster_centers = swap(pointcloud, clusters, n_cluster, pointcloud.shape[0],
                                                          max_cluster_size)
        # print("Performed ", number_of_swaps, "swaps")

        if number_of_swaps == 0:
            # print(np.bincount(clusters, minlength=1))
            # plt.scatter(pointcloud[:, 0], pointcloud[:, 1], c=clusters, cmap='jet')
            # from scipy.spatial import ConvexHull
            # for i in range(n_cluster):
            #     points = pointcloud[clusters == i][..., :2]
            #     hull = ConvexHull(points)
            #     plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='#ff0000', alpha=0.2)
            # plt.show()
            break

    return clusters, cluster_centers


@numba.njit
def process(clusters, n_clusters):
    stacked_clusters = []
    for t in range(clusters.shape[0]):
        c = [np.where(clusters[t] == i)[0] for i in range(clusters[t].max() + 1)]
        for i in range(len(c)):
            while len(c[i]) != n_clusters:
                c[i] = np.append(c[i], -1)
                # c[i] = np.concatenate([c[i], -np.ones((n_clusters - len(c[i])))])
        stacked_clusters.append(c)
    return stacked_clusters


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataloader = FluentDataset()
    for i in range(len(dataloader)):
        print(f"Processing {i} ({args.max_cluster_size}):")
        path = os.path.join(dataloader.dataloc[i], f"constrained_kmeans_{args.max_cluster_size}.npy")

        if not os.path.exists(path):
            x = dataloader[i]

            mesh_pos = x['mesh_pos']

            # state = np.concatenate([mesh_pos, node_type], axis=-1).astype(np.float32)
            state = mesh_pos.astype(np.float32)
            n_clusters = int(np.ceil(state.shape[1] / args.max_cluster_size)) + 1
            init_cluster_centers = None

            label_list = []
            for t in tqdm(range(state.shape[0])):
                pointcloud = state[t]
                labels, init_cluster_centers = constrained_clustering_numpy(pointcloud, init_cluster_centers,
                                                                            n_clusters, args.max_cluster_size)
                label_list.append(labels)
            labels = np.stack(label_list, axis=0)
            stacked_clusters = process(labels, args.max_cluster_size)
            stacked_clusters = np.array(stacked_clusters).astype(np.int32)
            np.save(path, stacked_clusters)


if __name__ == '__main__':
    main()
