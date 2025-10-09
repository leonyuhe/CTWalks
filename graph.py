import numpy as np
import random
import torch
import math
import matplotlib.pyplot as plt
from bisect import bisect_left
from sample import *

PRECISION = 5


class NeighborFinder:
    def __init__(self, adj_list, community_labels, bridging_nodes, temporal_bias=0, spatial_bias=0, ee_bias=0, ts_precision=PRECISION, use_cache=False,
                 sample_method='multinomial', limit_ngh_span=False, ngh_span=None):
        self.limit_ngh_span = limit_ngh_span
        self.ngh_span_list = ngh_span
        self.temporal_bias = temporal_bias
        self.spatial_bias = spatial_bias
        self.ee_bias = ee_bias
        node_idx_l, node_ts_l, edge_idx_l, binary_prob_l, off_set_l, self.nodeedge2idx = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.binary_prob_l = binary_prob_l
        self.off_set_l = off_set_l
        self.use_cache = use_cache
        self.cache = {}
        self.ts_precision = ts_precision
        self.ngh_lengths = []
        self.ngh_time_lengths = []
        self.sample_method = sample_method
        self.community_labels = community_labels
        self.bridging_nodes = bridging_nodes
        self.adj_list = adj_list

    def init_off_set(self, adj_list):
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        binary_prob_l = []
        off_set_l = [0]
        nodeedge2idx = {}
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            ts_l = [x[2] for x in curr]
            n_ts_l.extend(ts_l)
            binary_prob_l.append(self.compute_binary_prob(np.array(ts_l)))
            off_set_l.append(len(n_idx_l))
            nodeedge2idx[i] = self.get_ts2idx(curr)
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        binary_prob_l = np.concatenate(binary_prob_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, binary_prob_l, off_set_l, nodeedge2idx

    def compute_binary_prob(self, ts_l):
        if len(ts_l) == 0:
            return np.array([])
        ts_l = ts_l - np.max(ts_l)
        exp_ts_l = np.exp(self.temporal_bias * ts_l)
        exp_ts_l /= np.cumsum(exp_ts_l)
        return exp_ts_l

    def get_ts2idx(self, sorted_triples):
        ts2idx = {}
        if len(sorted_triples) == 0:
            return ts2idx
        tie_ts_e_indices = []
        last_ts = -1
        last_e_idx = -1
        for i, (n_idx, e_idx, ts_idx) in enumerate(sorted_triples):
            ts2idx[e_idx] = i

            if ts_idx == last_ts:
                if len(tie_ts_e_indices) == 0:
                    tie_ts_e_indices = [last_e_idx, e_idx]
                else:
                    tie_ts_e_indices.append(e_idx)

            if (not (ts_idx == last_ts)) and (len(tie_ts_e_indices) > 0):
                tie_len = len(tie_ts_e_indices)
                for j, tie_ts_e_idx in enumerate(tie_ts_e_indices):
                    ts2idx[tie_ts_e_idx] -= j
                tie_ts_e_indices = []
            last_ts = ts_idx
            last_e_idx = e_idx
        return ts2idx



    def find_before(self, src_idx, cut_time, e_idx=None, return_binary_prob=False):
        if self.use_cache:
            result = self.check_cache(src_idx, cut_time)
            if result is not None:
                return result[0], result[1], result[2], result[3]

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        binary_prob_l = self.binary_prob_l
        start = off_set_l[src_idx]
        end = off_set_l[src_idx + 1]
        neighbors_idx = node_idx_l[start: end]
        neighbors_ts = node_ts_l[start: end]
        neighbors_e_idx = edge_idx_l[start: end]

        assert (len(neighbors_idx) == len(neighbors_ts) and len(neighbors_idx) == len(
            neighbors_e_idx))
        if e_idx is None:
            cut_idx = bisect_left_adapt(neighbors_ts, cut_time)
        else:
            cut_idx = self.nodeedge2idx[src_idx].get(e_idx) if src_idx > 0 else 0
            if cut_idx is None:
                raise IndexError('e_idx {} not found in edge list of {}'.format(e_idx, src_idx))
        if not return_binary_prob:
            result = (neighbors_idx[:cut_idx], neighbors_e_idx[:cut_idx], neighbors_ts[:cut_idx], None)
        else:
            neighbors_binary_prob = binary_prob_l[start: end]
            result = (
            neighbors_idx[:cut_idx], neighbors_e_idx[:cut_idx], neighbors_ts[:cut_idx], neighbors_binary_prob[:cut_idx])

        if self.use_cache:
            self.update_cache(src_idx, cut_time, result)

        return result

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbor=20, e_idx_l=None):
        """
        Perform community-aware temporal neighbor sampling.
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            # 根据节点角色筛选有效邻居
            src_community = self.community_labels[src_idx] if src_idx < len(self.community_labels) else -1
            
            # 孤立节点（社区标签为-1）：采样所有邻居
            if src_community == -1:
                valid_neighbors = self.adj_list[src_idx]
            elif src_idx in self.bridging_nodes:
                # 桥接节点：只采样跨社区的邻居
                valid_neighbors = [n for n in self.adj_list[src_idx] if
                                   n[0] < len(self.community_labels) and 
                                   self.community_labels[n[0]] != src_community]
            else:
                # 非桥接节点：只采样社区内部的邻居
                valid_neighbors = [n for n in self.adj_list[src_idx] if
                                   n[0] < len(self.community_labels) and 
                                   self.community_labels[n[0]] == src_community]

            # 如果没有有效邻居，跳过
            if len(valid_neighbors) == 0:
                continue

            # 按时间排序并提取信息
            valid_neighbors = sorted(valid_neighbors, key=lambda x: x[2])
            ngh_idx = np.array([n[0] for n in valid_neighbors])
            ngh_ts = np.array([n[2] for n in valid_neighbors])
            ngh_eidx = np.array([n[1] for n in valid_neighbors])

            # 计算采样权重（基于时间动态特性）
            time_delta = cut_time - ngh_ts
            sampling_weight = np.exp(-self.temporal_bias * time_delta)
            sampling_weight /= sampling_weight.sum()

            # 进行采样
            sampled_idx = np.random.choice(len(ngh_idx), num_neighbor, replace=True, p=sampling_weight)
            out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
            out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
            out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors, e_idx_l=None):
        """Sampling the k-hop sub graph in tree struture
        """
        if k == 0:
            return ([], [], [])
        batch = len(src_idx_l)
        layer_i = 0
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors[layer_i], e_idx_l=e_idx_l)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for layer_i in range(1, k):
            ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_records[-1]
            ngh_node_est = ngh_node_est.flatten()
            ngh_e_est = ngh_e_est.flatten()
            ngh_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est,
                                                                                                 ngh_t_est,
                                                                                                 num_neighbors[layer_i],
                                                                                                 e_idx_l=ngh_e_est)
            out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1)
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(batch, -1)
            out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)

        return (node_records, eidx_records, t_records)  # each of them is a list of k numpy arrays, each in shape (batch,  num_neighbors ** hop_variable)

    def save_ngh_stats(self, save_dir):
        ngh_lengths, ngh_time_lengths = np.array(self.ngh_lengths), np.array(self.ngh_time_lengths)
        plt.scatter(ngh_lengths, ngh_time_lengths)
        avg_ngh_num = int(ngh_lengths.mean())
        avg_ngh_time_span = int(ngh_time_lengths.mean())
        avg_time_span_per_ngh = int((ngh_time_lengths / ngh_lengths).mean())
        plt.title('avg ngh num:{}, avg ngh time span: {}, avg time span/ngh: {}'.format(avg_ngh_num, avg_ngh_time_span,
                                                                                        avg_time_span_per_ngh))
        plt.xlabel('number of neighbors')
        plt.ylabel('number of neighbor time span')
        plt.savefig('/'.join([save_dir, 'ngh_num_span.png']), dpi=200)

    def find_k_hop_neighbors(self, node, k, community_constrained=True):
        """
        Sample neighbors up to k hops, with optional community constraints.
        :param node: Node to start from
        :param k: Number of hops
        :param community_constrained: If True, restrict sampling within the same community
        :return: List of neighbors
        """
        visited = set()
        to_visit = [(node, 0)]
        neighbors = []

        while to_visit:
            current_node, current_hop = to_visit.pop(0)
            if current_hop == k:
                continue
            for dst, ts, weight in self.adj_list[current_node]:
                if dst not in visited:
                    if community_constrained and current_node not in self.bridging_nodes:
                        # Only allow intra-community neighbors
                        current_community = self.community_labels[current_node] if current_node < len(self.community_labels) else -1
                        dst_community = self.community_labels[dst] if dst < len(self.community_labels) else -1
                        if current_community != -1 and dst_community != current_community:
                            continue
                    visited.add(dst)
                    neighbors.append((dst, ts, weight))
                    to_visit.append((dst, current_hop + 1))
        return neighbors

    def find_k_hop_walk(self, k, src_idx_l, cut_time_l, n_walk=100, e_idx_l=None, recent_bias=1.0):
        if len(src_idx_l) == 0:
            return None, None, None
        n_idx_batch, e_idx_batch, ts_batch = [], [], []
        for sample_idx, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            e_idx = None if e_idx_l is None else e_idx_l[sample_idx]
            walks_n_idx, walks_e_idx, walks_ts = self.get_random_walks(src_idx, cut_time, n_walk=n_walk, len_walk=k,
                                                                       e_idx=e_idx, recent_bias=recent_bias)
            n_idx_batch.append(walks_n_idx)
            e_idx_batch.append(walks_e_idx)
            ts_batch.append(walks_ts)
        n_idx_batch, e_idx_batch, ts_batch = np.stack(n_idx_batch), np.stack(e_idx_batch), np.stack(ts_batch)
        return n_idx_batch, e_idx_batch, ts_batch

    def get_random_walks(self, src_idx, cut_time, n_walk=100, len_walk=5, e_idx=None, recent_bias=1.0):
        walks_n_idx, walks_e_idx, walks_ts = [], [], []
        for _ in range(n_walk):
            walk_n_idx, walk_e_idx, walk_ts = self.get_random_walk(src_idx, cut_time, seed=-1,
                                                                   len_walk=len_walk, e_idx=e_idx,
                                                                   recent_bias=recent_bias, packed=False)
            walks_n_idx.append(walk_n_idx)
            walks_e_idx.append(walk_e_idx)
            walks_ts.append(walk_ts)
        walks_n_idx, walks_e_idx, walks_ts = np.stack(walks_n_idx), np.stack(walks_e_idx), np.stack(walks_ts)
        return walks_n_idx, walks_e_idx, walks_ts

    def get_random_walk(self, src_idx, cut_time, seed=0, len_walk=5, e_idx=None, packed=False, recent_bias=1.0):
        if seed >= 0:
            random.seed(seed)
        cur_n_idx, cur_time, cur_e_idx = src_idx, cut_time, e_idx
        if packed:
            random_walk = [(src_idx, cut_time)]
            for hop in range(len_walk):
                n_idx_l, e_idx_l, ts_l = self.find_before(cur_n_idx, cur_time, e_idx=cur_e_idx)
                cur_len = len(n_idx_l)
                if cur_len == 0:
                    random_walk += [[0, 0.0]] * (len_walk - hop)
                    return random_walk
                r = random.random()
                r = -(1 - r) ** recent_bias + 1
                idx_picked = int(r * cur_len)
                cur_n_idx, cur_time, cur_e_idx = n_idx_l[idx_picked], ts_l[idx_picked], e_idx_l[
                    idx_picked]
                random_walk.append((cur_n_idx, cur_e_idx))
                return random_walk
        else:
            walk_n_idx, walk_e_idx, walk_ts = [cur_n_idx], [e_idx if e_idx is not None else -1], [cur_time]
            for hop in range(len_walk):
                n_idx_l, e_idx_l, ts_l = self.find_before(cur_n_idx, cur_time, e_idx=cur_e_idx)
                cur_len = len(n_idx_l)
                if cur_len == 0:
                    walk_n_idx.extend([0] * (len_walk - hop))
                    walk_e_idx.extend([0] * (len_walk - hop))
                    walk_ts.extend([0.0] * (len_walk - hop))
                    break
                r = random.random()
                r = -(1 - r) ** recent_bias + 1
                idx_picked = int(r * cur_len)
                cur_n_idx, cur_time, cur_e_idx = n_idx_l[idx_picked], ts_l[idx_picked], e_idx_l[
                    idx_picked]
                walk_n_idx.append(cur_n_idx)
                walk_e_idx.append(cur_e_idx)
                walk_ts.append(cur_time)
            walks_n_idx, walks_e_idx, walks_ts = np.array(walk_n_idx, dtype=int), \
                                                 np.array(walk_e_idx, dtype=int), np.array(walk_ts, dtype=float)
            return walks_n_idx, walks_e_idx, walks_ts

    def update_cache(self, node, ts, results):
        ts_str = str(round(ts, PRECISION))
        key = (node, ts_str)
        if key not in self.cache:
            self.cache[key] = results

    def check_cache(self, node, ts):
        ts_str = str(round(ts, PRECISION))
        key = (node, ts_str)
        return self.cache.get(key)

    def compute_degs(self):
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        degs = []
        for n_idx, ts, e_idx in zip(node_idx_l, node_ts_l, edge_idx_l):
            deg = len(self.find_before(n_idx, ts, e_idx=e_idx)[0])
            degs.append(deg)
        degs = np.array(degs)
        return degs.mean(), degs

    def compute_2hop_degs(self, progress_bar=False, n_workers=1):

        def float2str(ts):
            return str(round(ts, 5))

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        degs = []
        if progress_bar:
            from tqdm import tqdm_notebook as tqdm
            iterable = tqdm(zip(node_idx_l, node_ts_l, edge_idx_l), total=len(node_idx_l))
        else:
            iterable = zip(node_idx_l, node_ts_l, edge_idx_l)
        for n_idx, n_ts, e_idx in iterable:
            one_hop_n_idx, one_hop_e_idx, one_hop_ts = self.find_before(n_idx, n_ts, e_idx)
            one_hop_node_l = set([(n, float2str(ts)) for n, ts in zip(one_hop_n_idx, one_hop_ts)])
            two_hop_node_l = []
            for n, ts, e in zip(one_hop_n_idx, one_hop_ts, one_hop_e_idx):
                two_hop_n_idx, _, two_hop_ts = self.find_before(n, ts, e_idx=e)
                two_hop_node_l.extend([(two_n, float2str(two_ts)) for two_n, two_ts in zip(two_hop_n_idx, two_hop_ts)])
            two_hop_node_l = set(two_hop_node_l) - one_hop_node_l
            degs.append((len(one_hop_node_l), two_hop_node_l))
        return degs
