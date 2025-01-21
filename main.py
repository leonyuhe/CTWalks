import pandas as pd
from log import *
from utils import *
from train import *
from module import CTWalks
from graph import NeighborFinder
import community as community_louvain
import networkx as nx
import resource

def compute_edge_weights(df, weight_type='frequency'):
    """
    Compute edge weights for a temporal graph.
    :param df: DataFrame containing edges with columns [src, dst, ts]
    :param weight_type: Method to compute weights ('frequency', 'recency', 'custom')
    :return: Updated DataFrame with edge weights
    """
    if weight_type == 'frequency':
        # Count the number of occurrences of each edge
        weights = df.groupby(['u', 'i']).size().reset_index(name='weight')
    elif weight_type == 'recency':
        # Compute weights inversely proportional to recency
        current_time = df['ts'].max()
        weights = df.groupby(['u', 'i']).agg({'ts': lambda x: 1 / (current_time - x.mean() + 1)}).reset_index()
        weights.rename(columns={'ts': 'weight'}, inplace=True)
    elif weight_type == 'custom':
        # Define your own custom weight logic here
        weights = df.groupby(['u', 'i']).size().reset_index(name='weight')
        weights['weight'] *= 2  # Example custom rule
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    return weights


def community_detection(weighted_adj_list):
    """
    Perform community detection on a weighted temporal graph using Louvain algorithm.
    :param weighted_adj_list: Adjacency list with weights [(dst, ts, weight), ...]
    :return: Dictionary mapping each node to its community
    """
    G = nx.Graph()
    for src, neighbors in enumerate(weighted_adj_list):
        for dst, ts, weight in neighbors:
            G.add_edge(src, dst, weight=weight)

    # Use Louvain algorithm for community detection
    partition = community_louvain.best_partition(G, weight='weight')
    return partition

def identify_bridging_nodes(weighted_adj_list, community_labels):
    """
    Identify bridging and non-bridging nodes based on community labels.
    :param weighted_adj_list: Adjacency list with weights [(dst, ts, weight), ...]
    :param community_labels: List mapping each node to its community
    :return: Set of bridging nodes
    """
    bridging_nodes = set()
    for src, neighbors in enumerate(weighted_adj_list):
        src_community = community_labels[src]
        for dst, _, _ in neighbors:
            if community_labels[dst] != src_community:
                bridging_nodes.add(src)
                break
    return bridging_nodes
def assign_community_and_update(new_nodes, test_src_l, test_dst_l, community_labels, bridging_nodes):
    """
    根据测试数据中成对的节点信息，推断新节点的社区并更新数据结构。
    :param new_nodes: Set[int] 测试数据中不在 weighted_adj_list 中的节点
    :param test_src_l: List[int] 测试数据中的源节点列表
    :param test_dst_l: List[int] 测试数据中的目标节点列表
    :param community_labels: List[int] 节点的社区标签
    :param bridging_nodes: Set[int] 桥接节点集合
    :return: 更新后的 community_labels 和 bridging_nodes
    """
    for node in new_nodes:
        # 找到和这个节点相关的所有对（邻居）
        neighbors = set()
        for src, dst in zip(test_src_l, test_dst_l):
            if src == node:
                neighbors.add(dst)
            elif dst == node:
                neighbors.add(src)

        if not neighbors:
            # 如果没有邻居，无法推断社区，设置默认值（例如 -1）
            community_labels[node] = -1
            continue

        # 统计邻居的社区信息
        neighbor_communities = {}
        for neighbor in neighbors:
            if neighbor in community_labels:
                community = community_labels[neighbor]
                neighbor_communities[community] = neighbor_communities.get(community, 0) + 1  # 默认权重为1

        if not neighbor_communities:
            # 邻居都没有社区标签，无法推断社区
            community_labels[node] = -1
        elif len(neighbor_communities) == 1:
            # 所有邻居都在同一个社区
            assigned_community = next(iter(neighbor_communities.keys()))
            community_labels[node] = assigned_community
        else:
            # 邻居分属于多个社区，根据权重概率分配社区
            total_weight = sum(neighbor_communities.values())
            probabilities = {community: count / total_weight for community, count in neighbor_communities.items()}
            assigned_community = max(probabilities, key=probabilities.get)  # 按概率分配
            community_labels[node] = assigned_community
            bridging_nodes.add(node)  # 多个社区则为桥接节点

    return community_labels, bridging_nodes


args, sys_argv = get_args()

assert(args.cpu_cores >= -1)
set_random_seed(args.seed)

device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')

g_df = pd.read_csv('./data/ml_{}.csv'.format(args.data))
if args.data_usage < 1:
    g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
    print('use partial data, ratio: {}'.format(args.data_usage), flush=True)
e_feat = np.load('./data/ml_{}.npy'.format(args.data))
n_feat = np.load('./data/ml_{}_node.npy'.format(args.data))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())

assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))
assert(n_feat.shape[0] == max_idx + 1 or ~math.isclose(1, args.data_usage))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])),
                                      int(0.1 * num_total_unique_nodes)))  # mask 10% nodes for inductive evaluation
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(
        len(mask_node_set)))

train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = \
    src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], \
    e_idx_l[valid_train_flag], label_l[valid_train_flag]

# Step 1: 基于训练数据构建 Weighted Temporal Graph
# Compute edge weights
train_g_df = pd.DataFrame({'u': train_src_l, 'i': train_dst_l, 'ts': train_ts_l})
train_g_df = compute_edge_weights(train_g_df, weight_type='frequency')

# 更新训练数据的加权邻接表
weighted_adj_list = [[] for _ in range(max(src_l.max(), dst_l.max()) + 1)]
for _, row in train_g_df.iterrows():
    src, dst, ts, weight = row['u'], row['i'], row['ts'], row['weight']
    weighted_adj_list[src].append((dst, ts, weight))
    weighted_adj_list[dst].append((src, ts, weight))

# Step 2: 执行社区划分
partition = community_detection(weighted_adj_list)
community_labels = [partition[node] for node in range(len(weighted_adj_list))]
np.save('./data/{}_community_labels.npy'.format(args.data), community_labels)

# Step 3: 标记桥接节点和非桥接节点
bridging_nodes = identify_bridging_nodes(weighted_adj_list, community_labels)
non_bridging_nodes = set(range(len(weighted_adj_list))) - bridging_nodes

# 记录社区标签到节点特征
n_feat_with_community = np.hstack([n_feat, np.array(community_labels).reshape(-1, 1)])


val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = \
    src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], \
    e_idx_l[valid_val_flag], label_l[valid_val_flag]
val_new_nodes = set(val_src_l).union(set(val_dst_l)) - set(weighted_adj_list.keys())
# 更新社区标签和桥接节点
community_labels, bridging_nodes = assign_community_and_update(
    val_new_nodes, val_src_l, val_dst_l, community_labels, bridging_nodes
)
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = \
    src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], \
    e_idx_l[valid_test_flag], label_l[valid_test_flag]

test_new_nodes = set(test_src_l).union(set(test_dst_l)) - set(weighted_adj_list.keys())
# 更新社区标签和桥接节点
community_labels, bridging_nodes = assign_community_and_update(
    test_new_nodes, test_src_l, test_dst_l, community_labels, bridging_nodes
)

if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = \
        src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], \
        e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]

    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = \
        src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], \
        e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

full_ngh_finder = NeighborFinder(full_adj_list, bridging_nodes,community_labels, temporal_bias=args.temporal_bias, spatial_bias=args.spatial_bias,
                                 ee_bias=args.ee_bias, use_cache=args.ngh_cache, sample_method=args.pos_sample,
                                 limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span)

partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))

partial_ngh_finder = NeighborFinder(partial_adj_list, bridging_nodes,community_labels,temporal_bias=args.temporal_bias, spatial_bias=args.spatial_bias,
                                    ee_bias=args.ee_bias, use_cache=args.ngh_cache, sample_method=args.pos_sample,
                                    limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span)

ngh_finders = partial_ngh_finder, full_ngh_finder
logger.info('Sampling module - temporal bias: {}, spatial bias: {}, E&E bias: {}'.format(args.temporal_bias,
                                                                                         args.spatial_bias,
                                                                                         args.ee_bias))

train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
rand_samplers = train_rand_sampler, val_rand_sampler

model = CTWalks(n_feat=n_feat_with_community, e_feat=e_feat, walk_mutual=args.walk_mutual, walk_linear_out=args.walk_linear_out,
                pos_enc=args.pos_enc, pos_dim=args.pos_dim, num_layers=args.n_layer, num_neighbors=args.n_degree,
                tau=args.tau, negs=args.negs, solver=args.solver, step_size=args.step_size, drop_out=args.drop_out,
                cpu_cores=args.cpu_cores, verbosity=args.verbosity, get_checkpoint_path=get_checkpoint_path, community_labels=community_labels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopper = EarlyStopMonitor(tolerance=args.tolerance)

train_val(train_val_data, model, args.mode, args.bs, args.n_epoch, optimizer, early_stopper,
          ngh_finders, rand_samplers, logger, args.negs)

model.update_ngh_finder(full_ngh_finder)
test_ap, test_auc = eval_one_epoch(model, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
logger.info('Test statistics: {} all nodes -- auc: {}, ap: {}'.format(args.mode, test_auc, test_ap))

test_new_new_ap, test_new_new_auc, test_new_old_ap, test_new_old_auc = [-1]*4
if args.mode == 'i':
    test_new_new_ap, test_new_new_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_new_l,
                                                       test_dst_new_new_l, test_ts_new_new_l,
                                                       test_label_new_new_l, test_e_idx_new_new_l)
    logger.info('Test statistics: {} new-new nodes -- auc: {}, ap: {}'.format(args.mode, test_new_new_auc,
                                                                              test_new_new_ap))
    test_new_old_ap, test_new_old_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_old_l,
                                                       test_dst_new_old_l, test_ts_new_old_l,
                                                       test_label_new_old_l, test_e_idx_new_old_l)
    logger.info('Test statistics: {} new-old nodes -- auc: {}, ap: {}'.format(args.mode, test_new_old_auc,
                                                                              test_new_old_ap))

logger.info('Saving model...')
torch.save(model.state_dict(), best_model_path)
logger.info('Saved model to {}'.format(best_model_path))
logger.info('model saved')