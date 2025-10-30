import numpy as np
import json
import os
from collections import defaultdict


def check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    """Check if dataset already exists"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        if (config['num_clients'] == num_clients and 
            config['niid'] == niid and 
            config['balance'] == balance and 
            config['partition'] == partition):
            return True
    return False


def separate_data(dataset, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    """
    Separate data for federated learning - adapted for object detection
    """
    X, y = dataset
    X = np.array(X)
    y = np.array(y)
    
    if not niid:
        # IID case - simple random split
        idxs = np.random.permutation(len(X))
        batch_idxs = np.array_split(idxs, num_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
        
        client_data = [X[idx] for idx in batch_idxs]
        client_labels = [y[idx] for idx in batch_idxs]
        
    else:
        # Non-IID case
        if partition == "dirichlet":
            client_data, client_labels, net_dataidx_map = dirichlet_partition(X, y, num_clients, num_classes, alpha=0.5)
        else:
            # Default: assign based on dominant classes
            client_data, client_labels, net_dataidx_map = class_based_partition(X, y, num_clients, num_classes, class_per_client)
    
    # Calculate statistics
    statistic = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        client_y = client_labels[client_id]
        for class_id in range(num_classes):
            count = np.sum(client_y == class_id)
            if count > 0:
                statistic[client_id].append((class_id, count))
    
    return client_data, client_labels, statistic


def class_based_partition(X, y, num_clients, num_classes, class_per_client):
    """Partition data based on class distribution"""
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, class_id in enumerate(y):
        class_indices[class_id].append(idx)
    
    net_dataidx_map = {i: np.array([], dtype=int) for i in range(num_clients)}
    
    # Assign classes to clients
    classes_per_client = max(1, class_per_client)
    all_classes = list(range(num_classes))
    np.random.shuffle(all_classes)
    
    for client_id in range(num_clients):
        client_classes = all_classes[client_id * classes_per_client: (client_id + 1) * classes_per_client]
        
        for class_id in client_classes:
            if class_id in class_indices:
                selected_indices = np.random.choice(
                    class_indices[class_id], 
                    size=min(len(class_indices[class_id]), len(class_indices[class_id]) // num_clients + 1),
                    replace=False
                )
                net_dataidx_map[client_id] = np.concatenate([net_dataidx_map[client_id], selected_indices])
    
    # Convert to client data format
    client_data = [X[indices] for indices in net_dataidx_map.values()]
    client_labels = [y[indices] for indices in net_dataidx_map.values()]
    
    return client_data, client_labels, net_dataidx_map


def dirichlet_partition(X, y, num_clients, num_classes, alpha=0.5):
    """Partition data using Dirichlet distribution"""
    idx_batch = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        idxs = np.where(y == class_id)[0]
        np.random.shuffle(idxs)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < len(idxs) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idxs, proportions))]
    
    net_dataidx_map = {i: np.array(idx_batch[i]) for i in range(num_clients)}
    client_data = [X[indices] for indices in net_dataidx_map.values()]
    client_labels = [y[indices] for indices in net_dataidx_map.values()]
    
    return client_data, client_labels, net_dataidx_map


def split_data(X, y, train_ratio=0.8):
    """Split data into train and test sets for each client"""
    train_data = []
    test_data = []
    
    for client_X, client_y in zip(X, y):
        n_train = int(len(client_X) * train_ratio)
        
        indices = np.random.permutation(len(client_X))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_data.append((client_X[train_idx], client_y[train_idx]))
        test_data.append((client_X[test_idx], client_y[test_idx]))
    
    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
              statistic, niid, balance, partition):
    """Save dataset configuration"""
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'niid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for each client': [len(train_data[i][0]) for i in range(num_clients)],
        'statistic': statistic
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Configuration file saved!")