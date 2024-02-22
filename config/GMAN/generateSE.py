import os
import pickle
import argparse
import numpy as np
import networkx as nx
from .node2vec import Graph
from gensim.models import Word2Vec

from metadata import N_SERIES

p = 2
q = 1
iter = 1000
dimensions = 64
num_walks = 100
walk_length = 80
window_size = 10
is_directed = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default='pems03')
args = parser.parse_args()

adj_file = f'./data/raw/{args.dataset_name}/Adj_{args.dataset_name}.txt'
se_file = f'./data/raw/{args.dataset_name}/SE_{args.dataset_name}.txt'

def _load_adj_mat(dataset_name: str) -> np.ndarray:
    """
    Load hand-crafted adjacency matrix.

    See https://github.com/nnzhan/Graph-WaveNet/ .

    Return:
        adj_mat: hand-crafted (pre-defined) adjacency matrix
    """
    adj_mat_file_path = os.path.join("./data/raw", dataset_name, f"{dataset_name}_adj.pkl")

    try:
        with open(adj_mat_file_path, "rb") as f:
            adj_mat = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(adj_mat_file_path, "rb") as f:
            *_, adj_mat = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Fail to load the hand-crafted adjacency matrix...")
        print("Err:", e)
        raise

    return adj_mat

def generate_graph(adj_file):
    adj = _load_adj_mat(args.dataset_name)
    adj = np.where(adj > 0, 1.0, 0.0)
    n_series = N_SERIES[args.dataset_name]
    for i in range(n_series):
        adj[i][i] = 1
    
    with open(adj_file, 'w') as file:
        for i in range(n_series):
            for j in range(n_series):
                file.write(f"{i} {j} {adj[i][j]}\n")

def read_graph(adj_file):
    G = nx.read_edgelist(
        adj_file, 
        nodetype=int,
        data=(('weight',float),),
        create_using=nx.DiGraph()
    )
    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=10,
        min_count=0,
        sg=1,
        workers=8,
        epochs=iter
    )
    model.wv.save_word2vec_format(output_file)

    f = open(output_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
    np.savetxt(output_file, SE, delimiter=',')
	
    return

print('Generate grpah...')
generate_graph(adj_file)
print('Read grpah...')
nx_G = read_graph(adj_file)
G = Graph(nx_G, is_directed, p, q)
print('Preprocess...')
G.preprocess_transition_probs()
print('Simulate walks...')
walks = G.simulate_walks(num_walks, walk_length)
print('Learn embeddings...')
learn_embeddings(walks, dimensions, se_file)
print('Finish!')