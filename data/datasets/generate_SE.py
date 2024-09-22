"""
Script for generating spatial embeddings for GAMN.
Author: ChunWei Shen
"""
import os
import pickle
import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

from metadata import N_SERIES

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		"""Simulate a random walk starting from start node."""
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		"""Repeatedly simulate random walks from each node."""
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print ("Walk iteration:")
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), "/", str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		"""Get the alias edge setup lists for a given edge."""
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]["weight"])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob) / norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		"""Preprocessing of transition probabilities for guiding the random walks."""
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob) / norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return
	
def alias_setup(probs):
	"""
	Compute utility lists for non-uniform sampling from discrete distributions. Refer to 
	https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details.
	"""
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K * prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()
		
		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	"""Draw sample from a non-uniform discrete distribution using alias sampling."""
	K = len(J)

	kk = int(np.floor(np.random.rand() * K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]

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

def generate_graph(dataset_name: str, adj_save_path: str):
    adj = _load_adj_mat(dataset_name)
    adj = np.where(adj > 0, 1.0, 0.0)
    n_series = N_SERIES[dataset_name]
    for i in range(n_series):
        adj[i][i] = 1
    
    with open(adj_save_path, 'w') as file:
        for i in range(n_series):
            for j in range(n_series):
                file.write(f"{i} {j} {adj[i][j]}\n")
				
    return

def read_graph(adj_save_path):
    G = nx.read_edgelist(
        adj_save_path, 
        nodetype=int,
        data=(("weight",float),),
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

    f = open(output_file, mode="r")
    lines = f.readlines()
    temp = lines[0].split(" ")
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(" ")
        index = int(temp[0])
        SE[index] = temp[1 :]
    np.savetxt(output_file, SE, delimiter=",")
	
    return


def gen_se(dataset_name: str, is_directed: bool):
	print(f"Generating {dataset_name} spatial embeddings ...")

	adj_save_path = f"./data/raw/{dataset_name}/adj_{dataset_name}.txt"
	se_file = f"./data/raw/{dataset_name}/SE_{dataset_name}.txt"

	print("Generating grpah ...")
	generate_graph(dataset_name=dataset_name, adj_save_path=adj_save_path)
	print("Reading grpah ...")
	nx_G = read_graph(adj_save_path=adj_save_path)
	G = Graph(nx_G, is_directed, p, q)
	print("Preprocessing ...")
	G.preprocess_transition_probs()
	print("Simulating walks ...")
	walks = G.simulate_walks(num_walks, walk_length)
	print("Learning embeddings ...")
	learn_embeddings(walks, dimensions, se_file)
	print("Finish!")
	print("-" * 50)

	return
	

if __name__ == "__main__":
    p = 2
    q = 1
    iter = 1000
    dimensions = 64
    num_walks = 100
    walk_length = 80
    window_size = 10

    gen_se(dataset_name="metr_la", is_directed=True)
    gen_se(dataset_name="pems_bay", is_directed=True)
    gen_se(dataset_name="pems03", is_directed=False)
    gen_se(dataset_name="pems04", is_directed=False)
    gen_se(dataset_name="pems07", is_directed=False)
    gen_se(dataset_name="pems08", is_directed=False)