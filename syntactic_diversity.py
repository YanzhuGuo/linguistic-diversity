import os
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from nltk.tokenize import sent_tokenize
import stanza
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import zscore
import scipy.stats as stats

# Download and initialize Stanza pipeline for NLP processing
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,mwt,lemma,depparse')

# Function to calculate the mean confidence interval
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
    return mean, margin_of_error

# Function to create a dependency graph from a Stanza-processed sentence
def create_graph(doc):
    sent = doc.sentences[0]
    G = nx.Graph()

    for word in sent.to_dict():
        if isinstance(word['id'], tuple):
            continue

        # Add the current word to the graph
        if word['id'] not in G.nodes():
            G.add_node(word['id'])
            
        G.nodes[word['id']]['label'] = word['upos']

        # Add the head of the word (the word it depends on)
        if word['head'] not in G.nodes():
            G.add_node(word['head'])

        G.add_edge(word['id'], word['head'])

    # Set the root node label
    G.nodes[0]['label'] = 'none'
    
    return G

# Parameters
n = 3000  # Number of sentences to sample
output_file = 'syn.txt'

# List of directories to process
directories = ['output']

# Open the output file for writing
with open(output_file, 'w') as f_out:
    for dir in directories:
        f_out.write(f'******************** {dir} ************************\n')
        f_out.flush()

        # Get list of files in the directory
        files_name = [fname for fname in os.listdir(dir) if os.path.isfile(os.path.join(dir, fname))]
        files = [os.path.join(dir, fname) for fname in files_name]

        # Process each file
        for fname in files:
            graphs = []

            with open(fname) as f:
                lines = f.readlines()
                lines = [l.replace('<newline>', '\n') for l in lines if l.strip() != ""]

                # Tokenize sentences
                sentences = []
                for t in lines:
                    sents = sent_tokenize(t)
                    #ideally discard the first and last sentences because they might not be complete
                    if len(sents) > 2:
                        sentences += sents[1:-1]
                    elif len(sents) > 1:
                        sentences += [sents[0]]
                    else:
                        sentences += sents

                # Randomly sample sentences
                random.seed(42)
                sentences = random.sample(sentences, n)

                # Process each sentence and create graph
                for s in tqdm(sentences):
                    doc = nlp(s)
                    graphs.append(create_graph(doc))

            # Convert NetworkX graphs to Grakel format
            G = list(graph_from_networkx(graphs, node_labels_tag='label'))

            # Initialize Weisfeiler-Lehman kernel
            gk = WeisfeilerLehman(n_iter=2, normalize=True, base_graph_kernel=VertexHistogram)

            # Compute kernel matrix
            K = gk.fit_transform(G)
            K = torch.tensor(K).to(torch.float).to("cuda:0")

            # Extract non-diagonal elements from kernel matrix
            mask = ~torch.eye(K.size(0), dtype=bool)
            non_diag_elements = K[mask]
            non_diag_array = non_diag_elements.cpu().numpy()

            # Rescale values and compute mean and confidence interval
            res = 1 - non_diag_array
            mean, error = mean_confidence_interval(res)

            # Write results to the output file
            f_out.write(f'{fname}\n')
            f_out.write(f'{mean} +- {error}\n')
            f_out.flush()

            # Clear GPU memory
            torch.cuda.empty_cache()

# Done processing all files
print(f'Results written to {output_file}')
