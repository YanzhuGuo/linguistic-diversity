import os
import torch
import random
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import numpy as np
from scipy.stats import zscore
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Function to calculate the mean confidence interval
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
    return mean, margin_of_error

# Parameters
n = 3000  # Number of sentences to sample
output_file = 'sem.txt'

# List of directories to process
directories = ['../story/outputs', '../dialogue/outputs', '../summary/outputs', '../translation/outputs', '../wiki/outputs']

# Open the output file for writing
with open(output_file, 'w') as f_out:
    for dir in directories:
        f_out.write(f'******************** {dir} ************************\n')

        # Get list of files in the directory
        files_name = [fname for fname in os.listdir(dir) if os.path.isfile(os.path.join(dir, fname))]
        files = [os.path.join(dir, fname) for fname in files_name]

        # Process each file
        for fname in files:
            torch.cuda.empty_cache()

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

                # Load the sentence transformer model
                model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:1')
                
                # Encode sentences into embeddings
                sentence_embeddings = model.encode(sentences, batch_size=128, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
                
                # Convert embeddings to tensor and move to CUDA
                x = torch.tensor(sentence_embeddings).to(torch.float).to("cuda:0")

                # Clear memory by moving model and embeddings to CPU
                model.to('cpu')
                del model
                sentence_embeddings.to('cpu')
                del sentence_embeddings
                torch.cuda.empty_cache()

                # Compute cosine similarity matrix
                with torch.no_grad():
                    x_cosine_similarity = torch.nn.functional.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)

                # Extract non-diagonal elements from the cosine similarity matrix
                mask = ~torch.eye(x_cosine_similarity.size(0), dtype=bool)
                non_diag_elements = x_cosine_similarity[mask]
                non_diag_array = non_diag_elements.cpu().numpy()
                
                # Rescale cosine similarity values to range [0, 1]
                res = (1 - non_diag_array) / 2
                
                # Calculate mean and confidence interval
                mean, error = mean_confidence_interval(res)

                # Write results to the output file
                f_out.write(f'{fname}\n')
                f_out.write(f'{mean} +- {error}\n')
                f_out.flush()

# Done processing all files
print(f'Results written to {output_file}')
