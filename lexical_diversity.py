import os
import re
from nltk import ngrams
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import string
import random

# Set the random seed for reproducibility
random.seed(42)

# Initialize the tokenizer and other utilities
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()

# Load stopwords and punctuation
stopwords = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)

# Function to compute Type-Token Ratio (TTR)
def ttr(tokens):
    return len(set(tokens)) / len(tokens)

# Output file for writing results
output_file = 'unique-n.txt'

# List of directories to process
directories = ['../story/outputs', '../dialogue/outputs', '../summary/outputs', '../translation/outputs', '../wiki/outputs']

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
            f_out.write(f'{fname}\n')
            f_out.flush()

            tokens = []
            bi_grams = []
            tri_grams = []

            # Read and process the file
            with open(fname) as f:
                lines = f.readlines()
                lines = [l.replace('<newline>', '\n') for l in lines if l.strip() != ""]

            # Tokenize the lines and extract n-grams
            for l in lines:
                token = tokenizer.tokenize(l)
                token = [t.lower() for t in token if t not in punctuation]
                tokens += token
                bi_grams += ngrams(token, 2)
                tri_grams += ngrams(token, 3)

            # Randomly sample tokens and n-grams (if there are enough tokens)
            n_tokens = 200000 #modify according to dataset
            tokens = random.sample(tokens, min(n_tokens, len(tokens)))
            bi_grams = random.sample(bi_grams, min(n_tokens, len(bi_grams)))
            tri_grams = random.sample(tri_grams, min(n_tokens, len(tri_grams)))

            # Calculate the TTR for unigrams, bigrams, and trigrams
            unique_1 = ttr(tokens)
            unique_2 = ttr(bi_grams)
            unique_3 = ttr(tri_grams)

            # Compute the average uniqueness across n-grams
            average_uniqueness = (unique_1 + unique_2 + unique_3) / 3

            # Write the result to the output file
            f_out.write(f'{average_uniqueness}\n')
            f_out.flush()

# Done processing all files
print(f'Results written to {output_file}')