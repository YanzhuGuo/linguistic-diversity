
## README

This repo contains Python scripts for analyzing the linguistic diversity among collections of texts, specifically measuring lexical, semantic, and syntactic diversity.

### Setup

1. Install the required dependencies using the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the input data. Input text files should be placed in subdirectories under the `data/` directory. The folder structure should be as follows:

   ```
   data/
   ├── story/outputs/
   ├── dialogue/outputs/
   ├── summary/outputs/
   ├── translation/outputs/
   └── wiki/outputs/
   ```

### Input File Format

- Input files should contain plain text.
- Each line in an input file corresponds to one sample.
- Sentences within the same sample are separated by `<newline>` markers.
- The scripts assume UTF-8 encoding for the text files.

### Scripts

1. **`semantic_diversity.py`**: 
   - Computes sentence embeddings using transformer models and calculates cosine similarity to measure semantic diversity.
   - Results are saved to `sem.txt`.

2. **`syntactic_diversity.py`**: 
   - Parses sentences to generate syntactic dependency graphs and computes graph kernel similarities to assess syntactic diversity.
   - Results are saved to `syn.txt`.

3. **`lexical_diversity.py`**: 
   - Tokenizes text and calculates the Type-Token Ratio (TTR) for unigrams, bigrams, and trigrams. The average TTR is reported as lexical diversity.
   - Results are saved to `unique-n.txt`.