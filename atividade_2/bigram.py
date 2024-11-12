import torch
import numpy as np
from nltk import sent_tokenize
from typing import List
from tqdm import tqdm

class BigramLanguageModel:
    def __init__(self, tokenizer, vocab_size: int, corpus_lang: str="portuguese"):
        """
        Initializes the bigram language model.

        Parameters
        ----------
        tokenizer : BPETokenization
            The tokenizer object for encoding and decoding text.
        vocab_size : int
            The size of the vocabulary to use for the bigram model.
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.bigram_freqs = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
        self.bigram_probs = None
        self.corpus_lang = corpus_lang

    def build_bigram_matrix(self, corpus: List[str]):
        """
        Builds the bigram frequency matrix from a list of text data.

        Parameters
        ----------
        corpus : List[str]
            The list of text data to train the bigram model on.
        """
        for text in tqdm(corpus, desc="Training bigram matrix"):
            sentences = sent_tokenize(text, self.corpus_lang)
            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence)
                for i in range(len(tokens) - 1):
                    if tokens[i] < self.vocab_size and tokens[i + 1] < self.vocab_size:
                        self.bigram_freqs[tokens[i], tokens[i + 1]] += 1
        
        # smoothing para evitar valores infinitos em steps posteriores
        self.bigram_probs = (self.bigram_freqs + 1e-9) / (self.bigram_freqs.sum(dim=1, keepdim=True) + 1e-9)

    def generate_text(self, initial_text: str, num_tokens: int = 10) -> str:
        """
        Generates text starting from initial_text using bigram probabilities.

        Parameters
        ----------
        initial_text : str
            The initial text to start generation from.
        num_tokens : int, optional
            Number of tokens to generate, by default 10.

        Returns
        -------
        str
            The generated text.
        """
        tokens = self.tokenizer.encode(initial_text)
        for _ in range(num_tokens):
            current_token = tokens[-1]
            if current_token >= self.vocab_size:
                break 
            next_token = torch.multinomial(self.bigram_probs[current_token], 1).item()
            tokens.append(next_token)
            if next_token == self.tokenizer.encode("<EOS>")[0]:
                break
        return self.tokenizer.decode(tokens)

    def calculate_perplexity(self, test_corpus: List[str]) -> float:
        """
        Calculates the perplexity of the model on a test corpus.

        Parameters
        ----------
        test_corpus : List[str]
            The list of text data to evaluate the perplexity on.

        Returns
        -------
        float
            The perplexity score.
        """
        log_likelihood = 0
        token_count = 0

        for text in test_corpus:
            tokens = self.tokenizer.encode(text)
            for i in range(len(tokens) - 1):
                current_token = tokens[i]
                next_token = tokens[i + 1]
                if current_token >= self.vocab_size or next_token >= self.vocab_size:
                    continue
                prob = self.bigram_probs[current_token, next_token].item()
                # 1e-9 adicionado para evitar log(0)
                log_likelihood += np.log(prob + 1e-9)
                token_count += 1

        avg_log_likelihood = log_likelihood / token_count
        perplexity = np.exp(-avg_log_likelihood)
        return perplexity
