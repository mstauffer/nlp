from typing import Iterable, List, Tuple, Dict
from tqdm import tqdm

class BPETokenization:
    """
    A class that implements the BPE algorithm with a vocabulary size target.

    Parameters
    ----------
    vocab_size : int, default=1000
        The target vocabulary size for the BPE algorithm.

    Methods
    -------
    text_to_byte_ids(text: str)
        Convert text into a list of byte IDs.

    get_pair_counts(token_ids: List[int])
        Count the frequency of adjacent pairs in the token list.

    find_most_frequent_pair(pair_counts: Dict[Tuple[int, int], int])
        Find the most frequent adjacent pair in the token list.

    merge_pair(token_ids: List[int], pair: Tuple[int, int], new_id: int)
        Merge all occurrences of the most frequent pair into a single ID.

    train(data: List[str])
        Encode all the text provided and updates the object's vocab.

    encode(text: str)
        Apply all the sub-methods to encode text using the BPE algorithm.

    decode(ids: List[int])
        Decode a list of IDs into the original text.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}

        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        self.BOS_ID = 0
        self.EOS_ID = 1
        self.next_id = 2
        
        self.vocab[self.BOS_TOKEN] = self.BOS_ID
        self.vocab[self.EOS_TOKEN] = self.EOS_ID
        self.reverse_vocab[self.BOS_ID] = self.BOS_TOKEN
        self.reverse_vocab[self.EOS_ID] = self.EOS_TOKEN

    def text_to_byte_ids(self, text: str) -> List[int]:
        """Convert text into a list of byte IDs."""
        return list(text.encode("utf-8"))

    def get_pair_counts(self, token_ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count the frequency of adjacent pairs in the token list."""
        counts = {}
        for pair in zip(token_ids, token_ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def find_most_frequent_pair(self, pair_counts: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
        """Find the most frequent adjacent pair in the token list."""
        return max(pair_counts, key=pair_counts.get)

    def merge_pair(self, token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Merge all occurrences of the most frequent pair into a single ID."""
        new_ids = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(token_ids[i])
                i += 1
        return new_ids

    def update_vocab(self, pair: Tuple[int, int], new_id: int):
        """Add the new pair to the vocabulary."""
        self.vocab[pair] = new_id
        self.reverse_vocab[new_id] = pair

    def bpe_step(self, token_ids: List[int], next_id: int) -> Tuple[List[int], int]:
        """Perform one BPE merge operation on token_ids."""
        pair_counts = self.get_pair_counts(token_ids)
        if not pair_counts:
            return token_ids, next_id

        most_frequent_pair = self.find_most_frequent_pair(pair_counts)
        self.update_vocab(most_frequent_pair, next_id)
        token_ids = self.merge_pair(token_ids, most_frequent_pair, next_id)
        return token_ids, next_id + 1

    def encode(self, text: str) -> List[int]:
        """Encode text using the BPE algorithm."""
        token_ids = [self.BOS_ID] + self.text_to_byte_ids(text) + [self.EOS_ID]
        next_id = max(token_ids) + 1

        while len(self.vocab) < self.vocab_size:
            token_ids, next_id = self.bpe_step(token_ids, next_id)
            if not token_ids:
                break
        return token_ids

    def train(self, data: Iterable[str], progress_bar: bool = True):
        """
        Train the tokenizer on a large corpus, accepting an iterable of text instances.
        """
        next_id = 256
        iterable = tqdm(data) if progress_bar else data

        for text in iterable:
            token_ids = self.text_to_byte_ids(text)
            while len(self.vocab) < self.vocab_size:
                token_ids, next_id = self.bpe_step(token_ids, next_id)
                if not token_ids:
                    break

    def decode(self, ids: List[int]) -> str:
        """Decode a list of IDs into the original text, ignoring special tokens."""
        decoded = []
        for i in ids:
            if i == self.BOS_ID or i == self.EOS_ID:
                continue
            elif i < 256:
                decoded.append(i)
            else:
                decoded.append(ord('?'))
        
        return bytes(decoded).decode("utf-8", errors="replace")
