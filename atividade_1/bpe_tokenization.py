from typing import List, Tuple, Dict

class BPETokenization:
    """
    A class that implements the BPE algorithm.

    Parameters
    ----------
    k : int, default=10
        The number of merge operations performed throughout the algorithm.

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
    
    encode(text: str)
        Apply all the sub-methods to encode text using the BPE algorithm.
    
    """
    def __init__(self, k: int=10):
        self.k = k
        self.vocab = {}
        self.reverse_vocab = {}

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

    def encode(self, text: str) -> List[int]:
        """Encode text using the BPE algorithm."""
        # Convert text to byte IDs
        token_ids = self.text_to_byte_ids(text)
        # Start new IDs above the byte range
        next_id = max(token_ids) + 1

        # for k merges, we
        for _ in range(self.k):
            # Build and count the token pairs
            pair_counts = self.get_pair_counts(token_ids)
            if not pair_counts:
                break
            
            # Find the most frequent pair
            most_frequent_pair = self.find_most_frequent_pair(pair_counts)

            # Assign a new int ID to the most frequent pair
            self.vocab[most_frequent_pair] = next_id
            self.reverse_vocab[next_id] = most_frequent_pair

            # Merge the most frequent pair
            token_ids = self.merge_pair(token_ids, most_frequent_pair, next_id)
            next_id += 1

        return token_ids
