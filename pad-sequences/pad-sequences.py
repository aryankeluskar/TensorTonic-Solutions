import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if not seqs:
        return np.zeros((0,0))
    
    if not max_len:
        max_len = max(len(seq) for seq in seqs)

    out = np.full((len(seqs), max_len), pad_value)
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        out[i, :length] = seq[:length]

    return out