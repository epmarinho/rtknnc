import numpy as np

class KNNGraph:
    """Directed K-NN graph com suporte a round-trip."""

    def __init__(self, k: int = 10, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.adjacency_ = None   # matriz esparsa N×N

    def fit(self, X: np.ndarray) -> "KNNGraph":
        """Constrói o grafo dirigido a partir dos dados."""
        ...
        return self

    def roundtrip_edges(self) -> np.ndarray:
        """Retorna arestas de round-trip (mútuas)."""
        ...