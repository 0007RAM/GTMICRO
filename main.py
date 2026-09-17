"""
GTMicro: Candidate Microservice Identification from Software Use Cases
------------------------------------------------------------------------
Implements the original GTMicro approach only:

    Use Cases -> Sentence Transformer -> Embeddings -> Cosine Similarity
    -> Distance Matrix -> Hierarchical Clustering -> Candidate Microservices

No LLM-based naming, no diagrams beyond the required heatmap/dendrogram,
no code generation. Pipeline stops at candidate microservice extraction.
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # safe for headless environments; also works with a display
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit(
        "ERROR: sentence-transformers is not installed.\n"
        "Install with: pip install sentence-transformers --break-system-packages"
    )
# =====================================================================
# CONFIG
# =====================================================================

MODEL_NAME: str = "all-MiniLM-L6-v2"   # preferred model per spec
# Alternative per spec: "bert-base-nli-mean-tokens"

LINKAGE_METHOD: str = "average"
HEATMAP_PATH: str = "similarity_heatmap.png"
DENDROGRAM_PATH: str = "dendrogram.png"

USE_CASES: List[str] = [
    "user login using email and password",
    "new user registration",
    "reset forgotten password",
    "search products",
    "view product details",
    "add product to cart",
    "place order",
    "process payment",
]

NUMBER_OF_CLUSTERS: int = 3


# =====================================================================
# STEP 0: LOAD MODEL (once)
# =====================================================================

def load_model(model_name: str) -> SentenceTransformer:
    """Load a sentence-transformer model. Raises a clear error on failure."""
    try:
        print(f"Loading model '{model_name}'...")
        model = SentenceTransformer(model_name)
        return model
    except Exception as exc:
        sys.exit(f"ERROR: failed to load model '{model_name}'.\nDetails: {exc}")


# =====================================================================
# STEP 1: GENERATE EMBEDDINGS
# =====================================================================

def generate_embeddings(model: SentenceTransformer, use_cases: List[str]) -> np.ndarray:
    """Generate sentence embeddings for all use cases."""
    if not use_cases:
        raise ValueError("use_cases list is empty; nothing to embed.")

    try:
        embeddings = model.encode(use_cases, show_progress_bar=False)
    except Exception as exc:
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    print("\n--- STEP 1: EMBEDDINGS ---")
    print(f"Number of use cases   : {len(use_cases)}")
    print(f"Embedding dimension   : {embeddings.shape[1]}")
    print(f"Embedding matrix shape: {embeddings.shape}")

    return embeddings


# =====================================================================
# STEP 2: COSINE SIMILARITY MATRIX
# =====================================================================

def compute_similarity_matrix(
    embeddings: np.ndarray, use_cases: List[str]
) -> pd.DataFrame:
    """Compute an NxN cosine similarity matrix as a labeled DataFrame."""
    try:
        sim_matrix = cosine_similarity(embeddings)
    except Exception as exc:
        raise RuntimeError(f"Similarity computation failed: {exc}") from exc

    # Guard against floating point noise pushing values slightly outside [-1, 1]
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    sim_df = pd.DataFrame(sim_matrix, index=use_cases, columns=use_cases)

    print("\n--- STEP 2: SIMILARITY MATRIX ---")
    print(sim_df.round(3))

    return sim_df


# =====================================================================
# VISUALIZATION 1: SIMILARITY HEATMAP
# =====================================================================

def generate_heatmap(sim_df: pd.DataFrame, path: str = HEATMAP_PATH) -> None:
    """Plot and save a seaborn heatmap of the similarity matrix."""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="viridis", square=True)
        plt.title("Use Case Similarity Matrix")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\nSimilarity heatmap saved to: {path}")
    except Exception as exc:
        raise RuntimeError(f"Heatmap generation failed: {exc}") from exc


# =====================================================================
# STEP 3 & 4: DISTANCE MATRIX + HIERARCHICAL CLUSTERING
# =====================================================================

def perform_clustering(
    sim_df: pd.DataFrame, method: str = LINKAGE_METHOD
) -> np.ndarray:
    """
    Convert similarity to distance (distance = 1 - similarity), build a
    valid condensed distance matrix via squareform(), and run hierarchical
    (agglomerative) clustering with average linkage.
    """
    try:
        distance_matrix = 1.0 - sim_df.values

        # Force exact zero diagonal and perfect symmetry before squareform,
        # since squareform() requires a valid symmetric distance matrix and
        # floating point rounding can otherwise violate that by a hair.
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

        # IMPORTANT: do NOT pass the NxN distance matrix directly to linkage().
        # Convert it to condensed form first.
        condensed_distance = squareform(distance_matrix, checks=False)

        linkage_matrix = linkage(condensed_distance, method=method)
    except Exception as exc:
        raise RuntimeError(f"Clustering failed: {exc}") from exc

    print("\n--- STEP 3 & 4: DISTANCE MATRIX + HIERARCHICAL CLUSTERING ---")
    print(f"Linkage method: {method}")
    print(f"Linkage matrix shape: {linkage_matrix.shape}")

    return linkage_matrix


# =====================================================================
# STEP 5: DENDROGRAM
# =====================================================================

def plot_dendrogram(
    linkage_matrix: np.ndarray, use_cases: List[str], path: str = DENDROGRAM_PATH
) -> None:
    """Plot and save the hierarchical clustering dendrogram."""
    try:
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=use_cases, leaf_rotation=90)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Use Cases")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\nDendrogram saved to: {path}")
    except Exception as exc:
        raise RuntimeError(f"Dendrogram generation failed: {exc}") from exc


# =====================================================================
# STEP 6 & 7: CANDIDATE MICROSERVICE EXTRACTION + FINAL OUTPUT
# =====================================================================

def extract_microservices(
    linkage_matrix: np.ndarray,
    use_cases: List[str],
    number_of_clusters: int,
) -> dict[int, List[str]]:
    """
    Cut the hierarchical tree into `number_of_clusters` flat clusters and
    return them as candidate microservices, keyed by cluster id.
    """
    if number_of_clusters < 1 or number_of_clusters > len(use_cases):
        raise ValueError(
            f"number_of_clusters={number_of_clusters} is invalid for "
            f"{len(use_cases)} use cases."
        )

    try:
        cluster_labels = fcluster(
            linkage_matrix, t=number_of_clusters, criterion="maxclust"
        )
    except Exception as exc:
        raise RuntimeError(f"Cluster extraction failed: {exc}") from exc

    microservices: dict[int, List[str]] = {}
    for cluster_id in sorted(set(cluster_labels)):
        members = [
            uc for uc, label in zip(use_cases, cluster_labels) if label == cluster_id
        ]
        microservices[cluster_id] = members

    print("\n------------------------------------")
    print("FINAL MICROSERVICE CLUSTERS")
    print("------------------------------------")
    for cluster_id, members in microservices.items():
        print(f"\nMicroservice {cluster_id}")
        for use_case in members:
            print(f"- {use_case}")

    return microservices


# =====================================================================
# MAIN EXECUTION BLOCK
# =====================================================================

def main() -> None:
    try:
        model = load_model(MODEL_NAME)
        embeddings = generate_embeddings(model, USE_CASES)
        sim_df = compute_similarity_matrix(embeddings, USE_CASES)

        generate_heatmap(sim_df, HEATMAP_PATH)

        linkage_matrix = perform_clustering(sim_df, LINKAGE_METHOD)
        plot_dendrogram(linkage_matrix, USE_CASES, DENDROGRAM_PATH)

        extract_microservices(linkage_matrix, USE_CASES, NUMBER_OF_CLUSTERS)

    except Exception as exc:
        sys.exit(f"ERROR: pipeline failed -> {exc}")


if __name__ == "__main__":
    main()