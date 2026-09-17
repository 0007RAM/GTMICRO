"""
GTMicro: Candidate Microservice Identification from Software Use Cases
------------------------------------------------------------------------
Implements the original GTMicro approach:

    Use Cases -> Sentence Transformer -> Embeddings -> Cosine Similarity
    -> Distance Matrix -> Hierarchical Clustering -> Candidate Microservices

Enhancement over the base GTMicro pipeline: instead of a hardcoded
`number_of_clusters`, the optimal cluster count is chosen automatically
via Silhouette Score analysis over a range of candidate k values.

No LLM-based naming, no diagrams beyond the required heatmap/dendrogram,
no code generation. Pipeline stops at candidate microservice extraction.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # safe for headless environments; also works with a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

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

CSV_FILE: str = "requirements.csv"
USE_CASE_COLUMN: str = "use_case_name"

# Fallback cluster count, used only if silhouette analysis can't run
# (e.g. too few use cases to evaluate more than one candidate k).
DEFAULT_NUMBER_OF_CLUSTERS: int = 3

# Upper bound on k tried during silhouette analysis.
MAX_CLUSTERS_TO_TRY: int = 10

# Sample data written automatically if CSV_FILE does not exist, so the
# script is runnable end-to-end without any manual setup.
SAMPLE_USE_CASES: List[str] = [
    "user login using email and password",
    "new user registration",
    "reset forgotten password",
    "search products",
    "view product details",
    "add product to cart",
    "place order",
    "process payment",
]


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
# DATA LOADING
# =====================================================================

def ensure_sample_csv(csv_file: str, column_name: str) -> None:
    """
    Create a sample CSV file if it doesn't already exist, so the pipeline
    is runnable without requiring the user to hand-craft input data first.
    """
    if os.path.exists(csv_file):
        return

    print(f"'{csv_file}' not found. Creating a sample CSV with default use cases...")
    pd.DataFrame({column_name: SAMPLE_USE_CASES}).to_csv(csv_file, index=False)


def load_use_cases_from_csv(csv_file: str, column_name: str) -> List[str]:
    """Load use cases from a CSV file, validating the target column exists."""
    try:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

        df = pd.read_csv(csv_file)

        if column_name not in df.columns:
            raise ValueError(
                f"Column '{column_name}' not found in '{csv_file}'.\n"
                f"Available columns: {list(df.columns)}"
            )

        use_cases = df[column_name].dropna().astype(str).str.strip().tolist()
        use_cases = [uc for uc in use_cases if uc]  # drop empty strings

        if not use_cases:
            raise ValueError(f"No non-empty use cases found in column '{column_name}'.")

        if len(use_cases) < 2:
            raise ValueError(
                "At least 2 use cases are required to compute similarity/clustering."
            )

        print(f"\nLoaded {len(use_cases)} use cases from '{csv_file}'.")
        return use_cases

    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV '{csv_file}': {exc}") from exc


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert similarity to distance (distance = 1 - similarity), build a
    valid condensed distance matrix via squareform(), and run hierarchical
    (agglomerative) clustering with average linkage.

    Returns:
        (linkage_matrix, square_distance_matrix)
        The square distance matrix is returned too so silhouette analysis
        can reuse the *same* cosine-distance metric the clustering was
        actually performed on, instead of silently switching metrics.
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

    return linkage_matrix, distance_matrix


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
# OPTIMAL CLUSTER COUNT: SILHOUETTE ANALYSIS
# =====================================================================

def find_optimal_clusters(
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    n_use_cases: int,
    max_clusters: int = MAX_CLUSTERS_TO_TRY,
) -> int:
    """
    Find the optimal number of clusters using the Silhouette Score,
    evaluated on the SAME cosine distance matrix the clustering used
    (metric='precomputed'), so the score is consistent with how the
    tree was actually built.

    Silhouette Score is only defined for 2 <= n_clusters <= n_samples - 1,
    so the search range is bounded accordingly. If there are too few use
    cases to evaluate more than one candidate k, this falls back to
    DEFAULT_NUMBER_OF_CLUSTERS.
    """
    upper_bound = min(max_clusters, n_use_cases - 1)

    if upper_bound < 2:
        print(
            "\nNot enough use cases to run silhouette analysis "
            f"(need >= 3, got {n_use_cases}). "
            f"Falling back to DEFAULT_NUMBER_OF_CLUSTERS={DEFAULT_NUMBER_OF_CLUSTERS}."
        )
        return min(DEFAULT_NUMBER_OF_CLUSTERS, n_use_cases)

    best_k = 2
    best_score = -1.0
    found_valid_k = False

    print("\n--- SILHOUETTE ANALYSIS ---")

    for k in range(2, upper_bound + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")

        # fcluster can occasionally collapse to fewer than k distinct labels;
        # silhouette_score requires at least 2 distinct labels to be valid.
        if len(set(labels)) < 2:
            print(f"Clusters = {k} | skipped (produced < 2 distinct groups)")
            continue

        try:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
        except Exception as exc:
            print(f"Clusters = {k} | skipped (silhouette failed: {exc})")
            continue

        found_valid_k = True
        print(f"Clusters = {k} | Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    if not found_valid_k:
        print(
            f"\nNo valid k produced a scoreable clustering. "
            f"Falling back to DEFAULT_NUMBER_OF_CLUSTERS={DEFAULT_NUMBER_OF_CLUSTERS}."
        )
        return min(DEFAULT_NUMBER_OF_CLUSTERS, n_use_cases)

    print(f"\nBest Number of Clusters: {best_k}")
    print(f"Best Silhouette Score  : {round(best_score, 4)}")

    return best_k


# =====================================================================
# STEP 6 & 7: CANDIDATE MICROSERVICE EXTRACTION + FINAL OUTPUT
# =====================================================================

def extract_microservices(
    linkage_matrix: np.ndarray,
    use_cases: List[str],
    number_of_clusters: int,
) -> Dict[int, List[str]]:
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

    microservices: Dict[int, List[str]] = {}
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

        ensure_sample_csv(CSV_FILE, USE_CASE_COLUMN)
        use_cases = load_use_cases_from_csv(CSV_FILE, USE_CASE_COLUMN)

        embeddings = generate_embeddings(model, use_cases)
        sim_df = compute_similarity_matrix(embeddings, use_cases)

        generate_heatmap(sim_df, HEATMAP_PATH)

        linkage_matrix, distance_matrix = perform_clustering(sim_df, LINKAGE_METHOD)
        plot_dendrogram(linkage_matrix, use_cases, DENDROGRAM_PATH)

        optimal_clusters = find_optimal_clusters(
            linkage_matrix, distance_matrix, n_use_cases=len(use_cases)
        )

        extract_microservices(linkage_matrix, use_cases, optimal_clusters)

    except Exception as exc:
        sys.exit(f"ERROR: pipeline failed -> {exc}")


if __name__ == "__main__":
    main()