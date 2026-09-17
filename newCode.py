"""
GTMicro: Candidate Microservice Identification + LLM Naming + Dependency Discovery
------------------------------------------------------------------------------------
Pipeline:

    Requirements (CSV)
        |
        v
    Sentence Transformer Embeddings
        |
        v
    Cosine Similarity Matrix
        |
        v
    Hierarchical Clustering
        |
        v
    Candidate Microservices  (clusters)
        |
        v
    LLM Naming                 <-- ONLY naming is done by the LLM.
        |                           Clustering itself stays algorithmic.
        v
    Dependency Discovery      (rule-based ontology, no LLM)
        |
        v
    Dependency Graph (JSON + PNG)

The LLM is used ONLY to generate a name for each already-formed cluster.
It never creates, merges, splits, or moves use cases between clusters.
Dependency discovery stays rule-based (keyword -> domain -> ontology),
so it works the same regardless of what name the LLM picked.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # safe for headless environments; also works with a display
import matplotlib.pyplot as plt
import seaborn as sns

from getpass import getpass

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

try:
    from openai import OpenAI
except ImportError:
    sys.exit(
        "ERROR: openai package is not installed.\n"
        "Install with: pip install openai --break-system-packages"
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
DEPENDENCIES_JSON_PATH: str = "dependencies.json"
DEPENDENCY_GRAPH_IMAGE_PATH: str = "dependency_graph.png"
MICROSERVICES_JSON_PATH: str = "microservices_output.json"

# Fallback cluster count, used only if silhouette analysis can't run
# (e.g. too few use cases to evaluate more than one candidate k).
DEFAULT_NUMBER_OF_CLUSTERS: int = 3

# Upper bound on k tried during silhouette analysis.
MAX_CLUSTERS_TO_TRY: int = 10

# LLM model used ONLY for naming clusters (no other pipeline step calls it).
LLM_MODEL: str = "gpt-5.6-luna"

# Sample data written automatically if CSV_FILE does not exist, so the
# script is runnable end-to-end without any manual setup. Mirrors the
# food-ordering example used to validate Dependency Discovery.
SAMPLE_USE_CASES: List[str] = [
    "Customer Registration",
    "Customer Login",
    "Password Reset",
    "Food Search",
    "Restaurant Search",
    "View Menu",
    "Place Order",
    "Order Tracking",
    "Order Cancellation",
    "Payment Processing",
    "Refund Processing",
]


# =====================================================================
# OPENAI API KEY
# =====================================================================

def get_openai_client() -> OpenAI:
    """
    Look for OPENAI_API_KEY in the environment first; if not found, prompt
    the user securely (input is not echoed to the terminal).
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\nOpenAI API key not found in environment.")
        print("Enter your API key below.")
        print("Your key will not be printed on the screen.\n")
        api_key = getpass("Enter OpenAI API key: ")

    if not api_key:
        sys.exit("ERROR: An OpenAI API key is required for LLM-based naming.")

    return OpenAI(api_key=api_key)


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

        # Remove duplicates while preserving order.
        use_cases = list(dict.fromkeys(use_cases))

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
    (metric='precomputed').
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
# STEP 6: CANDIDATE MICROSERVICE EXTRACTION
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
    print("CANDIDATE MICROSERVICE CLUSTERS (pre-naming)")
    print("------------------------------------")
    for cluster_id, members in microservices.items():
        print(f"\nCluster {cluster_id}")
        for use_case in members:
            print(f"  - {use_case}")

    return microservices


# =====================================================================
# STEP 7: LLM-BASED MICROSERVICE NAMING
# =====================================================================
# The LLM is used ONLY to name an already-formed cluster. It does NOT:
#   - create clusters
#   - merge clusters
#   - split clusters
#   - move use cases between clusters

def generate_microservice_name(llm_client: OpenAI, use_cases_in_cluster: List[str]) -> str:
    """Ask the LLM for one concise microservice name for a given cluster."""

    use_case_text = "\n".join(f"- {uc}" for uc in use_cases_in_cluster)

    prompt = f"""
You are naming a microservice identified by a clustering algorithm.

The following use cases already belong to ONE cluster:

{use_case_text}

Give one concise and meaningful microservice name.

Rules:
1. Return ONLY the microservice name.
2. Do not explain.
3. Do not create multiple names.
4. Do not move or remove use cases.
5. Do not merge or split the cluster.
6. Use a software architecture naming style.
7. Prefer names such as:
   Account Service
   Catalog Service
   Order Service
   Payment Service
   Shopping Cart Service
   Authentication Service

Return only the name.
"""

    try:
        response = llm_client.responses.create(model=LLM_MODEL, input=prompt)
        name = response.output_text.strip()
        name = name.strip('"').strip("'")

        if not name:
            name = "Microservice"

        return name

    except Exception as exc:
        print(f"\nWarning: LLM naming failed: {exc}")
        return "Microservice"


def name_microservices_with_llm(
    llm_client: OpenAI, microservices: Dict[int, List[str]]
) -> Dict[str, List[str]]:
    """
    Convert numeric cluster ids into LLM-generated service names.
    Duplicate names are disambiguated with a numeric suffix so the
    resulting dict keys stay unique.
    """
    named: Dict[str, List[str]] = {}
    name_counts: Dict[str, int] = {}

    for cluster_id, use_cases in microservices.items():
        print(f"\nNaming Cluster {cluster_id}...")
        base_name = generate_microservice_name(llm_client, use_cases)
        print(f"Generated name: {base_name}")

        if base_name in name_counts:
            name_counts[base_name] += 1
            final_name = f"{base_name} ({name_counts[base_name]})"
        else:
            name_counts[base_name] = 1
            final_name = base_name

        named[final_name] = use_cases

    print("\n------------------------------------")
    print("NAMED MICROSERVICES")
    print("------------------------------------")
    for service_name, use_cases in named.items():
        print(f"\n{service_name}")
        for use_case in use_cases:
            print(f"  - {use_case}")

    return named


def save_named_microservices(
    named_microservices: Dict[str, List[str]], path: str = MICROSERVICES_JSON_PATH
) -> None:
    """
    Persist the named clusters to a JSON file in the GTMicro output format.
    Not printed to the console — the human-readable listing is handled by
    name_microservices_with_llm() above.
    """
    output = {
        "microservices": [
            {"microservice_name": name, "use_cases": use_cases}
            for name, use_cases in named_microservices.items()
        ]
    }

    try:
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(output, file_handle, indent=2, ensure_ascii=False)
        print(f"\nMicroservices JSON saved to: {path}")
    except OSError as exc:
        raise RuntimeError(f"Failed to write microservices JSON to '{path}': {exc}") from exc


# =====================================================================
# RULE-BASED DOMAIN CLASSIFICATION (used ONLY for dependency discovery)
# =====================================================================
# Naming is done by the LLM (above). This classifier is kept purely to
# figure out each service's business domain (auth/catalog/order/...)
# from its use cases, so the dependency ontology below can be applied.
# It reads the (unchanged) use case text, so it works the same
# regardless of what name the LLM picked for the cluster.

DEFAULT_DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "auth": {
        "login", "logout", "register", "registration", "signup", "signin",
        "password", "reset", "authentication", "auth", "account", "verify",
        "verification",
    },
    "catalog": {
        "search", "product", "products", "item", "items", "catalog",
        "menu", "restaurant", "food", "view", "browse", "listing",
    },
    "order": {
        "order", "orders", "cart", "checkout", "place", "tracking",
        "track", "cancellation", "cancel",
    },
    "payment": {
        "payment", "pay", "refund", "billing", "invoice", "charge",
        "transaction", "processing",
    },
    "shipping": {
        "ship", "shipping", "delivery", "deliver", "dispatch", "courier",
    },
    "notification": {
        "notify", "notification", "email", "sms", "alert", "reminder",
    },
    "review": {
        "review", "rating", "rate", "feedback", "comment",
    },
}

# Domain-level dependency ontology: domain -> set of domains it depends on.
DEFAULT_DEPENDENCY_RULES: Dict[str, Set[str]] = {
    "auth": set(),
    "catalog": set(),
    "order": {"catalog", "payment"},
    "payment": set(),
    "shipping": {"order"},
    "notification": {"order"},
    "review": {"catalog", "order"},
}
def save_services_json(
    named_microservices: dict,
    output_file: str = "services.json",
) -> None:
    """
    Persist the named microservices dictionary to a JSON file.

    Validates the input before writing:
      - the dictionary is not empty
      - every key (service name) is a non-empty string
      - every value (use cases) is a list
      - duplicate use cases within a service are removed (order preserved)

    Args:
        named_microservices: service_name -> list of use case names,
            as already produced by the existing naming step.
        output_file: destination path for the JSON file. Defaults to
            "services.json" and safely overwrites any existing file there.

    Raises:
        ValueError: if the input dictionary is empty or malformed.
        RuntimeError: if the file cannot be written.
    """
    if not named_microservices:
        raise ValueError("named_microservices is empty; nothing to save.")

    validated: Dict[str, List[str]] = {}

    for service_name, use_cases in named_microservices.items():
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError(
                f"Invalid service name: {service_name!r} (must be a non-empty string)."
            )

        if not isinstance(use_cases, list):
            raise ValueError(
                f"Use cases for '{service_name}' must be a list, "
                f"got {type(use_cases).__name__}."
            )

        # Remove duplicate use cases while preserving original order.
        deduped_use_cases = list(dict.fromkeys(use_cases))
        validated[service_name] = deduped_use_cases

    try:
        with open(output_file, "w", encoding="utf-8") as file_handle:
            json.dump(validated, file_handle, indent=4, ensure_ascii=False)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write services JSON to '{output_file}': {exc}"
        ) from exc

    print(f"\nServices JSON saved successfully:\n{output_file}")
    print(f"\nTotal Services: {len(validated)}")

class DomainClassifier:
    """
    Deterministic, rule-based classifier that maps a service's name and
    use cases to a business domain using keyword overlap. No ML model or
    external API call is used; classification is fully explainable and
    reproducible. Used only to drive dependency discovery, not naming.
    """

    DEFAULT_STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "for", "of", "to", "in", "on",
        "with", "by", "via", "using", "new", "user", "users", "customer",
        "allow", "allows", "enable", "system", "feature", "functionality",
        "service",
    }

    def __init__(
        self,
        domain_keywords: Optional[Dict[str, Set[str]]] = None,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        self.domain_keywords: Dict[str, Set[str]] = (
            domain_keywords if domain_keywords is not None else DEFAULT_DOMAIN_KEYWORDS
        )
        self.stopwords: Set[str] = (
            stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        )

    def tokenize(self, text: str) -> Set[str]:
        """Lowercase, strip punctuation, and split into a set of significant words."""
        try:
            cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
            return {tok for tok in cleaned.split() if tok and tok not in self.stopwords}
        except Exception as exc:
            raise ValueError(f"Failed to tokenize text '{text}': {exc}") from exc

    def classify(self, texts: List[str]) -> Optional[str]:
        """
        Classify a list of texts (e.g. a service name plus its use cases)
        into the single best-matching domain based on keyword overlap.
        """
        if not texts:
            return None

        tokens: Set[str] = set()
        for text in texts:
            tokens |= self.tokenize(text)

        if not tokens:
            return None

        scores: Dict[str, int] = {}
        for domain, keywords in self.domain_keywords.items():
            lowered_keywords = {kw.lower() for kw in keywords}
            overlap = tokens & lowered_keywords
            if overlap:
                scores[domain] = len(overlap)

        if not scores:
            return None

        best_domain = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return best_domain


# =====================================================================
# DEPENDENCY DISCOVERY
# =====================================================================

class DependencyDiscovery:
    """
    Discovers directed dependency relationships between candidate
    microservices using a rule-based domain ontology. No LLM is used
    here; all inference comes from keyword-based domain classification
    plus a configurable domain dependency ontology.
    """

    def __init__(
        self,
        classifier: Optional[DomainClassifier] = None,
        dependency_rules: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.classifier: DomainClassifier = classifier or DomainClassifier()
        self.dependency_rules: Dict[str, Set[str]] = (
            dependency_rules if dependency_rules is not None else DEFAULT_DEPENDENCY_RULES
        )

    def _classify_services(
        self, service_use_cases: Dict[str, List[str]]
    ) -> Dict[str, Optional[str]]:
        """Classify each service (name + use cases) into a business domain."""
        service_domains: Dict[str, Optional[str]] = {}
        for service_name, use_cases in service_use_cases.items():
            texts = [service_name] + list(use_cases)
            service_domains[service_name] = self.classifier.classify(texts)
        return service_domains

    def _build_raw_graph(
        self,
        service_use_cases: Dict[str, List[str]],
        service_domains: Dict[str, Optional[str]],
    ) -> Dict[str, List[str]]:
        """Build the dependency graph (with possible cycles) from domain rules."""
        domain_to_services: Dict[str, List[str]] = {}
        for service, domain in service_domains.items():
            if domain:
                domain_to_services.setdefault(domain, []).append(service)

        graph: Dict[str, List[str]] = {service: [] for service in service_use_cases}

        for service, domain in service_domains.items():
            if not domain:
                continue

            required_domains = self.dependency_rules.get(domain, set())
            for required_domain in required_domains:
                for target_service in domain_to_services.get(required_domain, []):
                    if target_service == service:
                        continue  # ignore self-dependencies
                    if target_service not in graph[service]:
                        graph[service].append(target_service)  # dedup

        return graph

    @staticmethod
    def _remove_cycles(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Detect and break circular dependencies using DFS with a
        white/gray/black coloring scheme. Any edge that would close a
        cycle is dropped and reported; the rest of the graph is kept.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {node: WHITE for node in graph}
        cleaned_graph: Dict[str, List[str]] = {
            node: list(edges) for node, edges in graph.items()
        }

        def visit(node: str) -> None:
            color[node] = GRAY
            for neighbor in list(cleaned_graph[node]):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    print(
                        f"WARNING: circular dependency detected "
                        f"({node} -> {neighbor}); edge removed to keep the "
                        f"graph acyclic."
                    )
                    cleaned_graph[node].remove(neighbor)
                elif color[neighbor] == WHITE:
                    visit(neighbor)
            color[node] = BLACK

        for node in list(graph.keys()):
            if color[node] == WHITE:
                visit(node)

        return cleaned_graph

    def discover(self, service_use_cases: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Run full dependency discovery on a service_name -> use_cases map."""
        if not service_use_cases:
            raise ValueError("service_use_cases is empty; nothing to analyze.")

        try:
            service_domains = self._classify_services(service_use_cases)
            raw_graph = self._build_raw_graph(service_use_cases, service_domains)
            acyclic_graph = self._remove_cycles(raw_graph)
        except Exception as exc:
            raise RuntimeError(f"Dependency discovery failed: {exc}") from exc

        return acyclic_graph

    @staticmethod
    def print_report(graph: Dict[str, List[str]]) -> None:
        """Print the dependency graph in a human-readable format."""
        print("\n====================================")
        print("DEPENDENCY DISCOVERY RESULTS")
        print("====================================")
        for service, dependencies in graph.items():
            print(f"\n{service}")
            if dependencies:
                for dependency in dependencies:
                    print(f"  -> {dependency}")
            else:
                print("  (no dependencies)")

    @staticmethod
    def to_json(graph: Dict[str, List[str]], path: str = DEPENDENCIES_JSON_PATH) -> None:
        """Export the dependency graph as JSON."""
        try:
            with open(path, "w", encoding="utf-8") as file_handle:
                json.dump(graph, file_handle, indent=4)
            print(f"\nDependency graph exported to: {path}")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write dependency JSON to '{path}': {exc}"
            ) from exc

    @staticmethod
    def visualize(
        graph: Dict[str, List[str]], path: str = DEPENDENCY_GRAPH_IMAGE_PATH
    ) -> None:
        """Render the dependency graph as a directed graph image."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "networkx is required for dependency graph visualization.\n"
                "Install with: pip install networkx --break-system-packages"
            ) from exc

        try:
            directed_graph = nx.DiGraph()
            directed_graph.add_nodes_from(graph.keys())
            for service, dependencies in graph.items():
                for dependency in dependencies:
                    directed_graph.add_edge(service, dependency)

            plt.figure(figsize=(10, 7))
            layout = nx.spring_layout(directed_graph, seed=42, k=1.2)

            nx.draw_networkx_nodes(
                directed_graph, layout, node_color="#4C72B0",
                node_size=2400, alpha=0.9,
            )
            nx.draw_networkx_labels(
                directed_graph, layout, font_size=9, font_color="white",
                font_weight="bold",
            )
            nx.draw_networkx_edges(
                directed_graph, layout, arrowstyle="-|>", arrowsize=22,
                edge_color="#333333", width=1.6,
                connectionstyle="arc3,rad=0.08",
            )

            plt.title("Microservice Dependency Graph")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Dependency graph visualization saved to: {path}")
        except Exception as exc:
            raise RuntimeError(f"Dependency graph visualization failed: {exc}") from exc


# =====================================================================
# MAIN EXECUTION BLOCK
# =====================================================================

def main() -> None:
    try:
        # --- OpenAI client (used only for naming) ---
        llm_client = get_openai_client()

        # --- Candidate microservice identification (GTMicro core) ---
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

        microservices_by_id = extract_microservices(
            linkage_matrix, use_cases, optimal_clusters
        )

        # --- LLM-based naming (clusters are already fixed; LLM only names them) ---
        named_microservices = name_microservices_with_llm(llm_client, microservices_by_id)
        save_named_microservices(named_microservices, MICROSERVICES_JSON_PATH)

        # --- Dependency Discovery (rule-based, unaffected by LLM naming) ---
        classifier = DomainClassifier()
        dependency_discovery = DependencyDiscovery(classifier=classifier)
        dependency_graph = dependency_discovery.discover(named_microservices)

        dependency_discovery.print_report(dependency_graph)
        dependency_discovery.to_json(dependency_graph, DEPENDENCIES_JSON_PATH)
        dependency_discovery.visualize(dependency_graph, DEPENDENCY_GRAPH_IMAGE_PATH)

    except Exception as exc:
        sys.exit(f"ERROR: pipeline failed -> {exc}")


if __name__ == "__main__":
    main()