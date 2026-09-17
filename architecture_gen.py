"""
GTMicro+ : Architecture Generation Phase
------------------------------------------------------------------------
Input:
    dependencies.json   (service_name -> list of service names it depends on)

Outputs:
    architecture.mmd            Mermaid diagram source
    architecture.png            NetworkX + Matplotlib directed graph image
    architecture_summary.json   Computed architecture metrics
    architecture_report.txt     Human-readable report

This module is self-contained and drop-in: add this file next to your
existing GTMicro+ pipeline script and call `ArchitectureGenerator().run()`
right after your Dependency Discovery phase writes dependencies.json.
See the bottom of this file for the exact one-line integration snippet.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # safe for headless environments; also works with a display
import matplotlib.pyplot as plt


class ArchitectureGenerator:
    """
    Generates a Mermaid diagram, PNG visualization, JSON summary, and a
    human-readable text report from a microservice dependency graph.

    This class only reads `dependencies.json` and produces static
    architecture artifacts; it does not call any LLM or external API.

    Typical usage:
        generator = ArchitectureGenerator()
        generator.run()
    """

    def __init__(
        self,
        dependencies_path: str = "dependencies.json",
        mermaid_path: str = "architecture.mmd",
        png_path: str = "architecture.png",
        summary_path: str = "architecture_summary.json",
        report_path: str = "architecture_report.txt",
    ) -> None:
        """
        Args:
            dependencies_path: path to the input dependency graph JSON.
            mermaid_path: output path for the Mermaid diagram source.
            png_path: output path for the rendered architecture diagram.
            summary_path: output path for the computed architecture metrics.
            report_path: output path for the human-readable report.
        """
        self.dependencies_path: str = dependencies_path
        self.mermaid_path: str = mermaid_path
        self.png_path: str = png_path
        self.summary_path: str = summary_path
        self.report_path: str = report_path

        self.dependency_graph: Dict[str, List[str]] = {}
        self.graph: Optional[object] = None  # networkx.DiGraph, set by build_graph()
        self._id_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load_dependencies(self) -> Dict[str, List[str]]:
        """
        Load and validate the dependency graph from `dependencies_path`.

        Ensures every dependency target also exists as a node (even if it
        has no outgoing edges of its own), de-duplicates edges, and drops
        any accidental self-dependencies.

        Returns:
            service_name -> list of service names it depends on.

        Raises:
            RuntimeError: if the file is missing, unreadable, or malformed.
        """
        try:
            if not os.path.exists(self.dependencies_path):
                raise FileNotFoundError(
                    f"'{self.dependencies_path}' not found. "
                    f"Run the Dependency Discovery phase first."
                )

            with open(self.dependencies_path, "r", encoding="utf-8") as file_handle:
                raw_data = json.load(file_handle)

            if not isinstance(raw_data, dict):
                raise ValueError("dependencies.json must contain a JSON object.")

            cleaned: Dict[str, List[str]] = {}
            for service, dependencies in raw_data.items():
                if not isinstance(dependencies, list):
                    raise ValueError(f"Dependencies for '{service}' must be a list.")

                seen: set = set()
                clean_deps: List[str] = []
                for dependency in dependencies:
                    if dependency == service or dependency in seen:
                        continue  # skip self-dependency / duplicate
                    seen.add(dependency)
                    clean_deps.append(dependency)

                cleaned[service] = clean_deps

            # Defensive: any dependency target not declared as its own
            # top-level key still becomes a node (with no outgoing edges),
            # so the graph and diagrams never silently drop a service.
            referenced_targets = {dep for deps in cleaned.values() for dep in deps}
            for target in referenced_targets:
                cleaned.setdefault(target, [])

            if not cleaned:
                raise ValueError("dependencies.json is empty; nothing to visualize.")

            self.dependency_graph = cleaned
            print(f"Loaded {len(cleaned)} services from '{self.dependencies_path}'.")
            return cleaned

        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read '{self.dependencies_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load dependency graph: {exc}") from exc

    # ------------------------------------------------------------------
    # MERMAID GENERATION
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_id(name: str, fallback_seed: int = 0) -> str:
        """Convert a service name into an alphanumeric-only Mermaid/NetworkX node id."""
        sanitized = re.sub(r"[^0-9a-zA-Z]", "", name)
        if not sanitized:
            sanitized = f"Service{fallback_seed}"
        return sanitized

    def _build_id_map(self) -> Dict[str, str]:
        """Build a unique, Mermaid-safe node id for every service name."""
        id_map: Dict[str, str] = {}
        used_ids: Dict[str, int] = {}

        for index, service in enumerate(self.dependency_graph):
            base_id = self._sanitize_id(service, fallback_seed=index)
            if base_id in used_ids:
                used_ids[base_id] += 1
                unique_id = f"{base_id}{used_ids[base_id]}"
            else:
                used_ids[base_id] = 0
                unique_id = base_id
            id_map[service] = unique_id

        return id_map

    def generate_mermaid(self) -> str:
        """
        Build Mermaid flowchart source ("graph TD") from the loaded
        dependency graph. Every service is declared as a labeled node
        (so isolated services still render), followed by an edge line
        per dependency, e.g.:

            graph TD
                OrderService["Order Service"]
                CatalogService["Catalog Service"]

                OrderService --> CatalogService

        Returns:
            The complete Mermaid diagram source as a string.

        Raises:
            RuntimeError: if called before load_dependencies().
        """
        if not self.dependency_graph:
            raise RuntimeError("No dependency graph loaded. Call load_dependencies() first.")

        try:
            self._id_map = self._build_id_map()
            lines: List[str] = ["graph TD"]

            for service, node_id in self._id_map.items():
                safe_label = service.replace('"', "'")
                lines.append(f'    {node_id}["{safe_label}"]')

            lines.append("")

            for service, dependencies in self.dependency_graph.items():
                source_id = self._id_map[service]
                for dependency in dependencies:
                    target_id = self._id_map.get(dependency)
                    if target_id is None:
                        continue  # defensive; load_dependencies() should prevent this
                    lines.append(f"    {source_id} --> {target_id}")

            return "\n".join(lines)

        except Exception as exc:
            raise RuntimeError(f"Mermaid generation failed: {exc}") from exc

    def save_mermaid(self, mermaid_code: str) -> None:
        """
        Write Mermaid diagram source to `mermaid_path`.

        Args:
            mermaid_code: output of generate_mermaid().

        Raises:
            RuntimeError: if the file cannot be written.
        """
        try:
            with open(self.mermaid_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(mermaid_code + "\n")
            print(f"Mermaid architecture saved to: {self.mermaid_path}")
        except OSError as exc:
            raise RuntimeError(f"Failed to write '{self.mermaid_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # NETWORKX GRAPH + PNG VISUALIZATION
    # ------------------------------------------------------------------

    def build_graph(self):
        """
        Build a NetworkX directed graph from the loaded dependency graph.

        Returns:
            The constructed networkx.DiGraph.

        Raises:
            RuntimeError: if networkx is missing or the graph isn't loaded.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "networkx is required. Install with: "
                "pip install networkx --break-system-packages"
            ) from exc

        if not self.dependency_graph:
            raise RuntimeError("No dependency graph loaded. Call load_dependencies() first.")

        try:
            directed_graph = nx.DiGraph()
            directed_graph.add_nodes_from(self.dependency_graph.keys())
            for service, dependencies in self.dependency_graph.items():
                for dependency in dependencies:
                    directed_graph.add_edge(service, dependency)

            self.graph = directed_graph
            return directed_graph

        except Exception as exc:
            raise RuntimeError(f"Failed to build architecture graph: {exc}") from exc

    def generate_png(self) -> None:
        """
        Render the architecture graph to `png_path` using NetworkX +
        Matplotlib: directed edges with visible arrowheads, spaced-out
        nodes, visible service-name labels, and services with the most
        outgoing dependencies highlighted as core services.

        Raises:
            RuntimeError: if build_graph() hasn't been called, or rendering fails.
        """
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "networkx is required. Install with: "
                "pip install networkx --break-system-packages"
            ) from exc

        try:
            node_count = max(self.graph.number_of_nodes(), 1)

            # Scale figure size and layout spacing with node count so
            # labels stay legible as the architecture grows.
            fig_width = max(10, node_count * 1.8)
            fig_height = max(7, node_count * 1.2)
            plt.figure(figsize=(fig_width, fig_height))

            layout = nx.spring_layout(
                self.graph, seed=42, k=1.6 / (node_count ** 0.5)
            )

            out_degrees = dict(self.graph.out_degree())
            max_out_degree = max(out_degrees.values()) if out_degrees else 0
            node_colors = [
                "#C44E52" if max_out_degree > 0 and out_degrees[node] == max_out_degree
                else "#4C72B0"
                for node in self.graph.nodes()
            ]

            nx.draw_networkx_nodes(
                self.graph, layout, node_color=node_colors,
                node_size=2800, alpha=0.92, edgecolors="#222222", linewidths=1.0,
            )
            nx.draw_networkx_labels(
                self.graph, layout, font_size=9, font_color="white", font_weight="bold",
            )
            nx.draw_networkx_edges(
                self.graph, layout, arrowstyle="-|>", arrowsize=24,
                edge_color="#333333", width=1.6, node_size=2800,
                connectionstyle="arc3,rad=0.1",
            )

            plt.title("Microservice Architecture", fontsize=15, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(self.png_path, dpi=150)
            plt.close()
            print(f"Architecture diagram saved to: {self.png_path}")

        except Exception as exc:
            raise RuntimeError(f"Architecture PNG generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------

    def generate_summary(self) -> Dict[str, object]:
        """
        Compute architecture metrics and write them to `summary_path`:
        total services, total dependencies, core services (highest
        outgoing dependency count), leaf services (no outgoing
        dependencies), and per-service dependency counts.

        Returns:
            The computed summary dictionary.

        Raises:
            RuntimeError: if the graph isn't loaded or the file can't be written.
        """
        if not self.dependency_graph:
            raise RuntimeError("No dependency graph loaded. Call load_dependencies() first.")

        try:
            dependency_counts: Dict[str, int] = {
                service: len(dependencies)
                for service, dependencies in self.dependency_graph.items()
            }

            total_services = len(self.dependency_graph)
            total_dependencies = sum(dependency_counts.values())

            max_count = max(dependency_counts.values()) if dependency_counts else 0
            core_services = (
                [service for service, count in dependency_counts.items() if count == max_count]
                if max_count > 0 else []
            )
            leaf_services = [
                service for service, count in dependency_counts.items() if count == 0
            ]

            summary: Dict[str, object] = {
                "total_services": total_services,
                "total_dependencies": total_dependencies,
                "core_services": core_services,
                "leaf_services": leaf_services,
                "dependency_count_per_service": dependency_counts,
            }

            with open(self.summary_path, "w", encoding="utf-8") as file_handle:
                json.dump(summary, file_handle, indent=4)
            print(f"Architecture summary saved to: {self.summary_path}")

            return summary

        except OSError as exc:
            raise RuntimeError(f"Failed to write '{self.summary_path}': {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Summary generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    def generate_report(self, summary: Dict[str, object]) -> None:
        """
        Write a human-readable architecture report to `report_path`.

        Args:
            summary: output of generate_summary().

        Raises:
            RuntimeError: if the graph isn't loaded or the file can't be written.
        """
        if not self.dependency_graph:
            raise RuntimeError("No dependency graph loaded. Call load_dependencies() first.")

        try:
            lines: List[str] = ["ARCHITECTURE REPORT", "-" * 19, ""]

            lines.append(f"Total Services: {summary['total_services']}")
            lines.append("")
            lines.append(f"Total Dependencies: {summary['total_dependencies']}")
            lines.append("")

            lines.append("Core Services:")
            core_services = summary.get("core_services") or []
            if core_services:
                lines.extend(f"- {service}" for service in core_services)
            else:
                lines.append("- (none)")
            lines.append("")

            lines.append("Leaf Services:")
            leaf_services = summary.get("leaf_services") or []
            if leaf_services:
                lines.extend(f"- {service}" for service in leaf_services)
            else:
                lines.append("- (none)")
            lines.append("")

            lines.append("Dependencies:")
            lines.append("")
            for service, dependencies in self.dependency_graph.items():
                lines.append(service)
                if dependencies:
                    lines.extend(f" -> {dependency}" for dependency in dependencies)
                else:
                    lines.append(" (none)")
                lines.append("")

            report_text = "\n".join(lines).rstrip() + "\n"

            with open(self.report_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(report_text)
            print(f"Architecture report saved to: {self.report_path}")

        except OSError as exc:
            raise RuntimeError(f"Failed to write '{self.report_path}': {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Report generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # ORCHESTRATION
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, object]:
        """
        Execute the full Architecture Generation phase end-to-end:
        load dependencies -> Mermaid -> PNG -> summary -> report.

        Returns:
            The computed architecture summary dictionary.

        Raises:
            RuntimeError: if any stage of the pipeline fails.
        """
        try:
            self.load_dependencies()

            mermaid_code = self.generate_mermaid()
            self.save_mermaid(mermaid_code)

            self.build_graph()
            self.generate_png()

            summary = self.generate_summary()
            self.generate_report(summary)

            return summary

        except Exception as exc:
            raise RuntimeError(f"Architecture Generation pipeline failed: {exc}") from exc


# =====================================================================
# STANDALONE ENTRY POINT
# =====================================================================
#
# Integration into your existing GTMicro+ script: after your Dependency
# Discovery phase calls `dependency_discovery.to_json(...)`, just add:
#
#     from architecture_generator import ArchitectureGenerator
#     ArchitectureGenerator().run()
#
# Or run this file directly against an existing dependencies.json:
#     python3 architecture_generator.py
#
if __name__ == "__main__":
    try:
        generator = ArchitectureGenerator()
        generator.run()
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
