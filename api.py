"""
GTMicro+ : API Suggestion Engine
------------------------------------------------------------------------
Input:
    services.json   (service_name -> list of use case names)

Outputs:
    api_contracts.json   Full REST API contracts per microservice
    api_report.txt       Human-readable API contract report

Pipeline position:
    services.json
        |
        v
    API Suggestion Engine   <-- THIS MODULE
        |
        v
    api_contracts.json + api_report.txt

Design notes:
    - Rule-based + NLP inference only. No OpenAI / Gemini / any external API.
    - Fully deterministic and explainable endpoint generation.
    - Follows REST conventions and SOLID principles.
    - Safe to run standalone or as part of the larger GTMicro+ pipeline.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


# =====================================================================
# CONSTANTS
# =====================================================================

SERVICES_JSON_PATH: str = "services.json"
API_CONTRACTS_JSON_PATH: str = "api_contracts.json"
API_REPORT_TXT_PATH: str = "api_report.txt"


# =====================================================================
# PREDEFINED USE-CASE → API RULES
# =====================================================================
# Each rule maps a normalized use-case phrase to a (method, endpoint,
# description) triple. Endpoints here are RELATIVE to the service base
# path; the final endpoint written to api_contracts.json is the
# absolute path (base_path + endpoint).
#
# Keys are lowercase, whitespace-normalized phrases for robust matching.
# =====================================================================

USE_CASE_RULES: Dict[str, Tuple[str, str, str]] = {
    # ---------------------------------------------------------------
    # Authentication related
    # ---------------------------------------------------------------
    "customer registration": ("POST", "/register", "Register a new customer"),
    "customer login":        ("POST", "/login",    "Authenticate customer"),
    "password reset":        ("POST", "/reset-password", "Reset customer password"),
    "user registration":     ("POST", "/register", "Register a new user"),
    "user login":            ("POST", "/login",    "Authenticate user"),
    "logout":                ("POST", "/logout",   "Log the user out"),

    # ---------------------------------------------------------------
    # Catalog related
    # ---------------------------------------------------------------
    "view menu":              ("GET",  "/menu",              "View the menu"),
    "menu creation":          ("POST", "/menu",              "Create a new menu item"),
    "menu update":            ("PUT",  "/menu/{id}",         "Update an existing menu item"),
    "food search":            ("GET",  "/foods/search",      "Search for foods"),
    "restaurant search":      ("GET",  "/restaurants/search","Search for restaurants"),
    "view restaurant details":("GET",  "/restaurants/{id}",  "View restaurant details"),

    # ---------------------------------------------------------------
    # Cart related
    # ---------------------------------------------------------------
    "add to cart":        ("POST",   "/cart",       "Add an item to the cart"),
    "update cart":        ("PUT",    "/cart/{id}",  "Update a cart item"),
    "remove from cart":   ("DELETE", "/cart/{id}",  "Remove an item from the cart"),
    "view cart":          ("GET",    "/cart",       "View the current cart"),

    # ---------------------------------------------------------------
    # Order related
    # ---------------------------------------------------------------
    "place order":         ("POST",   "/orders",         "Place a new order"),
    "order tracking":      ("GET",    "/orders/{id}",    "Track an order"),
    "order cancellation":  ("DELETE", "/orders/{id}",    "Cancel an order"),
    "order confirmation":  ("POST",   "/orders/confirm", "Confirm an order"),

    # ---------------------------------------------------------------
    # Payment related
    # ---------------------------------------------------------------
    "online payment":       ("POST", "/payments",         "Process an online payment"),
    "payment verification": ("POST", "/payments/verify",  "Verify a payment"),
    "refund processing":    ("POST", "/refunds",          "Process a refund"),

    # ---------------------------------------------------------------
    # Notification related
    # ---------------------------------------------------------------
    "email notifications": ("POST", "/notifications/email", "Send email notifications"),
    "sms notifications":   ("POST", "/notifications/sms",   "Send SMS notifications"),
    "push notifications":  ("POST", "/notifications/push",  "Send push notifications"),
}


# =====================================================================
# BASE PATH RULES
# =====================================================================
# Known service names → base path. Unknown names are converted
# automatically via slugification (see `infer_base_path`).
# =====================================================================

BASE_PATH_RULES: Dict[str, str] = {
    "auth service":         "/auth",
    "catalog service":      "/catalog",
    "order service":        "/orders",
    "payment service":      "/payments",
    "notification service": "/notifications",
    "inventory service":    "/inventory",
    "support service":      "/support",
    "cart service":         "/cart",
    "shipping service":     "/shipping",
    "review service":       "/reviews",
}


# =====================================================================
# NLP INFERENCE VOCABULARY
# =====================================================================
# Used when a use case is NOT found in USE_CASE_RULES.
#
# Verbs are checked first to pick an HTTP method:
#   - read-like verbs   → GET
#   - create-like verbs → POST
#   - update-like verbs → PUT
#   - delete-like verbs → DELETE
#
# Nouns are then slugified into a resource segment. Domain-specific
# noun mappings let "Generate Invoice" produce `/invoices` rather than
# `/invoice-generation`.
# =====================================================================

READ_VERBS = {
    "view", "get", "fetch", "list", "show", "display", "search",
    "find", "browse", "track", "check", "retrieve", "query",
}
CREATE_VERBS = {
    "create", "add", "register", "generate", "submit", "place",
    "send", "issue", "make", "start", "initiate", "open", "book",
    "schedule", "process", "confirm", "verify", "publish", "upload",
    "resolve", "assign", "apply", "accept", "approve", "reject",
}
UPDATE_VERBS = {
    "update", "edit", "modify", "change", "set", "adjust", "patch",
    "rename", "replace", "revise", "reset",
}
DELETE_VERBS = {
    "delete", "remove", "cancel", "destroy", "drop", "discard",
    "unregister", "revoke", "terminate", "close",
}

# Singular / plural noun normalization for common resources, so that
# "Generate Invoice" → /invoices (plural, REST convention).
NOUN_PLURAL_MAP: Dict[str, str] = {
    "invoice":       "invoices",
    "payment":       "payments",
    "refund":        "refunds",
    "order":         "orders",
    "customer":      "customers",
    "user":          "users",
    "product":       "products",
    "item":          "items",
    "menu":          "menu",
    "food":          "foods",
    "restaurant":    "restaurants",
    "review":        "reviews",
    "notification":  "notifications",
    "email":         "notifications/email",
    "sms":           "notifications/sms",
    "push":          "notifications/push",
    "ticket":        "tickets",
    "support":       "support",
    "report":        "reports",
    "category":      "categories",
    "address":       "addresses",
    "cart":          "cart",
    "shipment":      "shipments",
    "delivery":      "deliveries",
    "coupon":        "coupons",
    "discount":      "discounts",
    "inventory":     "inventory",
    "stock":         "stock",
    "session":       "sessions",
    "token":         "tokens",
}

# Stopwords removed before extracting the resource noun.
STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on",
    "with", "by", "via", "using", "new", "all", "any", "my", "our",
    "their", "this", "that", "these", "those", "from", "into", "at",
    "is", "are", "be", "been", "being", "do", "does", "did", "done",
}


# =====================================================================
# API SUGGESTION ENGINE
# =====================================================================

class APISuggestionEngine:
    """
    Analyzes services.json (service_name -> list of use case names) and
    produces REST API contracts for every microservice.

    The engine is fully rule-based + NLP-driven:
        1. Exact rule lookup in USE_CASE_RULES for known use cases.
        2. Fallback NLP inference (verb → HTTP method, noun → resource)
           for unknown use cases.
        3. Base-path resolution via BASE_PATH_RULES or auto-slugified
           from the service name.

    Outputs:
        - api_contracts.json
        - api_report.txt
    """

    # ------------------------------------------------------------------
    # CONSTRUCTION
    # ------------------------------------------------------------------

    def __init__(
        self,
        services_path: str = SERVICES_JSON_PATH,
        contracts_path: str = API_CONTRACTS_JSON_PATH,
        report_path: str = API_REPORT_TXT_PATH,
        use_case_rules: Optional[Dict[str, Tuple[str, str, str]]] = None,
        base_path_rules: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            services_path: path to the input services.json.
            contracts_path: output path for api_contracts.json.
            report_path: output path for api_report.txt.
            use_case_rules: override the default USE_CASE_RULES table.
            base_path_rules: override the default BASE_PATH_RULES table.
        """
        self.services_path: str = services_path
        self.contracts_path: str = contracts_path
        self.report_path: str = report_path

        self.use_case_rules: Dict[str, Tuple[str, str, str]] = (
            use_case_rules if use_case_rules is not None else USE_CASE_RULES
        )
        self.base_path_rules: Dict[str, str] = (
            base_path_rules if base_path_rules is not None else BASE_PATH_RULES
        )

        self.services: Dict[str, List[str]] = {}
        self.contracts: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load_services(self) -> Dict[str, List[str]]:
        """
        Load and validate services.json.

        Returns:
            A dict mapping service_name -> list of use case names.

        Raises:
            RuntimeError: if the file is missing, malformed, or empty.
        """
        try:
            if not os.path.exists(self.services_path):
                raise FileNotFoundError(
                    f"'{self.services_path}' not found. "
                    f"Run the candidate-microservice phase first."
                )

            with open(self.services_path, "r", encoding="utf-8") as file_handle:
                raw = json.load(file_handle)

            if not isinstance(raw, dict) or not raw:
                raise ValueError(
                    "services.json must be a non-empty JSON object mapping "
                    "service names to lists of use cases."
                )

            cleaned: Dict[str, List[str]] = {}
            for service_name, use_cases in raw.items():
                if not isinstance(service_name, str) or not service_name.strip():
                    continue
                if not isinstance(use_cases, list):
                    raise ValueError(
                        f"Use cases for '{service_name}' must be a list."
                    )

                # De-duplicate while preserving order; drop empties.
                seen: set = set()
                unique: List[str] = []
                for uc in use_cases:
                    if not isinstance(uc, str):
                        continue
                    stripped = uc.strip()
                    if not stripped or stripped in seen:
                        continue
                    seen.add(stripped)
                    unique.append(stripped)

                if unique:
                    cleaned[service_name.strip()] = unique

            if not cleaned:
                raise ValueError("services.json contained no valid services.")

            self.services = cleaned
            print(
                f"Loaded {len(cleaned)} service(s) from '{self.services_path}'."
            )
            return cleaned

        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read '{self.services_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load services: {exc}") from exc

    # ------------------------------------------------------------------
    # TEXT NORMALIZATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, collapse whitespace, strip punctuation noise."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert a phrase into a URL-safe hyphenated slug."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", text.lower())
        return re.sub(r"[\s-]+", "-", cleaned).strip("-")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Split normalized text into significant tokens."""
        return [
            tok for tok in APISuggestionEngine._normalize(text).split()
            if tok and tok not in STOPWORDS
        ]

    # ------------------------------------------------------------------
    # INFERENCE: HTTP METHOD
    # ------------------------------------------------------------------

    def infer_http_method(self, use_case: str) -> str:
        """
        Determine the HTTP method for a use case.

        First checks the predefined rules table. If no match, falls back
        to NLP verb analysis:
            - read verbs    → GET
            - create verbs  → POST
            - update verbs  → PUT
            - delete verbs  → DELETE
            - unknown       → POST (safe default for a business action)

        Args:
            use_case: the raw use case string.

        Returns:
            One of "GET", "POST", "PUT", "DELETE".
        """
        if not use_case or not use_case.strip():
            return "POST"

        normalized = self._normalize(use_case)
        if normalized in self.use_case_rules:
            return self.use_case_rules[normalized][0]

        tokens = self._tokenize(use_case)
        if not tokens:
            return "POST"

        # The leading verb carries the strongest signal; fall back to
        # scanning the whole token set if the first token isn't a verb.
        first = tokens[0]
        if first in DELETE_VERBS:
            return "DELETE"
        if first in UPDATE_VERBS:
            return "PUT"
        if first in READ_VERBS:
            return "GET"
        if first in CREATE_VERBS:
            return "POST"

        token_set = set(tokens)
        if token_set & DELETE_VERBS:
            return "DELETE"
        if token_set & UPDATE_VERBS:
            return "PUT"
        if token_set & READ_VERBS:
            return "GET"

        return "POST"

    # ------------------------------------------------------------------
    # INFERENCE: ENDPOINT
    # ------------------------------------------------------------------

    def infer_endpoint(self, use_case: str, method: Optional[str] = None) -> str:
        """
        Determine the relative endpoint (without base path) for a use
        case, e.g. "/register", "/menu/{id}", "/invoices".

        Predefined rules take precedence. Otherwise the endpoint is
        inferred from the leading verb + resource noun:
            "Generate Invoice"   → "/invoices"
            "Support Resolution" → "/support/resolve"

        Args:
            use_case: the raw use case string.
            method: optional pre-computed HTTP method, used to decide
                whether an "{id}" segment or "/search" suffix should be
                appended.

        Returns:
            A relative endpoint string beginning with "/".
        """
        if not use_case or not use_case.strip():
            return "/"

        normalized = self._normalize(use_case)
        if normalized in self.use_case_rules:
            return self.use_case_rules[normalized][1]

        if method is None:
            method = self.infer_http_method(use_case)

        tokens = self._tokenize(use_case)
        if not tokens:
            return "/"

        # Identify resource noun: prefer the last noun-like token that
        # has a plural mapping, otherwise fall back to the last token.
        verb = tokens[0]
        noun_candidates = tokens[1:] if len(tokens) > 1 else tokens

        resource: Optional[str] = None
        for token in reversed(noun_candidates):
            if token in NOUN_PLURAL_MAP:
                resource = NOUN_PLURAL_MAP[token]
                break
            if token.endswith("s") and len(token) > 2:
                resource = token
                break

        if resource is None:
            # Use the last token as the resource, slugified.
            resource = self._slugify(noun_candidates[-1]) if noun_candidates else ""
            resource = NOUN_PLURAL_MAP.get(resource, resource)

        if not resource:
            return "/"

        # Handle special read-verb suffixes.
        if method == "GET":
            if "search" in tokens:
                return f"/{resource}/search"
            return f"/{resource}"

        if method in {"PUT", "DELETE"}:
            # Mutating an existing resource → include an id placeholder.
            return f"/{resource}/{{id}}"

        # POST: if the verb is a distinct action verb (e.g. "resolve"),
        # append it as a sub-path on the resource, mirroring the
        # "Support Resolution → /support/resolve" rule.
        if verb in CREATE_VERBS and verb not in {
            "create", "add", "register", "generate", "place", "submit",
            "send", "issue", "make", "start", "initiate", "open", "book",
            "schedule", "process", "publish", "upload",
        }:
            return f"/{resource}/{verb}"

        return f"/{resource}"

    # ------------------------------------------------------------------
    # INFERENCE: BASE PATH
    # ------------------------------------------------------------------

    def infer_base_path(self, service_name: str) -> str:
        """
        Determine the base path for a service.

        Checks BASE_PATH_RULES first (case-insensitive). For unknown
        service names, slugifies the meaningful part:
            "Customer Service" → "/customer"
            "Loyalty Service"  → "/loyalty"
            "Analytics"        → "/analytics"

        Args:
            service_name: the microservice name.

        Returns:
            A base path beginning with "/".
        """
        if not service_name or not service_name.strip():
            return "/"

        normalized = self._normalize(service_name)

        if normalized in self.base_path_rules:
            return self.base_path_rules[normalized]

        # Auto-slugify: strip the trailing "service" word if present.
        tokens = normalized.split()
        if tokens and tokens[-1] == "service":
            tokens = tokens[:-1]

        # Singularize the first remaining token when possible.
        if tokens:
            first = tokens[0]
            if first.endswith("s") and len(first) > 2:
                first = first[:-1]
            return f"/{self._slugify(first)}"

        return f"/{self._slugify(service_name)}"

    # ------------------------------------------------------------------
    # DESCRIPTION
    # ------------------------------------------------------------------

    def _infer_description(self, use_case: str, method: str) -> str:
        """
        Produce a human-readable description for a use case.

        Predefined rules supply curated descriptions. For inferred use
        cases, the description is generated from the use case itself.
        """
        normalized = self._normalize(use_case)
        if normalized in self.use_case_rules:
            return self.use_case_rules[normalized][2]

        verb_map = {
            "GET":    "Retrieve",
            "POST":   "Create",
            "PUT":    "Update",
            "DELETE": "Delete",
        }
        prefix = verb_map.get(method, "Handle")
        return f"{prefix} {use_case.strip().lower()}"

    # ------------------------------------------------------------------
    # CONTRACT GENERATION
    # ------------------------------------------------------------------

    def generate_api_contracts(self) -> Dict[str, Dict[str, object]]:
        """
        Generate REST API contracts for every loaded service.

        Returns:
            A dict of the form:
                {
                    "<Service Name>": {
                        "base_path": "/...",
                        "apis": [
                            {"method": "...", "endpoint": "...",
                             "description": "..."},
                            ...
                        ]
                    },
                    ...
                }

        Raises:
            RuntimeError: if load_services() hasn't been called or the
                service map is empty.
        """
        if not self.services:
            raise RuntimeError(
                "No services loaded. Call load_services() first."
            )

        try:
            contracts: Dict[str, Dict[str, object]] = {}

            for service_name, use_cases in self.services.items():
                base_path = self.infer_base_path(service_name)
                apis: List[Dict[str, str]] = []
                seen_keys: set = set()

                for use_case in use_cases:
                    method = self.infer_http_method(use_case)
                    relative = self.infer_endpoint(use_case, method)

                    # Final endpoint = base_path + relative, with the
                    # base path stripped from the relative part if the
                    # rule already embedded it.
                    if relative.startswith(base_path + "/"):
                        absolute = relative
                    elif relative == base_path:
                        absolute = base_path
                    else:
                        absolute = f"{base_path}{relative}" if relative != "/" else base_path

                    # Normalize duplicate slashes.
                    absolute = re.sub(r"/{2,}", "/", absolute)

                    description = self._infer_description(use_case, method)

                    dedup_key = (method, absolute)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)

                    apis.append({
                        "method": method,
                        "endpoint": absolute,
                        "description": description,
                    })

                contracts[service_name] = {
                    "base_path": base_path,
                    "apis": apis,
                }

            self.contracts = contracts
            return contracts

        except Exception as exc:
            raise RuntimeError(f"API contract generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # EXPORT: JSON
    # ------------------------------------------------------------------

    def export_json(self, path: Optional[str] = None) -> None:
        """
        Write the generated contracts to api_contracts.json.

        Args:
            path: optional override for the output path.

        Raises:
            RuntimeError: if no contracts have been generated or the
                file cannot be written.
        """
        if not self.contracts:
            raise RuntimeError(
                "No contracts to export. Call generate_api_contracts() first."
            )

        output_path = path or self.contracts_path

        try:
            with open(output_path, "w", encoding="utf-8") as file_handle:
                json.dump(self.contracts, file_handle, indent=4, ensure_ascii=False)
            print(f"API contracts exported to: {output_path}")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write '{output_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # EXPORT: REPORT
    # ------------------------------------------------------------------

    def generate_report(self, path: Optional[str] = None) -> None:
        """
        Write a human-readable API contract report to api_report.txt.

        Args:
            path: optional override for the output path.

        Raises:
            RuntimeError: if no contracts exist or the file can't be written.
        """
        if not self.contracts:
            raise RuntimeError(
                "No contracts to report. Call generate_api_contracts() first."
            )

        output_path = path or self.report_path

        try:
            lines: List[str] = []
            divider = "=" * 36

            lines.append(divider)
            lines.append("API CONTRACT REPORT")
            lines.append(divider)
            lines.append("")

            for service_name, payload in self.contracts.items():
                base_path = payload.get("base_path", "/")
                apis = payload.get("apis", [])

                lines.append(service_name)
                lines.append(f"Base Path: {base_path}")
                lines.append("")

                if not apis:
                    lines.append("  (no APIs generated)")
                    lines.append("")
                    continue

                for api in apis:
                    method = api.get("method", "POST")
                    endpoint = api.get("endpoint", "/")
                    description = api.get("description", "")
                    lines.append(f"{method:<6} {endpoint}")
                    if description:
                        lines.append(f"       {description}")
                lines.append("")

            lines.append(divider)
            lines.append("")

            report_text = "\n".join(lines)

            with open(output_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(report_text)
            print(f"API report saved to: {output_path}")

        except OSError as exc:
            raise RuntimeError(
                f"Failed to write '{output_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # ORCHESTRATION
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Dict[str, object]]:
        """
        Execute the full API Suggestion pipeline end-to-end:
            load_services -> generate_api_contracts
            -> export_json -> generate_report

        Returns:
            The generated contracts dictionary.

        Raises:
            RuntimeError: if any stage fails.
        """
        try:
            self.load_services()
            contracts = self.generate_api_contracts()
            self.export_json()
            self.generate_report()
            return contracts
        except Exception as exc:
            raise RuntimeError(
                f"API Suggestion Engine pipeline failed: {exc}"
            ) from exc


# =====================================================================
# STANDALONE ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    try:
        engine = APISuggestionEngine()
        engine.run()
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")