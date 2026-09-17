"""
GTMicro+ : Spring Boot Skeleton Generator
------------------------------------------------------------------------
Input:
    services.json         (service_name -> list of use case names)
    api_contracts.json    (service_name -> {base_path, apis[]})
    dependencies.json     (service_name -> list of dependent services)

Output:
    generated_microservices/
        <service-folder>/
            pom.xml
            src/main/java/com/gtmicro/<pkg>/
                <Pkg>Application.java
                controller/<Pkg>Controller.java
                service/<Pkg>Service.java
                repository/<Pkg>Repository.java
                entity/<Entity>.java
                dto/<UseCase>Request.java
                config/OpenApiConfig.java
            src/main/resources/application.properties
    generation_report.txt

Pipeline position:
    api_contracts.json
            |
            v
    Spring Boot Skeleton Generator   <-- THIS MODULE
            |
            v
    generated_microservices/  (ready-to-develop Spring Boot projects)

Design notes:
    - Pure static code generation. No OpenAI / Gemini / external API.
    - Rule-based mapping from service names, use cases, and endpoints
      to Java package / class / method names.
    - Deterministic, idempotent, and safe to re-run (overwrites output).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple


# =====================================================================
# CONSTANTS
# =====================================================================

SERVICES_JSON_PATH: str = "services.json"
API_CONTRACTS_JSON_PATH: str = "api_contracts.json"
DEPENDENCIES_JSON_PATH: str = "dependencies.json"

OUTPUT_DIR: str = "generated_microservices"
REPORT_PATH: str = "generation_report.txt"

GROUP_ID: str = "com.gtmicro"
JAVA_VERSION: str = "21"
SPRING_BOOT_VERSION: str = "3.2.5"

# Port assignment: known service keywords → base port.
# Services are matched in order; first hit wins. Others get auto ports.
BASE_PORT: int = 8081
KNOWN_PORTS: Dict[str, int] = {
    "auth":         8081,
    "catalog":      8082,
    "order":        8083,
    "payment":      8084,
    "notification": 8085,
    "inventory":    8086,
    "support":      8087,
    "cart":         8088,
    "shipping":     8089,
    "review":       8090,
}

# Words stripped when deriving the Java package / class prefix from a
# service name. E.g. "Auth Service" → "auth" → "Auth".
SERVICE_NAME_NOISE: set = {"service", "microservice", "ms", "api"}

# Default entity name per known domain. Used to generate a placeholder
# JPA entity for each service.
DEFAULT_ENTITY_BY_DOMAIN: Dict[str, str] = {
    "auth":         "User",
    "catalog":      "CatalogItem",
    "order":        "Order",
    "payment":      "Payment",
    "notification": "Notification",
    "inventory":    "InventoryItem",
    "support":      "SupportTicket",
    "cart":         "CartItem",
    "shipping":     "Shipment",
    "review":       "Review",
}

# Words stripped from use cases when generating DTO class names so that
# "Customer Registration" → "Registration" → "RegistrationRequest".
USE_CASE_NOISE: set = {"customer", "user", "the", "a", "an", "of", "for"}


# =====================================================================
# SPRING BOOT SKELETON GENERATOR
# =====================================================================

class SpringBootSkeletonGenerator:
    """
    Generates production-ready Spring Boot microservice skeletons from
    the GTMicro+ pipeline artifacts (services, api contracts, dependencies).

    For each service the generator produces a Maven project with:
        - pom.xml (Java 21, Spring Boot 3.x, JPA, Validation, Lombok,
          PostgreSQL, DevTools, springdoc-openapi)
        - <Pkg>Application.java
        - controller/<Pkg>Controller.java (one @PostMapping/@GetMapping/
          @PutMapping/@DeleteMapping per API contract entry)
        - service/<Pkg>Service.java (with @Service and dependency stubs)
        - repository/<Pkg>Repository.java (JpaRepository interface)
        - entity/<Entity>.java (placeholder JPA entity)
        - dto/<UseCase>Request.java (one per API action)
        - config/OpenApiConfig.java
        - application.properties (unique port per service)

    Also writes a top-level `generation_report.txt` summarizing what
    was generated.
    """

    # ------------------------------------------------------------------
    # CONSTRUCTION
    # ------------------------------------------------------------------

    def __init__(
        self,
        services_path: str = SERVICES_JSON_PATH,
        contracts_path: str = API_CONTRACTS_JSON_PATH,
        dependencies_path: str = DEPENDENCIES_JSON_PATH,
        output_dir: str = OUTPUT_DIR,
        report_path: str = REPORT_PATH,
    ) -> None:
        """
        Args:
            services_path: path to services.json.
            contracts_path: path to api_contracts.json.
            dependencies_path: path to dependencies.json.
            output_dir: root folder for generated projects.
            report_path: path for the generation report.
        """
        self.services_path: str = services_path
        self.contracts_path: str = contracts_path
        self.dependencies_path: str = dependencies_path
        self.output_dir: str = output_dir
        self.report_path: str = report_path

        self.services: Dict[str, List[str]] = {}
        self.contracts: Dict[str, Dict[str, object]] = {}
        self.dependencies: Dict[str, List[str]] = {}

        # Filled during run(); used to build the report.
        self._stats: Dict[str, int] = {
            "services": 0,
            "controllers": 0,
            "services_layer": 0,
            "repositories": 0,
            "entities": 0,
            "dtos": 0,
            "dependencies": 0,
        }

    # ------------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------------

    def load_services(self) -> Dict[str, List[str]]:
        """
        Load services.json.

        Returns:
            service_name -> list of use cases.

        Raises:
            RuntimeError: if the file is missing, malformed, or empty.
        """
        try:
            raw = self._read_json(self.services_path)
            if not isinstance(raw, dict) or not raw:
                raise ValueError(
                    "services.json must be a non-empty JSON object."
                )

            cleaned: Dict[str, List[str]] = {}
            for name, use_cases in raw.items():
                if not isinstance(name, str) or not name.strip():
                    continue
                if not isinstance(use_cases, list):
                    raise ValueError(
                        f"Use cases for '{name}' must be a list."
                    )
                unique = list(dict.fromkeys(
                    uc.strip() for uc in use_cases
                    if isinstance(uc, str) and uc.strip()
                ))
                if unique:
                    cleaned[name.strip()] = unique

            if not cleaned:
                raise ValueError("services.json contained no valid services.")

            self.services = cleaned
            print(f"Loaded {len(cleaned)} service(s) from '{self.services_path}'.")
            return cleaned

        except Exception as exc:
            raise RuntimeError(f"Failed to load services: {exc}") from exc

    def load_api_contracts(self) -> Dict[str, Dict[str, object]]:
        """
        Load api_contracts.json.

        Returns:
            service_name -> {base_path, apis: [{method, endpoint, description}]}

        Raises:
            RuntimeError: if the file is missing or malformed.
        """
        try:
            raw = self._read_json(self.contracts_path)
            if not isinstance(raw, dict) or not raw:
                raise ValueError(
                    "api_contracts.json must be a non-empty JSON object."
                )

            cleaned: Dict[str, Dict[str, object]] = {}
            for name, payload in raw.items():
                if not isinstance(name, str) or not name.strip():
                    continue
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Contract for '{name}' must be a JSON object."
                    )

                base_path = payload.get("base_path", "/")
                if not isinstance(base_path, str):
                    base_path = "/"

                apis = payload.get("apis", [])
                if not isinstance(apis, list):
                    apis = []

                clean_apis: List[Dict[str, str]] = []
                for api in apis:
                    if not isinstance(api, dict):
                        continue
                    method = str(api.get("method", "POST")).upper()
                    endpoint = str(api.get("endpoint", "/"))
                    description = str(api.get("description", ""))
                    clean_apis.append({
                        "method": method,
                        "endpoint": endpoint,
                        "description": description,
                    })

                cleaned[name.strip()] = {
                    "base_path": base_path,
                    "apis": clean_apis,
                }

            if not cleaned:
                raise ValueError("api_contracts.json contained no valid contracts.")

            self.contracts = cleaned
            print(
                f"Loaded {len(cleaned)} contract(s) from '{self.contracts_path}'."
            )
            return cleaned

        except Exception as exc:
            raise RuntimeError(f"Failed to load API contracts: {exc}") from exc

    def load_dependencies(self) -> Dict[str, List[str]]:
        """
        Load dependencies.json. Missing file is tolerated (empty graph).

        Returns:
            service_name -> list of dependent service names.

        Raises:
            RuntimeError: if the file exists but is malformed.
        """
        try:
            if not os.path.exists(self.dependencies_path):
                print(
                    f"'{self.dependencies_path}' not found; "
                    f"proceeding without dependency awareness."
                )
                self.dependencies = {}
                return {}

            raw = self._read_json(self.dependencies_path)
            if not isinstance(raw, dict):
                raise ValueError(
                    "dependencies.json must be a JSON object."
                )

            cleaned: Dict[str, List[str]] = {}
            for name, deps in raw.items():
                if not isinstance(name, str) or not name.strip():
                    continue
                if not isinstance(deps, list):
                    raise ValueError(
                        f"Dependencies for '{name}' must be a list."
                    )
                unique = list(dict.fromkeys(
                    d.strip() for d in deps
                    if isinstance(d, str) and d.strip()
                ))
                cleaned[name.strip()] = unique

            self.dependencies = cleaned
            total_edges = sum(len(v) for v in cleaned.values())
            print(
                f"Loaded {len(cleaned)} dependency entr(ies) "
                f"({total_edges} edge(s)) from '{self.dependencies_path}'."
            )
            return cleaned

        except Exception as exc:
            raise RuntimeError(f"Failed to load dependencies: {exc}") from exc

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: str) -> object:
        """Read and parse a JSON file, raising a clear error on failure."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' not found.")
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @staticmethod
    def _to_pascal_case(text: str) -> str:
        """Convert any phrase to PascalCase (e.g. 'auth service' → 'AuthService')."""
        parts = re.split(r"[^a-zA-Z0-9]+", text)
        return "".join(p.capitalize() for p in parts if p)

    @staticmethod
    def _to_camel_case(text: str) -> str:
        """Convert any phrase to camelCase (e.g. 'auth service' → 'authService')."""
        pascal = SpringBootSkeletonGenerator._to_pascal_case(text)
        return pascal[:1].lower() + pascal[1:] if pascal else ""

    @staticmethod
    def _to_kebab_case(text: str) -> str:
        """Convert any phrase to kebab-case (e.g. 'Auth Service' → 'auth-service')."""
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
        return re.sub(r"-{2,}", "-", cleaned).strip("-")

    @staticmethod
    def _to_snake_case(text: str) -> str:
        """Convert any phrase to snake_case (used for package-safe names)."""
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
        return re.sub(r"_{2,}", "_", cleaned).strip("_")

    def _service_short_name(self, service_name: str) -> str:
        """
        Extract the meaningful part of a service name.

        'Auth Service'         → 'auth'
        'Notification Service' → 'notification'
        'Loyalty Program MS'   → 'loyalty-program'
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", service_name.lower())
        tokens = [t for t in tokens if t and t not in SERVICE_NAME_NOISE]
        return "-".join(tokens) if tokens else self._to_kebab_case(service_name)

    def _package_name(self, service_name: str) -> str:
        """Java sub-package for a service, e.g. 'Auth Service' → 'auth'."""
        short = self._service_short_name(service_name)
        return re.sub(r"[^a-z0-9]", "", short) or "service"

    def _domain_key(self, service_name: str) -> str:
        """Best-effort domain keyword for port/entity lookup."""
        short = self._service_short_name(service_name)
        first = short.split("-")[0]
        return first

    def _assign_port(self, service_name: str) -> int:
        """
        Determine the HTTP port for a service.

        Known domains use the KNOWN_PORTS map; unknown services get a
        deterministic port based on sorted position (starting at 8091).
        """
        domain = self._domain_key(service_name)
        if domain in KNOWN_PORTS:
            return KNOWN_PORTS[domain]

        # Deterministic auto-assignment: stable by alphabetical order of
        # all known service names.
        unknowns = sorted(
            name for name in self.services
            if self._domain_key(name) not in KNOWN_PORTS
        )
        next_port = max(KNOWN_PORTS.values()) + 1
        try:
            offset = unknowns.index(service_name)
        except ValueError:
            offset = 0
        return next_port + offset

    def _endpoint_to_method_name(
        self, endpoint: str, method: str, description: str
    ) -> str:
        """
        Derive a Java method name from an endpoint.

        Examples:
            POST   /register        → register
            GET    /menu            → getMenu
            GET    /menu/{id}       → getMenuById
            PUT    /menu/{id}       → updateMenu
            DELETE /menu/{id}       → deleteMenu
            GET    /foods/search    → searchFoods
        """
        # Strip the base path if present.
        path = endpoint
        if path.startswith("/"):
            path = path[1:]

        # Split into segments, ignoring path variables.
        segments = [
            seg for seg in path.split("/")
            if seg and not (seg.startswith("{") and seg.endswith("}"))
        ]
        has_id = "{" in endpoint and "}" in endpoint

        # Prefer description-based verb if endpoint has no clear verb.
        first = segments[0] if segments else ""

        verb_prefix = {
            "GET":    "get",
            "POST":   "create",
            "PUT":    "update",
            "PATCH":  "patch",
            "DELETE": "delete",
        }.get(method, "handle")

        # Read endpoints named "/search" or containing "search" should
        # use "search" as the verb.
        if method == "GET" and "search" in segments:
            base = "Search" + "".join(
                p.capitalize() for p in segments if p != "search"
            )
            return "search" + base[len("Search"):] if base != "Search" else "search"

        if first:
            camel = self._to_camel_case(first)
            method_name = camel
        else:
            method_name = "root"

        # Prefix with verb when the endpoint doesn't already imply it.
        # A leading resource (e.g. /menu) still gets a verb prefix.
        if method == "GET":
            method_name = "get" + method_name[:1].upper() + method_name[1:]
        elif method == "POST":
            # Keep user-specified verbs like "register", "login".
            if not method_name.lower().startswith(
                ("create", "register", "login", "logout", "place", "confirm",
                 "verify", "reset", "process", "send", "publish", "submit",
                 "search", "resolve", "generate", "add", "remove", "update",
                 "delete", "cancel", "track")
            ):
                method_name = "create" + method_name[:1].upper() + method_name[1:]
        elif method == "PUT":
            method_name = "update" + method_name[:1].upper() + method_name[1:]
        elif method == "PATCH":
            method_name = "patch" + method_name[:1].upper() + method_name[1:]
        elif method == "DELETE":
            method_name = "delete" + method_name[:1].upper() + method_name[1:]

        if has_id:
            method_name += "ById"

        return method_name

    def _dto_class_name(self, use_case: str) -> str:
        """
        Derive a DTO class name from a use case.

        'Customer Registration' → 'RegistrationRequest'
        'Place Order'           → 'PlaceOrderRequest'
        'Generate Invoice'      → 'GenerateInvoiceRequest'
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", use_case.strip())
        tokens = [t for t in tokens if t and t.lower() not in USE_CASE_NOISE]
        pascal = "".join(t.capitalize() for t in tokens) or "Action"
        return f"{pascal}Request"

    def _entity_class_name(self, service_name: str) -> str:
        """Pick the placeholder entity class name for a service."""
        domain = self._domain_key(service_name)
        if domain in DEFAULT_ENTITY_BY_DOMAIN:
            return DEFAULT_ENTITY_BY_DOMAIN[domain]
        return self._to_pascal_case(service_name.replace("Service", "").strip()) or "Entity"

    # ------------------------------------------------------------------
    # FILE WRITERS
    # ------------------------------------------------------------------
        @staticmethod
    def _write_file(self, path: str, content: str) -> None:
        try:

            directory = os.path.dirname(path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write(content)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to write '{path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # PROJECT STRUCTURE
    # ------------------------------------------------------------------

    def create_project_structure(self, service_name: str) -> Tuple[str, str, str]:
        """
        Create the folder skeleton for a service.

        Returns:
            (project_root, java_package_root, package_name)
        """
        folder = self._to_kebab_case(service_name)
        package = self._package_name(service_name)
        project_root = os.path.join(self.output_dir, folder)
        java_root = os.path.join(
            project_root,
            "src", "main", "java",
            *GROUP_ID.split("."),
            package,
        )

        for sub in (
            "", "controller", "service", "repository",
            "entity", "dto", "config",
        ):
            os.makedirs(os.path.join(java_root, sub), exist_ok=True)

        os.makedirs(
            os.path.join(project_root, "src", "main", "resources"),
            exist_ok=True,
        )

        return project_root, java_root, package

    # ------------------------------------------------------------------
    # POM.XML
    # ------------------------------------------------------------------

    def generate_pom(self, service_name: str, project_root: str) -> str:
        """
        Generate pom.xml for a service.

        Returns:
            The artifactId used in the pom.
        """
        artifact_id = self._to_kebab_case(service_name)
        class_name = self._to_pascal_case(service_name)

        pom = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
                             https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>{SPRING_BOOT_VERSION}</version>
        <relativePath/>
    </parent>

    <groupId>{GROUP_ID}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>{class_name}</name>
    <description>GTMicro+ generated Spring Boot microservice: {service_name}</description>

    <properties>
        <java.version>{JAVA_VERSION}</java.version>
        <springdoc.version>2.5.0</springdoc.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-validation</artifactId>
        </dependency>

        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>

        <dependency>
            <groupId>org.postgresql</groupId>
            <artifactId>postgresql</artifactId>
            <scope>runtime</scope>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>

        <dependency>
            <groupId>org.springdoc</groupId>
            <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
            <version>${{springdoc.version}}</version>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <configuration>
                    <excludes>
                        <exclude>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                        </exclude>
                    </excludes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
"""
        self._write_file(os.path.join(project_root, "pom.xml"), pom)
        return artifact_id

    # ------------------------------------------------------------------
    # APPLICATION CLASS
    # ------------------------------------------------------------------

    def generate_application_class(
        self, service_name: str, java_root: str, package: str
    ) -> str:
        """Generate <Pkg>Application.java. Returns the class name."""
        class_name = self._to_pascal_case(service_name) + "Application"
        content = f"""package {GROUP_ID}.{package};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class {class_name} {{

    public static void main(String[] args) {{
        SpringApplication.run({class_name}.class, args);
    }}
}}
"""
        self._write_file(
            os.path.join(java_root, f"{class_name}.java"), content
        )
        return class_name

    # ------------------------------------------------------------------
    # CONTROLLER
    # ------------------------------------------------------------------

    def generate_controller(
        self,
        service_name: str,
        java_root: str,
        package: str,
        contract: Dict[str, object],
        dependencies: List[str],
    ) -> Tuple[str, int]:
        """
        Generate the REST controller for a service based on its API contract.

        Returns:
            (controller_class_name, method_count)
        """
        class_prefix = self._to_pascal_case(service_name)
        controller_name = f"{class_prefix}Controller"
        base_path = contract.get("base_path", "/") or "/"
        apis: List[Dict[str, str]] = contract.get("apis", [])  # type: ignore

        # Build dependency-awareness comment.
        dep_lines: List[str] = []
        if dependencies:
            dep_lines.append("// ---------------------------------------------------------------")
            for dep in dependencies:
                dep_lines.append(f"// Depends on {dep}")
            dep_lines.append("// ---------------------------------------------------------------")

        imports = [
            "import org.springframework.http.ResponseEntity;",
            "import org.springframework.web.bind.annotation.*;",
            "import io.swagger.v3.oas.annotations.Operation;",
            "import io.swagger.v3.oas.annotations.tags.Tag;",
        ]

        method_blocks: List[str] = []
        seen_methods: set = set()

        for api in apis:
            method = api.get("method", "POST").upper()
            endpoint = api.get("endpoint", "/")
            description = api.get("description", "")

            # Convert the absolute endpoint to a relative one under base_path.
            relative = endpoint
            if base_path and endpoint.startswith(base_path):
                relative = endpoint[len(base_path):] or "/"
            if not relative.startswith("/"):
                relative = "/" + relative

            method_name = self._endpoint_to_method_name(endpoint, method, description)
            # Ensure uniqueness within the controller.
            base_method_name = method_name
            counter = 2
            while method_name in seen_methods:
                method_name = f"{base_method_name}{counter}"
                counter += 1
            seen_methods.add(method_name)

            annotation = {
                "GET":    "GetMapping",
                "POST":   "PostMapping",
                "PUT":    "PutMapping",
                "PATCH":  "PatchMapping",
                "DELETE": "DeleteMapping",
            }.get(method, "PostMapping")

            method_blocks.append(
                f"""    @Operation(summary = "{description or method_name}")
    @{annotation}("{relative}")
    public ResponseEntity<String> {method_name}() {{
        return ResponseEntity.ok("{method_name}");
    }}"""
            )

        methods_joined = "\n\n".join(method_blocks) if method_blocks else (
            "    // No APIs defined in api_contracts.json for this service."
        )

        dep_block = "\n".join(dep_lines)
        dep_block = f"{dep_block}\n" if dep_block else ""

        content = f"""package {GROUP_ID}.{package}.controller;

{chr(10).join(imports)}

{dep_block}@Tag(name = "{class_prefix}", description = "REST API for {service_name}")
@RestController
@RequestMapping("{base_path}")
public class {controller_name} {{

{methods_joined}
}}
"""
        self._write_file(
            os.path.join(java_root, "controller", f"{controller_name}.java"),
            content,
        )
        return controller_name, len(method_blocks)

    # ------------------------------------------------------------------
    # SERVICE LAYER
    # ------------------------------------------------------------------

    def generate_service(
        self,
        service_name: str,
        java_root: str,
        package: str,
        dependencies: List[str],
    ) -> str:
        """Generate the @Service layer class. Returns the class name."""
        class_prefix = self._to_pascal_case(service_name)
        service_class = f"{class_prefix}Service"

        dep_lines: List[str] = []
        if dependencies:
            dep_lines.append("    // -----------------------------------------------------------")
            for dep in dependencies:
                dep_lines.append(f"    // Depends on {dep}")
            dep_lines.append("    // -----------------------------------------------------------")

        dep_block = "\n".join(dep_lines)

        content = f"""package {GROUP_ID}.{package}.service;

import org.springframework.stereotype.Service;

@Service
public class {service_class} {{

{dep_block}
}}
"""
        self._write_file(
            os.path.join(java_root, "service", f"{service_class}.java"),
            content,
        )
        return service_class

    # ------------------------------------------------------------------
    # REPOSITORY
    # ------------------------------------------------------------------

    def generate_repository(
        self,
        service_name: str,
        java_root: str,
        package: str,
        entity_name: str,
    ) -> str:
        """Generate the JPA repository interface. Returns the class name."""
        class_prefix = self._to_pascal_case(service_name)
        repository_name = f"{class_prefix}Repository"

        content = f"""package {GROUP_ID}.{package}.repository;

import {GROUP_ID}.{package}.entity.{entity_name};
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface {repository_name} extends JpaRepository<{entity_name}, Long> {{
}}
"""
        self._write_file(
            os.path.join(java_root, "repository", f"{repository_name}.java"),
            content,
        )
        return repository_name

    # ------------------------------------------------------------------
    # ENTITY
    # ------------------------------------------------------------------

    def generate_entity(
        self,
        java_root: str,
        package: str,
        entity_name: str,
    ) -> str:
        """Generate a placeholder JPA entity. Returns the class name."""
        table_name = self._to_snake_case(entity_name) + "s"

        content = f"""package {GROUP_ID}.{package}.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "{table_name}")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class {entity_name} {{

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // TODO: add domain-specific fields for {entity_name}.
}}
"""
        self._write_file(
            os.path.join(java_root, "entity", f"{entity_name}.java"),
            content,
        )
        return entity_name

    # ------------------------------------------------------------------
    # DTOs
    # ------------------------------------------------------------------

    def generate_dto(
        self,
        java_root: str,
        package: str,
        dto_name: str,
    ) -> str:
        """Generate a placeholder DTO record. Returns the DTO class name."""
        # Strip trailing "Request" to derive a payload-free record name.
        content = f"""package {GROUP_ID}.{package}.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class {dto_name} {{

    // TODO: add request fields for {dto_name}.
}}
"""
        self._write_file(
            os.path.join(java_root, "dto", f"{dto_name}.java"),
            content,
        )
        return dto_name

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def generate_openapi_config(
        self,
        service_name: str,
        java_root: str,
        package: str,
    ) -> str:
        """Generate an OpenAPI configuration class. Returns class name."""
        class_prefix = self._to_pascal_case(service_name)
        config_name = f"{class_prefix}OpenApiConfig"

        content = f"""package {GROUP_ID}.{package}.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class {config_name} {{

    @Bean
    public OpenAPI {self._to_camel_case(service_name)}OpenAPI() {{
        return new OpenAPI()
                .info(new Info()
                        .title("{service_name} API")
                        .description("GTMicro+ generated API for {service_name}")
                        .version("1.0.0"));
    }}
}}
"""
        self._write_file(
            os.path.join(java_root, "config", f"{config_name}.java"),
            content,
        )
        return config_name

    # ------------------------------------------------------------------
    # APPLICATION.PROPERTIES
    # ------------------------------------------------------------------

    def generate_application_properties(
        self,
        service_name: str,
        project_root: str,
    ) -> int:
        """Generate application.properties. Returns the assigned port."""
        artifact_id = self._to_kebab_case(service_name)
        port = self._assign_port(service_name)

        properties = f"""spring.application.name={artifact_id}
server.port={port}

# ---------------------------------------------------------------
# Datasource (PostgreSQL) — adjust for your environment
# ---------------------------------------------------------------
spring.datasource.url=jdbc:postgresql://localhost:5432/{artifact_id.replace('-', '_')}_db
spring.datasource.username=postgres
spring.datasource.password=postgres
spring.datasource.driver-class-name=org.postgresql.Driver

spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect
spring.jpa.properties.hibernate.format_sql=true

# ---------------------------------------------------------------
# springdoc-openapi
# ---------------------------------------------------------------
springdoc.api-docs.path=/v3/api-docs
springdoc.swagger-ui.path=/swagger-ui.html
"""
        self._write_file(
            os.path.join(
                project_root, "src", "main", "resources", "application.properties"
            ),
            properties,
        )
        return port

    # ------------------------------------------------------------------
    # ORCHESTRATION
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, int]:
        """
        Execute the full Spring Boot skeleton generation pipeline.

        Returns:
            A dictionary of generation statistics.
        """
        try:
            print("=" * 60)
            print("GTMicro+ SPRING BOOT SKELETON GENERATOR")
            print("=" * 60)

            self.load_services()
            self.load_api_contracts()
            self.load_dependencies()

            # Reset output directory for idempotency.
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

            services_generated = 0
            controllers = 0
            services_layer = 0
            repositories = 0
            entities = 0
            dtos = 0
            total_deps = sum(len(v) for v in self.dependencies.values())

            for service_name, use_cases in self.services.items():
                print(f"\n--- Generating: {service_name} ---")

                project_root, java_root, package = self.create_project_structure(
                    service_name
                )

                self.generate_pom(service_name, project_root)
                self.generate_application_class(service_name, java_root, package)

                contract = self.contracts.get(
                    service_name, {"base_path": "/", "apis": []}
                )
                dependencies = self.dependencies.get(service_name, [])

                _, method_count = self.generate_controller(
                    service_name, java_root, package, contract, dependencies
                )
                self.generate_service(
                    service_name, java_root, package, dependencies
                )

                entity_name = self._entity_class_name(service_name)
                self.generate_entity(java_root, package, entity_name)
                self.generate_repository(
                    service_name, java_root, package, entity_name
                )

                self.generate_openapi_config(service_name, java_root, package)
                port = self.generate_application_properties(
                    service_name, project_root
                )

                # Generate one DTO per use case.
                seen_dtos: set = set()
                for use_case in use_cases:
                    dto_name = self._dto_class_name(use_case)
                    if dto_name in seen_dtos:
                        continue
                    seen_dtos.add(dto_name)
                    self.generate_dto(java_root, package, dto_name)
                    dtos += 1

                services_generated += 1
                controllers += 1
                services_layer += 1
                repositories += 1
                entities += 1

                print(
                    f"  -> folder: {self._to_kebab_case(service_name)} | "
                    f"port: {port} | APIs: {method_count} | DTOs: {len(seen_dtos)}"
                )

            self._stats = {
                "services": services_generated,
                "controllers": controllers,
                "services_layer": services_layer,
                "repositories": repositories,
                "entities": entities,
                "dtos": dtos,
                "dependencies": total_deps,
            }

            self.generate_report()
            print(
                f"\nDone. {services_generated} service(s) generated under "
                f"'{self.output_dir}/'."
            )
            return self._stats

        except Exception as exc:
            raise RuntimeError(
                f"Spring Boot skeleton generation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    def generate_report(self, path: Optional[str] = None) -> None:
        """
        Write generation_report.txt summarizing what was generated.

        Args:
            path: optional override for the report path.
        """
        output_path = path or self.report_path

        lines: List[str] = [
            "=" * 36,
            "SPRING BOOT GENERATION REPORT",
            "=" * 36,
            "",
            f"Services Generated:     {self._stats['services']}",
            f"Controllers Generated:  {self._stats['controllers']}",
            f"Service Classes:        {self._stats['services_layer']}",
            f"Repositories Generated: {self._stats['repositories']}",
            f"Entities Generated:     {self._stats['entities']}",
            f"DTOs Generated:         {self._stats['dtos']}",
            f"Dependencies Found:     {self._stats['dependencies']}",
            "",
            "=" * 36,
            "",
        ]

        try:
            self._write_file(output_path, "\n".join(lines))
            print(f"Generation report saved to: {output_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write generation report: {exc}"
            ) from exc


# =====================================================================
# STANDALONE ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    try:
        generator = SpringBootSkeletonGenerator()
        generator.run()
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")