import networkx as nx
import streamlit as st
import ast
import os
import re
from git import Repo, InvalidGitRepositoryError # Import InvalidGitRepositoryError
import pandas as pd
import json
import uuid # For unique node IDs when no natural ID exists

class KnowledgeGraph:
    """
    Manages the construction and querying of a graph-based representation
    of software artifacts.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0

    def add_node(self, node_type: str, name: str, attributes: dict = None, node_id: str = None):
        """
        Adds a node to the graph. If node_id is provided, uses it, otherwise generates one.
        Ensures node IDs are unique.
        """
        if node_id is None:
            node_id = f"{node_type}_{uuid.uuid4().hex[:8]}" # More robust unique ID
        
        # Ensure ID is unique, if not, try another
        while node_id in self.graph:
            node_id = f"{node_type}_{uuid.uuid4().hex[:8]}"

        attrs = {'type': node_type, 'name': name}
        if attributes:
            attrs.update(attributes)
        
        # Ensure serializable attributes (e.g., convert objects to strings if needed)
        for key, value in attrs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                attrs[key] = str(value) # Convert non-serializable objects to string

        self.graph.add_node(node_id, **attrs)
        return node_id

    def add_edge(self, source_id: str, target_id: str, relation: str, attributes: dict = None):
        """Adds an edge between two nodes, only if both nodes exist."""
        if source_id not in self.graph:
            # st.warning(f"Source node {source_id} not found for edge {relation}. Skipping edge.")
            return
        if target_id not in self.graph:
            # st.warning(f"Target node {target_id} not found for edge {relation}. Skipping edge.")
            return

        attrs = {'relation': relation}
        if attributes:
            attrs.update(attributes)
        
        # Ensure serializable attributes for edges too
        for key, value in attrs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                attrs[key] = str(value)

        self.graph.add_edge(source_id, target_id, **attrs)

    def get_nodes_by_type(self, node_type: str):
        """Returns all node IDs of a specific type."""
        return [node for node, attrs in self.graph.nodes(data=True) if attrs.get('type') == node_type]

    def get_node_by_attribute(self, attr_name: str, attr_value: str, node_type: str = None):
        """Returns the first node ID matching attribute, optionally filtered by type."""
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get(attr_name) == attr_value:
                if node_type is None or attrs.get('type') == node_type:
                    return node_id
        return None

    def get_neighbors(self, node_id: str, relation_type: str = None, outgoing: bool = True):
        """
        Returns neighbors of a node, optionally filtered by relation type and direction.
        outgoing: True for successors, False for predecessors.
        """
        if node_id not in self.graph:
            return []

        neighbors = []
        if outgoing:
            iterator = self.graph.successors(node_id)
        else:
            iterator = self.graph.predecessors(node_id)

        for neighbor_id in iterator:
            edge_data = self.graph.get_edge_data(node_id, neighbor_id) if outgoing else self.graph.get_edge_data(neighbor_id, node_id)
            if edge_data and (relation_type is None or edge_data.get('relation') == relation_type):
                neighbors.append(neighbor_id)
        return neighbors

    def construct_from_github_repo(self, repo_path: str, repo_name: str = "GitHub_Repo"):
        """
        Constructs parts of the knowledge graph from a cloned GitHub repository.
        Focuses on Python files, commits, developers, and basic dependencies.
        Handles shallow clones by focusing on latest commits and existing files.
        """
        st.info(f"Constructing knowledge graph from GitHub repo: {repo_name} at {repo_path}")
        try:
            repo = Repo(repo_path)
            # Ensure the repo is not empty (e.g., if clone failed entirely)
            if not repo.head.is_valid():
                st.error(f"Repository at '{repo_path}' is empty or invalid after cloning. Skipping KG construction.")
                return
        except InvalidGitRepositoryError:
            st.error(f"'{repo_path}' is not a valid Git repository. Skipping KG construction for this repo.")
            return
        except Exception as e:
            st.error(f"Error accessing Git repository at '{repo_path}': {e}. Skipping KG construction.")
            return

        # Add Repository node
        repo_node_id = self.add_node("repository", repo_name, {"path": repo_path})

        code_nodes = {} # Stores {relative_path: node_id}
        developer_nodes = {} # Stores {email: node_id}
        
        # 1. Add Code Modules (Python files) and parse AST for dependencies
        st.info("Parsing code files for modules, functions, classes and dependencies...")
        for root, _, files in os.walk(repo_path):
            # Skip common non-code directories and .git
            if ".git" in root or "venv" in root or "env" in root or "__pycache__" in root or "node_modules" in root:
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception as e:
                        st.warning(f"Could not read file {relative_path}: {e}")
                        content = ""

                    # Add code_file node
                    code_id = self.add_node("code_file", relative_path, 
                                            {"path": file_path, "language": "python", "content_preview": content[:500]})
                    code_nodes[relative_path] = code_id
                    self.add_edge(repo_node_id, code_id, "CONTAINS_FILE")

                    # Parse AST for basic function/class definitions and calls
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                func_code_snippet = ast.get_source_segment(content, node) if ast.get_source_segment(content, node) else ""
                                func_id = self.add_node("function", node.name, {"file": relative_path, "lineno": node.lineno, "code_snippet": func_code_snippet})
                                self.add_edge(code_id, func_id, "DEFINES_FUNCTION")
                                
                                # Basic call graph within the file (can be improved)
                                for sub_node in ast.walk(node):
                                    if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                                        called_func_name = sub_node.func.id
                                        # Heuristic: if a function with this name exists in this file
                                        for existing_func_node_id in self.get_nodes_by_type("function"):
                                            if self.graph.nodes[existing_func_node_id].get('name') == called_func_name and \
                                               self.graph.nodes[existing_func_node_id].get('file') == relative_path and \
                                               existing_func_node_id != func_id: # Avoid self-loops for direct definition
                                                self.add_edge(func_id, existing_func_node_id, "CALLS_FUNCTION")
                                                break
                            elif isinstance(node, ast.ClassDef):
                                class_code_snippet = ast.get_source_segment(content, node) if ast.get_source_segment(content, node) else ""
                                class_id = self.add_node("class", node.name, {"file": relative_path, "lineno": node.lineno, "code_snippet": class_code_snippet})
                                self.add_edge(code_id, class_id, "DEFINES_CLASS")
                                for method in [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                                    method_code_snippet = ast.get_source_segment(content, method) if ast.get_source_segment(content, method) else ""
                                    method_id = self.add_node("method", method.name, {"class": node.name, "file": relative_path, "lineno": method.lineno, "code_snippet": method_code_snippet})
                                    self.add_edge(class_id, method_id, "HAS_METHOD")

                            if isinstance(node, (ast.Import, ast.ImportFrom)):
                                module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                                if module_name and not module_name.startswith('.'): # Focus on non-relative imports for now
                                    potential_target_path = module_name.replace('.', os.sep) + ".py"
                                    for existing_file_path, existing_file_node_id in code_nodes.items():
                                        if existing_file_path.endswith(potential_target_path) and existing_file_node_id != code_id: # Avoid self-import
                                            self.add_edge(code_id, existing_file_node_id, "IMPORTS")
                                            break
                    except SyntaxError as e:
                        st.warning(f"SyntaxError in {relative_path}: {e}")
                    except Exception as e:
                        st.warning(f"Could not parse AST or extract features for {relative_path}: {e}")

        # 2. Add Commits and Developers (Iterate on commits carefully for shallow clones)
        st.info("Parsing commit history for commits and developers...")
        
        latest_commit = None
        try:
            latest_commit = repo.head.commit
        except ValueError: # No commits yet, or repo is empty (should be caught by repo.head.is_valid() above)
            st.warning("Repository appears to have no commits. Skipping commit/developer linking.")
            return # Exit if no commits
        except Exception as e:
            st.warning(f"Could not get latest commit: {e}. Skipping commit/developer linking.")
            return # Exit on other errors

        # If we have a latest_commit, proceed
        if latest_commit:
            committer_name = latest_commit.author.name
            committer_email = latest_commit.author.email
            
            developer_id = developer_nodes.get(committer_email)
            if not developer_id:
                developer_id = self.add_node("developer", committer_name, {"email": committer_email})
                developer_nodes[committer_email] = developer_id

            commit_id = self.add_node("commit", latest_commit.hexsha[:7], {
                "hash": latest_commit.hexsha,
                "message": latest_commit.message.splitlines()[0] if latest_commit.message else "",
                "full_message": latest_commit.message,
                "date": latest_commit.authored_datetime.isoformat(),
                "author_email": committer_email
            })
            self.add_edge(commit_id, developer_id, "AUTHORED_BY")
            self.add_edge(developer_id, commit_id, "HAS_AUTHORED")
            self.add_edge(repo_node_id, commit_id, "HAS_COMMIT")

            # For shallow clones, linking changes requires careful handling.
            # Instead of `commit.stats.files` which tries to diff,
            # we link the latest commit to all files currently in the repo as a simplification.
            st.info("Linking latest commit to all code files (simplification for shallow clones)...")
            for file_path_rel, file_node_id in code_nodes.items():
                self.add_edge(commit_id, file_node_id, "MODIFIED")
                self.add_edge(file_node_id, commit_id, "MODIFIED_BY_COMMIT")

        # 3. Simulate Requirements/Issues and link to code/commits (existing logic remains)
        st.info("Simulating requirements and linking heuristically...")
        dummy_issues = [
            {"id": "REQ-001", "summary": "Implement user authentication for `login.py`", "keywords": ["login", "authentication", "user", "auth"], "description": "Users should be able to log in securely."},
            {"id": "BUG-002", "summary": "Fix performance issue in `data_processor.py`", "keywords": ["performance", "data_processor", "speed"], "description": "The system slows down under high load."},
            {"id": "REQ-003", "summary": "Add user profile management", "keywords": ["profile", "user", "manage"], "description": "Allow users to view and update their profile details."},
            {"id": "DOC-001", "summary": "Update API documentation for user endpoints", "keywords": ["API", "documentation", "user"], "description": "Ensure all user-related API endpoints are documented correctly."}
        ]
        
        latest_commit_id_available = 'commit_id' in locals() # Check if commit_id was successfully created

        for issue in dummy_issues:
            # Ensure unique node ID for requirements if not provided explicitly by the data
            req_node_id = self.add_node("requirement", issue['summary'], attributes=issue, node_id=issue['id'])
            
            # Link requirements to code nodes heuristically
            for code_rel_path, code_node_id in code_nodes.items():
                if any(keyword.lower() in code_rel_path.lower() or keyword.lower() in self.graph.nodes[code_node_id].get('content_preview', '').lower() for keyword in issue['keywords']):
                    self.add_edge(req_node_id, code_node_id, "REQUIRES_IMPLEMENTATION_IN")
                    self.add_edge(code_node_id, req_node_id, "IMPLEMENTS")
            # Link to the latest commit heuristically
            if latest_commit_id_available and latest_commit: 
                if any(keyword.lower() in latest_commit.message.lower() for keyword in issue['keywords']):
                    self.add_edge(req_node_id, commit_id, "RELATED_TO_COMMIT")


        st.success(f"Knowledge graph built from GitHub repo with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")


    def construct_from_promise_jira(self, jira_data_df: pd.DataFrame, dataset_name: str = "PROMISE_Dataset"):
        """
        Constructs parts of the knowledge graph from a PROMISE-like JIRA dataset (DataFrame).
        Assumes columns like 'id' (or 'issue_id'), 'summary', 'description', 'status', 'classification' (e.g., 'buggy'/'not buggy').
        """
        st.info(f"Constructing knowledge graph from PROMISE JIRA dataset: {dataset_name}.")
        
        dataset_node_id = self.add_node("dataset", dataset_name)

        issue_nodes = {}
        for index, row in jira_data_df.iterrows():
            # Robustly get issue ID, handling different column names
            issue_id_col = next((col for col in ['id', 'issue_id', 'bug_id'] if col in row), None)
            summary_col = next((col for col in ['summary', 'title'] if col in row), None)
            description_col = next((col for col in ['description', 'text'] if col in row), None)
            status_col = next((col for col in ['status', 'state'] if col in row), None)
            class_col = next((col for col in ['class', 'defects', 'buggy'] if col in row), None) # For defect prediction

            issue_id = row[issue_id_col] if issue_id_col else f"issue_{uuid.uuid4().hex[:8]}"
            summary = row[summary_col] if summary_col else f"No summary for {issue_id}"
            description = row[description_col] if description_col else ""
            status = row[status_col] if status_col else "unknown"
            
            is_buggy = False
            if class_col and pd.notna(row[class_col]):
                # Handle boolean, int (0/1), or string ('true'/'false') representations
                if isinstance(row[class_col], str):
                    is_buggy = row[class_col].strip().lower() in ['true', 'yes', 'y', '1']
                elif isinstance(row[class_col], (bool, int, float)): # float for numpy bool converted to float
                    is_buggy = bool(row[class_col])

            # Determine issue type heuristically
            issue_type = "requirement"
            if "bug" in summary.lower() or "defect" in summary.lower() or is_buggy:
                issue_type = "bug"
            elif "feature" in summary.lower() or "add" in summary.lower():
                issue_type = "feature"
            
            node_attrs = {
                "original_id": str(issue_id), # Ensure ID is string for consistency
                "description": description,
                "status": status,
                "is_buggy": is_buggy,
                "type": issue_type
            }

            node_kg_id = self.add_node("issue", summary, attributes=node_attrs, node_id=f"issue_{issue_id}") # Use explicit issue ID for KG node
            issue_nodes[issue_id] = node_kg_id
            self.add_edge(dataset_node_id, node_kg_id, "CONTAINS_ISSUE")
            
            # Additional attributes for PROMISE datasets like CM1/JM1
            # These datasets often have many metrics. Add a subset to the node attributes.
            relevant_metrics_cols = [
                'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lóc',
                'l_o_c', 'cyclomatic_complexity', 'halstead_volume', 'halstead_difficulty'
            ]
            for col in relevant_metrics_cols:
                if col in row and pd.notna(row[col]):
                    self.graph.nodes[node_kg_id][col] = row[col] # Add directly to node attributes


        st.success(f"Knowledge graph built from PROMISE JIRA data with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")


    def construct_from_codesearchnet(self, codesearchnet_dataset, dataset_name: str = "CodeSearchNet_Python"):
        """
        Constructs parts of the knowledge graph from a CodeSearchNet-like dataset.
        Focus: Code snippets (functions), their documentation, and definitions.
        """
        st.info(f"Constructing knowledge graph from CodeSearchNet dataset: {dataset_name}.")
        
        dataset_node_id = self.add_node("dataset", dataset_name)
        
        func_nodes = {} # Stores {unique_func_identifier: node_id}
        
        num_processed_entries = 0
        # Iterate up to 200 entries to keep it manageable for demo
        for i, entry in enumerate(codesearchnet_dataset):
            if i >= 200: 
                st.warning(f"Processed {num_processed_entries} CodeSearchNet entries. Stopping at 200 for demo performance.")
                break
            
            func_code = entry.get('func_code') # This is the full code of the function
            docstring = entry.get('docstring')
            func_name = entry.get('func_name')
            repo_name = entry.get('repo')
            file_path = entry.get('path')

            if func_code and func_name:
                # Create a truly unique identifier for each function
                unique_func_identifier = f"csn_{repo_name}_{file_path}::{func_name}"
                
                if unique_func_identifier in func_nodes:
                    continue # Skip duplicates
                
                code_node_id = self.add_node("code_function", func_name, {
                    "code": func_code, # Store full code in this node
                    "language": "python",
                    "docstring": docstring,
                    "repo": repo_name,
                    "file_path": file_path,
                    "full_identifier": unique_func_identifier
                }, node_id=unique_func_identifier)
                
                func_nodes[unique_func_identifier] = code_node_id
                self.add_edge(dataset_node_id, code_node_id, "CONTAINS_FUNCTION")
                num_processed_entries += 1

                # Parse AST for basic calls within the snippet (using func_code)
                try:
                    tree = ast.parse(func_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                called_func_name = node.func.id
                                # Heuristic: if a function with this name exists from other CodeSearchNet entries
                                # This is a weak link, but better than nothing for a demo context.
                                for existing_func_id, attrs in self.graph.nodes(data=True):
                                    if attrs.get('type') == 'code_function' and attrs.get('name') == called_func_name and existing_func_id != code_node_id:
                                        self.add_edge(code_node_id, existing_func_id, "CALLS")
                                        break
                except SyntaxError as e:
                    st.warning(f"SyntaxError in function '{func_name}' (CodeSearchNet, {repo_name}/{file_path}): {e}")
                except Exception as e:
                    st.warning(f"Could not parse AST or extract calls for function '{func_name}' (CodeSearchNet): {e}")

        st.success(f"Knowledge graph built from CodeSearchNet data with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

    def query_graph(self, query_text: str, k: int = 5):
        """
        Performs a semantic search-like query over the knowledge graph (simplified).
        For now, it's keyword-based on node names and descriptions.
        """
        # st.info(f"Querying knowledge graph for: '{query_text}' (keyword-based fallback)")
        results = []
        query_text_lower = query_text.lower()

        # Simple keyword matching on node names and attributes
        for node_id, attrs in self.graph.nodes(data=True):
            match = False
            if query_text_lower in attrs.get('name', '').lower():
                match = True
            elif query_text_lower in attrs.get('description', '').lower():
                match = True
            elif query_text_lower in attrs.get('full_message', '').lower(): # For commit messages
                match = True
            elif attrs.get('code') and query_text_lower in attrs['code'].lower(): # For generated_code content
                match = True
            elif attrs.get('code_snippet') and query_text_lower in attrs['code_snippet'].lower(): # For function/class code snippets
                match = True
            elif attrs.get('content_preview') and query_text_lower in attrs['content_preview'].lower(): # For code_file previews
                match = True
            
            if match:
                results.append((node_id, attrs))

        # Prioritize certain types if query is specific
        if "requirement" in query_text_lower or "issue" in query_text_lower or "bug" in query_text_lower:
            results.sort(key=lambda x: 1 if x[1].get('type') in ['requirement', 'issue', 'bug', 'feature'] else 0, reverse=True)
        elif "code" in query_text_lower or "function" in query_text_lower or "class" in query_text_lower:
            results.sort(key=lambda x: 1 if x[1].get('type') in ['code_file', 'code_function', 'function', 'class', 'method', 'generated_code'] else 0, reverse=True)
        
        return results[:k]

    def get_context_for_node(self, node_id: str, depth: int = 1) -> str:
        """
        Extracts contextual information around a given node (its attributes and immediate neighbors).
        """
        if node_id not in self.graph:
            return f"Node '{node_id}' not found in knowledge graph."

        node_attrs = self.graph.nodes[node_id]
        context = [f"--- Node: {node_attrs.get('name', node_id)} (Type: {node_attrs.get('type')}) ---"]
        
        # Add key attributes
        for attr, value in node_attrs.items():
            if attr not in ['type', 'name', 'code', 'code_snippet', 'content_preview', 'raw_llm_response', 'full_identifier'] and value is not None:
                context.append(f"  {attr.replace('_', ' ').title()}: {str(value)[:150]}..." if len(str(value)) > 150 else f"  {attr.replace('_', ' ').title()}: {str(value)}")

        if node_attrs.get('code'): # For generated_code
            context.append(f"  Code snippet: ```python\n{node_attrs['code'][:500]}...\n```")
        elif node_attrs.get('code_snippet'): # For function/method/class nodes
            context.append(f"  Code Snippet: ```python\n{node_attrs['code_snippet'][:500]}...\n```")
        elif node_attrs.get('content_preview'): # For code_file nodes
            context.append(f"  File Content Preview: ```python\n{node_attrs['content_preview'][:500]}...\n```")
        
        # Add immediate neighbors (depth 1)
        if depth > 0:
            context.append("\n--- Related Nodes (Edges) ---")
            for neighbor in nx.neighbors(self.graph, node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                neighbor_attrs = self.graph.nodes[neighbor]
                context.append(f"  - {edge_data.get('relation', 'RELATED_TO')} -> {neighbor_attrs.get('name', neighbor)} (Type: {neighbor_attrs.get('type')})")
        
        return "\n".join(context)