import streamlit as st
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate # Specific imports
from langchain_core.output_parsers import StrOutputParser
from utils.llm_manager import initialize_llm, generate_response
from utils.rag_retriever import RAGRetriever
from utils.knowledge_graph import KnowledgeGraph
import time
import json
import re

class BaseAgent:
    def __init__(self, name: str, role_description: str, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        self.name = name
        self.role_description = role_description
        self.llm = initialize_llm(llm_model_name)
        self.rag_retriever = rag_retriever
        self.kg = kg
        self.workflow_history = [] # To log agent actions for the current run

        if not self.llm:
            st.error(f"Agent {self.name}: Failed to initialize LLM '{llm_model_name}'. Agent will not function.")

    def log_action(self, action: str, output_summary: str, details: Dict[str, Any] = None):
        log_entry = {
            "agent_name": self.name,
            "action": action,
            "timestamp": time.time(),
            "output_summary": output_summary[:500],
            "details": details if details else {}
        }
        self.workflow_history.append(log_entry)
        
        # --- TEMPORARY DEBUGGING OUTPUT ---
        st.write(f"**Agent {self.name}:** {action}. Output Summary: {output_summary[:100]}...")
        if self.name in ["CodeAgent", "RequirementAgent", "ArchitectureAgent", "TestingAgent", "ConflictAgent", "TraceabilityAgent"]: # Focus on critical agents
             st.expander(f"Raw LLM Response for {self.name} - {action}").code(details.get("llm_response", "No raw response found."))
        # --- END TEMPORARY DEBUGGING OUTPUT ---

    def execute_task(self, task_description: str, context: str = "", optimization_params: List[float] = None) -> str:
        if not self.llm:
            return "Agent LLM not initialized. Cannot execute task."

        full_prompt = f"You are a {self.role_description}.\n\n"
        if context:
            full_prompt += f"Here is relevant context from the knowledge base:\n{context}\n\n"
        
        if optimization_params is not None:
            # Provide specific interpretation of params to the LLM
            # Parameters from `se_objective_function`:
            # [0] -> arch_modularity_preference (0-1)
            # [1] -> code_complexity_tolerance (0-1)
            # [2] -> test_coverage_goal (0-1)

            # Ensure we're extracting scalar values from the list
            # The .item() is not needed here as optimization_params is already a list of floats
            modularity_pref = optimization_params[0] if len(optimization_params) > 0 else 0.5
            complexity_tolerance = optimization_params[1] if len(optimization_params) > 1 else 0.5
            coverage_goal = optimization_params[2] if len(optimization_params) > 2 else 0.5
            
            # The previous check `if isinstance(modularity_pref, list):` was only needed if `optimization_params` 
            # might sometimes be a list containing a list (which should not happen if optimizer returns list of floats).
            # If optimization_params is always `List[float]`, then direct indexing is fine.

            guidance_msgs = [f"Considering the current optimization parameters (Modularity Preference: {modularity_pref:.2f}, Code Complexity Tolerance: {complexity_tolerance:.2f}, Test Coverage Goal: {coverage_goal:.2f}):"]
            
            if self.name == "ArchitectureAgent":
                if modularity_pref > 0.7:
                    guidance_msgs.append("Prioritize a highly modular, potentially microservice-oriented architecture.")
                elif modularity_pref < 0.3:
                    guidance_msgs.append("Consider a more integrated, possibly monolithic or service-oriented architecture for simplicity.")
                else:
                    guidance_msgs.append("Seek a balanced architecture that allows for flexibility without excessive complexity.")
            
            if self.name == "CodeAgent":
                if complexity_tolerance < 0.3:
                    guidance_msgs.append("Strive for extremely simple, highly readable code, even if it means slightly more verbose implementation. Avoid complex patterns.")
                elif complexity_tolerance > 0.7:
                    guidance_msgs.append("You have more tolerance for complexity. Focus on highly optimized, concise code, potentially using advanced patterns or algorithms for performance.")
                else:
                    guidance_msgs.append("Balance readability and performance/optimization for the generated code.")
            
            if self.name == "TestingAgent":
                if coverage_goal > 0.7:
                    guidance_msgs.append("Generate comprehensive tests, aiming for high code coverage, including edge cases, negative scenarios, and integration points.")
                elif coverage_goal < 0.3:
                    guidance_msgs.append("Focus on generating essential tests that cover core functional paths for quick validation.")
                else:
                    guidance_msgs.append("Aim for a balanced test suite that prioritizes critical paths and reasonable coverage.")

            if self.name == "RequirementAgent":
                if complexity_tolerance < 0.5: # Example use of tolerance in requirements
                    guidance_msgs.append("When refining requirements, emphasize clarity and simplicity, avoiding highly intricate features that might lead to complex implementations.")
                if modularity_pref > 0.5: # Example use of modularity preference in requirements
                    guidance_msgs.append("When refining requirements, consider how they contribute to a modular system, potentially breaking down large features.")

            full_prompt += "\n".join(guidance_msgs) + "\n\n"
            full_prompt += "Your task: " + task_description + "\n\nProvide your detailed response and outcome, following any specific output format requests."
        else:
            full_prompt += f"Your task: {task_description}\n\nProvide your detailed response and outcome, following any specific output format requests."

        response = generate_response(self.llm, full_prompt, system_message=self.role_description)
        self.log_action(f"Executed: {task_description}", response, details={"prompt": full_prompt, "llm_response": response})
        return response

# Helper for robust JSON parsing
def parse_llm_json_response(response: str, expected_type: type = list):
    """
    Attempts to parse JSON from LLM response, prioritizing markdown code blocks,
    then aggressive extraction, and cleaning common LLM JSON errors like trailing commas.
    """
    cleaned_response = response.strip()

    # --- Helper to clean common LLM JSON errors ---
    def _clean_json_string(json_str: str) -> str:
        # Remove trailing commas from objects and arrays
        # This regex looks for a comma followed by whitespace and a closing brace/bracket
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # More aggressive: remove any trailing comma before a new line that's followed by a closing brace/bracket
        json_str = re.sub(r',\s*\n\s*([}\]])', r'\n\1', json_str, flags=re.MULTILINE)
        return json_str
    # --- End Helper ---

    # Strategy 1: Look for JSON within a markdown block (```json ... ```)
    json_match = re.search(r"```json\n(.*?)```", cleaned_response, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
        json_content = _clean_json_string(json_content) # Apply cleaning
        try:
            parsed_json = json.loads(json_content)
            if isinstance(parsed_json, expected_type):
                return parsed_json
            else:
                st.warning(f"JSON block parsed, but type mismatch (Expected {expected_type}, Got {type(parsed_json)}). Content: {json_content[:100]}...")
                # If type is wrong but it's valid JSON, let's still return it for debugging,
                # the caller can handle type validation more strictly.
                return parsed_json
        except json.JSONDecodeError:
            st.warning(f"JSON block found but content is invalid JSON even after cleaning. Content: {json_content[:100]}...")
            # Fall through if the markdown block's content itself is invalid JSON

    # Strategy 2: Aggressive extraction - find the first '{' or '[' and the last '}' or ']'
    first_open_brace = cleaned_response.find('{')
    first_open_bracket = cleaned_response.find('[')

    start_idx = -1
    if first_open_brace != -1 and (first_open_brace < first_open_bracket or first_open_bracket == -1):
        start_idx = first_open_brace
        last_close_brace = cleaned_response.rfind('}')
        if last_close_brace != -1 and last_close_brace > start_idx:
            potential_json_str = cleaned_response[start_idx : last_close_brace + 1]
            potential_json_str = _clean_json_string(potential_json_str) # Apply cleaning
            try:
                parsed_json = json.loads(potential_json_str)
                if isinstance(parsed_json, expected_type):
                    return parsed_json
                else:
                    st.warning(f"Aggressive JSON (dict) extract parsed, but type mismatch (Expected {expected_type}, Got {type(parsed_json)}). Content: {potential_json_str[:100]}...")
                    return parsed_json
            except json.JSONDecodeError:
                pass # Fall through
    
    elif first_open_bracket != -1: # Array start found
        start_idx = first_open_bracket
        last_close_bracket = cleaned_response.rfind(']')
        if last_close_bracket != -1 and last_close_bracket > start_idx:
            potential_json_str = cleaned_response[start_idx : last_close_bracket + 1]
            potential_json_str = _clean_json_string(potential_json_str) # Apply cleaning
            try:
                parsed_json = json.loads(potential_json_str)
                if isinstance(parsed_json, expected_type):
                    return parsed_json
                else:
                    st.warning(f"Aggressive JSON (list) extract parsed, but type mismatch (Expected {expected_type}, Got {type(parsed_json)}). Content: {potential_json_str[:100]}...")
                    return parsed_json
            except json.JSONDecodeError:
                pass # Fall through

    st.error(f"Failed to parse LLM response as JSON. Expected {expected_type}. Raw response: {response[:500]}...")
    return None # Indicate failure

# --- Agent implementations (no major changes to their core logic, just how they use optimization_params) ---
class RequirementAgent(BaseAgent):
    def __init__(self, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        super().__init__("RequirementAgent", "An expert in extracting, refining, and validating software requirements.", llm_model_name, rag_retriever, kg)

    def extract_and_refine_requirements(self, initial_spec: str, optimization_params: List[float] = None) -> Dict[str, Any]:
        context_docs_list = self.rag_retriever.retrieve_context(f"existing requirements related to {initial_spec[:50]}", k=3)
        context_str = context_docs_list['full_context'] if context_docs_list else ""

        prompt = (f"Analyze the following initial software specification and extract clear, concise, "
                  f"and unambiguous functional and non-functional requirements. "
                  f"Refine them to be testable. Present them as a JSON list of objects, "
                  f"each with '{{id}}' (e.g., 'REQ-001'), '{{type}}' (functional/non-functional), '{{description}}', '{{priority}}' (High/Medium/Low), "
                  f"and '{{estimated_effort_points}}' (integer, e.g., 1-5 for small, 5-10 for medium). "
                  f"Ensure the '{{id}}' is unique (e.g., 'REQ-001').\n\n" # Added escaping for 'id'
                  f"**Crucially, respond ONLY with the JSON list and NO other text or explanations.**\n\n"
                  f"Example format:\n"
                  f"```json\n"
                  f"[\n"
                  f"  {{\"id\": \"REQ-001\", \"type\": \"functional\", \"description\": \"Users must be able to register with a unique email.\", \"priority\": \"High\", \"estimated_effort_points\": 5}},\n"
                  f"  {{\"id\": \"NFR-001\", \"type\": \"non-functional\", \"description\": \"The system must respond to user login requests within 2 seconds.\", \"priority\": \"Medium\", \"estimated_effort_points\": 3}}\n"
                  f"]\n"
                  f"```\n\n"
                  f"Initial Specification:\n{initial_spec}\n\n"
                  f"Context from Knowledge Graph:\n{context_str}")
        
        response = self.execute_task("Extracting and refining requirements", prompt, optimization_params)
        reqs = parse_llm_json_response(response, list)
        
        if reqs:
            # We will store the fixed requirements here to return them
            processed_reqs = []
            
            for req in reqs:
                # FIX: Check if the LLM returned a string and convert to dict
                if isinstance(req, str):
                    req = {
                        "id": None,
                        "description": req,
                        "type": "functional",
                        "priority": "Medium",
                        "estimated_effort_points": 1
                    }
                
                # Now req is guaranteed to be a dictionary
                if 'id' not in req or not req['id']:
                    req['id'] = f"REQ-{self.kg.node_counter:03d}"
                
                # Check if requirement already exists in the Knowledge Graph
                existing_node_id = self.kg.get_node_by_attribute('id', req['id'], 'requirement')
                
                if existing_node_id:
                    # Update existing requirement node
                    self.kg.graph.nodes[existing_node_id].update(req)
                    self.log_action(f"Updated Requirement {req.get('id', '')} in KG", 
                                   f"Req: {req.get('description', '')}", details=req)
                else:
                    # Add new requirement node
                    self.kg.add_node("requirement", req.get('description', 'Untitled Req'), 
                                    attributes=req, node_id=req['id'])
                    self.log_action(f"Added Requirement {req.get('id', '')} to KG", 
                                   f"Req: {req.get('description', '')}", details=req)
                
                processed_reqs.append(req)

            return {"requirements": processed_reqs, "raw_llm_response": response}
        
        return {"requirements": [], "raw_llm_response": response}


class ArchitectureAgent(BaseAgent):
    def __init__(self, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        super().__init__("ArchitectureAgent", "An expert in designing scalable, maintainable, and robust software architectures.", llm_model_name, rag_retriever, kg)

    def propose_architecture(self, requirements: List[Dict[str, Any]], optimization_params: List[float] = None) -> Dict[str, Any]:
        req_summary = "\n".join([f"- {r['id']}: {r['description']} (Priority: {r['priority']}, Effort: {r.get('estimated_effort_points', 'N/A')})" for r in requirements])
        context_docs_list = self.rag_retriever.retrieve_context(f"architecture patterns for {req_summary[:50]}", k=3)
        context_str = context_docs_list['full_context'] if context_docs_list else ""

        prompt = (f"Given the following refined requirements, propose a suitable software architecture. "
                  f"Describe the architecture in terms of components, their responsibilities, "
                  f"interactions, and key technologies. Also, discuss design patterns applied. "
                  f"Present the architecture as a JSON object with '{{name}}' (e.g., 'UserManagementService'), " # Escaped
                  f"'{{components}}' (list of {{name}}, {{responsibility}}, {{tech}}, {{type}}), " # Escaped
                  f"'{{interactions}}' (list of {{source_component_name}}, {{target_component_name}}, {{type}}), and '{{patterns}}' (list of {{name}}, {{description}}). " # Escaped
                  f"Emphasize scalability and maintainability.\n\n"
                  f"Requirements:\n{req_summary}\n\n"
                  f"Context from Knowledge Graph:\n{context_str}")
        response = self.execute_task("Proposing architecture", prompt, optimization_params)
        architecture = parse_llm_json_response(response, dict)
        
        if architecture:
            arch_name = architecture.get('name', 'Proposed Architecture')
            arch_node_id = self.kg.get_node_by_attribute('name', arch_name, 'architecture')
            if arch_node_id: # Update
                self.kg.graph.nodes[arch_node_id].update({"summary": req_summary, **architecture})
            else: # Add new
                arch_node_id = self.kg.add_node("architecture", arch_name, attributes={"summary": req_summary, **architecture})
            
            for comp in architecture.get('components', []):
                comp_name = comp['name']
                comp_node_id = self.kg.get_node_by_attribute('name', comp_name, 'component')
                if comp_node_id: # Update
                    self.kg.graph.nodes[comp_node_id].update(comp)
                else: # Add new
                    comp_node_id = self.kg.add_node("component", comp_name, attributes=comp, node_id=f"COMP-{comp_name}")
                self.kg.add_edge(arch_node_id, comp_node_id, "HAS_COMPONENT")

            # Link requirements to architecture components (heuristic)
            for req in requirements:
                req_node_id = self.kg.get_node_by_attribute('id', req['id'], 'requirement')
                if req_node_id:
                    for comp in architecture.get('components', []):
                        if any(k.lower() in comp['name'].lower() or k.lower() in comp['responsibility'].lower() for k in req['description'].split() if len(k) > 2): 
                            comp_node_id = self.kg.get_node_by_attribute('name', comp['name'], 'component')
                            if comp_node_id:
                                self.kg.add_edge(req_node_id, comp_node_id, "SATISFIED_BY")
                                self.kg.add_edge(comp_node_id, req_node_id, "SATISFIES")

            self.log_action(f"Proposed architecture with {len(architecture.get('components',[]))} components", f"Architecture: {arch_name}", details=architecture)
            return {"architecture": architecture, "raw_llm_response": response}
        return {"architecture": {}, "raw_llm_response": response}


class CodeAgent(BaseAgent):
    def __init__(self, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        super().__init__("CodeAgent", "An expert in generating and refactoring high-quality, efficient code.", llm_model_name, rag_retriever, kg)

    def generate_code(self, architecture: Dict[str, Any], requirements: List[Dict[str, Any]], optimization_params: List[float] = None) -> Dict[str, Any]:
        arch_summary = json.dumps(architecture.get('components',[]))
        req_summary = "\n".join([r['description'] for r in requirements])
        context_docs_list = self.rag_retriever.retrieve_context(f"code examples for {arch_summary[:50]} and {req_summary[:50]}", k=5)
        context_str = context_docs_list['full_context'] if context_docs_list else ""

        guidance_text = ""
        if optimization_params is not None:
            # Correctly extract scalar from the list
            complexity_tolerance = optimization_params[1] if len(optimization_params) > 1 else 0.5 # THIS LINE IS NOW CORRECTED
            if complexity_tolerance < 0.3:
                guidance_text = "Strive for extremely simple, highly readable code, even if it means slightly more verbose implementation. Avoid complex patterns."
            elif complexity_tolerance > 0.7:
                guidance_text = "You have more tolerance for complexity. Focus on highly optimized, concise code, potentially using advanced patterns or algorithms for performance."
            else:
                guidance_text = "Balance readability and performance/optimization for the generated code."

        prompt = (f"Based on the following architecture proposal and requirements, "
                  f"generate a Python code snippet (e.g., a class or function) that implements a core part of the system. "
                  f"Focus on a single, well-defined module/component, preferably related to user management or authentication (e.g., a `UserManager` class or `AuthService` functions). "
                  f"Include docstrings, type hints, and comments. "
                  f"**Your response MUST contain ONLY the Python code, enclosed in a markdown code block starting with ```python and ending with ```. Do NOT add any other text, explanations, or conversational filler.** {guidance_text}\n\n"
                  f"Architecture Summary: {arch_summary}\n\n"
                  f"Requirements Summary: {req_summary}\n\n"
                  f"Context from Knowledge Graph:\n{context_str}")
        
        response = self.execute_task("Generating code", prompt, optimization_params)
        
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        generated_code = code_match.group(1).strip() if code_match else "No code block found."
        
        if generated_code != "No code block found.":
            code_node_id = self.kg.add_node("generated_code", f"Code_Iter_{self.kg.node_counter}", attributes={"code": generated_code, "context_prompt": prompt})
            
            for req in requirements:
                req_node_id = self.kg.get_node_by_attribute('id', req['id'], 'requirement')
                if req_node_id:
                    self.kg.add_edge(code_node_id, req_node_id, "IMPLEMENTS")
            for comp in architecture.get('components', []):
                comp_node_id = self.kg.get_node_by_attribute('name', comp['name'], 'component')
                if comp_node_id and any(k.lower() in generated_code.lower() for k in comp['name'].split() if len(k) > 2): # Simple heuristic
                     self.kg.add_edge(code_node_id, comp_node_id, "PART_OF_ARCHITECTURE")

            self.log_action(f"Generated code", f"Code length: {len(generated_code)} chars", details={"code_preview": generated_code[:200]})

        return {"generated_code": generated_code, "raw_llm_response": response}


class TestingAgent(BaseAgent):
    def __init__(self, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        super().__init__("TestingAgent", "An expert in designing and generating comprehensive unit and integration tests.", llm_model_name, rag_retriever, kg)

    def generate_tests(self, generated_code: str, requirements: List[Dict[str, Any]], optimization_params: List[float] = None) -> Dict[str, Any]:
        req_summary = "\n".join([r['description'] for r in requirements])
        context_docs_list = self.rag_retriever.retrieve_context(f"testing patterns for {generated_code[:50]} and {req_summary[:50]}", k=3)
        context_str = context_docs_list['full_context'] if context_docs_list else ""

        guidance_text = ""
        if optimization_params is not None:
            # Correctly extract scalar from the list
            coverage_goal = optimization_params[2] if len(optimization_params) > 2 else 0.5 # THIS LINE IS NOW CORRECTED
            if coverage_goal > 0.7:
                guidance_text = "Generate comprehensive tests, aiming for high code coverage, including edge cases, negative scenarios, and integration points."
            elif coverage_goal < 0.3:
                guidance_text = "Focus on generating essential tests that cover core functional paths for quick validation."
            else:
                guidance_text = "Aim for a balanced test suite that prioritizes critical paths and reasonable coverage."

        prompt = (f"Given the following Python code and requirements, generate unit tests using `pytest`. "
                  f"Ensure tests cover functional aspects derived from requirements. "
                  f"**Your response MUST contain ONLY the test code, enclosed in a markdown code block starting with ```python and ending with ```. Do NOT add any other text, explanations, or conversational filler.**\n\n" # <-- STRONGER INSTRUCTION
                  f"Code:\n```python\n{generated_code}\n```\n\n"
                  f"Requirements Summary: {req_summary}\n\n"
                  f"Context from Knowledge Graph:\n{context_str}")
        response = self.execute_task("Generating tests", prompt, optimization_params)
        
        test_code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        generated_test_code = test_code_match.group(1).strip() if test_code_match else "No test code block found."

        if generated_test_code != "No test code block found.":
            test_node_id = self.kg.add_node("test_case", f"Tests_Iter_{self.kg.node_counter}", attributes={"test_code": generated_test_code})
            # Link to generated code node (find latest generated code node)
            code_nodes = self.kg.get_nodes_by_type("generated_code")
            if code_nodes:
                latest_code_node_id = code_nodes[-1] # Assuming latest is the one we want to test
                self.kg.add_edge(test_node_id, latest_code_node_id, "TESTS")
            
            self.log_action(f"Generated tests", f"Test length: {len(generated_test_code)} chars", details={"test_code_preview": generated_test_code[:200]})

        return {"generated_test_code": generated_test_code, "raw_llm_response": response}

class ConflictAgent(BaseAgent):
    def __init__(self, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        super().__init__("ConflictAgent", "An expert in detecting inconsistencies and conflicts between software artifacts (requirements, architecture, code).", llm_model_name, rag_retriever, kg)

    def detect_inconsistencies(self, requirements: List[Dict[str, Any]], architecture: Dict[str, Any], generated_code: str, optimization_params: List[float] = None) -> Dict[str, Any]:
        req_summary = "\n".join([r['description'] for r in requirements])
        arch_summary = json.dumps(architecture.get('components',[]))
        
        context_docs_list = self.rag_retriever.retrieve_context(f"conflict resolution patterns for {req_summary[:50]}", k=2)
        context_str = context_docs_list['full_context'] if context_docs_list else ""

        prompt = (f"Analyze the following requirements, proposed architecture, and generated code snippet. "
                  f"Identify any inconsistencies, ambiguities, or conflicts between them. "
                  f"For each detected inconsistency, provide a detailed description and suggest potential resolutions. "
                  f"Present the findings as a JSON list of objects, each with '{{type}}' (e.g., requirement-architecture conflict, code-requirement mismatch), " # Escaped
                  f"'{{description}}', '{{severity}}' (High/Medium/Low), and '{{suggested_resolution}}'. " # Escaped
                  f"**Crucially, respond ONLY with the JSON list and NO other text or explanations.**\n\n" # Added instruction
                  f"Requirements:\n{req_summary}\n\n"
                  f"Architecture:\n{arch_summary}\n\n"
                  f"Generated Code:\n```python\n{generated_code}\n```\n\n"
                  f"Context from Knowledge Graph:\n{context_str}")
        response = self.execute_task("Detecting inconsistencies", prompt, optimization_params)
        conflicts = parse_llm_json_response(response, list)
        
        if conflicts is not None:
            if conflicts:
                self.log_action(f"Detected {len(conflicts)} conflicts", f"First conflict: {conflicts[0].get('description', '')}" if conflicts else "N/A", details={"conflicts": conflicts})
            else:
                self.log_action("No conflicts detected", "System appears consistent.")
            return {"conflicts": conflicts, "raw_llm_response": response}
        return {"conflicts": [], "raw_llm_response": response}

class TraceabilityAgent(BaseAgent):
    def __init__(self, llm_model_name: str, rag_retriever: RAGRetriever, kg: KnowledgeGraph):
        super().__init__("TraceabilityAgent", "An expert in establishing and verifying traceability links between requirements, design, and code.", llm_model_name, rag_retriever, kg)

    def link_artifacts(self, requirements: List[Dict[str, Any]], architecture: Dict[str, Any], generated_code: str, optimization_params: List[float] = None) -> Dict[str, Any]:
        req_ids = [r['id'] for r in requirements]
        
        context_docs_list = self.rag_retriever.retrieve_context(f"traceability links for requirements {req_ids} to code", k=3)
        context_str = context_docs_list['full_context'] if context_docs_list else ""

        prompt = (f"Analyze the provided requirements, architecture, and generated code. "
                  f"Identify and establish direct traceability links between: "
                  f"1. Requirements and Architecture components. "
                  f"2. Requirements and Generated Code segments. "
                  f"3. Architecture components and Generated Code segments. "
                  f"List these links as a JSON list of objects, each with '{{source_type}}', '{{source_id}}', " # Escaped
                  f"'{{target_type}}', '{{target_id}}', '{{relation_type}}' (e.g., 'IMPLEMENTS', 'SATISFIES', 'IS_PART_OF'). " # Escaped
                  f"**Crucially, respond ONLY with the JSON list and NO other text or explanations.**\n\n" # Added instruction
                  f"Requirements: {json.dumps(requirements, indent=2)}\n\n"
                  f"Architecture Components: {json.dumps(architecture.get('components',[]), indent=2)}\n\n"
                  f"Generated Code:\n```python\n{generated_code}\n```\n\n"
                  f"Context from Knowledge Graph:\n{context_str}")
        response = self.execute_task("Linking software artifacts for traceability", prompt, optimization_params)
        links = parse_llm_json_response(response, list)
        
        if links is not None:
            self.log_action(f"Identified {len(links)} traceability links", f"First link: {links[0] if links else 'N/A'}", details={"links": links})
            return {"traceability_links": links, "raw_llm_response": response}
        return {"traceability_links": [], "raw_llm_response": response}