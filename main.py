import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import time
import json
import itertools
import random
from typing import List, Dict, Any, Tuple

# Import utility functions
from utils.data_loader import download_and_load_dataset, get_available_datasets, DATA_DIR
from utils.llm_manager import get_ollama_models, initialize_llm, generate_response
from utils.optimizers import get_available_optimizers, OptimizationProblem, run_optimizer
from utils.knowledge_graph import KnowledgeGraph
from utils.rag_retriever import RAGRetriever
from utils.agent_roles import (
    RequirementAgent, ArchitectureAgent, CodeAgent,
    TestingAgent, ConflictAgent, TraceabilityAgent, BaseAgent
)
from utils.evaluation_metrics import calculate_code_metrics, evaluate_defect_prediction, evaluate_traceability, \
    simulate_test_execution, calculate_system_metrics
from utils.plotting import (
    plot_results_dataframe,
    plot_agent_workflow,
    plot_knowledge_graph_summary,
    plot_optimizer_history,
    # ADD THESE NEW IMPORTS:
    plot_optimizer_parameter_trajectories,
    plot_metric_evolution_per_cycle,
    display_summary_statistics_table,
    display_best_performing_configurations
)
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent RAG for Autonomous SE",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💡 Multi-Agent RAG Framework for Autonomous Software Engineering")
st.markdown("---")

# --- Session State Initialization ---
if 'experiment_results' not in st.session_state:
    st.session_state['experiment_results'] = pd.DataFrame(columns=[
        'Run ID', 'Optimization Cycle', 'Dataset', 'LLM Config Name', 'LLM Config Details', 'Optimizer',
        'Maintainability Index', 'Cyclomatic Complexity', 'LOC', 'Num Functions', 'Num Classes',
        'Comment Ratio', 'Defect Prediction Accuracy (F1)', 'Defect Prediction Precision',
        'Defect Prediction Recall', 'Requirement Traceability Accuracy (F1)',
        'Requirement Traceability Precision', 'Requirement Traceability Recall',
        'Test Coverage', 'Test Pass Rate',
        'Architecture Quality Score', 'Code Complexity Score',
        'Requirement Satisfaction Score', 'Conflict Count',
        'Optimizer Best Fitness', 'Optimizer Best Position',
        'Latency (s)', 'Total Workflow Latency (s)', 'KG Node Count', 'KG Edge Count', 'Scalability Metric',
        'Distributed Performance Metric', 'Success', 'Experiment Start Time', 'Experiment End Time',
        'Initial Specification', 'Requirements LLM Output', 'Architecture LLM Output',
        'Generated Code', 'Generated Test Code', 'Conflicts LLM Output',
        'Traceability Links LLM Output', 'Simulated Test Raw Output',
        'KG State at Cycle Start (Nodes)', 'KG State at Cycle Start (Edges)' # Store KG info
    ])
if 'run_id_counter' not in st.session_state:
    st.session_state['run_id_counter'] = 0
if 'full_optimizer_histories' not in st.session_state:
    st.session_state['full_optimizer_histories'] = {} # Store full optimizer histories separately
if 'full_workflow_histories' not in st.session_state:
    st.session_state['full_workflow_histories'] = {} # Store full workflow histories separately

# --- Sidebar Configuration ---
st.sidebar.header("🚀 Experiment Configuration")

# Dataset Selection
st.sidebar.subheader("1. Select Datasets")
available_datasets = get_available_datasets()
selected_datasets = st.sidebar.multiselect(
    "Choose datasets for the experiment (select at least one)",
    options=available_datasets,
    default=available_datasets if available_datasets else []
)

# LLM Model Selection
st.sidebar.subheader("2. Configure LLM Models (Ollama)")
ollama_models = get_ollama_models()
if not ollama_models:
    st.sidebar.warning("No Ollama models found. Please ensure Ollama is running and models are downloaded (`ollama pull <model_name>`).")
selected_llms = st.sidebar.multiselect(
    "Select LLM models for agents (e.g., 'llama2', 'mistral', 'codellama')",
    options=ollama_models,
    default=ollama_models[:min(2, len(ollama_models))] if ollama_models else [] # Default to first 2 or less
)

num_llm_combos_per_run = st.sidebar.number_input(
    "Number of distinct LLM models to use for agent roles in each combination",
    min_value=1,
    max_value=len(selected_llms) if selected_llms else 1,
    value=min(1, len(selected_llms)) if selected_llms else 1,
    help="If you select N LLMs and set this to K, each agent role will be assigned one of the K randomly chosen LLMs from your selection. A value of 1 means all agents in a run use the same chosen LLM. Max value is the number of selected LLMs."
)

# Optimizer Selection
st.sidebar.subheader("3. Select Optimization Algorithms")
available_optimizers = get_available_optimizers()
selected_optimizers = st.sidebar.multiselect(
    "Choose metaheuristic optimizers to evaluate (select at least one)",
    options=available_optimizers,
    default=available_optimizers if available_optimizers else []
)
num_optimizer_iterations = st.sidebar.number_input(
    "Number of internal iterations for each optimizer",
    min_value=10,
    value=50,
    step=10,
    help="This is the `max_iter` parameter for the metaheuristic optimizers. Higher iterations mean potentially better optimization but take longer."
)
num_optimization_cycles = st.sidebar.number_input(
    "Number of Agent-Optimizer Cycles",
    min_value=1,
    value=3,
    help="How many times the agents re-evaluate and refine their outputs based on optimizer guidance. (Each cycle involves a full agent workflow and one optimizer run.)"
)

# Experiment Repetitions
st.sidebar.subheader("4. Experiment Repetitions")
num_workflow_repetitions = st.sidebar.number_input(
    "Number of full agent workflow repetitions per combination",
    min_value=1,
    value=1,
    help="How many times the entire agent workflow (Req->Arch->Code->Test...) is executed for each unique combination of LLM(s), optimizer, and dataset."
)
st.sidebar.markdown("---")

# --- Run Experiment Button ---
if st.sidebar.button("▶️ Run Experiment", type="primary"):
    if not selected_datasets:
        st.sidebar.error("Please select at least one dataset.")
        st.stop()
    if not selected_llms:
        st.sidebar.error("Please select at least one LLM model.")
        st.stop()
    if not selected_optimizers:
        st.sidebar.error("Please select at least one optimizer.")
        st.stop()

    st.session_state['run_id_counter'] += 1
    current_run_id_prefix = f"RUN-{st.session_state['run_id_counter']}"
    st.subheader(f"Starting Experiment: {current_run_id_prefix}")
    st.info("Experiment in progress... This may take a while depending on selected options, dataset sizes, and LLM inference.")

    llm_configs_to_test = []
    agent_roles_list = ["RequirementAgent", "ArchitectureAgent", "CodeAgent", "TestingAgent", "ConflictAgent", "TraceabilityAgent"]

    if num_llm_combos_per_run == 1:
        for llm_name in selected_llms:
            llm_configs_to_test.append({
                "config_name": f"All Agents: {llm_name}",
                "llm_assignments": {role: llm_name for role in agent_roles_list}
            })
    else:
        unique_llm_assignment_sets = set()
        # Create a pool of all possible permutations, then sample for larger numbers.
        # For simplicity, we randomly assign selected LLMs to each agent.
        # This can still generate duplicates, so we use a set for uniqueness.
        
        # Max reasonable random combinations to try and ensure some diversity without exploding runtime
        max_random_combinations = min(10, len(selected_llms)**min(len(agent_roles_list), num_llm_combos_per_run) ) 
        
        attempts = 0
        while len(unique_llm_assignment_sets) < max_random_combinations and attempts < max_random_combinations * 10: # More attempts to find unique
            # Each agent gets a random choice from the `selected_llms` list
            current_llm_assignment = {role: random.choice(selected_llms) for role in agent_roles_list}
            assignment_tuple = frozenset(current_llm_assignment.items()) # Hashable for set
            
            if assignment_tuple not in unique_llm_assignment_sets:
                unique_llm_assignment_sets.add(assignment_tuple)
                # Generate a descriptive name for the config
                assigned_llms_for_name = sorted(list(set(current_llm_assignment.values())))
                config_name = f"Mixed: ({', '.join(assigned_llms_for_name)})"
                
                llm_configs_to_test.append({
                    "config_name": config_name,
                    "llm_assignments": current_llm_assignment
                })
            attempts += 1
        st.sidebar.info(f"Generated {len(llm_configs_to_test)} unique LLM agent configurations for testing.")


    total_true_combinations = len(selected_datasets) * len(llm_configs_to_test) * len(selected_optimizers) * num_workflow_repetitions
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_true_step_count = 0

    for dataset_name in selected_datasets:
        st.markdown(f"## 📦 Processing Dataset: **{dataset_name}**")
        
        # 1. Data Loading and Knowledge Graph Construction (once per dataset)
        st.subheader("1. Data Loading and Knowledge Graph Construction")
        st.info(f"Loading dataset: {dataset_name}")
        dataset_info = download_and_load_dataset(dataset_name)
        if not dataset_info:
            st.error(f"Failed to load dataset: {dataset_name}. Skipping to next dataset.")
            continue
        st.success(f"Dataset '{dataset_name}' loaded successfully.")

        # Re-initialize KG for each dataset to avoid cross-contamination if KG is global
        # Or, if datasets are cumulative, you'd extend the KG. For clear experiments, separate is better.
        kg = KnowledgeGraph()
        if dataset_info['type'] == "git_repo":
            kg.construct_from_github_repo(dataset_info['path'], dataset_info['name'])
        elif dataset_info['type'] == "csv_jira_like":
            kg.construct_from_promise_jira(dataset_info['data'], dataset_info['name'])
        elif dataset_info['type'] == "huggingface_dataset":
            kg.construct_from_codesearchnet(dataset_info['data'], dataset_info['name'])
        
        st.success(f"Knowledge Graph for '{dataset_name}' constructed with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges.")
        
        if len(kg.graph.nodes) > 0 and st.checkbox(f"Show Knowledge Graph for {dataset_name}", value=False, key=f"kg_viz_{dataset_name}"):
             plot_knowledge_graph_summary(kg.graph)

        # 2. Initialize RAG Retriever (once per dataset after KG is built)
        st.subheader("2. Initializing RAG Retriever")
        embedding_model_name = "nomic-embed-text"
        st.info(f"Initializing RAG Retriever with embedding model: `{embedding_model_name}`. Please ensure it's pulled (`ollama pull {embedding_model_name}`).")
        rag_retriever = RAGRetriever(kg, embedding_model_name=embedding_model_name)
        rag_retriever.build_vector_store()
        if not rag_retriever.vectorstore:
            st.error("RAG retriever could not build vector store. Check OllamaEmbeddings initialization. Skipping this dataset.")
            continue

        for llm_config in llm_configs_to_test:
            current_llm_config_name = llm_config["config_name"]
            llm_assignments = llm_config["llm_assignments"]
            st.markdown(f"### 🧠 LLM Configuration: **{current_llm_config_name}**")
            
            # Initialize agents for this configuration (once per LLM config)
            st.subheader("3. Initializing Agents")
            agents = {}
            for role, llm_model_name in llm_assignments.items():
                if role == "RequirementAgent": agents[role] = RequirementAgent(llm_model_name, rag_retriever, kg)
                elif role == "ArchitectureAgent": agents[role] = ArchitectureAgent(llm_model_name, rag_retriever, kg)
                elif role == "CodeAgent": agents[role] = CodeAgent(llm_model_name, rag_retriever, kg)
                elif role == "TestingAgent": agents[role] = TestingAgent(llm_model_name, rag_retriever, kg)
                elif role == "ConflictAgent": agents[role] = ConflictAgent(llm_model_name, rag_retriever, kg)
                elif role == "TraceabilityAgent": agents[role] = TraceabilityAgent(llm_model_name, rag_retriever, kg)
            
            if any(agent.llm is None for agent in agents.values()):
                st.error(f"One or more agents failed to initialize LLM for config '{current_llm_config_name}'. Skipping this configuration.")
                continue
            
            for optimizer_name in selected_optimizers:
                st.markdown(f"#### ⚙️ Optimizer: **{optimizer_name}**")
                
                for rep_idx in range(num_workflow_repetitions):
                    current_true_step_count += 1
                    progress = current_true_step_count / total_true_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Running: Dataset '{dataset_name}', LLM Config '{current_llm_config_name}', Optimizer '{optimizer_name}', Repetition {rep_idx + 1}/{num_workflow_repetitions}...")
                    
                    current_run_sub_id = f"{current_run_id_prefix}_{dataset_name.replace(' ', '_')}_{current_llm_config_name.replace(' ', '_')}_{optimizer_name.replace(' ', '_')}_Rep{rep_idx+1}"
                    st.subheader(f"Running Agent-Optimizer Cycles for: {current_run_sub_id}")
                    
                    # Store optimization history for this specific run
                    optimizer_history_for_this_run = []
                    # Store full workflow history for this specific run
                    full_workflow_history_for_this_run = []

                    initial_spec_global = "Develop a Python-based web service for user management (create, read, update, delete users) with a RESTful API and basic authentication."
                    
                    # Initialize with a default 'neutral' optimization parameter set
                    # [0] -> arch_modularity_preference (0-1)
                    # [1] -> code_complexity_tolerance (0-1)
                    # [2] -> test_coverage_goal (0-1)
                    current_optimization_params = [0.5, 0.5, 0.5] 

                    for cycle_idx in range(num_optimization_cycles):
                        st.markdown(f"##### Optimization Cycle {cycle_idx + 1}/{num_optimization_cycles}")
                        cycle_start_time = time.time()
                        
                        # Store KG state at the beginning of the cycle
                        kg_nodes_at_cycle_start = len(kg.graph.nodes)
                        kg_edges_at_cycle_start = len(kg.graph.edges)

                        # --- Agent Workflow Execution with current optimization parameters ---
                        st.info(f"Executing Agent Workflow (Cycle {cycle_idx + 1})...")
                        
                        # Clear workflow history for this cycle's agents (to capture current cycle's actions)
                        for agent in agents.values():
                            agent.workflow_history = []

                        # 1. Requirement Agent
                        req_output = agents["RequirementAgent"].extract_and_refine_requirements(initial_spec_global, optimization_params=current_optimization_params)
                        requirements = req_output["requirements"]
                        full_workflow_history_for_this_run.extend(agents["RequirementAgent"].workflow_history)
                        if not requirements:
                            st.error("Requirement agent failed to generate requirements. Skipping cycle.")
                            break
                        
                        # 2. Architecture Agent
                        arch_output = agents["ArchitectureAgent"].propose_architecture(requirements, optimization_params=current_optimization_params)
                        architecture = arch_output["architecture"]
                        full_workflow_history_for_this_run.extend(agents["ArchitectureAgent"].workflow_history)
                        if not architecture or not architecture.get('components'):
                            st.error("Architecture agent failed to propose architecture. Skipping cycle.")
                            break
                        
                        # 3. Code Agent
                        code_output = agents["CodeAgent"].generate_code(architecture, requirements, optimization_params=current_optimization_params)
                        generated_code = code_output["generated_code"]
                        full_workflow_history_for_this_run.extend(agents["CodeAgent"].workflow_history)
                        if generated_code == "No code block found.":
                            st.error("Code agent failed to generate code. Skipping cycle.")
                            break
                        
                        # 4. Testing Agent
                        test_output = agents["TestingAgent"].generate_tests(generated_code, requirements, optimization_params=current_optimization_params)
                        generated_test_code = test_output["generated_test_code"]
                        full_workflow_history_for_this_run.extend(agents["TestingAgent"].workflow_history)

                        # 5. Conflict Agent
                        conflict_output = agents["ConflictAgent"].detect_inconsistencies(requirements, architecture, generated_code, optimization_params=current_optimization_params)
                        conflicts = conflict_output["conflicts"]
                        full_workflow_history_for_this_run.extend(agents["ConflictAgent"].workflow_history)

                        # 6. Traceability Agent
                        trace_output = agents["TraceabilityAgent"].link_artifacts(requirements, architecture, generated_code, optimization_params=current_optimization_params)
                        traceability_links = trace_output["traceability_links"]
                        full_workflow_history_for_this_run.extend(agents["TraceabilityAgent"].workflow_history)
                        
                        cycle_end_time = time.time()
                        total_workflow_latency_this_cycle = cycle_end_time - cycle_start_time

                        # --- Collect latest agent outputs for the objective function ---
                        # We need to compute metrics on the *current* state of generated artifacts.
                        current_code_metrics = calculate_code_metrics(generated_code)
                        current_test_exec_results = simulate_test_execution(generated_code, generated_test_code)

                        current_cycle_agent_outputs_for_obj_func = {
                            "generated_code": generated_code,
                            "requirements": requirements,
                            "architecture": architecture,
                            "conflicts": conflicts,
                            "traceability_links": traceability_links,
                            "generated_test_code": generated_test_code,
                            "code_metrics": current_code_metrics,
                            "test_results": current_test_exec_results
                        }

                        # --- Optimization Phase ---
                        st.info(f"Running Optimizer '{optimizer_name}' (Cycle {cycle_idx + 1})...")
                        
                        # Define the objective function for the optimizer here to capture latest state
                        # We are minimizing a 'cost' score here.
                        def se_objective_function_for_optimizer(params: List[float], current_agent_outputs: Dict[str, Any], current_kg: KnowledgeGraph) -> float:
                            # params: [0]->arch_modularity_preference (0-1), [1]->code_complexity_tolerance (0-1), [2]->test_coverage_goal (0-1)
                            
                            # Get the most recent agent outputs and metrics calculated
                            current_generated_code_in_obj = current_agent_outputs.get('generated_code', '')
                            current_requirements_in_obj = current_agent_outputs.get('requirements', [])
                            current_architecture_in_obj = current_agent_outputs.get('architecture', {})
                            current_conflicts_in_obj = current_agent_outputs.get('conflicts', [])
                            current_test_exec_results_in_obj = current_agent_outputs.get('test_results', {})
                            current_code_metrics_in_obj = current_agent_outputs.get('code_metrics', {})

                            # --- Cost Components from Agent Artifacts ---
                            
                            # 1. Conflict Penalty: Higher conflicts -> higher cost
                            conflict_penalty = len(current_conflicts_in_obj) * 15 

                            # 2. Maintainability Cost: Lower MI -> higher cost
                            maintainability_cost = (100 - current_code_metrics_in_obj.get('maintainability_index', 0.0)) * 0.7

                            # 3. Cyclomatic Complexity Cost: Higher CC -> higher cost
                            complexity_cost = current_code_metrics_in_obj.get('cyclomatic_complexity', 1) * 0.3

                            # 4. Requirement Satisfaction Cost: Unmet requirements -> higher cost (heuristic)
                            req_satisfaction_score = 0
                            if current_generated_code_in_obj:
                                for req in current_requirements_in_obj:
                                    if any(word.lower() in current_generated_code_in_obj.lower() for word in req['description'].split() if len(word) > 3):
                                        req_satisfaction_score += 1
                            req_satisfaction_cost = (len(current_requirements_in_obj) - req_satisfaction_score) * 8

                            # 5. Architecture Complexity Cost: Too many/few components vs ideal -> higher cost
                            num_components = len(current_architecture_in_obj.get('components', []))
                            # Ideal range (heuristic): 3-7 components for a simple web service
                            arch_complexity_cost = 0
                            if num_components < 3: arch_complexity_cost = (3 - num_components) * 5
                            elif num_components > 7: arch_complexity_cost = (num_components - 7) * 5
                            # else: within ideal, minimal cost
                            
                            # 6. Test Failure Cost: Lower test pass rate -> higher cost
                            test_failure_cost = (100 - current_test_exec_results_in_obj.get('test_pass_rate', 0.0)) * 0.6


                            # --- Cost Adjustments based on Optimization Parameters (params) ---
                            # These parameters guide the optimization towards specific trade-offs.
                            arch_modularity_pref_param = params[0]
                            code_complexity_tolerance_param = params[1]
                            test_coverage_goal_param = params[2]

                            # 7. Modularity Alignment Cost: Penalize if actual architecture modularity deviates from preference
                            # Assume num_components/10 is a proxy for modularity (more components = more modular)
                            actual_modularity_proxy = num_components / 10.0 # Scale to 0-1 range (heuristic)
                            modularity_alignment_cost = np.abs(arch_modularity_pref_param - actual_modularity_proxy) * 20 # Penalize mismatch

                            # 8. Complexity Tolerance Adjustment: Reward if complexity is within tolerance
                            # If actual complexity (CC) is high, but tolerance is also high, then less penalty
                            # Scale CC to a 0-1 range for comparison, e.g., CC/100
                            actual_complexity_proxy = current_code_metrics_in_obj.get('cyclomatic_complexity', 1) / 100.0
                            # If actual_complexity > tolerance_param, add cost. Else, subtract a small reward.
                            complexity_tolerance_adjustment = max(0, actual_complexity_proxy - code_complexity_tolerance_param) * 10
                            
                            # 9. Test Coverage Gap Cost: Penalize if actual coverage is far below the goal
                            actual_coverage = current_test_exec_results_in_obj.get('test_coverage', 0.0) / 100.0 # Scale to 0-1
                            test_coverage_gap_cost = max(0, (test_coverage_goal_param - actual_coverage)) * 25 # Penalize if goal not met

                            # Total objective value to minimize
                            total_cost = conflict_penalty + maintainability_cost + complexity_cost + \
                                         req_satisfaction_cost + arch_complexity_cost + test_failure_cost + \
                                         modularity_alignment_cost + complexity_tolerance_adjustment + test_coverage_gap_cost
                            
                            return max(0.1, total_cost) # Ensure positive cost
                        
                        optimizer_problem = OptimizationProblem(
                            objective_function=se_objective_function_for_optimizer,
                            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                            dimensions=3
                        )
                        
                        optimizer_result_cycle = run_optimizer(
                            optimizer_name,
                            optimizer_problem,
                            current_cycle_agent_outputs_for_obj_func, # Pass agent outputs to objective function
                            kg, # Pass knowledge graph state
                            num_optimizer_iterations # Number of internal iterations for the metaheuristic
                        )
                        
                        st.write(f"Optimizer Cycle {cycle_idx + 1} Result: Best Fitness = **{optimizer_result_cycle['best_fitness']:.4f}**, Best Position = **{optimizer_result_cycle['best_position']}**")
                        
                        optimizer_history_for_this_run.append(
                            {"cycle": cycle_idx, **optimizer_result_cycle}
                        )

                        # Update current_optimization_params for the next cycle with the optimizer's best
                        current_optimization_params = optimizer_result_cycle['best_position']
                        
                        # --- Evaluation Metrics for this Cycle (using the outputs from this cycle) ---
                        # Code metrics are already calculated in current_code_metrics
                        # Test exec results are already calculated in current_test_exec_results
                        
                        # Defect Prediction (still relies on heuristics for 'actuals' but 'predicted' comes from agent)
                        actual_defects_from_kg = [True for node_id in kg.get_nodes_by_type("issue") if kg.graph.nodes[node_id].get('is_buggy')]
                        # If no actual bugs from KG, generate some based on code complexity for evaluation purposes
                        if not actual_defects_from_kg and current_code_metrics['loc'] > 100: # If code is substantial
                            num_sim_actual_defects = int(current_code_metrics['cyclomatic_complexity'] / 50 + np.random.randint(0,2))
                            actual_defects_from_kg = [True] * num_sim_actual_defects + [False] * (max(0, 5 - num_sim_actual_defects))
                            actual_defects_from_kg = actual_defects_from_kg[:5] # Max 5 for simulation
                        elif not actual_defects_from_kg:
                            actual_defects_from_kg = [False]
                        
                        # Predicted defects: Heuristically link higher complexity/more conflicts to more predicted defects
                        # This should ideally come from ConflictAgent's more granular output
                        num_predicted_defects = int(current_code_metrics['cyclomatic_complexity'] / 40 + len(conflicts) * 0.8)
                        predicted_defects_sim = [True] * num_predicted_defects + [False] * (max(0, len(actual_defects_from_kg) - num_predicted_defects))
                        predicted_defects_sim = predicted_defects_sim[:len(actual_defects_from_kg)]
                        while len(predicted_defects_sim) < len(actual_defects_from_kg): predicted_defects_sim.append(False) # Pad if needed
                        
                        defect_eval_results = evaluate_defect_prediction(predicted_defects_sim, actual_defects_from_kg)
                        st.write(f"Defect Prediction (F1): {defect_eval_results['f1']:.2f}")

                        # Traceability Evaluation (against heuristically inferred actuals from KG)
                        actual_trace_links_kg = []
                        for req_node_id in kg.get_nodes_by_type("requirement"):
                            for code_node_id in kg.get_nodes_by_type("code_file") + kg.get_nodes_by_type("generated_code"):
                                # Check for edges like IMPLEMENTS, SATISFIED_BY between reqs and code
                                # Check if edge exists AND if the 'relation' attribute matches
                                has_satisfied = kg.graph.has_edge(req_node_id, code_node_id) and kg.graph[req_node_id][code_node_id].get('relation') == "SATISFIED_BY"
                                has_implements = kg.graph.has_edge(code_node_id, req_node_id) and kg.graph[code_node_id][req_node_id].get('relation') == "IMPLEMENTS"

                                if has_satisfied or has_implements:
                                    actual_trace_links_kg.append((req_node_id, code_node_id))
                                    
                        # Also infer links from architecture components to code or requirements
                        for arch_comp_node_id in kg.get_nodes_by_type("component"):
                            for req_node_id in kg.get_nodes_by_type("requirement"):
                                if kg.graph.has_edge(req_node_id, arch_comp_node_id) and kg.graph[req_node_id][arch_comp_node_id].get('relation') == "SATISFIED_BY":
                                    
                                    actual_trace_links_kg.append((req_node_id, arch_comp_node_id))
                        
                        trace_eval_results = evaluate_traceability(traceability_links, actual_trace_links_kg)
                        st.write(f"Traceability (F1): {trace_eval_results['f1']:.2f}")

                        # Test Execution Results
                        st.write(f"Test Coverage: {current_test_exec_results['test_coverage']:.2f}%, Pass Rate: {current_test_exec_results['test_pass_rate']:.2f}%")
                        if current_test_exec_results['raw_output']:
                            st.expander("Show Simulated Test Output").code(current_test_exec_results['raw_output'])
                        
                        # System Metrics
                        system_metrics = calculate_system_metrics(
                            cycle_start_time,
                            cycle_end_time,
                            num_agents=len(agents),
                            num_kg_nodes=len(kg.graph.nodes),
                            num_kg_edges=len(kg.graph.edges)
                        )
                        st.write(f"Latency: {system_metrics['latency_s']:.2f}s, Scalability: {system_metrics['scalability_metric']:.2f}, Distributed Perf: {system_metrics['distributed_performance_metric']:.2f}")

                        # --- Store Results for this Cycle ---
                        new_result = {
                            'Run ID': current_run_sub_id,
                            'Optimization Cycle': cycle_idx + 1,
                            'Dataset': dataset_name,
                            'LLM Config Name': current_llm_config_name,
                            'LLM Config Details': json.dumps(llm_assignments),
                            'Optimizer': optimizer_name,
                            # Software Metrics
                            'Maintainability Index': current_code_metrics['maintainability_index'],
                            'Cyclomatic Complexity': current_code_metrics['cyclomatic_complexity'],
                            'LOC': current_code_metrics['loc'],
                            'Num Functions': current_code_metrics['num_functions'],
                            'Num Classes': current_code_metrics['num_classes'],
                            'Comment Ratio': current_code_metrics['comment_ratio'],
                            'Defect Prediction Accuracy (F1)': defect_eval_results['f1'],
                            'Defect Prediction Precision': defect_eval_results['precision'],
                            'Defect Prediction Recall': defect_eval_results['recall'],
                            'Requirement Traceability Accuracy (F1)': trace_eval_results['f1'],
                            'Requirement Traceability Precision': trace_eval_results['precision'],
                            'Requirement Traceability Recall': trace_eval_results['recall'],
                            'Test Coverage': current_test_exec_results['test_coverage'],
                            'Test Pass Rate': current_test_exec_results['test_pass_rate'],
                            # Scores influenced by Optimizer (interpreted from optimizer parameters/fitness)
                            'Architecture Quality Score': (1 - current_optimization_params[0]) * 100, # Higher modularity param preferred by cost, so (1-param) * 100 as heuristic for quality
                            'Code Complexity Score': current_optimization_params[1] * 100, # Higher tolerance param means higher score
                            'Requirement Satisfaction Score': (1 - optimizer_result_cycle['best_fitness'] / 500) * 100, # Lower fitness means higher satisfaction heuristically
                            'Conflict Count': len(conflicts),
                            'Optimizer Best Fitness': optimizer_result_cycle['best_fitness'],
                            'Optimizer Best Position': json.dumps(optimizer_result_cycle['best_position']),
                            # System Metrics
                            'Latency (s)': system_metrics['latency_s'],
                            'Total Workflow Latency (s)': total_workflow_latency_this_cycle,
                            'KG Node Count': len(kg.graph.nodes),
                            'KG Edge Count': len(kg.graph.edges),
                            'Scalability Metric': system_metrics['scalability_metric'],
                            'Distributed Performance Metric': system_metrics['distributed_performance_metric'],
                            'Success': True, # Mark as success if no critical errors stopped workflow
                            'Experiment Start Time': pd.to_datetime(cycle_start_time, unit='s'),
                            'Experiment End Time': pd.to_datetime(cycle_end_time, unit='s'),
                            # Raw Agent Outputs (last iteration's outputs are stored in DF)
                            'Initial Specification': initial_spec_global,
                            'Requirements LLM Output': req_output["raw_llm_response"],
                            'Architecture LLM Output': arch_output["raw_llm_response"],
                            'Generated Code': generated_code,
                            'Generated Test Code': generated_test_code,
                            'Conflicts LLM Output': conflict_output["raw_llm_response"],
                            'Traceability Links LLM Output': trace_output["raw_llm_response"],
                            'Simulated Test Raw Output': current_test_exec_results['raw_output'],
                            'KG State at Cycle Start (Nodes)': kg_nodes_at_cycle_start,
                            'KG State at Cycle Start (Edges)': kg_edges_at_cycle_start
                        }
                        st.session_state['experiment_results'] = pd.concat(
                            [st.session_state['experiment_results'], pd.DataFrame([new_result])],
                            ignore_index=True
                        )
                        st.info(f"Cycle {cycle_idx + 1} completed. Metrics recorded.")
                    
                    # Store full optimizer history for this run
                    st.session_state['full_optimizer_histories'][current_run_sub_id] = optimizer_history_for_this_run
                    # Store full workflow history for this run
                    st.session_state['full_workflow_histories'][current_run_sub_id] = full_workflow_history_for_this_run
                    
                    st.success(f"Run {current_run_sub_id} (all cycles) completed and results stored.")
                    
    st.success("All experiments finished!")

# --- Display Results ---
st.header("📊 Overall Experiment Results")
if not st.session_state['experiment_results'].empty:
    st.dataframe(st.session_state['experiment_results'])

    st.subheader("Visualizations of Key Metrics")
    plot_results_dataframe(st.session_state['experiment_results'])

    # --- NEW PLOTTING INTEGRATION ---
    st.subheader("Detailed Run Analysis")
    # Get unique Run IDs that have completed at least one cycle
    unique_run_ids = st.session_state['experiment_results']['Run ID'].unique().tolist()
    if unique_run_ids:
        selected_detailed_run_id = st.selectbox(
            "Select a specific run for detailed analysis:",
            options=unique_run_ids,
            key="detailed_run_selector"
        )
        if selected_detailed_run_id:
            # Fetch details for the selected run (any row for descriptive info)
            run_details_row = st.session_state['experiment_results'][st.session_state['experiment_results']['Run ID'] == selected_detailed_run_id].iloc[0]
            
            st.markdown(f"**Analyzing Run ID:** `{selected_detailed_run_id}`")
            st.markdown(f"- **Dataset:** `{run_details_row['Dataset']}`")
            st.markdown(f"- **LLM Config:** `{run_details_row['LLM Config Name']}`")
            st.markdown(f"- **Optimizer:** `{run_details_row['Optimizer']}`")

            # Plot Optimizer Parameter Trajectories
            st.markdown("##### Optimizer Parameter Trajectories Over Cycles")
            plot_optimizer_parameter_trajectories(
                st.session_state['experiment_results'],
                selected_detailed_run_id,
                run_details_row['Optimizer'],
                run_details_row['LLM Config Name'],
                run_details_row['Dataset']
            )

            # Plot Metric Evolution Per Cycle
            st.markdown("##### Metric Evolution Over Optimization Cycles")
            plot_metric_evolution_per_cycle(
                st.session_state['experiment_results'],
                selected_detailed_run_id,
                run_details_row['Optimizer'],
                run_details_row['LLM Config Name'],
                run_details_row['Dataset']
            )

            # Display Raw Agent Outputs for Selected Run/Cycle
            st.markdown("##### Raw Agent Outputs for Selected Cycle")
            cycle_options = sorted(st.session_state['experiment_results'][
                st.session_state['experiment_results']['Run ID'] == selected_detailed_run_id
            ]['Optimization Cycle'].unique().tolist())
            selected_cycle_for_raw_output = st.selectbox(
                "Select Optimization Cycle to view raw outputs:",
                options=cycle_options,
                key=f"raw_output_cycle_selector_{selected_detailed_run_id}"
            )
            if selected_cycle_for_raw_output:
                raw_output_row = st.session_state['experiment_results'][
                    (st.session_state['experiment_results']['Run ID'] == selected_detailed_run_id) &
                    (st.session_state['experiment_results']['Optimization Cycle'] == selected_cycle_for_raw_output)
                ].iloc[0]

                st.json({
                    "Initial Specification": raw_output_row['Initial Specification'],
                    "Requirements LLM Output": raw_output_row['Requirements LLM Output'],
                    "Architecture LLM Output": raw_output_row['Architecture LLM Output'],
                    "Generated Code": raw_output_row['Generated Code'],
                    "Generated Test Code": raw_output_row['Generated Test Code'],
                    "Conflicts LLM Output": raw_output_row['Conflicts LLM Output'],
                    "Traceability Links LLM Output": raw_output_row['Traceability Links LLM Output'],
                    "Simulated Test Raw Output": raw_output_row['Simulated Test Raw Output'],
                })
            else:
                st.info("Select a cycle to view its raw agent outputs.")

    else:
        st.info("No completed runs to display detailed analysis.")

    # --- EXISTING PLOTTING INTEGRATION (moved below detailed analysis) ---
    st.subheader("Optimization Progress (Internal Iterations)")
    run_ids_with_opt_history = list(st.session_state['full_optimizer_histories'].keys())
    if run_ids_with_opt_history: # Typo here, should be run_ids_with_opt_history
        selected_run_for_opt_plot = st.selectbox(
            "Select a run to view its internal optimizer history:",
            options=run_ids_with_opt_history,
            key="opt_history_selector"
        )
        if selected_run_for_opt_plot:
            # history_data here is the list of dictionaries for each *cycle*
            # Each dictionary in this list contains a 'history' key which is the actual optimizer history
            full_cycle_history = st.session_state['full_optimizer_histories'][selected_run_for_opt_plot]
            
            # We need to extract the 'history' from the *last* cycle, assuming that's what we want to plot
            # If you want to plot history for each cycle separately, that's a different plot.
            # For "Optimization Progress (Internal Iterations)", we usually plot ONE internal history.
            
            # Let's plot the optimizer's internal history from the FINAL optimization cycle of the selected run.
            if full_cycle_history:
                # The 'history' key holds the list of internal iterations for that cycle's optimizer run
                optimizer_internal_history = full_cycle_history[-1].get('history', []) 
                
                run_details_row = st.session_state['experiment_results'][
                    st.session_state['experiment_results']['Run ID'] == selected_run_for_opt_plot
                ].iloc[0] # Get details from first cycle for consistency
                
                plot_optimizer_history(optimizer_internal_history, f"{run_details_row['Optimizer']} for {run_details_row['LLM Config Name']} on {run_details_row['Dataset']} (Last Cycle)")
            else:
                st.info("No internal optimizer history data found for the selected run.")
    else:
        st.info("No optimizer history available to plot.")

    st.subheader("Agent Workflow Visualizations")
    run_ids_with_workflow = list(st.session_state['full_workflow_histories'].keys())
    if run_ids_with_workflow:
        selected_run_for_workflow_plot = st.selectbox(
            "Select a run to view its agent workflow:",
            options=run_ids_with_workflow,
            key="workflow_history_selector"
        )
        if selected_run_for_workflow_plot:
            workflow_data = st.session_state['full_workflow_histories'][selected_run_for_workflow_plot]
            plot_agent_workflow(workflow_data)
    else:
        st.info("No agent workflow history available to plot.")

    # --- NEW TABLES INTEGRATION ---
    display_summary_statistics_table(st.session_state['experiment_results'])
    display_best_performing_configurations(st.session_state['experiment_results'])

    # --- Download Buttons ---
    csv_data = st.session_state['experiment_results'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download All Experiment Results as CSV",
        data=csv_data,
        file_name="multi_agent_se_experiment_results.csv",
        mime="text/csv",
    )
    json_histories = json.dumps(st.session_state['full_optimizer_histories'], indent=2)
    st.download_button(
        label="Download All Optimizer Histories (JSON)",
        data=json_histories.encode('utf-8'),
        file_name="multi_agent_se_optimizer_histories.json",
        mime="application/json",
    )
    json_workflow_histories = json.dumps(st.session_state['full_workflow_histories'], indent=2)
    st.download_button(
        label="Download All Workflow Histories (JSON)",
        data=json_workflow_histories.encode('utf-8'),
        file_name="multi_agent_se_workflow_histories.json",
        mime="application/json",
    )
else:
    st.info("Run an experiment to see results here.")

# --- Development Status ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Development Status:**")
st.sidebar.markdown("- **Step 5.2 (Current):** Implemented 6 robust metaheuristic optimizers. Objective function refined with more comprehensive heuristics and direct use of calculated metrics. Evaluation metrics for defect prediction and traceability have improved ground truth inference. Simulated test execution is more detailed.")
st.sidebar.markdown("- **Next Steps (Final Review):** Focus on advanced prompt engineering (fine-tuning agent personalities, output formats), and thorough testing across various LLM/Optimizer/Dataset combinations to gather solid results for the paper.")