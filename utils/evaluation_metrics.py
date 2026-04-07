import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
import ast # For parsing code to estimate complexity
import io
import contextlib
import sys
import os
import uuid # For generating temporary module names
from typing import List,Dict,Any,Tuple
# --- Code Metrics ---
def calculate_maintainability_index(loc: int, cyclomatic_complexity: int, comment_ratio: float) -> float:
    """
    Estimates Maintainability Index (MI) based on a simplified formula (0-100).
    Higher is better.
    """
    if loc <= 0: return 0.0

    # Weights are heuristic but aim to penalize complexity and reward comments/readability
    mi = 100 - (0.15 * cyclomatic_complexity) - (0.008 * loc) + (25 * comment_ratio)
    return max(0.0, min(100.0, mi)) # Clamp between 0 and 100

def get_ast_complexity_metrics(code_snippet: str) -> Tuple[int, int, int, int]:
    """
    Calculates basic AST-based metrics for a Python code snippet.
    - Lines of Code (LOC)
    - Cyclomatic Complexity (simplified, counts branches/loops/conditions)
    - Number of functions
    - Number of classes
    """
    loc = len(code_snippet.splitlines())
    num_functions = 0
    num_classes = 0
    cyclomatic_complexity = 1 # Start with 1 for the function/module itself (entry point)

    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                num_functions += 1
            elif isinstance(node, ast.ClassDef):
                num_classes += 1
            # Add 1 for each control flow statement that increases complexity
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith)):
                cyclomatic_complexity += 1
            elif isinstance(node, ast.BoolOp): # 'and', 'or' operators
                cyclomatic_complexity += len(node.values) - 1 # Each additional condition adds complexity
            elif isinstance(node, ast.Compare) and len(node.ops) > 1: # Chained comparisons (a < b < c)
                cyclomatic_complexity += len(node.ops) - 1

    except SyntaxError:
        # st.warning("SyntaxError in code for metric calculation. Returning defaults.")
        return 0, 1, 0, 0
    except Exception as e:
        # st.warning(f"Error parsing AST for metrics: {e}. Returning defaults.")
        return 0, 1, 0, 0

    return loc, cyclomatic_complexity, num_functions, num_classes

def calculate_code_metrics(code_snippet: str) -> Dict[str, Any]:
    """
    Calculates various metrics for a given code snippet.
    """
    if not code_snippet.strip():
        return {
            "loc": 0, "cyclomatic_complexity": 1, "num_functions": 0, "num_classes": 0,
            "comment_ratio": 0.0, "maintainability_index": 0.0, "code_len_chars": 0
        }

    loc, cyclomatic_complexity, num_functions, num_classes = get_ast_complexity_metrics(code_snippet)
    
    # Count comments (more robust heuristic including multiline docstrings)
    comment_lines = len(re.findall(r'^\s*#', code_snippet, re.MULTILINE))
    docstring_matches = re.findall(r'^\s*(?:"""|\'\'\')[\s\S]*?(?:"""|\'\'\')', code_snippet, re.MULTILINE)
    for match in docstring_matches:
        comment_lines += len(match.splitlines())
        
    comment_ratio = comment_lines / loc if loc > 0 else 0.0

    maintainability_index = calculate_maintainability_index(loc, cyclomatic_complexity, comment_ratio)

    return {
        "loc": loc,
        "cyclomatic_complexity": cyclomatic_complexity,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "comment_ratio": comment_ratio,
        "maintainability_index": maintainability_index,
        "code_len_chars": len(code_snippet)
    }

# --- Defect Prediction Evaluation ---
def evaluate_defect_prediction(predicted_defects: List[bool], actual_defects: List[bool]) -> Dict[str, Any]:
    """
    Calculates classification metrics for defect prediction.
    Predicted/actual are lists of boolean or 0/1.
    """
    if not actual_defects or not predicted_defects or len(actual_defects) != len(predicted_defects):
        # st.warning("Insufficient or mismatched data for defect prediction evaluation. Returning 0s.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": {}}

    try:
        actual_defects_bool = [bool(d) for d in actual_defects]
        predicted_defects_bool = [bool(d) for d in predicted_defects]

        acc = accuracy_score(actual_defects_bool, predicted_defects_bool)
        prec = precision_score(actual_defects_bool, predicted_defects_bool, zero_division=0)
        rec = recall_score(actual_defects_bool, predicted_defects_bool, zero_division=0)
        f1 = f1_score(actual_defects_bool, predicted_defects_bool, zero_division=0)
        
        # Convert confusion matrix to a serializable format (list of lists)
        cm_array = confusion_matrix(actual_defects_bool, predicted_defects_bool)
        cm_dict = {
            "TN": cm_array[0,0] if cm_array.shape == (2,2) else 0,
            "FP": cm_array[0,1] if cm_array.shape == (2,2) else 0,
            "FN": cm_array[1,0] if cm_array.shape == (2,2) else 0,
            "TP": cm_array[1,1] if cm_array.shape == (2,2) else 0,
        }
        
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm_dict}
    except Exception as e:
        # st.error(f"Error in defect prediction evaluation: {e}. Returning 0s.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": {}}

# --- Traceability Evaluation ---
def evaluate_traceability(predicted_links: List[Dict[str, Any]], actual_links: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Calculates precision, recall, and F1 for traceability links.
    predicted_links: list of dictionaries (from TraceabilityAgent)
    actual_links: list of tuples (source_id, target_id) (from KG inference)
    """
    
    # Convert predicted_links (list of dicts) to a set of hashable tuples (source_id, target_id, relation_type)
    processed_predicted_links = []
    for link_dict in predicted_links:
        source_id = link_dict.get('source_id')
        target_id = link_dict.get('target_id')
        relation_type = link_dict.get('relation_type')
        if source_id and target_id and relation_type:
            processed_predicted_links.append((str(source_id), str(target_id), str(relation_type))) # Ensure strings for consistency
    predicted_set = set(processed_predicted_links)

    # Convert actual_links (which are already tuples, but might need relation_type to match)
    # The actual_trace_links_kg from main.py is currently (req_node_id, code_node_id).
    # To compare apples-to-apples, we either need to:
    # 1. Simplify predicted_set to (source_id, target_id)
    # 2. Enrich actual_set to include a dummy/inferred relation_type
    
    # For now, let's simplify predicted_set for direct (source, target) comparison
    # This might lose some granularity but allows comparison with current actual_trace_links_kg
    predicted_set_simplified = set([(link[0], link[1]) for link in processed_predicted_links])


    # Prepare actual_set (which is already a list of tuples from KG)
    actual_set = set(actual_links)

    if not actual_set and not predicted_set_simplified: # Use simplified for consistency
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not actual_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0} if predicted_set_simplified else {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_set_simplified:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(predicted_set_simplified.intersection(actual_set))
    false_positives = len(predicted_set_simplified.difference(actual_set))
    false_negatives = len(actual_set.difference(predicted_set_simplified))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

# --- Test Execution Simulation ---
@contextlib.contextmanager
def captured_output():
    """Context manager to capture stdout and stderr."""
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield new_stdout, new_stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
# Helper for pytest-like assertions within the simulated context
def _pytest_assert_raises(exc_type, func, *args, **kwargs):
    """
    Simulates pytest.raises for the exec environment.
    Checks if a specific exception is raised by a function call.
    """
    try:
        func(*args, **kwargs)
        raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised.")
    except exc_type as e:
        # Expected exception was raised
        pass
    except Exception as e:
        # A different unexpected exception was raised
        raise AssertionError(f"Expected {exc_type.__name__} but caught {type(e).__name__}: {e}")

def simulate_test_execution(generated_code: str, generated_test_code: str) -> Dict[str, Any]:
    """
    Simulates running pytest-like tests against generated code in a basic sandboxed way.
    Attempts to count lines covered and assertions passed.
    """
    if not generated_code.strip() or not generated_test_code.strip():
        return {"test_coverage": 0.0, "test_pass_rate": 0.0, "raw_output": "No code or tests to execute."}

    # Generate unique module names to avoid conflicts in exec context
    module_id = uuid.uuid4().hex
    code_module_name = f"__temp_gen_code_{module_id}"
    test_module_name = f"__temp_test_code_{module_id}"

    exec_globals = {
        '__builtins__': __builtins__,
        'np': np, 'pd': pd, 're': re, 'os': os, 'sys': sys,
        'AssertionError': AssertionError, # Ensure AssertionError is available
        'Exception': Exception, # Ensure Exception is available
        # Provide the helper function to the exec context
        'pytest_raises': _pytest_assert_raises
    }
def simulate_test_execution(generated_code: str, generated_test_code: str) -> Dict[str, Any]:
    """
    Simulates running pytest-like tests against generated code in a basic sandboxed way.
    Attempts to count lines covered and assertions passed.
    """
    if not generated_code.strip() or not generated_test_code.strip():
        return {"test_coverage": 0.0, "test_pass_rate": 0.0, "raw_output": "No code or tests to execute."}

    # st.write("Simulating test execution (simplified)...")
    
    # Generate unique module names to avoid conflicts in exec context
    module_id = uuid.uuid4().hex
    code_module_name = f"__temp_gen_code_{module_id}"
    test_module_name = f"__temp_test_code_{module_id}"

    # Create dummy files for dynamic import simulation if needed, but exec works for simple cases
    # For robust sandboxing, consider `exec(code, {'__builtins__': {}})` but it's too restrictive for many LLM-generated codes.
    
    exec_globals = {
        '__builtins__': __builtins__,
        'np': np, 'pd': pd, 're': re, 'os': os, 'sys': sys, # Provide common modules
        # Custom assertion for simpler testing (equivalent to pytest.raises)
        'pytest_raises': lambda exc_type, func, *args, **kwargs: _pytest_assert_raises(exc_type, func, *args, **kwargs)
    }

    test_results_list = [] # Store (test_name, success, output_log, covered_lines_in_code)
    coverage_lines_count = 0
    total_code_lines = len(generated_code.splitlines())
    covered_lines_set = set() # Track actual lines executed in generated code

    try:
        # 1. Execute generated code to make its definitions available
        with captured_output() as (code_stdout, code_stderr):
            exec(generated_code, exec_globals)
        
        # Merge captured output from code execution
        code_exec_log = code_stdout.getvalue() + code_stderr.getvalue()
        
        # 2. Iterate through and execute test functions
        test_func_signatures = re.findall(r'(def test_.*?)\(', generated_test_code)
        num_tests_detected = len(test_func_signatures)

        if num_tests_detected == 0:
            return {"test_coverage": 0.0, "test_pass_rate": 0.0, "raw_output": "No tests detected in generated test code."}

        for test_func_signature in test_func_signatures:
            test_func_name = test_func_signature.split('def ')[1].strip()

            # Create a separate execution context for each test to isolate
            test_exec_context = exec_globals.copy()
            
            # Helper for pytest-like assertions within the simulated context
            def simulated_assert(condition, message="Assertion failed"):
                if not condition:
                    raise AssertionError(message)
            test_exec_context['assert_that'] = simulated_assert # Custom helper for tests

            # Attempt to extract test function body and execute
            test_func_body_match = re.search(f"{re.escape(test_func_signature)}\\s*:\\s*\\n(.*?)(?:\\ndef |\\nclass |\\Z)", generated_test_code, re.DOTALL)
            if not test_func_body_match:
                test_results_list.append((test_func_name, False, "Could not extract test function body.", 0))
                continue

            test_func_body = test_func_body_match.group(1)
            # Indent test body for proper function definition
            indented_test_body = "\n".join(["    " + line for line in test_func_body.splitlines() if line.strip()])
            full_test_func_code = f"{test_func_signature}):\n{indented_test_body}"

            try:
                # Compile and execute the individual test function
                exec(full_test_func_code, test_exec_context)
                
                # If test function exists in context, call it
                if test_func_name in test_exec_context:
                    with captured_output() as (test_stdout, test_stderr):
                        # Simple line coverage estimation for the generated code
                        # This is a VERY crude heuristic: check if any function from `generated_code` was called
                        # In reality, this requires tracing tools (like `coverage.py`).
                        # For this simulation, we'll assume calling the test function implies
                        # some interaction with the generated code.
                        
                        # --- Crude "Coverage" Trace ---
                        # We cannot do real line-by-line coverage with exec without external tools.
                        # Instead, we'll check if the test function calls any functions defined in `generated_code`.
                        # This implies that a test that passes and calls a code function "covers" it.
                        initial_kg_nodes = set(exec_globals.keys()) # Names defined by generated code
                        test_exec_context[test_func_name]() # Run the test!
                        # After test, check if specific functions/methods were accessed.
                        # For a very rough heuristic: any function from original code called by name is 'covered'
                        
                        # This part is highly speculative and needs context specific to generated code.
                        # For now, if the test runs without error, it contributes to covered lines.
                        
                        covered_lines_set.update(range(1, total_code_lines + 1)) # If test passes, assume full coverage for now
                        # More realistically: LLM can predict coverage based on test code.
                        # For robustness, we will keep it simple here.
                    
                    test_results_list.append((test_func_name, True, test_stdout.getvalue() + test_stderr.getvalue(), total_code_lines)) # For now, assume full coverage if passes

                else:
                    test_results_list.append((test_func_name, False, "Test function definition missing or not callable.", 0))

            except AssertionError as ae:
                test_results_list.append((test_func_name, False, f"Assertion Error: {ae}", 0))
            except Exception as e:
                test_results_list.append((test_func_name, False, f"Runtime Error during test: {e}", 0))
        
        # Calculate final metrics
        num_passed_tests = sum(1 for _, success, _, _ in test_results_list if success)
        pass_rate = (num_passed_tests / num_tests_detected) * 100 if num_tests_detected > 0 else 0.0
        
        # Coverage: A very crude estimate based on passed tests.
        # If any test passes, we might claim some base coverage.
        # A more realistic estimate: LLM-predicted coverage based on prompt.
        # For a more robust simulation, count unique lines executed or referenced.
        # Here, if pass rate > 0, assume minimum coverage.
        test_coverage = pass_rate * 0.9 + (10 if pass_rate > 0 else 0) # Heuristic
        test_coverage = min(100.0, test_coverage) # Cap at 100%

        full_log = code_exec_log + "\n\n--- Test Results ---\n" + "\n".join([
            f"Test '{name}': {'PASSED' if success else 'FAILED'}\nOutput: {out}"
            for name, success, out, _ in test_results_list
        ])

        return {"test_coverage": test_coverage, "test_pass_rate": pass_rate, "raw_output": full_log}

    except SyntaxError as se:
        return {"test_coverage": 0.0, "test_pass_rate": 0.0, "raw_output": f"Syntax Error in generated code/tests: {se}"}
    except Exception as e:
        return {"test_coverage": 0.0, "test_pass_rate": 0.0, "raw_output": f"Error during simulated execution: {e}"}

# --- System Metrics ---
def calculate_system_metrics(start_time: float, end_time: float,
                             num_agents: int, num_kg_nodes: int, num_kg_edges: int) -> Dict[str, Any]:
    """
    Calculates system-level metrics like latency and (simulated) scalability/distributed performance.
    """
    latency = end_time - start_time
    
    # Scalability heuristic: penalize for more agents, larger KG
    # The 'distributed' aspect implies overhead, which is captured by higher cost.
    # Higher metric value means better scalability.
    scalability_metric = (1000 / (num_agents * np.log(num_kg_nodes + 1) + np.log(num_kg_edges + 1) + 1e-6))
    scalability_metric = max(0.0, min(100.0, scalability_metric * 0.1)) # Clamp to 0-100 and scale

    # Distributed performance metric (simulated)
    # This metric should reflect efficiency under distributed load.
    # It will inversely relate to latency and directly to scalability.
    distributed_performance_metric = (scalability_metric * 0.5) + (1000 / (latency + 1e-6) * 0.05)
    distributed_performance_metric = max(0.0, min(100.0, distributed_performance_metric)) # Clamp to 0-100

    return {
        "latency_s": latency,
        "scalability_metric": scalability_metric,
        "distributed_performance_metric": distributed_performance_metric
    }