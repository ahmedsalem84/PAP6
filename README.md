# PAP6

# MA-GraphRAG-OSE: An Adaptive Multi-Agent GraphRAG Framework for Autonomous Software Engineering via Metaheuristic Optimization

This repository contains the full implementation and experimental setup for the **MA-GraphRAG-OSE** (Multi-Agent Graph Retrieval-Augmented Generation for Optimized Software Engineering) framework. This novel cybernetic framework integrates specialized LLM-powered agents, a dynamic Knowledge Graph (KG) with GraphRAG retrieval, and an iterative metaheuristic optimization engine to autonomously generate and refine software artifacts.

The core innovation lies in its closed-loop feedback mechanism, which translates quantitative software quality metrics (like maintainability, cyclomatic complexity, and test coverage) into targeted natural language prompt constraints. This "math-to-prompt" translation enables the agent system to iteratively self-correct and structurally evolve generated artifacts towards industry-ready standards.

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Key Features](#2-key-features)
3.  [Framework Architecture](#3-framework-architecture)
4.  [Experimental Setup](#4-experimental-setup)
    *   [Hardware Requirements](#hardware-requirements)
    *   [Software Requirements](#software-requirements)
    *   [Installation](#installation)
    *   [LLM Configuration](#llm-configuration)
    *   [Datasets](#datasets)
    *   [Baseline Configurations](#baseline-configurations)
    *   [Metaheuristic Optimizers](#metaheuristic-optimizers)
    *   [Evaluation Metrics](#evaluation-metrics)
5.  [Usage and Reproducibility](#5-usage-and-reproducibility)
    *   [Step-by-Step Execution Guide](#step-by-step-execution-guide)
    *   [Running Specific Configurations](#running-specific-configurations)
    *   [Collecting Results](#collecting-results)
6.  [Project Structure](#6-project-structure)
7.  [Citation](#7-citation)
8.  [License](#8-license)
9.  [Contact](#9-contact)

---

## 1. Introduction

Modern software development faces significant challenges due to increasing complexity, distributed teams, and massive codebases. While Large Language Models (LLMs) offer powerful AI-assisted development tools, they typically operate as static, "open-loop" systems, lacking deep structural context and autonomous self-correction capabilities against rigorous engineering standards.

MA-GraphRAG-OSE addresses this limitation by establishing a cybernetic control loop:
*   **Multi-Agent System:** A swarm of specialized AI agents (Requirement, Architecture, Code, Testing, Conflict, Traceability) collaborate on software development tasks.
*   **Dynamic Knowledge Graph (GraphRAG):** Agents are grounded in a dynamic Knowledge Graph that stores interconnected software artifacts (ASTs, commits, issues, requirements, etc.), enabling multi-hop reasoning and eliminating architectural hallucinations.
*   **Metaheuristic Optimization Engine:** A closed-loop optimizer continuously evaluates generated artifacts against a composite mathematical objective function (e.g., maintainability, cyclomatic complexity, test coverage). It then dynamically translates optimal parameters into targeted natural language prompt constraints for subsequent agent cycles, forcing iterative refinement and structural evolution.

This framework demonstrates significant improvements in defect detection, code maintainability, and test pass rates, validated by extensive empirical studies and human expert assessments.

---

## 2. Key Features

*   **Closed-Loop Cybernetic AI:** A self-regulating system that autonomously refines software artifacts based on quantitative quality metrics.
*   **Mathematical-to-Behavioral Translation:** Translates optimizer-derived parameters into natural language prompt directives, guiding LLM agent behavior.
*   **Deep Structural Grounding:** Utilizes a dynamic Knowledge Graph with GraphRAG for contextual, multi-hop reasoning over complex codebases.
*   **Specialized Multi-Agent Collaboration:** A team of expert LLM agents collaborates on the entire software development lifecycle (SDLC).
*   **Empirical Validation:** Extensive evaluation across real-world datasets (Flask, PROMISE JM1, CodeSearchNet) and human expert validation.
*   **Metaheuristic Optimization:** Incorporates SSA, GA, PSO, GWO, WOA, and FA for adaptive guidance.

---

## 3. Framework Architecture

The MA-GraphRAG-OSE framework is composed of five interconnected layers:

1.  **Data Sources:** Raw software artifacts (Git repositories, requirements docs, issue trackers, test cases, project documentation).
2.  **Dynamic Knowledge Graph (KG):** Structured, interlinked representation of all relevant software artifacts, populated automatically from data sources.
3.  **GraphRAG Retrieval Layer:** Empowers LLM agents to retrieve highly relevant, multi-hop information from the KG via semantic embeddings and graph traversal.
4.  **Multi-Agent LLM System:** A team of specialized LLM agents (Requirement, Architecture, Code, Testing, Conflict, Traceability) collaboratively executes SDLC tasks.
5.  **Optimization Engine:** The central nervous system employing metaheuristic algorithms to iteratively evaluate and mathematically steer agent behavior towards optimal engineering trade-offs.

A detailed overview of the framework architecture can be found in `framework.png` in the `docs` folder.

---

## 4. Experimental Setup

This section details the requirements and setup for replicating the experiments.

### Hardware Requirements

*   **GPU-enabled Workstation:** Essential for efficient local LLM inference.
*   **RAM:** Minimum 32GB RAM (64GB recommended for smoother concurrent LLM operations).
*   **Disk Space:** At least 100GB of free disk space for LLM models and dataset storage.

### Software Requirements

*   **Operating System:** Linux (Ubuntu 20.04+ recommended) or macOS. Windows Subsystem for Linux (WSL2) may work but is not officially tested for full GPU support with Ollama.
*   **Python:** Python 3.9+ (Python 3.10 recommended).
*   **Docker (Optional but Recommended for Ollama):** For easier management of Ollama LLM models.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/MA-GraphRAG-OSE.git
    cd MA-GraphRAG-OSE
    ```

2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` is updated with all necessary packages, e.g., `langchain`, `ollama`, `networkx`, `streamlit`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `faiss-cpu`, `python-dotenv`, `dataclasses`, `types-requests`, `matplotlib`, `seaborn`, `astunparse`, `radon`, `pytest`, `coverage`, `gitpython`).

### LLM Configuration

We utilize `Ollama` for local LLM inference, ensuring data privacy and controlled latency.

1.  **Install Ollama:** Follow the instructions on the [Ollama website](https://ollama.com/download).
2.  **Download LLM Models:**
    The experiments were conducted using 7B to 8B parameter instruction-tuned variants, with 4-bit quantization to accommodate typical workstation hardware.
    *   `llama2:7b-chat`
    *   `mistral:7b-instruct-v0.2`
    *   `codellama:7b-instruct`

    Download them using Ollama:
    ```bash
    ollama pull llama2:7b-chat
    ollama pull mistral:7b-instruct-v0.2
    ollama pull codellama:7b-instruct
    ollama pull nomic-embed-text # For embeddings in GraphRAG
    ```
    Ensure Ollama is running (`ollama serve`) before starting experiments.

3.  **Configure LLM Assignments:**
    The framework supports both **Homogeneous** (all agents use the same LLM) and **Heterogeneous** (agents use specialized LLMs) configurations. This is controlled via a configuration file (e.g., `config.py` or `.env`).
    *   **Homogeneous Example (all Llama2):**
        ```python
        # config.py
        LLM_CONFIG = {
            "default": "llama2:7b-chat",
            "requirement_agent": "llama2:7b-chat",
            "architecture_agent": "llama2:7b-chat",
            "code_agent": "llama2:7b-chat",
            "testing_agent": "llama2:7b-chat",
            "conflict_agent": "llama2:7b-chat",
            "traceability_agent": "llama2:7b-chat"
        }
        ```
    *   **Heterogeneous Example (Mixed - optimal configuration):**
        ```python
        # config.py
        LLM_CONFIG = {
            "default": "mistral:7b-instruct-v0.2", # Fallback or for general reasoning
            "requirement_agent": "mistral:7b-instruct-v0.2",
            "architecture_agent": "mistral:7b-instruct-v0.2",
            "code_agent": "codellama:7b-instruct",
            "testing_agent": "codellama:7b-instruct", # CodeLlama often good for tests too
            "conflict_agent": "llama2:7b-chat", # General reasoning/analysis
            "traceability_agent": "llama2:7b-chat"
        }
        ```
    The `nomic-embed-text` model is always used for embeddings and is internally managed by the GraphRAG retriever.

### Datasets

The experiments utilize three real-world datasets. These will be automatically downloaded and parsed into the knowledge graph structure by the framework.

1.  **Flask Web Framework:**
    *   **Source:** [Pallets Flask GitHub Repository](https://github.com/pallets/flask)
    *   **Purpose:** Code generation, architectural design, traceability in a realistic Python web development context.
2.  **PROMISE JM1 Dataset:**
    *   **Source:** [NASA's Metrics Data Program (MDP) JM1 dataset](https://www.openml.org/d/1053) (often available via `scikit-learn` datasets or similar, or direct download from public repositories like ApoorvaKrisna's GitHub link mentioned in the paper: `https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository`)
    *   **Purpose:** Defect prediction, impact of optimization on code quality attributes related to defects.
3.  **CodeSearchNet (Python Subset):**
    *   **Source:** [CodeSearchNet Challenge](https://github.com/github/CodeSearchNet) (Python subset)
    *   **Purpose:** Code understanding, comment generation, contextual GraphRAG retrieval for function-level tasks.

The framework expects a `data/` directory in the root. Data will be programmatically downloaded and processed into this directory.

### Baseline Configurations

The following configurations are used for comparative analysis:

1.  **SA-LLM (Single-Agent LLM Assistant):** A single `CodeAgent` only, no GraphRAG or optimization.
2.  **MA-LLM (Multi-Agent System w/o GraphRAG, w/o Optimization):** All specialized agents, but no GraphRAG for context and no optimization loop.
3.  **MA-GraphRAG (Multi-Agent System w/ GraphRAG, w/o Optimization):** All specialized agents with GraphRAG for context, but no iterative optimization loop (`num_optimization_cycles = 1`).
4.  **MA-GraphRAG-OSE (Full System):** The complete framework, including multi-agent collaboration, GraphRAG, and iterative metaheuristic optimization (`num_optimization_cycles > 1`).

### Metaheuristic Optimizers

Six distinct metaheuristic optimization algorithms are implemented and tested:

*   **SSA:** Sparrow Search Algorithm
*   **GA:** Genetic Algorithm
*   **PSO:** Particle Swarm Optimization
*   **GWO:** Grey Wolf Optimizer
*   **WOA:** Whale Optimization Algorithm
*   **FA:** Firefly Algorithm

Each optimizer is configured with:
*   `population_size = 20`
*   `max_internal_iterations = 30` (per optimization cycle)
*   The objective function is detailed in the paper's methodology section.

### Evaluation Metrics

The framework collects and evaluates a comprehensive suite of metrics:

*   **Software Quality Metrics:**
    *   Maintainability Index (MI)
    *   Cyclomatic Complexity (CC)
    *   Lines of Code (LOC)
    *   Number of Functions/Classes
    *   Comment Ratio
    *   Defect Prediction Accuracy (F1-score, Precision, Recall)
    *   Requirement Traceability Accuracy (F1-score, Precision, Recall)
    *   Test Coverage (%)
    *   Test Pass Rate (%)
    *   Conflict Count
    *   Optimization-Derived Scores (Architecture Quality, Code Complexity, Requirement Satisfaction)
*   **System Performance Metrics:**
    *   Latency (s) (total workflow cycle time)
    *   KG Node/Edge Count
    *   Scalability Metric (S, 0-100)
    *   Distributed Performance Metric (DP, 0-100)
    *   Optimizer Best Fitness

---

## 5. Usage and Reproducibility

### Step-by-Step Execution Guide

The framework is designed to be run via a Streamlit interface, allowing easy configuration and visualization.

1.  **Ensure all [Software Requirements](#software-requirements) and [LLM Configuration](#llm-configuration) steps are completed.** Specifically, Ollama should be running (`ollama serve`) and the necessary LLM models pulled.
2.  **Activate your Python virtual environment:**
    ```bash
    source venv/bin/activate
    ```
3.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    (Assuming `app.py` is your main Streamlit file for the UI.)

4.  **Access the Web Interface:** Open your web browser and navigate to the local address displayed in the terminal (usually `http://localhost:8501`).

5.  **Configure the Experiment in the UI:**
    The Streamlit UI will allow you to select:
    *   **Dataset:** Flask, PROMISE JM1, CodeSearchNet.
    *   **LLM Configuration:** Homogeneous (e.g., Llama2, Mistral) or Heterogeneous (Mixed).
    *   **Metaheuristic Optimizer:** SSA, GA, PSO, GWO, WOA, FA.
    *   **Number of Optimization Cycles:** Set to `1` for non-optimized baselines (MA-GraphRAG), or `5` (as used in the paper) for the full MA-GraphRAG-OSE.
    *   **Number of Repetitions:** For statistical robustness, set to `3` or `5` (as used in the paper, adjust based on computational budget).
    *   **Initial Software Specification:** Input your high-level requirement/problem statement for the agents.

6.  **Run the Experiment:** Click the "Start Simulation" or similar button in the Streamlit UI. The system will then:
    *   Initialize the Knowledge Graph and GraphRAG retriever.
    *   Instantiate LLM agents based on your configuration.
    *   Execute the iterative optimization cycles.
    *   Display real-time metrics, logs, and visualizations of artifact evolution and agent collaboration.

### Running Specific Configurations (for Ablation Studies)

To reproduce specific baseline or ablation study results:

*   **SA-LLM:** Select any dataset, then specify a single `CodeAgent` in your LLM configuration. Disable GraphRAG retrieval (if possible via a UI toggle, or manually adjust the `RAGRetriever` to return empty context) and set `num_optimization_cycles = 1`.
*   **MA-LLM:** Select any dataset, use the full suite of agents in `LLM_CONFIG`, but disable GraphRAG retrieval and set `num_optimization_cycles = 1`.
*   **MA-GraphRAG:** Select any dataset, use the full suite of agents, enable GraphRAG, but set `num_optimization_cycles = 1`.
*   **MA-GraphRAG-OSE:** Select any dataset, use the full suite of agents (Heterogeneous Mixed configuration with SSA is the optimal choice as per results), enable GraphRAG, and set `num_optimization_cycles = 5`.

### Collecting Results

All computed metrics, raw LLM outputs, optimizer parameters, best fitness values, and detailed workflow histories for each cycle are logged.

*   **Output Directory:** Results are saved in a structured format within a designated output directory (e.g., `results/`).
*   **File Format:** DataFrames (e.g., CSV) for quantitative metrics and JSON files for raw LLM outputs and workflow histories.
*   **Post-hoc Analysis:** Scripts for aggregating and analyzing these results (e.g., generating plots, calculating means/stds) will be provided in the `scripts/analysis` directory.

---

## 6. Project Structure

MA-GraphRAG-OSE/
├── app.py # Main Streamlit application for UI and experiment control
├── requirements.txt # Python dependencies
├── config.py # Configuration for LLM models, optimizers, etc.
├── src/
│ ├── agents/ # Definitions of specialized LLM agents
│ │ ├── base_agent.py
│ │ ├── requirement_agent.py
│ │ ├── architecture_agent.py
│ │ ├── code_agent.py
│ │ ├── testing_agent.py
│ │ ├── conflict_agent.py
│ │ └── traceability_agent.py
│ ├── kg/ # Knowledge Graph management and construction
│ │ ├── kg_builder.py # Algorithm 1 implementation
│ │ └── kg_schema.py # KG entities and relations definitions
│ ├── rag/ # GraphRAG retrieval layer implementation
│ │ ├── retriever.py # GraphRAG retriever logic
│ │ └── embeddings.py # Embedding model loading
│ ├── optimization/ # Metaheuristic optimization engine
│ │ ├── objective_function.py # Composite cost function definition
│ │ ├── ssa.py # Sparrow Search Algorithm implementation
│ │ ├── ga.py # Genetic Algorithm implementation
│ │ ├── pso.py # Particle Swarm Optimization implementation
│ │ ├── gwo.py # Grey Wolf Optimizer implementation
│ │ ├── woa.py # Whale Optimization Algorithm implementation
│ │ └── fa.py # Firefly Algorithm implementation
│ ├── utils/ # Utility functions (e.g., AST parsing, code execution sandbox, logging)
│ │ ├── code_analyzer.py
│ │ ├── sandbox_executor.py
│ │ └── logger.py
│ └── main_workflow.py # Orchestrates agent collaboration (Algorithm 2) and optimization loop
├── data/ # Directory for downloaded datasets (will be created automatically)
├── results/ # Directory for storing experiment outputs (logs, metrics, artifacts)
│ └── [timestamped_run_folders]/
├── scripts/
│ └── analysis/ # Scripts for post-hoc data analysis and visualization
├── docs/
│ ├── framework.png # Architectural overview diagram
│ ├── kg_schema.png # Knowledge Graph schema
│ ├── optimization_loop.png # Optimization feedback loop diagram
│ ├── baseline_boxplots.png # (and other result plots)
│ └── (all other figures from the paper)
└── README.md # This file

code
Code
download
content_copy
expand_less
---

7. Contact

For any questions or inquiries, please contact:

Ahmed Salem
a.salem@aast.edu
ORCID: 0000-0002-0456-2276

code
Code
download
content_copy
expand_less
