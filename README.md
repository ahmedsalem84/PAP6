# MA-GraphRAG-OSE: Closed-Loop Multi-Agent GraphRAG for Autonomous Software Engineering

MA-GraphRAG-OSE is a research framework for **quality-driven, iterative software engineering using multi-agent Large Language Models (LLMs), Graph Retrieval-Augmented Generation (GraphRAG), and metaheuristic optimization**.

The framework combines specialized software engineering agents with a repository-derived Knowledge Graph (KG). A supervisory optimization loop evaluates generated artifacts against quantitative software-quality objectives and translates continuous optimizer parameters into discrete natural-language prompt constraints. These constraints guide subsequent agent cycles toward improved software artifacts.

The central contribution is the **Continuous-to-Discrete Translation Engine**, denoted \(T(\mathbf{p})\), which connects mathematical optimization with prompt-level control. Rather than updating the neural-network weights of the underlying LLMs, the framework searches a low-dimensional parameter space and deterministically translates candidate parameters into prompt directives.

The study evaluates the framework using the Flask repository, PROMISE JM1 defect data, and the Python subset of CodeSearchNet. It also examines internal and external baselines, six metaheuristic optimizers, objective-independent software-quality metrics, expert assessments, and distributed scalability.

**Research repository:** [https://github.com/ahmedsalem84/PAP6](https://github.com/ahmedsalem84/PAP6)

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [Key Contributions](#2-key-contributions)
3. [Framework Architecture](#3-framework-architecture)
4. [Continuous-to-Discrete Translation](#4-continuous-to-discrete-translation)
5. [Experimental Design](#5-experimental-design)
6. [Datasets and Evaluation Targets](#6-datasets-and-evaluation-targets)
7. [Models and Experimental Configuration](#7-models-and-experimental-configuration)
8. [Baselines and Ablation Studies](#8-baselines-and-ablation-studies)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Installation and Execution](#10-installation-and-execution)
11. [Reproducibility and Results](#11-reproducibility-and-results)
12. [Limitations and Responsible Interpretation](#12-limitations-and-responsible-interpretation)
13. [Repository Structure](#13-repository-structure)
14. [Citation](#14-citation)
15. [License](#15-license)
16. [Contact](#16-contact)

---

## 1. Research Overview

Standalone LLMs can generate code and other software artifacts, but their outputs may not consistently satisfy repository-specific constraints, quantitative quality objectives, or cross-component dependencies. Multi-agent systems distribute software engineering tasks across specialized roles, while GraphRAG provides access to structured project context. Nevertheless, neither capability alone guarantees systematic improvement against measurable engineering objectives.

MA-GraphRAG-OSE addresses this problem through a closed-loop workflow:

1. **Repository grounding:** Software artifacts and their relationships are represented in a Knowledge Graph.
2. **Contextual retrieval:** GraphRAG retrieves relevant project information for the agents.
3. **Collaborative generation:** Specialized agents generate or revise software artifacts.
4. **Quantitative evaluation:** The resulting artifacts are evaluated using a composite objective and relevant software-quality metrics.
5. **Metaheuristic search:** An optimizer proposes candidate control parameters.
6. **Prompt translation:** The translation engine maps those parameters into natural-language constraints.
7. **Iterative refinement:** The updated directives guide the next generation cycle.

The framework is intended to study whether this combination improves software artifacts relative to open-loop and non-optimized configurations.

### Research objectives

The primary objective is **autonomous, quality-driven iterative refinement of software artifacts**.

Secondary objectives investigate:

- Defect prediction using labeled software metrics.
- Requirement-to-code traceability.
- The contribution of graph-based retrieval to structural grounding.
- Robustness of the continuous-to-discrete prompt translation.
- Comparative behavior of metaheuristic optimizers.
- External software-quality and held-out benchmark performance.
- Distributed execution cost and scalability.
- Human expert assessment of generated artifacts.

These objectives are evaluated separately because improvements in the primary optimization target do not automatically establish improvements in every secondary task.

---

## 2. Key Contributions

### 2.1 Closed-loop multi-agent software engineering

The framework integrates specialized agents into an iterative workflow. Generated artifacts are evaluated and revised instead of being accepted solely on the basis of a single generation.

### 2.2 Graph-based repository grounding

A dynamic Knowledge Graph represents relationships among software artifacts. GraphRAG retrieval provides relevant structural context to agents, helping reduce unsupported project-specific claims and inconsistencies.

Graph grounding is a mitigation strategy, not a guarantee that hallucinations or incorrect outputs are eliminated.

### 2.3 Continuous-to-Discrete Translation Engine

The translation engine \(T(\mathbf{p})\) maps continuous optimizer parameters into discrete natural-language prompt constraints. The manuscript specifies the mapping algorithm and thresholds and evaluates a graded variant, \(T_g\).

The design makes it possible to explore prompt-control settings with numerical optimizers without treating prompt text itself as an unconstrained search space.

### 2.4 Metaheuristic optimization

Six optimizers are examined:

- Sparrow Search Algorithm (SSA)
- Grey Wolf Optimizer (GWO)
- Particle Swarm Optimization (PSO)
- Whale Optimization Algorithm (WOA)
- Genetic Algorithm (GA)
- Firefly Algorithm (FA)

Their performance is compared under the experimental configurations described in the paper.

### 2.5 Multi-level empirical evaluation

The evaluation includes internal architectural baselines, external multi-agent or iterative baselines, held-out software-quality measures, benchmark tasks, expert ratings, and distributed scaling measurements.

The validity of each metric depends on its dataset, evaluation target, and experimental sampling unit.

---

## 3. Framework Architecture

The framework can be understood as five connected layers.

### Layer 1: Software data sources

The study uses software repositories, source code, repository history, documentation, issue information, and benchmark data as appropriate to each experiment.

### Layer 2: Knowledge Graph

The graph represents software entities and relationships, including structural and repository-level relationships where available. Its construction and contents must be consistent with the dataset and experiment being performed.

### Layer 3: GraphRAG retrieval

The retrieval component combines relevant graph context with agent prompts. This provides structured context for software engineering decisions and multi-hop retrieval.

### Layer 4: Specialized LLM agents

The framework assigns software engineering responsibilities to agent roles such as:

- Requirements
- Architecture
- Code generation
- Testing
- Conflict detection
- Traceability

The specific roles and model assignments are defined by the experimental configuration.

### Layer 5: Optimization and feedback

The optimization engine evaluates the current candidate, searches the control-parameter space, translates the selected parameters into prompt constraints, and directs subsequent agent cycles.

The optimizer controls the **prompt-level behavior** of the system; it does not perform gradient-based fine-tuning of the underlying LLM weights.

---

## 4. Continuous-to-Discrete Translation

The central mechanism maps a continuous parameter vector \(\mathbf{p}\) into discrete prompt directives:

\[
T:\mathbf{p}\longrightarrow\text{prompt constraints}.
\]

The optimizer searches the continuous parameter space, while the translation engine converts a candidate into instructions that the agents can use.

The manuscript also examines a graded translation variant, \(T_g\), alongside the threshold-based mapping. This allows the study to investigate the sensitivity of results to threshold settings, information loss, and translation granularity.

The translation mechanism is evaluated through:

- Threshold sensitivity analyses.
- Alternative translation variants.
- Exhaustive enumeration of the specified threshold combinations.
- Random-search and fixed-parameter controls.
- Robustness checks under changes to input phrasing and directive ordering.
- Analysis of variation across LLM runs.

For exact threshold values, equations, and the translation algorithm, consult the corresponding methodology section of the manuscript and the matching implementation. The implementation and published algorithm should agree exactly.

---

## 5. Experimental Design

The study evaluates architectural contributions and optimizer behavior using internal baselines, external comparisons, and targeted analyses.

### 5.1 Internal baselines

| Configuration | Description |
|---|---|
| SA-LLM | Single Code Agent without GraphRAG retrieval or optimization feedback. |
| MA-LLM | Multi-agent workflow without GraphRAG retrieval or iterative optimization. |
| MA-GraphRAG | Multi-agent workflow with GraphRAG retrieval but without iterative optimization. |
| MA-GraphRAG-OSE | Full framework with GraphRAG and iterative metaheuristic optimization. |

The full-system configuration uses five outer optimization cycles in the described experimental design. Comparisons should preserve the stated evaluation budgets and settings wherever applicable.

### 5.2 External baselines

The manuscript compares the proposed framework with:

- **MetaGPT:** A role-based multi-agent software development framework.
- **ChatDev:** A multi-agent software development framework based on agent communication.
- **Self-Refine:** An iterative refinement approach using self-feedback rather than the proposed quantitative metaheuristic controller.

External comparisons should be interpreted in light of each baseline's implementation, model configuration, compute budget, and applicable evaluation metric.

### 5.3 Optimizer controls

The study also considers:

- A fixed-parameter control.
- Random search over the optimizer parameter space.
- Exhaustive enumeration of the specified discrete prompt combinations.

These controls help distinguish the effect of active optimization from iterative generation alone.

### 5.4 Statistical analysis

The manuscript describes repeated experiments, assumption checks, effect sizes, confidence intervals, and multiple-comparison corrections.

The described protocol includes:

- Shapiro–Wilk tests for normality.
- Levene's test for variance homogeneity.
- Kruskal–Wallis tests and Dunn post-hoc comparisons for suitable multi-group analyses.
- Mann–Whitney \(U\) tests for independent samples.
- Wilcoxon signed-rank tests for paired samples.
- Holm correction for families of hypothesis tests.
- Hedges' \(g\), Cliff's \(\delta\), and bootstrap confidence intervals where applicable.

The manuscript describes 30 repetitions per configuration and dataset. A pooled sample size of 90 represents 30 repetitions across three datasets, not 90 independent software tasks. Analyses must use metric-specific sample sizes and account for the actual experimental design.

**Reproducibility requirement:** reported statistics should be regenerated from the run-level data and analysis scripts before being treated as independently verified.

---

## 6. Datasets and Evaluation Targets

The study uses three datasets, with different evaluation targets for each.

### 6.1 Flask repository

**Source:** [Pallets Flask repository](https://github.com/pallets/flask)

Flask provides a real software repository for examining code artifacts, repository structure, graph-based retrieval, and requirement-to-code traceability.

The manuscript describes a manually constructed traceability oracle and a repository-based defect-labeling procedure. These procedures should be reproduced from the documented annotation and preprocessing artifacts.

### 6.2 PROMISE JM1

**Source:** [OpenML PROMISE JM1 dataset](https://www.openml.org/d/1053)

JM1 provides module-level software metrics and defect labels. It is used for defect-prediction evaluation.

JM1 does not supply the original source code or requirement-to-code relationships required to independently evaluate source-code maintainability or requirement traceability. Consequently, JM1 defect labels should not be treated as ground truth for those other tasks.

The manuscript reports a cleaned dataset of 7,720 modules and an 80/20 stratified split. The dataset version, filtering rules, split, and absence of held-out-label leakage should be verified against the released preprocessing code and run manifests.

### 6.3 CodeSearchNet — Python subset

**Source:** [CodeSearchNet repository](https://github.com/github/CodeSearchNet)

CodeSearchNet provides function-level code and docstring pairs for software understanding and retrieval tasks.

The manuscript describes an evaluation sample drawn from the official Python test partition. Function–docstring pairs serve as benchmark reference labels; they should not be described as an independently created human oracle.

### Ground-truth independence

Evaluation targets differ by task:

- **JM1:** Labeled module-level defect data.
- **Flask:** Repository-derived evaluations and manually constructed traceability annotations where applicable.
- **CodeSearchNet:** Benchmark function–docstring reference pairs.

Graph-derived predictions should not be evaluated against labels constructed from the same graph in a way that makes the evaluation circular. The relevant ground-truth construction and data splits should therefore be documented and reproducible.

---

## 7. Models and Experimental Configuration

### 7.1 Local LLMs

The manuscript describes local inference using Ollama with the following 7B-parameter model variants:

| Model identifier | Role in the study |
|---|---|
| `llama2:7b-chat` | General conversational and reasoning tasks. |
| `mistral:7b-instruct-v0.2` | Instruction following and general reasoning. |
| `codellama:7b-instruct` | Code generation and testing-related tasks. |

The described setup uses 4-bit quantization.

The experiment includes homogeneous assignments, in which agents share a model backbone, and heterogeneous assignments, in which model roles are specialized.

Exact model digests, runtime versions, and library versions should be taken from the experiment's environment manifest rather than inferred from model names.

### 7.2 Generation settings

The manuscript specifies the following settings:

| Setting | Reported configuration |
|---|---|
| Temperature | 0.2 |
| Top-p | 0.9 |
| Context window | 4,096 tokens |
| Optimizer population | 20 |
| Internal optimizer iterations | 30 per cycle |
| Outer optimization cycles | 5 for the full system |
| Repetitions | 30 per configuration and dataset |

These values describe the reported experimental design. For exact reproduction, use the actual configuration files, seed schedules, model digests, and run manifests associated with the experiments.

### 7.3 Hardware and distributed setup

The manuscript describes a distributed evaluation using Ray and up to eight GPU workers across four physical nodes. The reported setup includes NVIDIA RTX 3090 GPUs, Ubuntu 22.04 LTS, and Ray 2.9.0.

These details describe the study's reported environment; they are not minimum hardware requirements for every component of the framework. A smaller setup may be suitable for testing individual components but should not be expected to reproduce the reported distributed performance.

---

## 8. Baselines and Ablation Studies

The principal comparisons examine the contributions of multi-agent collaboration, graph-based retrieval, and optimization.

| Research question | Main analysis |
|---|---|
| RQ1 | Comparison of internal configurations and external baselines. |
| RQ2 | Effectiveness and robustness of continuous-to-discrete prompt translation. |
| RQ3 | Optimizer convergence, rankings, and parameter sensitivity. |
| RQ4 | Contribution of GraphRAG retrieval to structural grounding. |
| RQ5 | Repository-scale behavior, latency, throughput, and distributed scaling. |
| RQ6 | Human expert assessment of generated artifacts. |

The manuscript also evaluates translation variants, fixed-parameter controls, random search, and exhaustive enumeration.

For a valid reproduction, preserve the baseline definitions and avoid changing multiple experimental variables at once unless the experiment is explicitly designed to measure their joint effect.

---

## 9. Evaluation Metrics

### 9.1 Primary software-quality metrics

The manuscript describes a composite objective using metrics such as:

- Maintainability Index (MI).
- Cyclomatic Complexity (CC).
- Test Pass Rate (TPR).
- Conflict count.
- Requirement Traceability F1-score.
- Defect-prediction F1-score where valid labels are available.

The objective's component definitions, normalization, and weights should be taken from the methodology and implementation.

Cyclomatic complexity must be interpreted at the same aggregation level as the objective function. A module-level sum and a per-function threshold are not interchangeable.

### 9.2 Objective-independent held-out metrics

The evaluation includes measures that are not directly optimized by the composite objective:

- Pylint score.
- Bandit security findings per thousand lines of code.
- Mutation score.
- Hidden-unit-test pass rate.
- HumanEval.
- MBPP.
- A subset of SWE-bench Lite.

These metrics are intended to test whether improvements extend beyond the objective used to guide optimization. They should not be assumed to measure the same construct, and benchmark-specific results should be reported separately.

### 9.3 Human expert assessment

The manuscript describes an expert study involving eight software engineering professionals, with matched artifact pairs and separate maintainability calibration.

Reported analyses include ordinal agreement and reliability measures, including Krippendorff's alpha and ICC, together with appropriate paired comparisons.

Human-study claims must match the actual study records, participant ratings, consent documentation, and applicable institutional ethics determination.

### 9.4 Scalability metrics

The distributed study measures:

- Workflow latency.
- Throughput.
- Memory consumption.
- Serialization and communication overhead.
- Parallel speedup and efficiency.
- Behavior as graph size, agent count, and optimization budget increase.

Scalability findings should be interpreted within the tested hardware and software configuration.

---

## 10. Installation and Execution

The following commands provide a conventional setup for a Python-based repository. **Confirm the actual entry-point and dependency filenames in the repository before using them.**

### 10.1 Clone the repository

```bash
git clone https://github.com/ahmedsalem84/PAP6.git
cd PAP6
```

### 10.2 Create a virtual environment

Python version requirements should be taken from the project's dependency files and environment manifest.

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 10.3 Install dependencies

If the repository contains a maintained `requirements.txt`, install the listed packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the environment versions recorded for the experiments when attempting exact reproduction. Installing the newest versions of all packages may produce a different runtime environment.

### 10.4 Configure Ollama

Install Ollama using the [official installation instructions](https://ollama.com/download).

Download the model variants used in the experiment, if they are available for your installed Ollama version:

```bash
ollama pull llama2:7b-chat
ollama pull mistral:7b-instruct-v0.2
ollama pull codellama:7b-instruct
```

Check the installed models:

```bash
ollama list
```

Ensure that the Ollama service is running before executing code that uses local inference. The exact model tags and digests should match the experimental manifest.

### 10.5 Run the project

Identify the documented entry point in the repository before starting the application.

If the repository contains a Streamlit entry point named `app.py`, launch it with:

```bash
streamlit run app.py
```

Otherwise, use the execution command documented for the actual entry point.

Do not assume that the existence of this README guarantees a particular application script, user interface, dataset downloader, or analysis script is present.

### 10.6 Configure an experiment

For a reproducible run, record at least:

- Dataset and dataset version.
- Preprocessing configuration and split manifest.
- Baseline or full-system configuration.
- LLM names and immutable model digests.
- Agent-to-model assignments.
- Optimizer and its hyperparameters.
- Translation method and thresholds.
- Random seeds.
- Generation settings.
- Number of outer cycles and repetitions.
- Hardware and software environment.
- Output paths and run identifiers.

Do not modify the evaluation split or select optimizer settings using held-out labels.

---

## 11. Reproducibility and Results

A reproducible release should contain the implementation and the artifacts needed to trace each reported number back to its source.

### Recommended reproducibility artifacts

| Artifact | Purpose |
|---|---|
| Environment manifest | Python, package, operating-system, Ollama, and model versions. |
| Configuration files | Optimizer settings, model assignments, generation parameters, and experiment options. |
| Seed and run manifests | Identification of repetitions and their experimental conditions. |
| Dataset and split manifests | Dataset versions, preprocessing, train/test splits, and filtering rules. |
| Prompt templates | System prompts, agent prompts, and \(T(\mathbf{p})\)/\(T_g\) directives. |
| Raw run-level results | Per-run measurements, outputs, failures, and relevant logs. |
| Statistical analysis scripts | Recalculation of summary statistics, confidence intervals, effect sizes, and adjusted \(p\)-values. |
| Ground-truth documentation | Label construction, annotation protocols, adjudication, and evaluation eligibility. |
| Figure-generation scripts | Reproduction of figures and tables from saved results. |

These are recommended and manuscript-described reproducibility materials; their presence in the public repository must be confirmed individually.

### Recommended workflow for validating reported results

1. Verify the environment and model digests.
2. Validate dataset versions, preprocessing, and held-out splits.
3. Run the documented configurations with the recorded seeds.
4. Preserve raw outputs and failures.
5. Recompute metrics from raw run-level results.
6. Reproduce confidence intervals, effect sizes, and multiplicity-adjusted tests.
7. Rebuild the tables and figures from the analysis scripts.
8. Compare the resulting artifacts with the manuscript.

If a result cannot be reproduced, investigate the discrepancy and correct the reported value where necessary.

### Interpreting reported findings

The manuscript reports improvements in selected software-quality metrics and external evaluation measures, along with differences between optimizers and limitations in distributed scaling.

The findings are conditional on the evaluated models, datasets, experimental protocols, and hardware. They should not be interpreted as proof that the framework will improve every software project, eliminate hallucinations, guarantee secure code, or operate efficiently at arbitrary repository sizes.

---

## 12. Limitations and Responsible Interpretation

The framework and its evaluation have important limitations.

- **Model dependence:** Results may differ with other LLM families, versions, quantization settings, and inference runtimes.
- **Translation discretization:** The continuous-to-discrete mapping can lose information and is sensitive to thresholds and directive design.
- **Evaluation-target dependence:** JM1, Flask, and CodeSearchNet support different claims; their metrics are not interchangeable.
- **Ground-truth limitations:** Repository-derived labels, manually annotated links, and benchmark reference pairs have different independence and reliability properties.
- **Statistical dependence:** Repeated runs do not necessarily represent independent software tasks. Pooled analyses must respect the actual sampling structure.
- **Compute and latency:** Iterative agent calls and graph retrieval introduce additional computation and communication overhead.
- **Scalability:** The reported experiments do not establish unlimited graph capacity or linear scaling.
- **Security and correctness:** Automated metrics and tests cannot guarantee the absence of vulnerabilities or semantic errors.
- **Human validation:** Conclusions from a limited expert panel should not be generalized to all developers or software engineering contexts.
- **Reproducibility:** Empirical conclusions depend on the availability and integrity of run-level data, scripts, configurations, and ground-truth artifacts.

The system is a research framework for quality-driven software engineering, not a replacement for professional code review, security testing, or human approval of production changes.

---

## 13. Repository Structure

The exact directory structure should reflect the files actually present in the repository. A useful organization for this project is:

```text
PAP6/
├── README.md
├── requirements.txt                 # If maintained by the project
├── config/                          # Experiment configurations, if present
├── src/                              # Framework implementation, if present
│   ├── agents/
│   ├── graph/
│   ├── retrieval/
│   ├── optimization/
│   └── evaluation/
├── data/                             # Dataset instructions/manifests;
│                                     # large datasets may be external
├── experiments/                      # Experiment entry points, if present
├── results/                          # Saved outputs, if distributed
├── scripts/                          # Analysis/figure-generation scripts
├── docs/
│   └── figures/                      # Manuscript figures, if distributed
└── supplementary/                    # Supplementary statistics and protocols
```

This is an organizational template, **not a claim that every directory or file already exists**. Update it to match the public repository before release.

Large datasets, generated artifacts, and raw model outputs may require separate downloads or repository releases. If so, provide stable links, version identifiers, and checksums.

---

## 14. Citation

If you use MA-GraphRAG-OSE in academic work, please cite the associated manuscript. Replace the provisional entry below with the final bibliographic details once publication information is available.

```bibtex
@unpublished{salem_magraph_rag_ose,
  author = {Salem, Ahmed and Hassan, Nada M.},
  title  = {Closed-Loop Multi-Agent GraphRAG for Autonomous Software Engineering via Metaheuristic Optimization},
  note   = {Manuscript under review}
}
```

If the manuscript is published, update this citation with the verified journal, year, volume, pages, and DOI.

---

## 15. License

A license has not been specified here. Check the repository's actual license file before copying, modifying, redistributing, or using the implementation commercially.

If no license is present, do not assume that the source code is released under an open-source license.

---

## 16. Contact

For questions concerning the research, please contact:

**Ahmed Salem**  
Email: [a.salem@aast.edu](mailto:a.salem@aast.edu)  
ORCID: [0000-0002-0456-2276](https://orcid.org/0000-0002-0456-2276)

**Repository:** [https://github.com/ahmedsalem84/PAP6](https://github.com/ahmedsalem84/PAP6)

---

### Final note

MA-GraphRAG-OSE investigates how graph-based repository grounding, specialized LLM agents, and metaheuristic prompt control can be combined in a closed-loop software engineering workflow. The repository should make the implementation, experimental settings, evaluation targets, and analysis artifacts sufficiently transparent for other researchers to inspect and reproduce the reported findings.
