import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
import json # For loading optimizer_best_position from JSON string
from typing import List, Dict, Any
import numpy as np
import random
# --- Existing plotting functions, updated to use Plotly where applicable ---

def plot_results_dataframe(df: pd.DataFrame):
    """
    Generates a set of interactive plots from the experiment results DataFrame using Plotly.
    Includes grouped box plots for key metrics, a correlation matrix, and trade-off scatter plots.
    """
    if df.empty:
        st.warning("No data to plot yet.")
        return # Return nothing if no data

    st.subheader("Interactive Performance Metrics by Configuration")

    # Ensure numeric types for plotting
    numeric_cols = [
        'Maintainability Index', 'Cyclomatic Complexity', 'LOC', 'Num Functions', 'Num Classes',
        'Comment Ratio', 'Defect Prediction Accuracy (F1)', 'Defect Prediction Precision',
        'Defect Prediction Recall', 'Requirement Traceability Accuracy (F1)',
        'Requirement Traceability Precision', 'Requirement Traceability Recall',
        'Test Coverage', 'Test Pass Rate',
        'Architecture Quality Score', 'Code Complexity Score',
        'Requirement Satisfaction Score', 'Conflict Count',
        'Optimizer Best Fitness', 'Latency (s)', 'Total Workflow Latency (s)',
        'KG Node Count', 'KG Edge Count', 'Scalability Metric', 'Distributed Performance Metric'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Grouped Box Plots for key metrics
    metrics_to_plot_box = [
        'Maintainability Index', 'Cyclomatic Complexity', 'Defect Prediction Accuracy (F1)',
        'Requirement Traceability Accuracy (F1)', 'Test Pass Rate', 'Latency (s)',
        'Architecture Quality Score', 'Requirement Satisfaction Score', 'Conflict Count',
        'Optimizer Best Fitness'
    ]
    
    for metric in metrics_to_plot_box:
        if metric in df.columns and not df[metric].isnull().all():
            fig = px.box(df, x='LLM Config Name', y=metric, color='Optimizer',
                         title=f'{metric} by LLM Configuration and Optimizer',
                         labels={'LLM Config Name': 'LLM Config', 'value': metric},
                         hover_data=['Dataset', 'Run ID', 'Optimization Cycle'])
            fig.update_layout(xaxis_title_text='LLM Configuration (Agents)',
                              xaxis_tickangle=-45,
                              legend_title_text='Optimizer',
                              height=550)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation Matrix
    numeric_df = df.select_dtypes(include=np.number).drop(columns=['Optimizer Best Fitness'], errors='ignore') # Remove fitness from general corr if minimizing
    if 'Optimizer Best Fitness' in numeric_df.columns:
         numeric_df = numeric_df.drop(columns=['Optimizer Best Fitness'], errors='ignore')

    if len(numeric_df.columns) > 1:
        st.subheader("Correlation Matrix of Numerical Metrics")
        fig_corr = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r',
                             title='Correlation Matrix of Experiment Metrics',
                             aspect="auto") # Adjust aspect for better display
        fig_corr.update_layout(height=800, width=900)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Trade-off Analysis (Scatter Plots)
    st.subheader("Trade-off Analysis (Scatter Plots)")
    trade_off_pairs = [
        ('Cyclomatic Complexity', 'Maintainability Index'),
        ('Test Pass Rate', 'Defect Prediction Accuracy (F1)'),
        ('Latency (s)', 'Scalability Metric'),
        ('Architecture Quality Score', 'Code Complexity Score'),
        ('Requirement Satisfaction Score', 'Conflict Count')
    ]

    for x_metric, y_metric in trade_off_pairs:
        if x_metric in df.columns and y_metric in df.columns and not df[[x_metric, y_metric]].isnull().any(axis=1).all():
            fig = px.scatter(df, x=x_metric, y=y_metric, color='Optimizer', symbol='LLM Config Name',
                             title=f'Trade-off: {x_metric} vs. {y_metric}',
                             hover_data=['Dataset', 'Run ID', 'Optimization Cycle'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def plot_agent_workflow(workflow_history: List[Dict[str, Any]]):
    """
    Visualizes the agent collaboration workflow using Plotly for interactivity.
    `workflow_history` is a list of dictionaries: (agent_name, action, timestamp, output_summary)
    """
    if not workflow_history:
        st.info("No agent workflow history to display.")
        return

    st.subheader("Agent Collaboration Workflow (Interactive)")
    
    workflow_df = pd.DataFrame(workflow_history)
    workflow_df['Time Elapsed (s)'] = workflow_df['timestamp'] - workflow_df['timestamp'].min()
    
    # Create nodes for agents and actions
    nodes = []
    edges = []
    
    # Keep track of agent positions for layout
    agent_x_positions = {}
    current_agent_x_offset = 0
    
    # Map agent name to unique color
    agent_names = sorted(list(workflow_df['agent_name'].unique()))
    agent_color_map = {agent: color for agent, color in zip(agent_names, px.colors.qualitative.Plotly)}

    # NetworkX graph for layout calculation
    G = nx.DiGraph()
    
    # Node mapping to ensure unique IDs for networkx
    node_id_counter = 0
    workflow_node_map = {} # Maps (agent_name, action_idx_in_flow) to networkx node ID

    for i, row in workflow_df.iterrows():
        current_nx_node_id = f"action_{node_id_counter}"
        workflow_node_map[(row['agent_name'], i)] = current_nx_node_id
        
        # Add node for the action
        G.add_node(current_nx_node_id, 
                   label=f"{row['agent_name']}:\n{row['action'].splitlines()[0][:25]}...",
                   title=f"Agent: {row['agent_name']}\nAction: {row['action']}\nSummary: {row['output_summary']}",
                   agent_name=row['agent_name'],
                   color=agent_color_map.get(row['agent_name'], '#555'))
        
        # Add edge from previous action in the overall flow
        if i > 0:
            prev_nx_node_id = workflow_node_map[(workflow_df.loc[i-1, 'agent_name'], i-1)]
            G.add_edge(prev_nx_node_id, current_nx_node_id, label=f"({row['action'].splitlines()[0][:15]}...)")
        
        node_id_counter += 1

    # Use a layout that spreads nodes, e.g., spring_layout or kamada_kawai_layout for larger graphs
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42) # k for optimal distance

    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_colors = []

    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node_id]['label'])
        node_hovertext.append(G.nodes[node_id]['title'])
        node_colors.append(G.nodes[node_id]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_colors,
            size=20,
            line_width=2))

    node_trace.text = node_text
    node_trace.hovertext = node_hovertext

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Agent Workflow Diagram (Click/Drag to Explore)',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=700
                    ))
    st.plotly_chart(fig, use_container_width=True)


def plot_knowledge_graph_summary(graph_data: nx.DiGraph):
    """
    Visualizes a summary of the knowledge graph using pyvis for interactivity.
    Limits nodes for performance on large graphs.
    """
    if not graph_data.nodes():
        st.info("Knowledge Graph is empty. Cannot visualize.")
        return

    st.subheader("Knowledge Graph Visualization (Interactive)")

    net = Network(height="750px", width="100%", notebook=True, directed=True,
                  cdn_resources='remote',
                  bgcolor="#222222", font_color="white")
    net.force_atlas_2based(central_gravity=0.01, spring_length=100, spring_constant=0.08, damping=0.9)

    max_nodes_for_viz = 500
    if len(graph_data.nodes) > max_nodes_for_viz:
        st.warning(f"Knowledge Graph has {len(graph_data.nodes)} nodes. Visualizing a random subset of {max_nodes_for_viz} nodes for performance. The displayed graph may not represent all relations.")
        
        # Preferential sampling: keep some critical nodes (e.g., requirements, generated code)
        critical_nodes = [n for n, d in graph_data.nodes(data=True) if d.get('type') in ['requirement', 'generated_code', 'architecture']]
        
        num_random_nodes = max_nodes_for_viz - len(critical_nodes)
        if num_random_nodes > 0:
            other_nodes = [n for n in graph_data.nodes if n not in critical_nodes]
            sampled_nodes = random.sample(other_nodes, min(num_random_nodes, len(other_nodes)))
            final_sampled_nodes = list(set(critical_nodes + sampled_nodes)) # Ensure unique
        else:
            final_sampled_nodes = critical_nodes[:max_nodes_for_viz] # Take top critical nodes

        subgraph = graph_data.subgraph(final_sampled_nodes)
    else:
        subgraph = graph_data

    node_types = sorted(list(set(nx.get_node_attributes(subgraph, 'type').values())))
    type_colors = {node_type: color for node_type, color in zip(node_types, px.colors.qualitative.G10)}

    for node_id, attrs in subgraph.nodes(data=True):
        node_type = attrs.get('type', 'N/A')
        title_attrs = {k: v for k, v in attrs.items() if k not in ['code', 'content_preview', 'raw_llm_response']} # Don't put large raw data in title
        title = "\n".join([f"{k}: {str(v)[:150]}..." if len(str(v)) > 150 else f"{k}: {str(v)}" for k, v in title_attrs.items()])
        
        color = type_colors.get(node_type, "#cccccc")
        
        net.add_node(node_id, label=attrs.get('name', str(node_id)), title=title, color=color,
                     size=15 if node_type in ['requirement', 'issue', 'generated_code'] else 10,
                     font={'color': 'white'})

    for u, v, attrs in subgraph.edges(data=True):
        relation = attrs.get('relation', 'RELATED_TO')
        net.add_edge(u, v, title=relation, label=relation, color='#888888', width=1, font={'size': 7, 'color': 'gray'})

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as fp:
            html_path = fp.name
            net.save_graph(html_path)
        st.components.v1.html(open(html_path, 'r').read(), height=750)
        os.remove(html_path)
    except Exception as e:
        st.error(f"Error generating knowledge graph visualization: {e}")
        st.info("Ensure `pyvis` is installed and there are enough nodes/edges to render meaningfully. If the graph is too dense, try simplifying for visualization.")
        st.text("Sample of graph data (first 5 nodes):")
        st.json(list(subgraph.nodes(data=True))[:5])


def plot_optimizer_history(optimizer_history: List[Dict[str, Any]], optimizer_name: str):
    """
    Plots the best fitness over iterations for an optimizer.
    """
    if not optimizer_history:
        st.info("No optimizer history to plot.")
        return

    history_df = pd.DataFrame(optimizer_history)
    
    fig = px.line(history_df, x='iteration', y='best_fitness',
                  title=f'Optimizer Progress: {optimizer_name} - Best Fitness Over Iterations',
                  labels={'iteration': 'Iteration', 'best_fitness': 'Best Fitness (to minimize)'},
                  markers=True) # Add markers for clarity
    fig.update_layout(xaxis_title_text='Iteration', yaxis_title_text='Best Fitness', height=450)
    st.plotly_chart(fig, use_container_width=True)


# --- NEW PLOTTING FUNCTIONS ---

def plot_optimizer_parameter_trajectories(df: pd.DataFrame, run_id: str, optimizer_name: str, llm_config_name: str, dataset_name: str):
    """
    Plots the trajectory of optimization parameters (best_position) over optimization cycles for a specific run.
    """
    run_df = df[df['Run ID'] == run_id].sort_values('Optimization Cycle')
    if run_df.empty:
        st.info(f"No data for run ID: {run_id} to plot parameter trajectories.")
        return

    # Extract parameters from 'Optimizer Best Position' JSON string
    param_data = []
    for _, row in run_df.iterrows():
        best_pos_str = row['Optimizer Best Position']
        if pd.notna(best_pos_str) and best_pos_str != 'None':
            try:
                params = json.loads(best_pos_str)
                param_data.append({'Optimization Cycle': row['Optimization Cycle'],
                                   'Modularity Preference': params[0] if len(params) > 0 else np.nan,
                                   'Complexity Tolerance': params[1] if len(params) > 1 else np.nan,
                                   'Coverage Goal': params[2] if len(params) > 2 else np.nan})
            except json.JSONDecodeError:
                st.warning(f"Could not parse Optimizer Best Position for {run_id}, Cycle {row['Optimization Cycle']}.")
                continue
        else:
            param_data.append({'Optimization Cycle': row['Optimization Cycle'],
                               'Modularity Preference': np.nan, 'Complexity Tolerance': np.nan, 'Coverage Goal': np.nan})

    param_df = pd.DataFrame(param_data)
    if param_df.empty or param_df.isnull().all().all():
        st.info(f"No valid parameter trajectory data for run ID: {run_id}.")
        return

    fig = px.line(param_df, x='Optimization Cycle', y=['Modularity Preference', 'Complexity Tolerance', 'Coverage Goal'],
                  title=f'Optimizer Parameter Trajectories for Run: {run_id}',
                  labels={'value': 'Parameter Value', 'variable': 'Parameter'},
                  markers=True)
    fig.update_layout(height=500, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def plot_metric_evolution_per_cycle(df: pd.DataFrame, run_id: str, optimizer_name: str, llm_config_name: str, dataset_name: str):
    """
    Plots the evolution of key software and system metrics over optimization cycles for a specific run.
    """
    run_df = df[df['Run ID'] == run_id].sort_values('Optimization Cycle')
    if run_df.empty:
        st.info(f"No data for run ID: {run_id} to plot metric evolution.")
        return

    metrics_for_evolution = [
        'Maintainability Index', 'Cyclomatic Complexity', 'Defect Prediction Accuracy (F1)',
        'Requirement Traceability Accuracy (F1)', 'Test Pass Rate', 'Conflict Count',
        'Optimizer Best Fitness', 'Latency (s)', 'Scalability Metric'
    ]
    
    # Ensure numeric types
    for col in metrics_for_evolution:
        if col in run_df.columns:
            run_df[col] = pd.to_numeric(run_df[col], errors='coerce')

    fig = px.line(run_df, x='Optimization Cycle', y=metrics_for_evolution,
                  title=f'Metric Evolution Over Cycles for Run: {run_id}',
                  labels={'value': 'Metric Value', 'variable': 'Metric'},
                  markers=True)
    fig.update_layout(height=600, legend_title_text="Metrics")
    st.plotly_chart(fig, use_container_width=True)


def display_summary_statistics_table(df: pd.DataFrame):
    """
    Displays a table with summary statistics (mean, std dev) for key metrics,
    grouped by Optimizer and LLM Configuration.
    """
    if df.empty:
        st.info("No data to generate summary statistics.")
        return

    st.subheader("Summary Statistics of Key Metrics")

    metrics_for_summary = [
        'Maintainability Index', 'Cyclomatic Complexity', 'Defect Prediction Accuracy (F1)',
        'Requirement Traceability Accuracy (F1)', 'Test Pass Rate', 'Conflict Count',
        'Latency (s)', 'Scalability Metric', 'Optimizer Best Fitness'
    ]
    
    # Ensure numeric types first
    for col in metrics_for_summary:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Group by Optimizer and LLM Config Name, then calculate mean and std
    summary_df = df.groupby(['Optimizer', 'LLM Config Name'])[metrics_for_summary].agg(['mean', 'std'])
    
    # Flatten the multi-level columns for better display
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    
    st.dataframe(summary_df.round(2), use_container_width=True)

def display_best_performing_configurations(df: pd.DataFrame, top_n: int = 5):
    """
    Displays a table of the top N performing configurations based on Optimizer Best Fitness (lowest is best).
    """
    if df.empty:
        st.info("No data to identify best performing configurations.")
        return

    st.subheader(f"Top {top_n} Performing Configurations (by Lowest Optimizer Best Fitness)")

    # Ensure Optimizer Best Fitness is numeric
    df['Optimizer Best Fitness'] = pd.to_numeric(df['Optimizer Best Fitness'], errors='coerce')
    
    # Group by unique run (Dataset, LLM Config, Optimizer), then take the minimum fitness (best over cycles)
    # And get the last cycle's metrics for consistency
    best_runs = df.loc[df.groupby(['Dataset', 'LLM Config Name', 'Optimizer'])['Optimizer Best Fitness'].idxmin()]
    
    # Sort by the best fitness found
    best_runs_sorted = best_runs.sort_values(by='Optimizer Best Fitness', ascending=True)

    # Select relevant columns for display
    display_cols = [
        'Run ID', 'Dataset', 'LLM Config Name', 'Optimizer', 'Optimizer Best Fitness',
        'Maintainability Index', 'Test Pass Rate', 'Conflict Count', 'Latency (s)'
    ]
    
    # Ensure all display_cols exist and are numeric where expected
    for col in display_cols:
        if col not in best_runs_sorted.columns:
            st.warning(f"Column '{col}' not found in results for best performing configurations.")
            display_cols.remove(col)
        elif col not in ['Run ID', 'Dataset', 'LLM Config Name', 'Optimizer']:
            best_runs_sorted[col] = pd.to_numeric(best_runs_sorted[col], errors='coerce')

    st.dataframe(best_runs_sorted[display_cols].head(top_n).round(2), use_container_width=True)