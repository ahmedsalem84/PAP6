import streamlit as st
import os
import git
from datasets import load_dataset
import shutil
import pandas as pd
import requests
import zipfile
from io import BytesIO

# Directory to store downloaded data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_data(show_spinner="Getting available datasets...")
def get_available_datasets():
    """
    Returns a list of real, accessible datasets.
    """
    return [
        "GitHub_Repo_Popular_Python", # A real, moderately sized GitHub Python project
        "PROMISE_JIRA_NASA_MDP",    # A PROMISE dataset, e.g., from NASA's MDP projects
        "CodeSearchNet_Python_Small", # A small subset of CodeSearchNet for Python
        # "Kaggle_Bug_Reports_Small" # Add if a suitable public Kaggle dataset is found without API key
    ]

@st.cache_data(show_spinner="Downloading and loading dataset...")
def download_and_load_dataset(dataset_name: str):
    """
    Downloads and loads a specific dataset based on its name.
    Returns a dictionary with 'type' and 'path'/'data'.
    """
    st.info(f"Attempting to download and load: {dataset_name}")
    dataset_target_path = os.path.join(DATA_DIR, dataset_name.replace(" ", "_").lower())
    os.makedirs(dataset_target_path, exist_ok=True)

    if dataset_name == "GitHub_Repo_Popular_Python":
        # Using a popular, relatively small-to-medium Python project
        repo_url = "https://github.com/pallets/flask.git" # Flask is a good example
        # repo_url = "https://github.com/scikit-learn/scikit-learn.git" # Too big for quick experiments
        # repo_url = "https://github.com/psf/requests.git" # Requests library, good size
        target_dir = os.path.join(dataset_target_path, "flask")

        if os.path.exists(target_dir):
            st.info(f"GitHub repo '{repo_url}' already exists at {target_dir}. Skipping clone.")
            # Ensure it's up-to-date if needed, or just return existing
            # try:
            #     repo = git.Repo(target_dir)
            #     origin = repo.remotes.origin
            #     origin.pull()
            #     st.success(f"Pulled latest changes for {repo_url}.")
            # except Exception as e:
            #     st.warning(f"Could not pull latest for {repo_url}: {e}")
            
        else:
            try:
                st.write(f"Cloning {repo_url} into {target_dir}...")
                git.Repo.clone_from(repo_url, target_dir, depth=1) # Use depth=1 for faster cloning of recent history
                st.success(f"Cloned {repo_url} to {target_dir}")
            except Exception as e:
                st.error(f"Error cloning GitHub repo {repo_url}: {e}")
                return None
        return {"type": "git_repo", "path": target_dir, "name": "Flask"}

    elif dataset_name == "PROMISE_JIRA_NASA_MDP":
        # Using a dataset from NASA's Metrics Data Program (MDP) as a PROMISE example
        # Often these are available as CSV or ARFF files.
        # We'll use a link to a CSV directly if available, or simulate a download.
        # Example: CM1 dataset (often used for defect prediction)
        csv_url = "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/jm1.csv"
        file_name = "jm1.csv"
        csv_path = os.path.join(dataset_target_path, file_name)

        if not os.path.exists(csv_path):
            try:
                st.write(f"Downloading {file_name} from {csv_url}...")
                response = requests.get(csv_url)
                response.raise_for_status() # Raise an exception for bad status codes
                with open(csv_path, 'wb') as f:
                    f.write(response.content)
                st.success(f"Downloaded PROMISE dataset to {csv_path}")
            except Exception as e:
                st.error(f"Error downloading PROMISE dataset from {csv_url}: {e}")
                return None
        else:
            st.info(f"PROMISE dataset already exists at {csv_path}. Skipping download.")

        try:
            # PROMISE datasets often have 'class' or 'defects' column
            df = pd.read_csv(csv_path)
            # Standardize column names if needed for general processing in KG
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            st.success(f"Loaded PROMISE dataset with {len(df)} entries.")
            return {"type": "csv_jira_like", "path": csv_path, "data": df, "name": "JM1"}
        except Exception as e:
            st.error(f"Error loading PROMISE dataset from {csv_path}: {e}")
            return None

    elif dataset_name == "CodeSearchNet_Python_Small":
        # Using HuggingFace datasets library for CodeSearchNet
        # We'll load a very small split for feasibility in an interactive app.
        try:
            # Using 'test' split to get some distinct data from 'train' for flexibility
            # Limiting to a very small number for quick load, can be increased for full runs
            dataset = load_dataset("code_search_net", "python", split='test[:100]') # Load first 100 samples from test split
            st.success(f"Downloaded CodeSearchNet Python test subset (100 samples).")
            return {"type": "huggingface_dataset", "data": dataset, "name": "CodeSearchNet_Python_Small"}
        except Exception as e:
            st.error(f"Error loading CodeSearchNet dataset: {e}")
            return None

    # elif dataset_name == "Kaggle_Bug_Reports_Small":
    #     # Example of a Kaggle dataset, usually requires API key or direct download.
    #     # For now, we'll keep this commented out unless a direct download link is found.
    #     # This would require user to provide Kaggle API key or manually download.
    #     # For simplicity in this demo, let's omit unless a very public, no-auth link exists.
    #     st.warning("Kaggle datasets usually require API key or manual download. This feature is not enabled for simplicity.")
    #     return None

    else:
        st.warning(f"Dataset '{dataset_name}' not recognized or implemented yet.")
        return None