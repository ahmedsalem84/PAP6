import streamlit as st
import ollama
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate # Keep these specific imports
from langchain_core.output_parsers import StrOutputParser

# List of known embedding-only models, extend as needed
EMBEDDING_ONLY_MODELS = {"nomic-embed-text"} 

def get_ollama_models():
    """
    Fetches a list of available Ollama models.
    """
    try:
        models_info = ollama.list()
        return [model['name'] for model in models_info['models']]
    except Exception as e:
        # st.error(f"Error connecting to Ollama: {e}. Is Ollama server running?") # Removed st.error for cleaner startup
        return []

def initialize_llm(model_name: str):
    """
    Initializes a LangChain Ollama LLM instance.
    Adds a check to prevent using embedding-only models for generation.
    """
    if model_name.split(':')[0] in EMBEDDING_ONLY_MODELS:
        st.error(f"Error: Model '{model_name}' is an embedding-only model and cannot be used for text generation. Please select a generative model.")
        return None
    try:
        llm = Ollama(model=model_name)
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM {model_name}: {e}")
        return None

def generate_response(llm, prompt: str, system_message: str = "") -> str:
    if llm is None:
        return "LLM instance is not valid. Generation skipped."
    
    messages = []
    if system_message:
        messages.append(SystemMessagePromptTemplate.from_template(system_message))
    
    # Use HumanMessagePromptTemplate, LangChain will infer variables if any (which should be 'input')
    messages.append(HumanMessagePromptTemplate.from_template("{input}")) 

    # ChatPromptTemplate infers input_variables from messages.
    # We ensure only '{input}' is a variable by escaping other curly braces in the *agent prompts*.
    prompt_template = ChatPromptTemplate.from_messages(messages) 

    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({"input": prompt}) 
        return response
    except Exception as e:
        st.error(f"Error during LLM inference: {e}")
        st.exception(e) # Show full traceback for LLM inference errors
        return "LLM inference failed."