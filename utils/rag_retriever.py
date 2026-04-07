import streamlit as st
from typing import List, Dict, Any
from utils.knowledge_graph import KnowledgeGraph # Ensure this is correct relative import
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import time
import numpy as np
import faiss # Make sure faiss-cpu or faiss-gpu is installed: pip install faiss-cpu
from langchain_community.docstore.in_memory import InMemoryDocstore # CORRECTED: Use langchain_community

class RAGRetriever:
    """
    Manages Retrieval Augmented Generation by querying the Knowledge Graph
    and potentially a vector store for semantic similarity.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, embedding_model_name: str = "nomic-embed-text"):
        self.knowledge_graph = knowledge_graph
        self.vectorstore = None
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.embedding_provider_available = False 

        try:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
            try:
                # Attempt a very small embedding query to check liveness
                # Use a very short, simple string for minimal overhead.
                test_embedding = self.embeddings.embed_query("connection test")
                if test_embedding and len(test_embedding) > 0: # Ensure not empty
                    self.embedding_provider_available = True
                    st.success(f"Initialized OllamaEmbeddings with model: {embedding_model_name} and successfully connected.")
                else:
                    st.error(f"OllamaEmbeddings with {embedding_model_name} initialized, but test query returned empty. Check Ollama server.")
            except Exception as e:
                st.error(f"OllamaEmbeddings with {embedding_model_name} initialized, but FAILED to make a test query: {e}")
                st.info("Ensure Ollama server is running and model is pulled (`ollama pull nomic-embed-text`).")

        except Exception as e:
            st.error(f"Failed to initialize OllamaEmbeddings with {embedding_model_name}. "
                     f"Please ensure the model is downloaded via `ollama pull {embedding_model_name}`. Error: {e}")
            self.embeddings = None


    def build_vector_store(self):
        """
        Builds a FAISS vector store from the Knowledge Graph nodes' content.
        This approach will manually embed documents and then build the FAISS index
        to separate the embedding process from the index building process.
        """
        if not self.embeddings or not self.embedding_provider_available:
            st.error("Embeddings model not initialized or provider unavailable. Cannot build vector store.")
            self.vectorstore = None
            return

        st.info("Building vector store from knowledge graph nodes...")
        documents_for_faiss = [] # This will hold LangChain Document objects with content and metadata

        # --- Step 1: Prepare Documents for Embedding ---
        for node_id, attrs in self.knowledge_graph.graph.nodes(data=True):
            content_parts = []
            content_parts.append(f"Type: {attrs.get('type', 'Unknown')}")
            content_parts.append(f"Name: {attrs.get('name', node_id)}")
            
            if attrs.get('description'): content_parts.append(f"Description: {attrs['description']}")
            if attrs.get('summary'): content_parts.append(f"Summary: {attrs['summary']}")
            if attrs.get('full_message'): content_parts.append(f"Commit Message: {attrs['full_message']}")
            if attrs.get('code_snippet'): content_parts.append(f"Code Snippet: {attrs['code_snippet'][:500]}...")
            elif attrs.get('content_preview'): content_parts.append(f"File Content Preview: {attrs['content_preview'][:500]}...")
            
            content = "\n".join([p for p in content_parts if p.strip()])

            if content.strip():
                # Store the full Document object, including metadata, for FAISS docstore
                documents_for_faiss.append(Document(page_content=content, metadata={"node_id": node_id, "node_type": attrs.get('type')}))
        
        if not documents_for_faiss:
            st.warning("No meaningful documents found in knowledge graph to embed. Vector store will be empty.")
            self.vectorstore = None
            self.embedding_provider_available = False # Can't fully claim available if no documents
            return

        st.write(f"Preparing to embed {len(documents_for_faiss)} knowledge graph nodes...")
        
        embedding_progress_text = st.empty()
        embedding_progress_bar = st.progress(0)
        
        try:
            # --- Step 2: Generate Embeddings in Chunks ---
            all_embeddings_vectors = [] # List of embedding vectors (list of floats)
            chunk_size = 50 

            for i in range(0, len(documents_for_faiss), chunk_size):
                chunk_docs = documents_for_faiss[i:i + chunk_size]
                chunk_contents = [doc.page_content for doc in chunk_docs]
                
                # This is the actual call to OllamaEmbeddings
                chunk_embeddings = self.embeddings.embed_documents(chunk_contents)
                all_embeddings_vectors.extend(chunk_embeddings)

                progress_percent = (i + len(chunk_docs)) / len(documents_for_faiss)
                embedding_progress_bar.progress(progress_percent)
                embedding_progress_text.text(f"Embedded {min(i+chunk_size, len(documents_for_faiss))} of {len(documents_for_faiss)} nodes.")
            
            embedding_progress_text.text(f"All {len(documents_for_faiss)} nodes embedded. Initiating FAISS index build...")
            embedding_progress_bar.progress(1.0)
            
            # --- Step 3: Build FAISS Index and Docstore ---
            
            # Check if faiss is available for direct use
            if not hasattr(faiss, 'IndexFlatL2'): # Check for a common FAISS class
                st.error("FAISS library (e.g., faiss-cpu) not fully installed or not accessible. Cannot build index.")
                self.vectorstore = None
                return

            if not all_embeddings_vectors:
                st.warning("No embeddings generated. Cannot build FAISS index.")
                self.vectorstore = None
                return

            embedding_dim = len(all_embeddings_vectors[0])
            embeddings_np = np.array(all_embeddings_vectors).astype('float32') # FAISS often prefers float32

            # Use st.spinner for the blocking FAISS index construction
            with st.spinner(f"Building FAISS index for {len(embeddings_np)} vectors (this might take a while)..."):
                index = faiss.IndexFlatL2(embedding_dim) # L2 distance
                index.add(embeddings_np) # Add the vectors to the index

                # Create the docstore mapping internal FAISS IDs to LangChain Documents
                # FAISS uses internal sequential IDs (0 to N-1) for documents added.
                # We need to map these to our original LangChain Document objects.
                # The ids_for_docstore should correspond to these internal FAISS IDs.
                ids_for_docstore = [str(i) for i in range(len(documents_for_faiss))] # Generate sequential string IDs
                
                # The docstore needs to map these internal IDs to the full Document objects
                docstore = InMemoryDocstore({ids_for_docstore[i]: documents_for_faiss[i] for i in range(len(documents_for_faiss))})

                # Create the LangChain FAISS vectorstore object
                self.vectorstore = FAISS(
                    embedding_function=self.embeddings.embed_query,
                    index=index,
                    docstore=docstore,
                    # This maps internal FAISS index IDs (0, 1, 2...) to our docstore IDs ("0", "1", "2"...)
                    index_to_docstore_id={i: ids_for_docstore[i] for i in range(len(ids_for_docstore))}
                )


            st.success(f"Vector store built with {len(documents_for_faiss)} documents. Index contains {index.ntotal} vectors.")
            self.embedding_provider_available = True
        except Exception as e:
            st.error(f"Error during embedding or building FAISS vector store: {e}")
            st.exception(e) # Display full traceback
            st.info("Possible causes: Ollama server disconnected, model unloaded, out of memory on Ollama, FAISS library issue, or very large index size.")
            self.vectorstore = None
            self.embedding_provider_available = False
        finally:
            embedding_progress_bar.empty() # Clear the progress bar when done or failed
            embedding_progress_text.empty() # Clear the text

    def retrieve_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves relevant context for a given query, first from the vector store
        (semantic search) and then enriching with graph traversal.
        Returns a dictionary with 'full_context' as a string.
        """
        if not self.vectorstore:
            st.warning("Vector store not available. Cannot retrieve context.")
            return {"full_context": "No context available (vector store not built)."}

        # st.info(f"Retrieving context for query: '{query}'") # Keep silent during iterative agent calls
        retrieved_docs = []
        
        try:
            # Perform semantic search using the vector store
            docs = self.vectorstore.similarity_search(query, k=k)
            for doc in docs:
                retrieved_docs.append({
                    "source": "vector_store",
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            # st.write(f"Retrieved {len(docs)} documents from vector store.") # Keep silent during iterative agent calls
        except Exception as e:
            st.error(f"Error during vector store retrieval: {e}")
            return {"full_context": f"Error during context retrieval: {e}"}

        # Further enrich context by traversing the graph around the top retrieved nodes
        enriched_context_items = []
        processed_node_ids = set()
        for doc in retrieved_docs:
            node_id = doc['metadata'].get('node_id')
            if node_id and node_id not in processed_node_ids:
                # Get rich context from the KG for the retrieved node
                graph_context = self.knowledge_graph.get_context_for_node(node_id, depth=1)
                enriched_context_items.append(f"--- Context from Knowledge Graph around Node ID: {node_id} ---\n{graph_context}")
                processed_node_ids.add(node_id)
            
            # Also add the original document content
            enriched_context_items.append(f"--- Retrieved Document Content (Node Type: {doc['metadata'].get('node_type', 'N/A')}) ---\n{doc['content']}")

        # Deduplicate and format for LLM
        final_context_str = []
        seen_contents = set()
        for item in enriched_context_items:
            if item not in seen_contents: # Check for exact string duplication
                final_context_str.append(item)
                seen_contents.add(item)
        
        return {"full_context": "\n\n".join(final_context_str)}