#!/usr/bin/env python
# coding: utf-8

# #### config.py

# Current vertion restored Chroma from blob container

# In[ ]:


## tools to upload chroma database 
from azure.storage.blob import BlobServiceClient
import os

import sys
try:
    import pysqlite3 # This loads the newer SQLite version packaged with pysqlite3-binary
    sys.modules['sqlite3'] = pysqlite3 #This literally replaces the 'sqlite3' entry in Python's module dictionary
except ImportError:
    pass


AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client_chromadb = blob_service_client.get_container_client("chromadb")

## restore chroma files 
def download_chromadb_from_blob(container_client, subfolder_name="chroma_db_tenant_blob_restored"):
    """
    Download ChromaDB files from blob storage to local directory
    Works on any OS including cloud environments
    """
    # Get current working directory and add subfolder
    base_path = os.getcwd()
    local_path = os.path.join(base_path, subfolder_name)
    
    print(f"Downloading ChromaDB to: {local_path}")
    
    # Create main directory
    os.makedirs(local_path, exist_ok=True)
    
    # Download all blobs
    for blob in container_client.list_blobs():
        # Create full local file path
        local_file_path = os.path.join(local_path, blob.name)
        
        # Create directory structure
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)
        
        # Download file
        blob_client = container_client.get_blob_client(blob.name)
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        print(f"Downloaded: {blob.name}")
    
    return {local_path}



# In[ ]:


# goal of the script is to prepare metadata for a database
# it will process documents, generate questions, and prepare metadata for storage


# === Standard Library ===
import sys
import gc
import re
import json
import datetime
import random

# === Third-Party Core Data Libraries ===
import pandas as pd
import numpy as np
import shutil

# === Third-Party ML/NLP Libraries ===
from sklearn.metrics.pairwise import cosine_similarity

# === Database and Vector Store Backends ===
import chromadb           # Direct Chroma Python interface, if needed

# === LangChain Integrations ===

# Vector Stores & Schema
from langchain_chroma import Chroma              # Vector DB wrapper
from langchain.schema import Document            # Standard document metadata interface

# Embeddings
from langchain_openai import OpenAIEmbeddings    # OpenAI API embeddings
#from langchain_openai import OpenAI              # OpenAI LLMs
#from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face transformer embeddings


# === Optional: LangChain Prompts/Chat Models (Uncomment if used) ===
#from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# import  bm25 for ranking
#from rank_bm25 import BM25Okapi


## upload env variables
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI

# === Tips ===
# - Keep only imports for libraries/components you actually use in this script/module.
# - Maintain logical groupings so updates or dependency changes are easy to manage.
# - If you have many environment-specific imports (e.g., for different DBs or vector stores), modularize them into utilities or separate scripts.


# === Configuration Paths ===

# location of databases
SUBFOLDER = 'chroma_db_tenant_blob_restored'
BASE_PATH = os.getcwd()
CHROMA_PATH_LAW_CORPUS = os.path.join(BASE_PATH, SUBFOLDER, 'law_corpus_chroma')
CHROMA_PATH_LAW_CORPUS_HQ = os.path.join(BASE_PATH, SUBFOLDER , 'law_corpus_chroma_hq')

# os.path.join majes the solution cross platform compatible
# NASE PATH with getcwd() is usefull for containerized solutions



api_key_var =  os.getenv('OPENAI_API_KEY')



# #### Define embeddings, llm , vector store

# In[ ]:


# https://python.langchain.com/docs/integrations/vectorstores/chroma/

# Load the model using HuggingFaceEmbeddings (passing the path, NOT the instance)
#embeddings_hf_1_chroma = HuggingFaceEmbeddings(model_name=MODEL_LLM_EMBED_OPT1)

# to encode query we require different hf class
#embeddings_hg_1 =  SentenceTransformer(MODEL_LLM_EMBED_OPT1)

# open ai embeddings
#embeddings_open_ai_chroma = OpenAIEmbeddings(model="text-embedding-3-small" , api_key = api_key_var )

# embeddings_main = embeddings_open_ai_chroma

# transform to function 
def get_embedding_model_instance_queries():
    ''' Inititalise and return open ai embedding model '''

    try: 
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small" , 
                                           api_key = api_key_var )
        
        print("OpenAIEmbeddings initialised")
        return embedding_model 
    
    except Exception as e:
        raise SystemExit("Embedding model initialization failed. Exiting.")

## test 
#embedding_main = get_embedding_model_instance_queries()

# buf = np.array( embedding_main.embed_query("What is main function of National Bank of Moldova") )
# print(buf.shape)
# buf.reshape(1,-1)  # (sample , features )

# buf = np.array( embedding_main.embed_documents(["What is main function of National Bank of Moldova", "What is weatehr on Jupiter" ]) )
# buf = [ x.reshape(1,-1) for x in buf ]
# buf = np.vstack(buf)
# print(buf.shape )



# In[ ]:


## get llm instance 

def get_llm_instance(): 
    """Initializes and returns the OpenAI Generative AI LLM."""

    try: # https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html
        llm = ChatOpenAI(model = 'gpt-4o-mini', 
                         api_key=api_key_var)
        print("LLM isinitialised ")
        return llm
    except Exception as e: 
        raise SystemExit("LLM initialization or connection failed. Exiting.")

# test 
#llm = get_llm_instance()
#llm.invoke("Hello, am i connected?.")


# #### Initiate databases

# In[ ]:


# Create or load the Chroma vector store

def get_vector_db_and_retriever_instance(_embeddings_instance):
    """Initializes and returns the Chroma DBs"""

    try:
        if not os.path.exists(CHROMA_PATH_LAW_CORPUS_HQ) or not os.path.exists(CHROMA_PATH_LAW_CORPUS):
            print(f"Chroma path not found.")
            # This is a critical dependency, so we exit if not found
            raise SystemExit(f"Vector databases not found")
        
        collection_metadata={ # https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration
                                "hnsw:space": "cosine",  
                                "hnsw:construction_ef":150,  
                                "hnsw:M": 32 ,
                                "hnsw:search_ef":150 }  
        # create databased
        vector_store_fin_law_hq  = Chroma( persist_directory = CHROMA_PATH_LAW_CORPUS_HQ,  ## currently we dont require persistent database
                        collection_name="codex_fin_law_hq" , 
                        embedding_function = _embeddings_instance)

        vector_store_fin_law  = Chroma( persist_directory = CHROMA_PATH_LAW_CORPUS,  ## currently we dont require persistent database
                        collection_name="codex_fin_law" , 
                        embedding_function = _embeddings_instance)

        print("Chroma DBs initialized")
        return vector_store_fin_law_hq, vector_store_fin_law

    except Exception as e:
         raise SystemExit("Failed to initialize Chromas DB")
    
#vector_store_fin_law_hq ,  vector_store_fin_law =  get_vector_db_and_retriever_instance(embedding_main)   

# test 
#vector_store_fin_law_hq._collection.get(include=["metadatas"]) 


# In[ ]:


## Prepare classes for downstream use 

# form a database
download_chromadb_from_blob(container_client_chromadb)

embedding_main = get_embedding_model_instance_queries()
llm = get_llm_instance()
vector_store_fin_law_hq ,  vector_store_fin_law =  get_vector_db_and_retriever_instance(embedding_main)  

