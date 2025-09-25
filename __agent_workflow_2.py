#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# goal of the script is to prepare metadata for a database
# it will process documents, generate questions, and prepare metadata for storage


# === Standard Library ===
import os
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
BASE_PATH = os.getcwd()
CHROMA_PATH_LAW_CORPUS = os.path.join(BASE_PATH, 'chroma_db_tenant_blob_restored', 'law_corpus_chroma')
CHROMA_PATH_LAW_CORPUS_HQ = os.path.join(BASE_PATH, 'chroma_db_tenant_blob_restored', 'law_corpus_chroma_hq')

# os.path.join majes the solution cross platform compatible
# NASE PATH with getcwd() is usefull for containerized solutions


from __config_2 import embedding_main, llm, vector_store_fin_law_hq ,  vector_store_fin_law



# #### Define instruments

# In[ ]:


## make a tool out of function 

from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage,ToolMessage, RemoveMessage
from langgraph.graph.message import add_messages 
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, List, Dict, Annotated, Any, Literal

@tool
def tool_1_retrieve_similar_questions(

    query:str,  # question asked by user
    hq_q_retrieve = 5,   # number of chunks to retrieve
    cs_threshold = 0.5   # cosine similarity threshold
):
    """
    Retrieve the most similar questions/chunks for a given query .
    Returns a DataFrame of results.

    Parameters:
    - query: string which contains the question asked by user
    - hq_q_retrieve: number of similar chunks to retrieve from the vector store
    - cs_threshold: minimum cosine similarity score for a chunk to be considered
    """
    # query to vector 
    query_embed_local = embedding_main.embed_query(query)

    # Step 1: Retrieve candidate chunks by vector similarity search (we use distance , 1 - similarity search )
    results = vector_store_fin_law_hq.similarity_search_by_vector_with_relevance_scores(
        embedding=query_embed_local , 
        k = hq_q_retrieve
    )

    result_pairs = []
    for doc, score in results:
        chunk_codex_id = doc.metadata['chunk_codex_ids_meta']
        page_content = doc.page_content
        result_pairs.append([chunk_codex_id, page_content,score])   

    ## sort from  lowest to highest distance ( highest to lowest similarity )
    result_pairs = sorted(result_pairs, key=lambda x: x[2])  # Sort by distance (x[2])    
    #print(result_pairs)

    # filter based on distance  
    result_pairs = [
        chunk_id for chunk_id, content, score in result_pairs if score <= cs_threshold
    ]       

    return result_pairs


## test the tool
#query = "Cum se calculează amenda aplicată de Banca Naționala"
#tool_1_retrieve_similar_questions.invoke({
#    "query": query,
#    "hq_q_retrieve": 5,
#    "cs_threshold": 0.5
#})


# In[ ]:


# metadata contains in hq database

# retrive all questions
#vector_store_fin_law_all_docs = vector_store_fin_law._collection.get(include=["metadatas"]) 
#print( vector_store_fin_law_all_docs.keys() )
#print( vector_store_fin_law_all_docs["metadatas"][0].keys() )   # check first 2 entries
  # dict_keys(['chunk_codex_ids_meta', 'original_chunk'])

# we require
# page_content 

# meta:
# page_number 
# document_title
# start_index  
# chunk_codex_ids_meta
# end_index


# In[ ]:


# make a tool out of function

@tool
def tool_2_retrieve_similar_content(

    query:str,       # string wihich contains the question asked by user
    hq_q_retrieve=5,        # number of chunks to retrieve
    cs_threshold=0.5         # cosine similarity threshold
    
):
    """
    Retrieve the most similar questions/chunks for a given query .
    Returns a DataFrame with columns: score, codex_id_meta, feature (="h_question"), content.

    Parameters:
    - query: string which contains the question asked by user
    - hq_q_retrieve: number of similar chunks to retrieve from the vector store
    - cs_threshold: minimum cosine similarity score for a chunk to be considered
    """

    query_embed_local = embedding_main.embed_query(query)

    # Candidate retrieval
    results = vector_store_fin_law.similarity_search_by_vector_with_relevance_scores(
        embedding=query_embed_local , 
        k = hq_q_retrieve )

    # Extract metadata and content
    result_pairs = []
    for doc, score in results:
        chunk_codex_id = doc.metadata['chunk_codex_ids_meta']
        page_content = doc.page_content
        result_pairs.append([chunk_codex_id, page_content,score]) 

    ## sort from  lowest to highest distance ( highest to lowest similarity )
    result_pairs = sorted(result_pairs, key=lambda x: x[2])  # Sort by distance (x[2])    
    #print(result_pairs)

    # filter based on distance  
    result_pairs = [
        chunk_id for chunk_id, content, score in result_pairs if score <= cs_threshold
    ]       

    return result_pairs

## test the tool
#query = "Cum se calculează amenda aplicată de Banca Naționala"
#tool_2_retrieve_similar_content.invoke({
#    "query": query,
#    "hq_q_retrieve": 5,
#    "cs_threshold": 0.5
#})



# In[ ]:


@tool 
def tool_3_retrieve_chunks_by_ids(chunk_id_list):
    """
    Retrieve documents from the vector store by a list of chunk IDs.

    Parameters:
    - chunk_id_list: list of string IDs, e.g., ['chunk_codex_117', ...]
    - vector_store: your vector store object, e.g., vector_store_fin_law

    Returns: 
        A list of dicts with 'chunk_id' and 'content' for each found chunk.
    """

    # Compose the 'where' filter for Chroma/Chroma-like interfaces
    where_filter = {"chunk_codex_ids_meta": {"$in": chunk_id_list}}
    
    # Query with a dummy embedding (not needed since filtering by meta)
    results = vector_store_fin_law._collection.get(where=where_filter, include=["documents", "metadatas"])

    # Compose output as list of dicts (order as returned by collection)
    chunks = []
    for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
        chunks.append({
            "chunk_id": meta.get("chunk_codex_ids_meta", None),
            "start_index": meta.get("start_index", None),
            "end_index": meta.get("end_index", None),
            "file_name": meta.get("file_name", None),
            "page_number": meta.get("page_number", None),
            "content": doc
        })
    return chunks



# #### agent_architecture

# define chat model 
# add tools to model 
# we call 2 tools in parallel to extract chunks 
# union of chunks id is used 
# 3-rd tool extracts chunks 
# 
# chunks become context to the final prompt to receive reply 

# In[ ]:


## define state as typed dict 

class State_MRR(TypedDict):
    """State for the MRR agent."""
    messages: Annotated[List[AnyMessage], add_messages]
    answer_repeat_ai_content_global: str
    question_type : str
    summary : str

    summary_ai_messages: str
    
tools = [ tool_1_retrieve_similar_questions , tool_2_retrieve_similar_content , tool_3_retrieve_chunks_by_ids  ]
llm_with_tools = llm.bind_tools(tools)

## dictionary of tools by name , used in case we dont use ToolNode and run tools externally 
# then append tools result with ToolMessage
tools_by_name = {tool.name:tool for tool in tools}
#print(tools_by_name)
#print(tools_by_name.keys())
#print(tools_by_name["tool_1_retrieve_similar_questions"])

# ToolMessage is defined by content and tool_call_id whicha re compulsory as each tool message 
# is bouded to ai message which regereces it id

## Add checkpointers 
  # this is build in persistent layer implemete via checkpointers

config ={"configurable": {"thread_id": "1"}}
memory_component = MemorySaver()



# In[ ]:


## scenario one, agents retrieves chunks id using tool 1 and 2

system_message = SystemMessage(
    content=(
        "You are a query classification agent. Your task is to call functions (tools) to retrieve relevant information. "
        "You must call BOTH tools named 'tool_1_retrieve_similar_questions' and 'tool_2_retrieve_similar_content' in parallel, each time you receive a query.\n"
        "VERY IMPORTANT: Do NOT change the wording of the user's query when calling the tools. "
        "Pass the exact query received from the user to each tool without modifying it. "
        "Never attempt to answer the user's question or summarize the result yourself—always return only the raw outputs of both tools."
    )
)


# node creates payload for tool calls, payload will be stored in ai message
# Tool calls: ['tool_1_retrieve_similar_questions', 'tool_2_retrieve_similar_content']
def tool_node_chunk_selection_payload(state: State_MRR):
    """
    TOOL-SELECTION STEP (no direct answering).
    You MUST:
      1) Call tool_1_retrieve_similar_questions with the latest user query.
      2) Call tool_2_retrieve_similar_content with the same query.

    """
    print("Executing tool_node_chunk_selection_payload...")

    # select last human message 
    latest_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    messages_for_llm = [system_message]
    if latest_human:
        messages_for_llm.append(latest_human)

    #print(llm_with_tools.invoke(messages_for_llm)) # for debugging
    print("Payload for chunks selection END")

    return {
        "messages": [
            llm_with_tools.invoke(messages_for_llm) # state message must contain latest human message
        ]                                                                #checkpoins already contain past conversation 
    }



# list with tools is stored externally, tool node is defined manually
# here we generate ToolMessages for each tool call in the last ai message , if emty, we skip generating response
def tool_node_chunk_selection_exec(state: State_MRR):
    """Performs the tool call"""
    print("Executing tool_node_chunk_selection_exec...")
    
    result = []
    for tool_call in state["messages"][-1].tool_calls: # AI message with payload , last ai message with too l calls
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
           
        result.append(ToolMessage( 
                       content = observation, 
                       tool_call_id=tool_call["id"] , 
                       name = tool_call['name']))    

    print("Executing tool_node_chunk_selection_exec END")
    return {"messages": result}



## Next we need to check if payload is not empty 
def check_chunks_found(state: State_MRR) -> Literal["llm_create_retrieve_tool_payload", "request_reformulate_question"]:
    """Check if any chunks were found by the tools"""

    print("Executing check_chunks_found...")
    
    # Look for recent ToolMessages from tools 1 and 2
    recent_tool_msgs = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name in ["tool_1_retrieve_similar_questions", "tool_2_retrieve_similar_content"]:
            recent_tool_msgs.append(msg)
        if len(recent_tool_msgs) >= 2:  # Found both tools
            break
    
    # Check if all tools returned empty results
    all_empty = all(not msg.content or len(msg.content) == 0 for msg in recent_tool_msgs)
    
    if all_empty:
        print("No chunks found by either tool.")
        return "request_reformulate_question"
    else:
        print("Chunks found, proceeding to final retrieval.")
        return "llm_create_retrieve_tool_payload"

     

## Create payload for chunks retrieval
## this is node which is first to take ans inoput human message
def llm_create_retrieve_tool_payload(state: State_MRR):
    """
    Node creates a tool payload for tool_3_retrieve_chunks_by_ids with the given chunk IDs.

    """
    print("Executing llm_create_retrieve_tool_payload...")

    ## Here we need to collect only the recent tool messages 
    ## we need messages after the last Human message otherwise we will collect all tool messages
    ## which have differnet return format

    # Find the index of the last HumanMessage
    last_human_idx = max(i for i, msg in enumerate(state["messages"]) if isinstance(msg, HumanMessage))

    # Gather all chunk IDs (flatten & deduplicate if needed)
    tool_messages = [ msg for msg in state["messages"][last_human_idx+1:] if isinstance(msg, ToolMessage)]

    ## merge chunks id together
    all_chunk_ids = []
    seen = set()
    for msg in tool_messages:
        for item in msg.content:
            if item not in seen:
                all_chunk_ids.append(item)
                seen.add(item)
    #print("All chunk IDs:", all_chunk_ids)
    # all_chunk_ids can be empty

    if not all_chunk_ids:
        print("No chunk IDs found; skipping retrieval payload.")
        return {}  # no state change

    # Craft messages for the LLM
    system_instructions = (
        "You are an LLM agent. If you see a list of legal chunk IDs, create a tool call for 'tool_3_retrieve_chunks_by_ids' "
        "with argument 'chunk_id_list' set to that list. Do NOT answer directly. Only request the tool call."
    )
    sys_msg = SystemMessage(content=system_instructions)
    human_msg = HumanMessage(content=f"chunk_id_list  = {all_chunk_ids}")

    # Invoke the LLM to get tool-call message
    ai_message = llm_with_tools.invoke([sys_msg, human_msg])

    print("Executing llm_create_retrieve_tool_payload END ")

    # Add this AI message to the state (LangGraph will process tool call in the next step)
    return {"messages":  [ai_message]}



# run tool_3 if the payload is crated 
def llm_create_retrieve_tool_exec(state):
    """
    Executes tools requested by LLM llm_create_retrieve_tool_payload messages,
    attaches results as ToolMessages.
    """
    print("Executing llm_create_retrieve_tool_exec...")

    results = []

    # Find the last message(s) with tool calls
    last_message = state["messages"][-1]
    # Support for possible multiple tool calls in one AIMessage
    for tool_call in getattr(last_message, "tool_calls", []): # multiple tools payloads are possible
        tool_name = tool_call["name"]
        args = tool_call["args"]
        # your tool registry should have your tool functions/objects
        tool = tools_by_name[tool_name]
        observation = tool.invoke(args)
        # attach as a ToolMessage:
        results.append(ToolMessage(
            content=observation,
            tool_call_id=tool_call["id"],
            name=tool_name
        ))

    print("Executing llm_create_retrieve_tool_exec END")
        
    return {"messages": results} # we dont need to return full state, add reduces will add messagfe


## generate final answer 
def final_answer_node(state):
    """
    Use the LLM to synthesize an answer from retrieved chunk content.
    Show the initial question, then answer, referencing file name, page, index.
    """

    print("Executing final_answer_node...")

    # Find the LAST human question (most recent)
    last_human_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg
            break
    
    if not last_human_msg:
        return {
            "messages": [AIMessage(content="No user question found.")]
        }
    
    question = last_human_msg.content
    print("User question:", question)

    # Find the most recent ToolMessage from tool_3_retrieve_chunks_by_ids
    # Search backwards from the end to get the most recent one
    chunks_tool_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "tool_3_retrieve_chunks_by_ids":
            chunks_tool_msg = msg
            break
    
    if not chunks_tool_msg or not chunks_tool_msg.content:
        return {
            "messages": [AIMessage(content="No retrieved content found.")]
        }

    retrieved_chunks = chunks_tool_msg.content  # This will be a list of dicts

    # Prepare context for LLM
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_lines.append(
            f"{i}. [File: {chunk['file_name']} | Page: {chunk.get('page_number', '?')} | Start Index: {chunk.get('start_index', '?')}]\n"
            f"{chunk['content']}\n"
        )
    context_text = "\n".join(context_lines)

    system_prompt = (
        "You are a legal assistant. Given the following user's question and passages with source references, "
        "write a clear, concise and SOURCED answer in Romanian. Always cite, for every part of your answer, "
        "the file name, pagina (Page) and start index as shown below. "
        "If multiple passages match, synthesize and cite accordingly."
    )
    user_prompt = (
        f"Întrebare utilizator:\n{question}\n\n"
        "Conținut relevant extras:\n"
        f"{context_text}\n"
        "Răspuns final (cu referințe bibliografice la fiecare afirmație):"
    )

    # Compose and run LLM
    fm = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(fm)

    print("Executing final_answer_node END")

    return { "messages": [AIMessage(content=response.content, additional_kwargs={"final_response":True} )] }


# define conditional node
def condition_final_answer_node(state: State_MRR) -> Literal["request_reformulate_question", "END"]:
    last_ai_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    if last_ai_msg and "No retrieved content found." in (last_ai_msg.content or ""):
        print("No retrieved content found.")
        return "request_reformulate_question"
        
    return "END"


# function which request user to repeat the query
def request_reformulate_question(state: State_MRR):
    """
    Asks the user to repeat/reformulate their question and resets the message list for a new start.
    """
    print("Executing request_reformulate_question...")

    # Add notification to messages
    updated_messages = state["messages"] + [
        AIMessage(content="Vă rugăm să repetați sau să reformulați întrebarea.", 
                  additional_kwargs={"final_response":True})  # Romanian
    ]
    # Reset or keep necessary state (here keeping condition_1 as is, adjust if needed)
    print("Executing request_reformulate_question END")
    return {"messages": updated_messages }


# we need to remove information from First human message to next human message(Excluding)
# the AI messages with content between will become summrized as special variable 

def ai_summarizer(state: State_MRR, n_summarise=6, keep=3):
    """
    Summarizes and removes old AI messages, keeping only the most recent ones.
    
    Args:
        state: Current state containing messages
        n_summarise: Minimum number of AI messages needed before summarization
        keep: Number of recent final responses to keep
    """
    # Global setup
    messages_list = state["messages"].copy()
    messages_list_id = [m.id for m in messages_list]

    print("Executing ai_summarizer...")
    print("number_of_messages_beginning:", len(messages_list))

    # Find all final responses (AI messages marked as final)
    final_response_messages = [m for m in messages_list 
                              if isinstance(m, AIMessage) and 
                              m.additional_kwargs.get('final_response', False)]
    
    final_response_ids = [m.id for m in final_response_messages]

    # Need at least 'keep' final responses to proceed
    print("Count of final responses:", len(final_response_ids))
    
    if len(final_response_ids) < n_summarise:
        print("summarizer not_required - not enough final responses")
        return {"answer_repeat_ai_content_global": "", "question_type": ""}

    # Debug: show final response content
    final_response_content = [m.content[:50] for m in final_response_messages]
    print("final_response_content:", final_response_content)

    # Calculate cutoff point: keep the last 'keep' final responses
    # Everything up to and including the (len-keep)th final response should be in summary window
    cutoff_final_response_index = len(final_response_ids) - keep - 1
    
    if cutoff_final_response_index < 0:
        print("summarizer not_required - cutoff index negative")
        return {"answer_repeat_ai_content_global": "", "question_type": ""}
    
    cutoff_final_response_id = final_response_ids[cutoff_final_response_index]
    
    # Find the position of this cutoff final response in the full message list
    cutoff_position = messages_list_id.index(cutoff_final_response_id)
    
    # Select all messages from start up to and including the cutoff position
    delete_id_window_list = messages_list_id[:cutoff_position + 1]
    
    # Get AI messages with content in this window for summarization
    ai_messages_to_summarize = [m for m in messages_list 
                               if isinstance(m, AIMessage) and 
                               getattr(m, "content", None) and 
                               m.id in delete_id_window_list]

    print(f"AI messages to summarize: {len(ai_messages_to_summarize)}")

    # Debug: show messages that will be summarized
    ai_content_list = [m.content for m in ai_messages_to_summarize]
    print(f"Messages to summarize: {len(ai_content_list)}")
    for i, content in enumerate(ai_content_list):
        print(f"  {i+1}. {content[:100]}...")

    try:
        # Create summary content
        contents_to_summarize = "\n\n".join(ai_content_list)
        
        # Create summary prompt
        summary_prompt = SystemMessage(content="""Summarize the following AI assistant messages concisely, 
        capturing the key information, decisions made, and context that might be relevant for future interactions. 
        Focus on actionable insights and important details:\n\n""")
        
        human_message = HumanMessage(content=contents_to_summarize)
        summary_result = llm.invoke([summary_prompt, human_message])

        # Create delete operations for all messages in the window
        delete_messages = [RemoveMessage(id=m_id) for m_id in delete_id_window_list]
        
        print(f"Messages to delete: {len(delete_messages)}")
        print(f"Summary created with {len(summary_result.content)} characters")
        
        # Add summary as a system message or AI message depending on your needs
        summary_message = AIMessage(
            content=f"[SUMMARY OF PREVIOUS CONVERSATION]\n{summary_result.content}",
            additional_kwargs={'is_summary': True}
        )
        
        print("Executing ai_summarizer END")
        return {
            "messages": [summary_message] + delete_messages,
            "answer_repeat_ai_content_global": "",
            "question_type": "",
            "summary": summary_result.content,
        }
        
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return {"answer_repeat_ai_content_global": "", "question_type": ""}



# In[ ]:


## form a node to check duplicated messages
## this is conditional node, it does not update a state information

def detect_duplicate_message(state:State_MRR):
    
    """
    If the last user message is a duplicate (cosine similarity > 0.9), 
    return 'END'. Otherwise, continue to 'tool_node_chunk_selection_payload'.

    """
    print("Executing detect_duplicate_message...")

    
    messages_list_global = state["messages"].copy() # collect all messages
    messages_list_global_ids = [m.id for m in messages_list_global] # list with all indices

    # set of all human messages
    human_messages = [m for m in messages_list_global if isinstance(m, HumanMessage)]
    human_messages = human_messages[-3:]  # keep only the last 3 human messages

    human_messages_with_content_id = [m.id for m in human_messages]
    human_messages_with_content_list = [m.content for m in human_messages]

    # debug
    print(human_messages_with_content_id)
    print(human_messages_with_content_list)

    if len(human_messages_with_content_list) < 2:
       print("Less than 2 human messages, proceeding to tool selection")
       return {"answer_repeat_ai_content_global": ""}  # No need to set answer_repeat_ai_content_global since it's already empty from reset

    # check if the question was already asked, use cosine similarity
    embeddings = np.array(embedding_main.embed_documents(human_messages_with_content_list))
    reshaped = [e.reshape(1, -1) for e in embeddings]

    last_emb = reshaped[-1] # last human message
    prev_embs = reshaped[:-1] # prev human messages
    embeddings_matrix = np.vstack(prev_embs) # stack as required by cosine similarity in sklearn
    scores = cosine_similarity(embeddings_matrix, last_emb).flatten()
    print("Cosine similarity scores:", scores)

    # Find indices with score > 0.9
    hit_indices_local = np.where(scores >= 0.9)[0] # local indexes of previous messages
    print("Hit indices (local):", hit_indices_local)
    if len(hit_indices_local) > 0:
        duplicate_idx_local = hit_indices_local[-1]  # last such message if needed
        print(duplicate_idx_local)

        ## locatiomn of duplicate_idx in the global list of messages messages_list_ids
        global_human_index = messages_list_global_ids.index(human_messages_with_content_id[duplicate_idx_local])
        print("Duplicate found at global index:", global_human_index)
   
        # now find next ai message after the duplicate human message
        duplicate_ai_id_global = None
        duplicate_ai_msg_global = None

        for msg in messages_list_global[global_human_index+1:]:
            if isinstance(msg, AIMessage) and getattr(msg, "content", None):
                duplicate_ai_id_global = msg.id
                duplicate_ai_msg_global = msg.content
                print("Content of duplicate AI message:", duplicate_ai_msg_global[:100])  # print first 100 characters
                break
            
        return {"answer_repeat_ai_content_global":duplicate_ai_msg_global}

    return {"answer_repeat_ai_content_global":""} # empty string


def check_should_repeat_answer(state: State_MRR) -> Literal["request_new_question", "tool_node_chunk_selection_payload"]:
    """Conditional node to check if we should return last message"""
    
    if state["answer_repeat_ai_content_global"]:
        return "request_new_question"
    else: 
        return "tool_node_chunk_selection_payload"


# function which request user to repeat the query
def request_new_question(state: State_MRR):
    """Return previously generated reply for repeated questions"""
    print("Executing request_new_question...")

    prior_messages = state["answer_repeat_ai_content_global"]
    print(prior_messages)
    
    updated_messages = [AIMessage(content=prior_messages, additional_kwargs={"final_response":True} )  ] 
    
    print("Executing request_new_question END")
    return {"messages": updated_messages}


# In[ ]:


## Test classifier if intent

def intent_classifier_simple(state: State_MRR) -> dict:
    """
    Simple intent classifier for RAG workflow routing.
    Classifies user question into one of 7 intent types.
    """
    
    messages = state["messages"]
    
    # Get the last human message
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break
    
    if not last_human_message or not last_human_message.strip():
        return {"intent_type": "NEW_FACTUAL_QUERY"}
    
    print(f"Classifying intent for: {last_human_message[:100]}...")
    
    # System prompt for intent classification
    classification_prompt = SystemMessage(content="""
You are an intent classifier for a RAG system. Analyze the user's question and classify it into ONE of these 7 categories:

**PULL DATA (run retrieval/tools):**

1. **NEW_FACTUAL_QUERY** - New information needed
   - Questions about topics, legal articles, regulations
   - Factual questions requiring data retrieval
   - Examples: "Ce reglementează legea X?", "Care sunt sancțiunile pentru Y?"

2. **VERIFY_PRIOR** - "Check/verify/prove/source that..."
   - Requests to verify or confirm information
   - Examples: "Poți verifica asta?", "Care este sursa?", "Confirm this info"

3. **UPDATE_RECENCY_CHECK** - "What's the latest/current status/version?"
   - Questions asking for current or updated information
   - Examples: "Care este ultima versiune?", "What's the current status?"

4. **CLARIFY_INFO_GAP** - Question lacks specifics
   - Vague questions needing more context
   - Examples: "Tell me about banking rules" (which rules? which law?)

**DON'T PULL DATA (comment/reuse/transform):**

5. **COMMENT_ONLY** - "Explain/analyze/comment on..."
   - Requests for explanation or analysis
   - Examples: "Explică asta", "Ce înseamnă?", "Add more details"

6. **REUSE_FROM_CONTEXT** - Follow-up questions about same topic
   - Questions continuing previous discussion
   - Examples: "And what about the penalties?", "What else?"

7. **FORMAT_TRANSFORM** - "Rewrite as bullets/code/email"
   - Formatting or transformation requests
   - Examples: "Make this shorter", "Rewrite as bullet points"

Respond with only the category name: NEW_FACTUAL_QUERY, VERIFY_PRIOR, UPDATE_RECENCY_CHECK, CLARIFY_INFO_GAP, COMMENT_ONLY, REUSE_FROM_CONTEXT, or FORMAT_TRANSFORM
""")
    
    classification_request = HumanMessage(content=f"Classify this question: {last_human_message}")
    
    try:
        # Get classification from LLM
        classification_response = llm.invoke([classification_prompt, classification_request])
        intent_raw = classification_response.content.strip().upper()
        
        # Valid intent types
        valid_intents = [
            "NEW_FACTUAL_QUERY",
            "VERIFY_PRIOR", 
            "UPDATE_RECENCY_CHECK",
            "CLARIFY_INFO_GAP",
            "COMMENT_ONLY",
            "REUSE_FROM_CONTEXT",
            "FORMAT_TRANSFORM"
        ]
        
        # Find matching intent
        classified_intent = "NEW_FACTUAL_QUERY"  # default
        for intent in valid_intents:
            if intent in intent_raw:
                classified_intent = intent
                break
        
        print(f"Intent classified as: {classified_intent}")
        
        return {
            "intent_type": classified_intent,
            "classification_reasoning": classification_response.content
        }
        
    except Exception as e:
        print(f"Error in intent classification: {str(e)}")
        return {"intent_type": "NEW_FACTUAL_QUERY"}


# Example usage for testing
def test_classifier():
    """Test the classifier with sample questions"""
    
    # Mock state with sample messages
    from langchain.schema import HumanMessage, AIMessage
    
    test_cases = [
        {"messages": [HumanMessage(content="Ce reglementează legea nr. 100/2023?")]},
        {"messages": [HumanMessage(content="Poți verifica această informație?")]},
        {"messages": [HumanMessage(content="Care este ultima versiune a legii?")]},
        {"messages": [HumanMessage(content="Explică-mi ce înseamnă asta")]},
        {"messages": [HumanMessage(content="Reformulează ca listă cu puncte")]},
        {"messages": [HumanMessage(content="Spune-mi despre regulile bancare")]},
    ]
    
    for i, state in enumerate(test_cases):
        result = intent_classifier_simple(state)
        question = state["messages"][-1].content
        print(f"Q: {question}")
        print(f"Intent: {result['intent_type']}\n")

# Uncomment to test
test_classifier()


# In[ ]:


## construct agent 
agent_builder = StateGraph(State_MRR)

# nodes
agent_builder.add_node("tool_node_chunk_selection_payload",tool_node_chunk_selection_payload)
agent_builder.add_node("tool_node_chunk_selection_exec", tool_node_chunk_selection_exec)  # Runs the correct function
agent_builder.add_node("llm_create_retrieve_tool_payload",llm_create_retrieve_tool_payload)
agent_builder.add_node("llm_create_retrieve_tool_exec", llm_create_retrieve_tool_exec)
agent_builder.add_node("final_answer_node", final_answer_node)

# nodes for side events
agent_builder.add_node("ai_summarizer",ai_summarizer)
agent_builder.add_node("detect_duplicate_message",detect_duplicate_message)
agent_builder.add_node("request_new_question",request_new_question)
agent_builder.add_node("request_reformulate_question", request_reformulate_question)


## edges
agent_builder.add_edge(START ,"ai_summarizer" )
agent_builder.add_edge("ai_summarizer" ,"detect_duplicate_message" )

agent_builder.add_conditional_edges(
                                   "detect_duplicate_message", 
                                    check_should_repeat_answer,  
                                    {
                                      "tool_node_chunk_selection_payload": "tool_node_chunk_selection_payload", 
                                      "request_new_question": "request_new_question" 
                                    }
                                    )

# execute payload
agent_builder.add_edge("tool_node_chunk_selection_payload", "tool_node_chunk_selection_exec")

agent_builder.add_conditional_edges(
    "tool_node_chunk_selection_exec",
    check_chunks_found,
    {
        "llm_create_retrieve_tool_payload": "llm_create_retrieve_tool_payload",
        "request_reformulate_question": "request_reformulate_question"
    }
)

agent_builder.add_edge( "llm_create_retrieve_tool_payload", "llm_create_retrieve_tool_exec" )
agent_builder.add_edge( "llm_create_retrieve_tool_exec", "final_answer_node" )
agent_builder.add_edge( "final_answer_node", END )



agent_builder.add_edge("request_reformulate_question", END)
agent_builder.add_edge("request_new_question", END)
#agent_builder.add_edge( "final_answer_node", END ) # now it relies on condition

# COMPILE THE AGENT - This is essential for memory to work!
agent = agent_builder.compile(checkpointer=memory_component)

# Remove the old static config and create a dynamic config function
def get_config_for_thread(thread_id: str):
    """Create configuration with specific thread_id for persistent memory"""
    return {"configurable": {"thread_id": thread_id}}

print("Agent compiled with checkpointer - memory persistence enabled!")


