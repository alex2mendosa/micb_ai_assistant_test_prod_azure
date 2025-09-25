#!/usr/bin/env python
# coding: utf-8

# #### Streamlit component

# In[ ]:


import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import logging
import os

# Import your modules - adjust these imports based on your actual module structure
try:
    from __config_2 import embedding_main, llm, vector_store_fin_law_hq, vector_store_fin_law
    from __agent_workflow import agent, get_config_for_thread, memory_component
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        # Create a unique thread ID for this session
        st.session_state.thread_id = str(uuid.uuid4())
        #logger.info(f"New session thread ID: {st.session_state.thread_id}")

@st.cache_resource
def get_agent_instance():
    """Cache the agent to avoid recompilation"""
    try:
        return agent
    except Exception as e:
        #logger.error(f"Failed to get agent: {e}")
        raise

def get_agent_response(user_input):
    """Get response from your LangGraph agent with persistent memory"""
    try:
        # Create config with the session's thread_id for persistent memory
        session_config = get_config_for_thread(st.session_state.thread_id)
        
        #logger.info(f"Invoking agent with thread_id: {st.session_state.thread_id}")
        #logger.info(f"User input: {user_input}")
        
        # Get the cached agent
        cached_agent = get_agent_instance()
        
        # Invoke the agent with the persistent thread_id
        result = cached_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=session_config
        )
        
        #logger.info(f"Agent returned {len(result.get('messages', []))} messages")
        
        # Extract the latest AI reply (last AIMessage in messages)
        ai_msg = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None
        )
        
        if ai_msg:
            #logger.info("Successfully extracted AI response")
            return ai_msg.content
        else:
            #logger.warning("No AI message found in agent response")
            return "Îmi pare rău, nu am putut genera un răspuns."
            
    except Exception as e:
        error_msg = f"Error getting agent response: {str(e)}"
        #logger.error(error_msg)
        st.error(error_msg)
        return "Îmi pare rău, a apărut o eroare la procesarea cererii dvs."

def display_chat_messages():
    """Display all chat messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="Asistent Juridic AI",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Asistent Juridic AI")
    st.markdown("*Asistent pentru legislația financiară cu memorie conversațională*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Panou de Control")
        
        # Display session info
        st.info(f"ID Sesiune: {st.session_state.thread_id[:8]}...")
        
        # Clear chat button
        if st.button("🗑️ Șterge Conversația", type="secondary"):
            st.session_state.messages = []
            # Create new thread ID for fresh conversation
            st.session_state.thread_id = str(uuid.uuid4())
            #logger.info(f"Chat cleared, new thread ID: {st.session_state.thread_id}")
            st.rerun()
        
        # Display message count
        st.metric("Mesaje", len(st.session_state.messages))
        
        # Agent status
        st.header("🔧 Status Agent")
        try:
            # Test if agent is working
            get_agent_instance()
            st.success("✅ Agent Pregătit")
        except Exception as e:
            st.error(f"❌ Eroare Agent: {str(e)}")
        
        st.divider()
        
        # Instructions
        st.markdown("""
        **Cum să folosești asistentul:**
        - Pune întrebări despre legislația financiară
        - Agentul își va aminti conversația anterioară
        - Poți face referiri la întrebările precedente
        """)
    
    # Main chat interface
    display_chat_messages()
    
    # Display welcome message if no conversation yet
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            welcome_msg = "Salut! Sunt asistentul tău juridic AI. Poți să mă întrebi orice despre legislația financiară din Moldova. De exemplu: 'Care sunt funcțiile principale ale Băncii Naționale a Moldovei?'"
            st.markdown(welcome_msg)
    
    # Chat input
    if prompt := st.chat_input("Întreabă despre legislația financiară..."):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Se procesează întrebarea..."):
                response_content = get_agent_response(prompt)
                st.markdown(response_content)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()

