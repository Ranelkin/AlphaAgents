import streamlit as st
import atexit
from src.util.log_config import setup_logging
from src.graph import run_debate, create_agents
from src.util import extract_ticker
from .stream import chat_stream

logger = setup_logging('chat.streamlit')


def extract_clean_text(content):
    """Extract clean text from various message formats"""
    if isinstance(content, str):
        if "structuredContent=" in content:
            import re
            match = re.search(r"'result':\s*'([^']+)'", content)
            if match:
                return match.group(1)
        return content
    return str(content)

def _shutdown():
    try:
        logger.info("MCP shutdown")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

atexit.register(_shutdown)

def chat_interface(): 
    # Streamlit config
    st.set_page_config(page_title="AlphaAgents", layout="wide")
    st.title("AlphaAgents")
    with st.sidebar:
        st.markdown(
            """
            - **Fundamental Agent**  
            - **Sentiment Agent**  
            - **Valuation Agent**  
            """
        )
        if st.button("End Session"):
            st.success("Session cleaned up")
            st.rerun()
    
    ####################
    # Chat configuration
    ####################
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    
    if prompt := st.chat_input("What would you like to do?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            ticker = extract_ticker(prompt)
            st.write_stream(chat_stream(f'Analyzing the following ticker: {ticker}'))
            try:
                
                fund_exp = st.expander("Fundamental Analyst", expanded=True)
                sent_exp = st.expander("Sentiment Analyst", expanded=True)
                val_exp = st.expander("Valuation Analyst", expanded=True)
                debate_exp = st.expander("Debate & Consensus", expanded=True)
                
                with st.spinner("Initializing agents..."):
                    agents = create_agents(ticker)
                
                with fund_exp:
                    st.write_stream(chat_stream("Analyzing 10-K/10-Q filings..."))
                    
                with sent_exp:
                    st.write_stream(chat_stream("Analyzing market sentiment and news..."))
                    
                with val_exp:
                    st.write_stream(chat_stream("Analyzing price trends and valuation..."))
                
                with st.spinner(f"Agents debating on {ticker}..."):
                    result = run_debate(ticker, mode="debate")
                
                chat_history = result.chat_history
                
                for msg in chat_history:
                    agent_name = msg.get('name', 'Unknown')
                    content = msg.get('content', '')
                    
                    if 'Fundamental' in agent_name:
                        with fund_exp:
                            st.markdown(content)
                    elif 'Sentiment' in agent_name:
                        with sent_exp:
                            st.markdown(content)
                    elif 'Valuation' in agent_name:
                        with val_exp:
                            st.markdown(content)
                    else:
                        with debate_exp:
                            st.markdown(f"**{agent_name}:** {content}")
                
                # Final rec
                final_answer = chat_history[-1].get('content', 'No recommendation')
                st.success(f"**Final Recommendation:** {final_answer}")
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                st.error(f"Error analyzing {ticker}: {e}")
        
        # Session saving 
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer #type: ignore 
        })