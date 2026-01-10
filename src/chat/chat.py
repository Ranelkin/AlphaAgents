import time
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
        logger.info("Shutting down application")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

atexit.register(_shutdown)

def chat_interface(): 
    # Streamlit config
    st.set_page_config(page_title="AlphaAgents", layout="wide")
    st.title("AlphaAgents")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

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
    
        # Set previous session conversations in sidebar
        st.markdown('---')
        st.markdown('Conversation History')

        if st.session_state.conversation_history:
            for idx, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
                if st.button(f"{conv['ticker']} - {conv['timestamp']}", key=f"hist_{idx}"):
                    st.session_state.selected_conversation = conv
        else:
            st.info("No conversation history yet")

        st.markdown("---")

        if st.button('Clear history'):
            st.session_state.conversation_history = []
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
            
        ticker = extract_ticker(prompt)
        st.write_stream(chat_stream(f'Analyzing the following ticker: {ticker}'))    
        with st.chat_message("assistant"):
            try:
                

                fund_exp = st.expander("Fundamental Analyst", expanded=True)
                sent_exp = st.expander("Sentiment Analyst", expanded=True)
                val_exp = st.expander("Valuation Analyst", expanded=True)
                debate_exp = st.expander("Debate & Consensus", expanded=True)
                
                with st.spinner("Initializing agents..."):
                    agents = create_agents(ticker)
                    time.sleep(0.5)
                
                with fund_exp:
                    status_fund = st.empty()
                    st.write_stream(chat_stream("Analyzing 10-K/10-Q filings..."))
                    
                with sent_exp:
                    status_sent = st.empty()
                    st.write_stream(chat_stream("Analyzing market sentiment and news..."))
                    
                with val_exp:
                    status_val = st.empty()
                    st.write_stream(chat_stream("Analyzing price trends and valuation..."))
                
                debate_status = st.empty()
                debate_status.info(f"🔄 Agents are debating on {ticker}...")

                result = run_debate(ticker, mode="debate")
                
                status_fund.empty()
                status_sent.empty()
                status_val.empty()
                debate_status.empty()

                agent_messages = {
                    'Fundamental_Analyst': [],
                    'Sentiment_Analyst': [],
                    'Valuation_Analyst': [],
                    'chat_manager': []
                }
                chat_history = result.chat_history
                
                for msg in chat_history:
                    agent_name = msg.get('name', 'Unknown')
                    content = msg.get('content', '')
                    agent_messages[agent_name].append(content)
                
                with fund_exp:
                    if agent_messages['Fundamental_Analyst']:
                        for msg in agent_messages['Fundamental_Analyst']:
                            # Stream the message word by word
                            msg_placeholder = st.empty()
                            streamed = ""
                            for word in chat_stream(msg):
                                streamed += word
                                msg_placeholder.markdown(streamed)
                    else:
                        st.info("No analysis from Fundamental Agent")
                
                with sent_exp:
                    if agent_messages['Sentiment_Analyst']:
                        for msg in agent_messages['Sentiment_Analyst']:
                            msg_placeholder = st.empty()
                            streamed = ""
                            for word in chat_stream(msg):
                                streamed += word
                                msg_placeholder.markdown(streamed)
                    else:
                        st.info("No analysis from Sentiment Agent")
                
                with val_exp:
                    if agent_messages['Valuation_Analyst']:
                        for msg in agent_messages['Valuation_Analyst']:
                            msg_placeholder = st.empty()
                            streamed = ""
                            for word in chat_stream(msg):
                                streamed += word
                                msg_placeholder.markdown(streamed)
                    else:
                        st.info("No analysis from Valuation Agent")
                
                with debate_exp:
                    if agent_messages['chat_manager']:
                        for agent_name, content in agent_messages['chat_manager']:
                            msg_placeholder = st.empty()
                            streamed = f"**{agent_name}:** "
                            for word in chat_stream(content):
                                streamed += word
                                msg_placeholder.markdown(streamed)
                    else:
                        st.info("No debate messages recorded")
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