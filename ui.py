import streamlit as st
from qa_agent import QAAgent

st.set_page_config(page_title="QA Agent", layout="centered")
st.title("QA Agent")

# INIT SESSION STATE
if "agent" not in st.session_state:
    st.session_state.agent = QAAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]

# RENDER CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# CHAT INPUT
prompt = st.chat_input("Ask a question")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.write(prompt)

    # Get agent response
    response = st.session_state.agent.run(prompt)

    # Show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    with st.chat_message("assistant"):
        st.write(response)