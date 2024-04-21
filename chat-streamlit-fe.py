import streamlit as st 
import  llm_bedrock_be as test_chat  

# title - https://docs.streamlit.io/library/api-reference/text/st.title
st.title("Test Chat") 

# langchain for the session State - https://docs.streamlit.io/library/api-reference/session-state
if 'memory' not in st.session_state: 
    st.session_state.memory = test_chat.memory_llm()

# UI chat history to the session  https://docs.streamlit.io/library/api-reference/session-state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# show chat history - to keep the earlier messages
for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 

# Enter the details for chatbot input box 
     
input_text = st.chat_input("Chat using Bedrock + LLM") 
if input_text: 
    
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({"role":"user", "text":input_text}) 

    chat_response = test_chat.conversation_llm(input_text=input_text, memory=st.session_state.memory) #** replace with ConversationChain Method name - call the model through the supporting library
    
    with st.chat_message("assistant"): 
        st.markdown(chat_response) 
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 