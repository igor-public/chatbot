#OS, BedrockChat, ConversationChain, ConversationBufferMemory Langchain Modules

import os
# from langchain.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory # memory support
from langchain.chains import ConversationChain # converstaiton support

#invoking model- client connection with Bedrock,
#profile - "default", with the region and repsonse format (JSON) defined
#model_id must be avaialble to the profile (region, requested) 
#Inference params- model_kwargs - check here: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html

MODEL_ID_LLAMA =  'meta.llama2-70b-chat-v1'
MODEL_ID_CLAUDE = 'anthropic.claude-3-sonnet-20240229-v1:0'
AWS_PROFILE = 'default'

# _____________________________________

MODEL_ID = MODEL_ID_CLAUDE
MESSAGE_PROMPT = "what is EPC in Europe in relation to Patents?"

# test function 

def retrieve_response(input_text):
    demo_llm = ChatBedrock(
       credentials_profile_name = AWS_PROFILE,
       model_id = MODEL_ID,
       model_kwargs= {
        "temperature": 0.1,
        "top_p": 0.9, 
        "top_k": 50,
        })
    return demo_llm.invoke(input_text)

# actual use
def llm_chat():
    llm = ChatBedrock(
       credentials_profile_name = AWS_PROFILE,
       model_id = MODEL_ID,
       model_kwargs= {
        "temperature": 0.1,
        "top_p": 0.9, 
        "top_k": 50,
        })
    return llm

response = retrieve_response(MESSAGE_PROMPT)
print(response)

# ConversationBufferMemory (llm and max token limit)
def memory_llm():
    llm_data=llm_chat()
    memory = ConversationBufferMemory(llm=llm_data, max_token_limit= 512)
    return memory

# Conversation Chain - Input + Memory
def conversation_llm(input_text,memory):
    llm_chain_data = llm_chat()
    llm_conversation= ConversationChain(llm=llm_chain_data,memory= memory,verbose=True)

#5 Response using Predict
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply
    
