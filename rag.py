from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os 

# 加载embedding模型，用于将query向量化
embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base') 

# 加载faiss向量库，用于知识召回
vector_db=FAISS.load_local('LLM.faiss',embeddings)
retriever=vector_db.as_retriever(search_kwargs={"k":5})

# 用vllm部署openai兼容的服务端接口，然后走ChatOpenAI客户端调用
os.environ['VLLM_USE_MODELSCOPE']='True'
chat=ChatOpenAI(
    model="qwen/Qwen-7B-Chat-Int4",
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>']
)

# Prompt模板
system_prompt=SystemMessagePromptTemplate.from_template('You are a helpful assistant.')
user_prompt=HumanMessagePromptTemplate.from_template('''
Answer the question based only on the following context:

{context}

Question: {query}
''')
full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="chat_history"),user_prompt])

'''
<|im_start|>system
You are a helpful assistant.
<|im_end|>
...
<|im_start|>user
Answer the question based only on the following context:

{context}

Question: {query}
<|im_end|>
<|im_start|>assitant
......
<|im_end|>
'''

# Chat chain
chat_chain={
        "context": itemgetter("query") | retriever,
        "query": itemgetter("query"),
        "chat_history":itemgetter("chat_history"),
    }|full_chat_prompt|chat

# 开始对话
chat_history=[]
while True:
    query=input('query:')
    response=chat_chain.invoke({'query':query,'chat_history':chat_history})
    chat_history.extend((HumanMessage(content=query),response))
    print(response.content)
    chat_history=chat_history[-20:] # 最新10轮对话