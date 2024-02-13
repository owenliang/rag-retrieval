from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 解析PDF，切成chunk片段
pdf_loader=PyPDFLoader('LLM.pdf',extract_images=True)   # 使用OCR解析pdf中图片里面的文字
chunks=pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)) # 每个chunk 500字, 相邻chunk之间有50字的重叠

# 加载embedding模型，用于将chunk向量化
embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base') 

# 将chunk插入到faiss本地向量数据库
vector_db=FAISS.from_documents(chunks,embeddings)
vector_db.save_local('LLM.faiss')

# 验证效果
chunks_with_score=vector_db.similarity_search_with_score('人工智能的发展历史',k=10)  # 搜索top 10结果
for chunk,score in chunks_with_score:
    print(chunk,score)