# configuration --------------------------------------------------------------------------------------------
MONGO_URL = 'mongodb://mongodb:27017'
QDRANT_URL = 'http://qdrant_db:6333'
# ELASTICSEARCH_URL = 'http://localhost:9200'
COLLECTION_NAME = 'stock-news'
ID_KEY = 'docs'


# document store -------------------------------------------------------------------------------------------
from langchain_community.storage import MongoDBStore
docstore = MongoDBStore(
    connection_string=MONGO_URL,
    db_name='mongodb',
    collection_name=COLLECTION_NAME
)


# vector store ---------------------------------------------------------------------------------------------
from embedding import biencoder_embedding, phobert_embedding, sbert_embedding
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# QDrant store with PhoBERT embedding
if f'{COLLECTION_NAME}-with-bi-encoder' not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=f'{COLLECTION_NAME}-with-bi-encoder',
        vectors_config=models.VectorParams(
            size=biencoder_embedding.get_embedding_dim(), 
            distance=models.Distance.COSINE
        )
    )
vectorstore_biencoder = Qdrant(
    client=qdrant_client,
    collection_name=f'{COLLECTION_NAME}-with-bi-encoder',
    embeddings=biencoder_embedding,
    distance_strategy='COSINE'
)

# QDrant store with PhoBERT embedding
if f'{COLLECTION_NAME}-with-phobert' not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=f'{COLLECTION_NAME}-with-phobert',
        vectors_config=models.VectorParams(
            size=phobert_embedding.get_embedding_dim(), 
            distance=models.Distance.COSINE
        )
    )
vectorstore_phobert = Qdrant(
    client=qdrant_client,
    collection_name=f'{COLLECTION_NAME}-with-phobert',
    embeddings=phobert_embedding,
    distance_strategy='COSINE'
)

# QDrant store with S-BERT embedding
if f'{COLLECTION_NAME}-with-sbert' not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=f'{COLLECTION_NAME}-with-sbert',
        vectors_config=models.VectorParams(
            size=sbert_embedding.get_embedding_dim(), 
            distance=models.Distance.COSINE
        )
    )
vectorstore_sbert = Qdrant(
    client=qdrant_client,
    collection_name=f'{COLLECTION_NAME}-with-sbert',
    embeddings=sbert_embedding,
    distance_strategy='COSINE'
)

# QDrant store with Generative AI embedding
# if f'{COLLECTION_NAME}-with-genai' not in [c.name for c in qdrant_client.get_collections().collections]:
#     qdrant_client.create_collection(
#         collection_name=f'{COLLECTION_NAME}-with-genai',
#         vectors_config=models.VectorParams(
#             size=sbert_embedding.get_embedding_dim(), 
#             distance=models.Distance.COSINE
#         )
#     )
# vectorstore_genai = Qdrant(
#     client=qdrant_client,
#     collection_name=f'{COLLECTION_NAME}-with-genai',
#     embeddings=sbert_embedding,
#     distance_strategy='COSINE'
# )


# initialize text splitters -----------------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

biencoder_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=biencoder_embedding.tokenizer,
    separators=["\n\n", "\n", ".", " ", ""], 
    chunk_size=round(0.5 * biencoder_embedding.get_max_seq_len()), 
    chunk_overlap=round(0.1 * biencoder_embedding.get_max_seq_len()),
    is_separator_regex=False
)

phobert_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=phobert_embedding.tokenizer,
    separators=["\n\n", "\n", ".", " ", ""], 
    chunk_size=round(0.5 * phobert_embedding.get_max_seq_len()), 
    chunk_overlap=round(0.1 * phobert_embedding.get_max_seq_len()),
    is_separator_regex=False
)

sbert_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=sbert_embedding.tokenizer,
    separators=["\n\n", "\n", ".", " ", ""], 
    chunk_size=round(0.5 * sbert_embedding.get_max_seq_len()), 
    chunk_overlap=round(0.1 * sbert_embedding.get_max_seq_len()),
    is_separator_regex=False
)


# bm25 elastic search ----------------------------------------------------------------------------------------
# from elasticsearch import Elasticsearch
# # Define the index settings and mappings
# settings = {
#     "analysis": {"analyzer": {"default": {"type": "standard"}}},
#     "similarity": {
#         "custom_bm25": {
#             "type": "BM25",
#             "k1": 2,
#             "b": 0.75,
#         }
#     },
# }
# mappings = {
#     "properties": {
#         "content": {
#             "type": "text",
#             "similarity": "custom_bm25",  # Use the custom BM25 similarity
#         }
#     }
# }
# es_client = Elasticsearch(ELASTICSEARCH_URL)
# if not es_client.indices.exists(index=COLLECTION_NAME):
#     es_client.indices.create(index=COLLECTION_NAME, settings=settings, mappings=mappings)



# add data to stores ---------------------------------------------------------------------------
from uuid import uuid4
from langchain_community.document_loaders import CSVLoader
# from elasticsearch.helpers import bulk

def add_data(file_path):
    try:
        documents = CSVLoader(file_path=file_path, metadata_columns=['link', 'time'], encoding='utf-8').load()
        doc_ids = [str(uuid4()) for _ in documents]

        biencoder_docs, phobert_docs, sbert_docs, full_docs = [], [], [], []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            
            biencoder_sub_docs = biencoder_text_splitter.split_documents([doc])
            phobert_sub_docs = phobert_text_splitter.split_documents([doc])
            sbert_sub_docs = sbert_text_splitter.split_documents([doc])
            for _biencoder_doc, _phobert_doc, _sbert_doc in zip(biencoder_sub_docs, phobert_sub_docs, sbert_sub_docs):
                _biencoder_doc.metadata[ID_KEY] = _id
                _phobert_doc.metadata[ID_KEY] = _id
                _sbert_doc.metadata[ID_KEY] = _id

            biencoder_docs.extend(biencoder_sub_docs)
            phobert_docs.extend(phobert_sub_docs)
            sbert_docs.extend(sbert_sub_docs)
            full_docs.append((_id, doc))
            # requests.append({
            #     "_op_type": "index",
            #     "_index": COLLECTION_NAME,
            #     "page_content": doc.page_content,
            #     "metadata": doc.metadata,
            #     "_id": _id,
            # })
        
        vectorstore_biencoder.add_documents(biencoder_docs)
        vectorstore_phobert.add_documents(phobert_docs)
        vectorstore_sbert.add_documents(sbert_docs)
        docstore.mset(full_docs)
        # vectorstore_genai.add_documents(documents)
        # bulk(es_client, requests)
        # es_client.indices.refresh(index=COLLECTION_NAME)

        return {
            'status': 'success',
            'number_of_documents': len(documents)
        }
    except Exception as e:
        return {
            'status': 'error',
            'except': e
        }
    
# chat history store ------------------------------------------------------------------
from typing import List, Optional
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory as RedisChatMessageHistoryBase

class RedisChatMessageHistory(RedisChatMessageHistoryBase):
    """Chat message history stored in a Redis database."""

    def __init__(
        self,
        session_id: str,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
        k: int=4
    ):
        super().__init__(session_id=session_id, url=url, key_prefix=key_prefix, ttl=ttl)
        self.k = k

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the K lastest messages from Redis"""
        return super().messages[-self.k:]


    def get_full_history(self):
        """Retrieve the messages from Redis"""
        return super().messages


# if __name__ == '__main__':
#     # file_paths = [
#     #     r'D:\NLP_Project\data\data_dantri.csv',
#     #     r'D:\NLP_Project\data\data_VNE.csv'
#     # ]
#     # for file_path in file_paths:
#     #     documents = CSVLoader(file_path=file_path, metadata_columns=['link', 'time'], encoding='utf-8').load()
#     #     print(add_data(documents))