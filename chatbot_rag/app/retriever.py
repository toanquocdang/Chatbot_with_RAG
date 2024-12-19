# Retrieved parameters ----------------------------------------------------------------------------
# search params
SEARCH_TYPE = 'mmr'
K = 10
FETCH_K = 20
LAMBDA_MULT = 0.35

# ensemble params
WEIGHTS = [0.4, 0.4, 0.2]
C = 60
NUM_RETRIEVES = 3

# LLM params
MODEL_NAME = "gemini-1.5-pro-latest"
API_KEY = 'AIzaSyAnVKQx0OY53BX0AFlviPDu7T6OYsFOCc8'
TEMPERATURE = 0.0
MAX_RETRIES = 4


# initialize storages -----------------------------------------------------------------------------
from store import docstore, vectorstore_biencoder, vectorstore_phobert, vectorstore_sbert, ID_KEY, COLLECTION_NAME

# build unique retrievers -------------------------------------------------------------
from langchain.retrievers import MultiVectorRetriever

biencoder_multivector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore_biencoder,
    docstore=docstore,
    id_key=ID_KEY,
    search_type='mmr',
    search_kwargs={
        'k': K,
        'fetch_k': FETCH_K,
        'lambda_mult': LAMBDA_MULT
    }
)

phobert_multivector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore_phobert,
    docstore=docstore,
    id_key=ID_KEY,
    search_type='mmr',
    search_kwargs={
        'k': K,
        'fetch_k': FETCH_K,
        'lambda_mult': LAMBDA_MULT
    }
)

sbert_multivector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore_sbert,
    docstore=docstore,
    id_key=ID_KEY,
    search_type='mmr',
    search_kwargs={
        'k': K,
        'fetch_k': FETCH_K,
        'lambda_mult': LAMBDA_MULT
    }
)

# genai_retriever = vectorstore_genai.as_retriever(
#     search_type='mmr',
#     search_kwargs={
#         'k': K,
#         'fetch_k': FETCH_K,
#         'lambda_mult': LAMBDA_MULT
#     }
# )
# BM25 Retriever -------------------------------------------------------------------------------
# from typing import List
# from elasticsearch import Elasticsearch
# from langchain_core.callbacks import CallbackManagerForRetrieverRun
# from langchain_core.documents import Document
# from langchain_core.retrievers import BaseRetriever

# class ElasticSearchBM25Retriever(BaseRetriever):
    
#     client: Elasticsearch
#     """Elasticsearch client."""
#     index_name: str
#     """Name of the index to use in Elasticsearch."""
#     k: int = 10
#     """Number of retrieved documents"""

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         query_dict = {"query": {"match": {"page_content": query}}}
#         res = self.client.search(index=self.index_name, body=query_dict)
        
#         docs = []
#         for r in res["hits"]["hits"][:self.k]:
#             docs.append(Document(page_content=r["_source"]["page_content"], metadata=r["_source"]["metadata"]))
#         return docs
    
#     async def _aget_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         query_dict = {"query": {"match": {"page_content": query}}}
#         res = self.client.search(index=self.index_name, body=query_dict)
        
#         docs = []
#         for r in res["hits"]["hits"][:self.k]:
#             docs.append(Document(page_content=r["_source"]["page_content"], metadata=r["_source"]["metadata"]))
#         return docs

# bm25_retriever = ElasticSearchBM25Retriever(client=es_client, index_name=COLLECTION_NAME, k=K)


# Ensemble Retriever ---------------------------------------------------------------------------
from typing import List, Optional, Any
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain.retrievers import EnsembleRetriever as EnsembleRetrieverBase

class EnsembleRetriever(EnsembleRetrieverBase):
    retrievers: List[RetrieverLike]
    weights: List[float]
    c: int = 60
    k: int = 4
    
    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        return super().invoke(input=input, config=config, **kwargs)[:self.k]

    async def ainvoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        results = await super().ainvoke(input=input, config=config, **kwargs)
        return results[:self.k]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        fused_documents = self.rank_fusion(query, run_manager)
        return fused_documents[:self.k]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        fused_documents = await self.arank_fusion(query, run_manager)
        return fused_documents[:self.k]

ensemble_retriever = EnsembleRetriever(
    retrievers=[ 
        biencoder_multivector_retriever, 
        phobert_multivector_retriever, 
        sbert_multivector_retriever,
        # genai_retriever
        # bm25_retriever
    ], 
    weights=WEIGHTS, c=C, k=NUM_RETRIEVES
)


# Flash ReRanking Order -------------------------------------------------------------------------
# from flashrank import Ranker
# from langchain.retrievers.document_compressors import FlashrankRerank
# flashrank_compressor = FlashrankRerank(
#     client=Ranker(
#         model_name="ms-marco-MultiBERT-L-12", 
#         cache_dir=r"D:\NLP_Project\models"
#     ), 
#     top_n=4, 
#     model="ms-marco-MultiBERT-L-12"
# )


# # Final Retriever --------------------------------------------------------------------------------
# from langchain.retrievers import ContextualCompressionRetriever
# retriever = ContextualCompressionRetriever(
#     base_retriever=ensemble_retriever, base_compressor=flashrank_compressor
# )

# History Aware Retriever -----------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
# define LLM
history_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME, 
    google_api_key=API_KEY,
    temperature=TEMPERATURE,
    max_retries=MAX_RETRIES,
)
# define prompt
contextualize_question_system_prompt = """
Với lịch sử trò chuyện và câu hỏi mới nhất của người dùng về thị trường chứng khoán Việt Nam đã được cho trước. 
Hãy dùng chúng như là ngữ cảnh của cuộc trò chuyện để tạo ra một câu hỏi độc lập mà có thể hiểu được 
mà không cần đến lịch sử cuộc trò chuyện. Vui lòng không trả lời câu hỏi, chỉ cần định dạng lại câu hỏi, 
và nếu không định dạng lại được thì hãy trả lại nguyên trạng câu hỏi ban đầu.

"""
contextualize_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_question_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# intialize retriever
history_aware_retriever = create_history_aware_retriever(
    llm=history_llm, retriever=ensemble_retriever, prompt=contextualize_question_prompt
)