# # Configuration ----------------------------------------------------------------------------------
# # LLM params
MODEL_NAME = 'gemini-1.5-pro-latest'
API_KEY = 'AIzaSyAnVKQx0OY53BX0AFlviPDu7T6OYsFOCc8'
TEMPERATURE = 0.7
MAX_RETRIES = 6


# ------------------------------------------------------------------------------------------------
from retriever import history_aware_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Define LLM -------------------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME, 
    google_api_key=API_KEY,
    temperature=TEMPERATURE,
    max_retries=MAX_RETRIES,
)

# Define Prompt -----------------------------------------------------------------------------------
qa_system_prompt = """
Bạn là một trợ lý ảo thông minh được thiết kế để cung cấp câu trả lời về thị trường chứng khoán Việt Nam dựa trên các tài liệu tham khảo đã cho. Vui lòng làm theo các bước sau:
- Đọc và phân tích tài liệu tham khảo.
- Sử dụng thông tin từ tài liệu để trả lời câu hỏi đã cho; hãy trả lời với số liệu, thời gian và dẫn chứng cụ thể.
- Nếu trong tài liệu có `Link truy cập`, hãy cung cấp link đó trong câu trả lời của mình.
- Nếu không tìm thấy đủ thông tin trong tài liệu, hãy báo rằng "không có thông tin đủ".

Tài liệu tham khảo:
{context}

"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# intialize question answer chain ------------------------------------------------------------------------------------
qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)


# set up Retrieval Augmented Chain -----------------------------------------------------------------------------------
rag_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=qa_chain)