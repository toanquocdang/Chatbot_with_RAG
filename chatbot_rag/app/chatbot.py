# configurations ---------------------------------------------------------------------------------------
REDIS_URL = 'redis://redis:6379'
COLLECTION_NAME = 'stock-news'
HISTORY_WINDOW_K = 10

# build chatbot ----------------------------------------------------------------------------------------
from rag_chain import rag_chain
from store import RedisChatMessageHistory

from typing import List, Iterator
from langchain_core.messages import BaseMessage
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ChatBot:
    def __init__(self):
        self.conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True
                ),
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True
                ),
            ],
            output_messages_key="answer"
        )

    
    def answer(self, question: str, user_id: str, session_id: str) -> str:
        return self.conversational_chain.invoke(                    
            {'input': question},
            config={
                'configurable': {'user_id': user_id, 'session_id': session_id}
            }
        )['answer']

    
    def stream(self, question: str, user_id: str, session_id: str):
        for chunk in self.conversational_chain.stream(
            {'input': question},
            config={
                'configurable': {'user_id': user_id, 'session_id': session_id}
            }
        ):
            yield chunk.get('answer', '')


    def get_session_history(self, user_id: str, session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL, key_prefix=f'{COLLECTION_NAME}_{user_id}_', k=HISTORY_WINDOW_K)

    
    def get_full_history(self, user_id: str, session_id: str) -> List[BaseMessage]:
        return self.get_session_history(user_id=user_id, session_id=session_id).get_full_history()
    

    def remove_session_history(self, user_id: str, session_id: str):
        self.get_session_history(user_id=user_id, session_id=session_id).clear()
    
chatbot = ChatBot()