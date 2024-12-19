from chatbot import chatbot
from store import add_data

from typing import Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

# initialize app object
app = FastAPI()

# def delete_data(field):
#     return ''

@app.put('/insert_data')
async def insert_data(csv_path: str):
    return add_data(file_path=csv_path)


# define processing function with a specific route
@app.get('/')
async def root():
    return {'message': 'Hello World!'}


@app.post('/chat/{user_id}/{session_id}')
async def answer(user_id: str, session_id: str, question: str=''):
    try:
        return StreamingResponse(chatbot.stream(question=question, user_id=user_id, session_id=session_id))
    except Exception as e:
        return e
    

@app.get('/chat/history/{user_id}/{session_id}')
async def get_full_history(user_id: str, session_id: str):
    try:
        return chatbot.get_full_history(user_id=user_id, session_id=session_id)
    except Exception as e:
        return e


@app.post('/chat/clear/{user_id}/{session_id}')
async def remove_session(user_id: str, session_id: str):
    try:
        chatbot.remove_session_history(user_id=user_id, session_id=session_id)
        return {
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'fail',
            'exception': e
        }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1=', port=5123)