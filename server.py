from fastapi import FastAPI, Form
from typing import Annotated
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama

llm = Llama(model_path='./models/codellama-7b.Q4_K_M.gguf')

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/healthcheck', response_class=PlainTextResponse)
async def healthcheck():
    return 'working'

@app.post('/llm', response_class=PlainTextResponse)
async def query_llm(input: Annotated[str, Form()]):
    output = llm(
        input,
        max_tokens=32,
        stop=["Q:", "\n"]
    )
    return output['choices'][0]['text']