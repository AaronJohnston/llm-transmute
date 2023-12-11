from fastapi import FastAPI, Form, Request
from typing import Annotated
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama

llm = Llama(model_path='./models/codellama-7b.Q4_K_M.gguf')

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')

templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/healthcheck', response_class=PlainTextResponse)
async def healthcheck():
    return 'working'

@app.post('/llm', response_class=HTMLResponse)
async def query_llm(request: Request, input: Annotated[str, Form()]):
    output = llm(
        input,
        max_tokens=32,
        stop=["Q:", "\n"]
    )
    return templates.TemplateResponse('llm_response.html', {'request': request, 'input': input, 'output': output['choices'][0]['text']})