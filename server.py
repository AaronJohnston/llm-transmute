from fastapi import FastAPI, Form, Request
from typing import Annotated
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from LLMS import LLMS

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

llms = LLMS()


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/llm', response_class=HTMLResponse)
async def query_llm(request: Request, input: Annotated[str, Form()]):
    outputs = llms.inference(input)
    return templates.TemplateResponse('llm_response.html', {'request': request, 'input': input, 'outputs': outputs})
