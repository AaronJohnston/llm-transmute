from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.responses import FileResponse

app = FastAPI()

@app.get('/')
async def index():
    return FileResponse('index.html')

@app.get('/healthcheck', response_class=PlainTextResponse)
async def test():
    return 'working'

@app.get('/llm', response_class=PlainTextResponse)
async def test():
    return 'LLM Transmute Test'