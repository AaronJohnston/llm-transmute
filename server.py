from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.responses import FileResponse
from llama_cpp import Llama

llm = Llama(model_path='./models/codellama-7b.Q4_K_M.gguf')

app = FastAPI()

@app.get('/')
async def index():
    return FileResponse('index.html')

@app.get('/healthcheck', response_class=PlainTextResponse)
async def healthcheck():
    return 'working'

@app.get('/llm', response_class=PlainTextResponse)
async def query_llm():
    output = llm(
        'Q: What is the peak of an orbit called? A:',
        max_tokens=32,
        stop=["Q:", "\n"]
    )
    return output['choices'][0]['text']