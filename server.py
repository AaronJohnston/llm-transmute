from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'sup'}

@app.get('/test')
async def test():
    return 'LLM Transmute Test'