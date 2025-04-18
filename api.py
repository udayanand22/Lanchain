# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from narad import get_narada_story  # 👈 Import the function from narad.py

app = FastAPI()

class StoryRequest(BaseModel):
    topic: str

@app.post("/get_story/")
async def get_story(request: StoryRequest):
    return get_narada_story(request.topic)
