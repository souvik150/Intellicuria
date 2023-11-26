from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ResearchRequest(BaseModel):
    userInput: str

@app.post("/generate_research/")
def generate_research_api(request: ResearchRequest):
    # Your existing code for generate_research_api function
    pass
