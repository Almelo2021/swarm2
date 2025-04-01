from fastapi import FastAPI
from pydantic import BaseModel
from agent import create_agent
from agents import Runner
import uvicorn

app = FastAPI(title="Local Sales Intelligence Agent API")

class QueryRequest(BaseModel):
    company: str
    query: str

class QueryResponse(BaseModel):
    result: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    agent = create_agent()
    formatted_query = f"Company: {request.company} Query: {request.query}"
    result = await Runner.run(agent, formatted_query)
    return {"result": result.final_output}

@app.get("/")
def read_root():
    return {"message": "Sales Intelligence Agent API is running. Send POST requests to /query"}

if __name__ == "__main__":
    uvicorn.run("local_api:app", host="127.0.0.1", port=8000, reload=True)