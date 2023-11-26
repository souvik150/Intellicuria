from fastapi import FastAPI
import pandas as pd
import sqlite3

app = FastAPI()

@app.get("/previous_research/")
def read_previous_research():
    # Your existing code for read_previous_research function
    pass
