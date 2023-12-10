
import os
import sqlite3
import pandas as pd
import json
from fastapi import FastAPI, Form
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import Chroma
import uvicorn
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

app = FastAPI()

os.environ['OPENAI_API_KEY'] = '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'


def create_research_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT
            )
        """)


def read_research_table():
    with sqlite3.connect('MASTER.db') as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df


def insert_research(user_input, introduction, quant_facts, publications, books):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()

        # Convert dictionaries to strings
        introduction_str = json.dumps(introduction)
        quant_facts_str = json.dumps(quant_facts)
        publications_str = json.dumps(publications)
        books_str = json.dumps(books)

        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books)
            VALUES (?, ?, ?, ?, ?)
        """, (user_input, introduction_str, quant_facts_str, publications_str, books_str))


def generate_research(userInput, TEMP):
    llm = OpenAI(temperature=TEMP, api_key='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')
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Wikipedia Research Tool",
            func=wiki.run,
            description="Useful for researching information on Wikipedia"
        ),
        Tool(
            name="Duck Duck Go Search Results Tool",
            func=DDGsearch.run,
            description="Useful for searching for information on the internet"
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    runAgent = initialize_agent(tools,
                                llm,
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                verbose=True,
                                memory=memory,
                                handle_parsing_errors=True)

    intro_response = runAgent(f'Write an academic introduction about {userInput}')
    try:
       intro = intro_response['output']['page_content']
    except KeyError:
                   intro = intro_response['output'][0]  
                   
    quant_facts_response = runAgent(f'''
        Considering user input: {userInput} and the intro paragraph: {intro} 
        \nGenerate a list of 3 to 5 quantitative facts about: {userInput}
        \nOnly return the list of quantitative facts
    ''')
    quant_facts = quant_facts_response['output'][0]

    papers_response = runAgent(f'''
        Consider user input: "{userInput}".
        \nConsider the intro paragraph: "{intro}",
        \nConsider these quantitative facts "{quant_facts}"
        \nNow Generate a list of 2 to 3 recent academic papers relating to {userInput}.
        \nInclude Titles, Links, Abstracts. 
    ''')
    papers_content = papers_response['output'][0]

    readings_response = runAgent(f'''
        Consider user input: "{userInput}".
        \nConsider the intro paragraph: "{intro}",
        \nConsider these quantitative facts "{quant_facts}"
        \nNow Generate a list of 5 relevant books to read relating to {userInput}.
    ''')
    readings_content = readings_response['output'][0]

    insert_research(userInput,
                    intro,
                    quant_facts,
                    papers_content,
                    readings_content)



@app.post("/generate_report")
def generate_report(user_input: str = Form(...), TEMP: float = Form(...)):
    generate_research(user_input, TEMP)
    research_df = read_research_table().tail(1)
    documents = research_df.apply(
        lambda row: {'user_input': row['user_input'],
                     'introduction': row['introduction'],
                     'quant_facts': row['quant_facts'],
                     'publications': row['publications'],
                     'books': row['books']}, axis=1).tolist()
    embeddings_db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory="./chroma_db")
    embeddings_db.persist()
    return {"status": "Report generated successfully", "user_input": user_input}


@app.get("/previous_research")
def previous_research():
    df = read_research_table().to_dict(orient="records")
    return df


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")