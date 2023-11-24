from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import YouTubeSearchTool
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from youtubesearchpython import VideosSearch
import sqlite3
import pandas as pd
import os

app = FastAPI()


class ResearchRequest(BaseModel):
    userInput: str


def create_research_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT,
                ytlinks TEXT
            )
        """)


def read_research_table():
    with sqlite3.connect('MASTER.db') as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df


def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks))


def generate_research(user_input):
    llm = OpenAI(temperature=TEMP)
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
    tools = [
        Tool(
            name="Wikipedia Research Tool",
            func=wiki.run,
            description="Useful for researching information on wikipedia"
        ),
        Tool(
            name="Duck Duck Go Search Results Tool",
            func=DDGsearch.run,
            description="Useful for search for information on the internet"
        ),
        Tool(
            name="YouTube Search Tool",
            func=YTsearch.run,
            description="Useful for gathering links on YouTube"
        )
    ]

    if st.session_state.embeddings_db:
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever=st.session_state.embeddings_db.as_retriever())
        tools.append(
            Tool(
                name='Previous Research Database Tool',
                func=qa.run,
                description="Useful for looking up previous research/information"
            )
        )

    memory = ConversationBufferMemory(memory_key="chat_history")
    runAgent = initialize_agent(tools,
                                llm,
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                verbose=True,
                                memory=memory,
                                )

    intro = runAgent(f'Write an academic introduction about {user_input}')
    quant_facts = runAgent(f'''
        Considering user input: {user_input} and the intro paragraph: {intro} 
        \nGenerate a list of 3 to 5 quantitative facts about: {user_input}
        \nOnly return the list of quantitative facts
    ''')
    papers = runAgent(f'''
        Consider user input: "{user_input}".
        \nConsider the intro paragraph: "{intro}",
        \nConsider these quantitative facts "{quant_facts}"
        \nNow Generate a list of 2 to 3 recent academic papers relating to {user_input}.
        \nInclude Titles, Links, Abstracts. 
    ''')
    readings = runAgent(f'''
        Consider user input: "{user_input}".
        \nConsider the intro paragraph: "{intro}",
        \nConsider these quantitative facts "{quant_facts}"
        \nNow Generate a list of 5 relevant books to read relating to {user_input}.
    ''')
    search = VideosSearch(user_input)
    ytlinks = ""
    for i in range(1, 6):
        ytlinks += (str(i) + ". Title: " + search.result()['result'][0]['title'] + "Link: https://www.youtube.com/watch?v=" + search.result()['result'][0]['id'] + "\n")
        search.next()

    insert_research(user_input,
                    intro['output'],
                    quant_facts['output'],
                    papers['output'],
                    readings['output'],
                    ytlinks)


class Document:
    def __init__(self, content, topic):
        self.page_content = content
        self.metadata = {"Topic": topic}


@app.post("/generate_research/", response_model=dict)
def generate_research_api(request: ResearchRequest):
    user_input = request.userInput
    generate_research(user_input)
    research_df = read_research_table().tail(1)
    return {"message": "Research generated successfully", "data": research_df.to_dict(orient='records')}


@app.get("/previous_research/", response_model=dict)
def read_previous_research():
    df = read_research_table()
    return {"data": df.to_dict(orient='records')}
