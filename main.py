from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import sqlite3
import pandas as pd
import os

app = Flask(__name__)

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
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books)
            VALUES (?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books))

def generate_research(user_input):
    llm = OpenAI(temperature=TEMP)
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

    # Replace st.session_state with a global variable or a class instance if necessary
    # If using a class, make sure to update it in the functions accordingly.

    if os.path.exists("./chroma_db"):
        embeddings_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
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
                                handle_parsing_errors=True)

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

    insert_research(user_input, intro['output'], quant_facts['output'], papers['output'], readings['output'])

@app.route('/generate_research', methods=['POST'])
def generate_research_endpoint():
    data = request.get_json()
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({"error": "User input is missing"}), 400

    try:
        generate_research(user_input)
        research_df = read_research_table().tail(1)
        documents = research_df.apply(lambda row: Document(' '.join([f'{idx}: {val}' for idx, val in zip(row.index, row.values.astype(str))]), row['user_input']), axis=1).tolist()
        embeddings_db = Chroma.from_documents(documents, embedding_function, persist_directory="./chroma_db")
        embeddings_db.persist()

        return jsonify({"success": True, "message": "Research generated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_previous_research', methods=['GET'])
def get_previous_research_endpoint():
    research_data = read_research_table().to_dict(orient='records')
    return jsonify(research_data)

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)



@app.get("/previous_research/", response_model=dict)
def read_previous_research():
    df = read_research_table()
    return {"data": df.to_dict(orient='records')}
