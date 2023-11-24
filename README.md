# Intellicuria backend

Imports:

We import the necessary modules and classes from FastAPI, SQLite, pandas, and other LangChain-related modules.
FastAPI App Initialization:

app = FastAPI(): Initializes a FastAPI instance as app.
Pydantic Model:

ResearchRequest is a Pydantic model defining the request structure for the generate_research_api endpoint. It expects a JSON body with a userInput field.
Database Functions:

create_research_db(): Creates or ensures the existence of the SQLite database table Research.
read_research_table(): Reads the entire Research table into a pandas DataFrame.
insert_research(...): Inserts a new research entry into the Research table.
Research Generation Function:

generate_research(user_input): This function generates research based on the user input.
Initializes LangChain tools, such as Wikipedia, DuckDuckGo, and YouTube search.
Creates a conversation agent (runAgent) and uses it to generate introduction, quantitative facts, recent publications, recommended books, and YouTube links.
Inserts the generated data into the database.
API Endpoints:

POST /generate_research/

Expects a JSON payload with a userInput field.
Calls generate_research function with the provided input.
Returns a JSON response with a success message and the generated research data.
GET /previous_research/

Retrieves all previous research entries from the database.
Returns a JSON response with the previous research data.
Explanation of Modifications:

I removed the HTML templates and used JSON responses for simplicity.
I replaced the st references with app since Streamlit's state (st.session_state) isn't available in FastAPI.
Ensure you have the necessary adjustments in your frontend or client to handle JSON responses instead of HTML.
Remember to handle the frontend accordingly to make API requests to these endpoints and process the JSON responses.
