To connect a frontend to your FastAPI application, you'll typically use an HTML template engine for rendering dynamic content. In this example, we'll use Jinja2 as the template engine. Create two HTML files in a folder named "templates" within your project directory: "index.html" and "report_generated.html".

index.html:

html
Copy code
<!DOCTYPE html>
<html>
<head>
    <title>Research Bot</title>
</head>
<body>
    <h1>GPT-4 LangChain Agents Research Bot</h1>
    <form action="/generate_report" method="post">
        <label for="user_input">User Input:</label>
        <input type="text" id="user_input" name="user_input" required>
        
        <label for="TEMP">Temperature:</label>
        <input type="number" id="TEMP" name="TEMP" step="0.01" required>
        
        <button type="submit">Generate Report</button>
    </form>
</body>
</html>
report_generated.html:

html
Copy code
<!DOCTYPE html>
<html>
<head>
    <title>Research Bot - Report Generated</title>
</head>
<body>
    <h1>Report Generated</h1>
    <p>User Input: {{ user_input }}</p>
    <!-- Add more content as needed -->
</body>
</html>
With these HTML templates, your frontend will have a simple form to input user data and a page to display the generated report.

Update your FastAPI app code to use these templates:

python
Copy code
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
# ... (other imports)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# ... (other code)

@app.get("/")
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate_report")
def generate_report(request: Request, user_input: str = Form(...), TEMP: float = Form(...)):
    generate_research(user_input, TEMP)
    research_df = read_research_table().tail(1)
    documents = research_df.apply(
        lambda row: Document(' '.join([f'{idx}: {val}' for idx, val in zip(row.index, row.values.astype(str))]),
                             row['user_input']), axis=1).tolist()
    embeddings_db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory="./chroma_db")
    embeddings_db.persist()
    return templates.TemplateResponse("report_generated.html", {"request": request, "user_input": user_input})
Ensure that the "templates" folder containing your HTML files is in the same directory as your FastAPI application.

Now, when you run your FastAPI app and navigate to the root URL (http://localhost:8000 by default), you'll see the form. Upon submitting the form, the generated report will be displayed on the "report_generated.html" page.
