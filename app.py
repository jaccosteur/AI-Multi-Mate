from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from elevenlabs import set_api_key
from flask import Flask, request, render_template
from flask_cors import CORS
from AIMateWebResearchAgent import AIMateWebResearchAgent
from AIMateLocalResearchAgent import AIMateLocalResearchAgent
from AIMateChatbot import AIMateChatbot
import os
import json


# Add mpv to the path. Needed to make the voice audible.
path = os.environ.get('PATH')
mpv_path = "/opt/homebrew/bin"
path += f":{mpv_path}"
os.environ['PATH'] = path

load_dotenv(find_dotenv())
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

set_api_key(eleven_labs_api_key)
enable_voice = False

embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="./data",embedding_function=embedding)

research_web_agent = AIMateWebResearchAgent(openai_api_key, google_search_api_key)
research_local_agent = AIMateLocalResearchAgent(vectordb)
chatbot = AIMateChatbot(enable_voice, eleven_labs_api_key)

app = Flask (__name__)
CORS (app)

@app.route('/chat', methods=['GET']) 
def chat():
    input = request.args.get('input') 
    jsonResponse = chatbot.chat_with_ai(input)
    print("---What to do BEGIN---")
    print(jsonResponse)
    print("---What to do END---")
    try:
        jsonData = json.loads(jsonResponse)
        response = jsonData['response']
        research_result = ""
        if 'command' in jsonData and 'web' in jsonData['command']:
            research_result =  research_web_agent.do_research(input)
            response = response + " \n\nThis is my research result done on the web: \n\n" + research_result['output']
        if 'command' in jsonData and 'local' in jsonData['command']:
            research_result = research_local_agent.do_research(input)
            response = response + " \n\nThis is my research result done in local files: \n\n" + research_result['answer']
        else:
            chatbot.text_to_speech(response)
    except json.JSONDecodeError:
        return { 'message': 'Oops. Something went wrong. What did you ask?'}
    
    return { 'message': response }

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
