
from dotenv import load_dotenv, find_dotenv
from elevenlabs import set_api_key
from flask import Flask, request, render_template
from flask_cors import CORS
import os
import json
from AIMateResearchAgent import AIMateResearchAgent
from AIMateChatbot import AIMateChatbot

llm_model="gpt-3.5-turbo-16k-0613"

load_dotenv(find_dotenv())
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

set_api_key(eleven_labs_api_key)
enable_voice = False


research_agent = AIMateResearchAgent(openai_api_key, google_search_api_key, browserless_api_key)
chatbot = AIMateChatbot(enable_voice, eleven_labs_api_key)

app = Flask (__name__)
CORS (app)


@app.route('/chat', methods=['GET']) 
def chat():
    input = request.args.get('input') 
    jsonResponse = chatbot.chat_with_ai(input)
    print(jsonResponse)
    jsonData = json.loads(jsonResponse)
    response = jsonData['response']
    research_result = ""
    if 'command' in jsonData and 'research' in jsonData['command']:
        research_result =  research_agent.do_research(input)
        response = response + " This is my research result: " + research_result['output']

    chatbot.text_to_speech(response)
    return { 'message': response }

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
