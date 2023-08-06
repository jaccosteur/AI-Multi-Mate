from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from langchain.chains.summarize import load_summarize_chain
from pydantic import BaseModel, Field
from typing import Type
from dotenv import load_dotenv, find_dotenv
from elevenlabs import generate, stream, set_api_key
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from flask_cors import CORS
import os
import json
import requests
import openai


class AIMateResearchAgent:
    def __init__(self, openai_api_key, google_search_api_key, browserless_api_key):
        self.openai_api_key = openai_api_key
        self.google_search_api_key = google_search_api_key
        self.browserless_api_key = browserless_api_key

    # All other methods related to research agent go here.
    # For example:
    def search(self, query):
        url = f"https://customsearch.googleapis.com/customsearch/v1?cx=f0bd3036b54ff440f&key={self.google_search_api_key}&q={query}"
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("GET", url, headers=headers)
        return response.text

    def scrape_website(self, objective, url):
    
        print("Scraping website...")
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }
        
        data = {
            "url": url
        }
        
        data_json = json.dumps(data)
        
        post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            print("CONTENT:", text)
        
            if len(text) > 10000:
                output = self.summary(objective, text)
                return output
            else:
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")
        
    def summary(self, objective, content):
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a summary of the following text for {objective}:
        "{text}"
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "objective"])

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True
        )

        output = summary_chain.run(input_documents=docs, objective=objective)

        return output

    def do_research(self, human_input):
        tools = [
            Tool(
              name="Search",
              func=self.search,
              description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
            ),
            ScrapeWebsiteTool(),
        ]

        system_message = SystemMessage(
            content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
                 you do not make things up, you will try as hard as possible to gather facts & data to back up the research

                 Please make sure you complete the objective above with the following rules:
                 1/ You should do enough research to gather as much information as possible about the objective
                 2/ If there are url of relevant links & articles, you will scrape it to gather more information
                 3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
                 4/ You should not make things up, you should only write facts & data that you have gathered
                 5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
                 6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
        )

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_message,
        }

        llmsearch = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
        memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llmsearch, max_token_limit=1000)

        agent = initialize_agent(
            tools,
            llmsearch,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

        researchResponse = agent({"input": human_input})
        return researchResponse


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")
