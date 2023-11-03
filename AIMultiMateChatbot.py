from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, stream
import json
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

# https://python.langchain.com/docs/modules/memory/types/vectorstore_retriever_memory

class AIMultiMateChatbot:
    def __init__(self):
        load_dotenv(find_dotenv())
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
        self.memory = ConversationBufferMemory()

        mates_infos = self.load_mates_from_file()
        destination_chains = {}
        self.default_chain = None
        for m_info in mates_infos:
            name = m_info["name"]
            prompt_file = m_info["prompt_file"]
            prompt_template = ""
            try:
                with open(prompt_file, 'r') as file:
                    prompt_template = file.read()
            except FileNotFoundError:
                print(f"File {prompt_file} not found.")

            # print(prompt_template)
            
            if m_info['default']:
                input_variables = ["history", "input"]
                prompt = PromptTemplate(template=prompt_template, input_variables = input_variables)
                self.default_chain = ConversationChain( 
	                llm=llm,
                    prompt=prompt, 
                    verbose=False,
                    memory=self.memory,
                    output_key="text"
                    )
                destination_chains[name] = self.default_chain
                continue
            input_variables = ["input"]
            prompt = PromptTemplate(template=prompt_template, input_variables = input_variables)
            chain = LLMChain(llm=llm,
                            prompt=prompt,
                            verbose=False,
                            )
            destination_chains[name] = chain
        
        destinations = [f"{p['name']}: {p['description']}" for p in mates_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        print(router_template)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)

        self.multi_mate_chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=self.default_chain,
            verbose=True
        )

    # Load the json file with chain definitions from file
    def load_mates_from_file(self):
        try:
            with open('config/prompts.json', 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print("File not found in the 'config' directory.")
        except json.JSONDecodeError:
            return None
    
    def chat_with_ai(self, human_input):
        
        return self.multi_mate_chain.run(human_input)

    def text_to_speech(self, message):
        if(self.enable_voice):
            audio_stream = generate(text=message, voice="Bella", model="eleven_monolingual_v1", stream=True)
            stream(audio_stream)