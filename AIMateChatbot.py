from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, stream
from langchain.prompts import PromptTemplate

# https://python.langchain.com/docs/modules/memory/types/vectorstore_retriever_memory


class AIMateChatbot:
    def __init__(self, enable_voice, eleven_labs_api_key ):
        self.eleven_labs_api_key = eleven_labs_api_key
        self.enable_voice = enable_voice
        self.memory = ConversationBufferMemory()

    def chat_with_ai(self, human_input):
        template = """
        Provide your response as a valid JSON object.
        The first key in this json format is the word: 'response', the value is your response to me.
        The second key in this json format is the word: 'command', the value is the word 'local' if my question contains something like a request to do local research or research in a pdf.
        The second key in this json format is the word: 'command', the value is the word 'web' if my question contains something like a request to do remote research or research on the web.
     
        Act as a very sarcastic character. your name is Diana. My name is Jacco.
        You are sarcastic. 
        You are working as a programmer and tell me a programmer joke every now and then.
        You also like cooking and tell me some easy to make recipes for pastas if you feel like it.

        If I ask you to look someting up or do some research you make the command field 'research'
        If I ask you what the weather is like or how hot it is then you make the command field 'weather'
     
        Following '===' is the conversation history.
	    Use this conversation history to make your decision.

	    Only use the text between first and second '===' to accomplish the task above.
	    ===
	    {history}
	    ===

	    Jacco: {human_input}
	    Diana:
	    """

        # Define the input variables
        input_variables = ["history", "human_input"]

        # Create the prompt template
        prompt_template = PromptTemplate(input_variables=input_variables, template=template)

        # Define the LLM chain 
        llm_chain = LLMChain( 
	        llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613"),
            # llm=ChatOpenAI(model="gpt-4"),
            prompt=prompt_template, 
            verbose=False,
            memory=self.memory
        )
        self.memory

        jsonResponse = llm_chain.predict(human_input=human_input)
        
        return jsonResponse

    def text_to_speech(self, message):
        if(self.enable_voice):
            audio_stream = generate(text=message, voice="Bella", model="eleven_monolingual_v1", stream=True)
            stream(audio_stream)