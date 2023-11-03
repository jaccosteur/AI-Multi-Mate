from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


class AIMateLocalResearchAgent:
    def __init__(self, vectordb):
        self.chat_history = []
        self.pdf_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),vectordb.as_retriever(search_kwargs={'k': 6}),return_source_documents=True,verbose=False)
        
        
    def do_research(self, input):
        result = self.pdf_qa({"question": input, "chat_history": self.chat_history})
        self.chat_history.append((input, result["answer"]))
        return result

