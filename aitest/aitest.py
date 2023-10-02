from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate, StringPromptTemplate
from git import Repo

import dotenv
dotenv.load_dotenv()

# Clone
repo_path = "Users/picker/Desktop/mo/todo-react"
# repo = Repo.clone_from("https://github.com/drehimself/todo-react", to_path=repo_path)

# Load
loader = GenericLoader.from_filesystem(
    repo_path + "/src/components",
    glob="**/*",
    suffixes=[".js"],
    parser=LanguageParser(language=Language.JS, parser_threshold=500)
)
documents = loader.load()
len(documents)
# print(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter
python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JS,
                                                                chunk_size=2000,
                                                                chunk_overlap=200)
texts = python_splitter.split_documents(documents)
len(texts)
print(len(texts))

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)


template = """You are a senior QA engineer that work with a frontend testing tool named cypress and having a conversation with human.
these are code files of the company, you might have to go through them and answer with code
Given the following extracted parts of long document and a question, create a final answer
{context}
{chat_history}
Human: {human_input}
aiTester:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)



#//////////////////////////////////////////////
# llm = ChatOpenAI(model_name="gpt-4")
# llm(messages)
# memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
# custom_prompt = PromptTemplate(user_input="question", template=template)


#//////////////////////////////////////////////
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
# retriever = retriever
# Create the multipurpose chain
qachat = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0),
    memory=memory,
    retriever=retriever, 
    condense_question_prompt=prompt
    # return_source_documents=True
)

result = qachat('please write several test using cypress for this project?')
print(result)



# question = "How can I initialize a ReAct agent?"
# result = qa(question)
# result['answer']
# print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
# print(result['answer'])