from .models import Employee
from .serializer import EmployeeSerializer
from codeinterpreterapi.config import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from git import Repo
import git
import shutil

# settings.MODEL = "gpt-4"
api_key = "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"
settings.OPENAI_API_KEY = api_key

class EmployeeList(APIView):
    def get(self, request):
        emp = Employee.objects.all()
        seri = EmployeeSerializer(emp, many=True)
        res = Response(seri.data)

        return res


class AiTest(APIView):
    def get(self, request):

        githubUrl = ''
        link = request.query_params.get('githubUrl', None)
        if link:
            githubUrl = link;
        else:
            githubUrl = "https://github.com/drehimself/todo-react"

        existing_path = "Users/picker/Desktop/mo/todo-react"
        try:
            shutil.rmtree(existing_path)
            print(f"Repository at {existing_path} has been removed.")
        except PermissionError as e:
            print(f"Permission error: {e}")
        except FileNotFoundError as e:
            print(f"Repository not found: {e}")

        destination_path = "Users/picker/Desktop/mo/todo-react"

        # try:
        #     repo = Repo.clone_from(githubUrl, to_path=destination_path)
        # except git.exc.GitCommandError as e:
        #     print(f"Git command error: {e}")

        # Load
        loader = GenericLoader.from_filesystem(
            destination_path + "/src",
            glob="**/[!.]*",
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
        # print(len(texts))

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

        # //////////////////////////////////////////////
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=retriever,
            condense_question_prompt=prompt
            # return_source_documents=True
        )

        result = qa('please write all test using cypress for this project?')
        result['answer']
        print(result['answer'])

        res = result['answer']
        return Response({'answer': res})

