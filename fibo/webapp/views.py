from django.shortcuts import render

from django.http import HttpResponse
from  django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Employee
from .serializer import EmployeeSerializer
from codeinterpreterapi import CodeInterpreterSession
from codeinterpreterapi.config import settings
from rest_framework.views import APIView
from rest_framework.response import Response

from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


# settings.MODEL = "gpt-4"
api_key = "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"
settings.OPENAI_API_KEY = api_key

class EmployeeList(APIView):
    def get(self, request):
        emp = Employee.objects.all()
        seri = EmployeeSerializer(emp, many=True)
        return Response(seri.data)


# class AiTest(APIView):
#     def get(self, request):
#         llm = OpenAI(temperature=0.9, openai_api_key=api_key)
#         response = llm.transform('transform this to something else')
#
#         return Response({'result': response})

# class AiTest(APIView):
#      def get(self, request):
#         # user_input = request.query_params.get("input", "")
#         user_input = "how to improve my programming skills"
#         with CodeInterpreterSession() as session:
#             response = session.generate_response(user_input)
#
#         return Response({'result': response})


class AiTest(APIView):
    def get(self, request):
        # Clone
        repo_path = "C:\Projects"
        # repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

        # Load
        loader = GenericLoader.from_filesystem(
            repo_path + "/libs/langchain/langchain",
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        documents = loader.load()
        len(documents)

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                       chunk_size=2000,
                                                                       chunk_overlap=200)
        texts = python_splitter.split_documents(documents)
        len(texts)

        db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
        retriever = db.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )

        llm = ChatOpenAI(model_name="gpt-3.5")
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

        question = "How can I initialize a ReAct agent?"
        result = qa(question)
        result['answer']

        res = result['answer']
        return Response({'result': res})



