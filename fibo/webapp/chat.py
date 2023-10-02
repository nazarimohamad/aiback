from codeinterpreterapi import CodeInterpreterSession, settings

settings.MODEL = "gpt-4"
settings.OPENAI_API_KEY = "sk-***************"

print(
    "hello i am fibo ai"
)

with CodeInterpreterSession() as session:
    while True:
        session.generate_response_sync(input("\nUser: ")).show()