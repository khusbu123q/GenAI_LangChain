from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage ,AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    )

model = ChatHuggingFace(llm=llm)

chat_hisstory=[
    SystemMessage(content="You are a helpful assitant "),
]
while True:
    user_input=input("you: ")
    chat_hisstory.append(HumanMessage(content=user_input))
    if user_input == "exit" :
        break
    result=model.invoke(chat_hisstory)
    chat_hisstory.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_hisstory)
