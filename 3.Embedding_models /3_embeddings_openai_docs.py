from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Bangalore is the capital of Karnataka",
    "Kolkata is the capital of West Bengal",
    "Amritsar is the capitar of Punjab"
]

result = embedding.embed_documents(documents)

print(str(result))