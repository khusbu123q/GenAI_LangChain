from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))  # Should print your key, not None

embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)


result=embedding.embed_query("Delhi is the capital of India")
print(str(result))