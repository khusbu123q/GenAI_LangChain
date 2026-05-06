from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
"Elon Musk is a visionary entrepreneur known for founding Tesla and SpaceX, pushing the boundaries of electric vehicles and space exploration."
"Jeff Bezos founded Amazon and transformed it from an online bookstore into the world's largest e-commerce and cloud computing giant."
"Steve Jobs co-founded Apple and revolutionized personal computing, music, and smartphones with iconic products like the iPhone and MacBook."
"Sundar Pichai, the CEO of Google and Alphabet, is known for leading advancements in artificial intelligence, cloud services, and Android."
"Mark Zuckerberg co-founded Facebook and built Meta into a social media empire connecting billions of people across the globe."
]

query = 'tell me about Elon Musk'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)