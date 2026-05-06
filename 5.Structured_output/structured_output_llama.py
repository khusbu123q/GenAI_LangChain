from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently started using Notion as my daily productivity and note-taking app, and it has completely changed the way I organize my life! 
The flexibility it offers is unmatched — I use it for study notes, project planning, habit tracking, and even journaling. 
Setting up templates is super easy and saves a lot of time once everything is in place.

The database feature is incredibly powerful for managing tasks and deadlines. I love how I can switch between table, board, and calendar views depending on what I need. 
The AI writing assistant built into Notion is also a nice bonus for drafting content quickly.

However, the app can feel overwhelming for beginners due to the sheer number of features. 
It also gets a bit slow when pages have too much content or embedded media. 
The free plan has limitations on file uploads which can be frustrating.

Pros:
Extremely flexible and customizable workspace
Powerful database and task management features
Clean and minimal UI
Built-in AI assistant is handy

Cons:
Steep learning curve for new users
Can slow down with heavy content
Free plan has upload limits

Review by Khusbu
""")

print(result)