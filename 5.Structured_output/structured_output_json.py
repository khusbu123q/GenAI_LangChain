from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
load_dotenv()

model = ChatOpenAI()

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""I recently joined a Python and AI course on Udemy, and it has been a fantastic learning experience! 
The course is well-structured, starting from Python basics and gradually moving into machine learning and deep learning concepts. 
The instructor explains complex topics in a very simple and engaging way, making it easy to follow even for beginners.

The hands-on projects are the highlight — building real-world AI applications gave me so much confidence. 
The section on LangChain and LLMs was especially exciting and very relevant to today's job market. 
Video quality is crisp and the subtitles are accurate, which helps a lot during tricky explanations.

However, some advanced topics feel a bit rushed, and I wish there were more coding exercises after each section. 
The course could also benefit from more frequent updates to keep up with the latest library versions.

Pros:
Well-structured curriculum from basics to advanced AI
Clear and beginner-friendly explanations
Exciting hands-on projects with real-world relevance
Great coverage of LangChain and modern LLM tools

Cons:
Some advanced sections feel rushed
Needs more practice exercises per section
Library versions occasionally outdated

Review by Khusbu
""")

print(result)