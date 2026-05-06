from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently purchased the Sony WH-1000XM5 noise-cancelling headphones, and they have completely transformed my listening experience! 
The active noise cancellation is simply the best I have ever tried — it blocks out everything from office chatter to loud traffic effortlessly. 
The sound quality is rich and balanced, with deep bass and crystal-clear highs that make both music and podcasts a joy to listen to.

The battery life of 30 hours is impressive, and the quick charge feature gives 3 hours of playback in just 3 minutes, which is a lifesaver. 
The headphones are incredibly lightweight and comfortable, even during long study or work sessions. 
Multipoint connection allows me to switch seamlessly between my laptop and phone without any hassle.

However, the touch controls on the earcup can be a bit too sensitive at times, triggering accidental pauses or skips. 
The carrying case, while sleek, feels a little flimsy for the price. 
At around $350, it is definitely a premium investment.

Pros:
Industry-leading noise cancellation
Excellent sound quality with balanced audio
30-hour battery life with quick charge
Lightweight and comfortable for long use
Seamless multipoint Bluetooth connection

Cons:
Touch controls are overly sensitive
Carrying case feels fragile
Premium price tag

Review by Khusbu
""")

print(result)