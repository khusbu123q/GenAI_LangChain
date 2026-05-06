from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently subscribed to Netflix and binged through several shows, and it has been an amazing entertainment experience! 
The content library is massive, with something for every mood — thrillers, documentaries, rom-coms, and international shows. 
The streaming quality is excellent, with crisp 4K HDR on supported devices and very rare buffering even on average internet speeds.

I especially loved the recommendation algorithm — it actually suggests shows I end up enjoying, unlike other platforms. 
The ability to download content for offline viewing is a huge plus during travel. 
Multiple screen support means my whole family can watch different things simultaneously without any conflict.

However, the subscription price has gone up quite a bit recently, which feels steep compared to competitors. 
Some popular shows get removed without notice, which is frustrating. 
The ad-supported plan has too many interruptions to enjoy comfortably.

Pros:
Massive and diverse content library
Excellent 4K HDR streaming quality
Smart and accurate recommendation system
Offline download feature is very convenient

Cons:
Price hikes are becoming frequent
Content removal without prior notice
Ad-supported plan is too interruption-heavy

Review by Khusbu
""")

print(result['name'])