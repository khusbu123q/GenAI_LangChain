from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="conversational",
    provider="auto",
)

prompt_1 = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)
prompt_2=PromptTemplate(
    template='Generate a 5 pointer summary from the following text\n{text}',
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain=prompt_1 | model | parser |prompt_2 |model |parser

result=chain.invoke({'topic':'sachintendulkar'})
print(result)