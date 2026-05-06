from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
llm2 = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Neural networks are a set of algorithms modeled loosely after the human brain, designed to recognize patterns.
The advantages of neural networks are:
Capable of learning complex and non-linear relationships between inputs and outputs.
Can handle large amounts of unstructured data such as images, audio, and text with high accuracy.
Once trained, neural networks can make predictions very quickly, making them suitable for real-time applications.
Highly flexible and can be applied to a wide range of tasks including classification, regression, and generation.
The disadvantages of neural networks include:
They require a large amount of labeled training data to perform well, which can be expensive and time-consuming to collect.
Neural networks are often called black boxes because it is difficult to interpret how they arrive at a decision.
Training deep neural networks requires significant computational resources such as GPUs and TPUs.
They are prone to overfitting when the model is too complex relative to the amount of training data available.
Neural networks consist of layers of interconnected nodes or neurons. The input layer receives raw data, hidden layers perform transformations, 
and the output layer produces the final result. Common types include Convolutional Neural Networks (CNNs) for image tasks, 
Recurrent Neural Networks (RNNs) for sequential data, and Transformers for natural language processing.
"""

result = chain.invoke({'text': text})
print(result)
chain.get_graph().print_ascii()