from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))

#if torch.cuda.is_available():
#    device = 0
#    print(torch.cuda.get_device_name(device))
#else:
#    device = -1
#    print("CUDA is not available")
device = -1 # My laptop does not have gpu :(

model = pipeline("text-generation",
                 model="mistralai/Mistral-7B-Instruct-v0.1",
                 device=device,
                 max_length=256,
                 truncation=True)

llm = HuggingFacePipeline(model)

# Create the prompt template
template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand.")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

# Execute the chain
response = chain.invoke({"topic": topic, "age": age})
print(response)

#model = pipeline("summarization", model="facebook/bart-large-cnn")
#response = model("text to summarize")

#print(response)