import transformers
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM 
import langchain.llms
import langchain.prompts
from langchain import PromptTemplate, LLMChain
import torch
import bitsandbytes

# Preparing model and tokenizer 
#model_id = "google/flan-t5-small"
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(
    model_id 
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map="auto",
)

#query = "What is GPU?"
#inputs = tokenizer(query, return_tensors="pt")
#outputs = model.generate(**inputs, max_length=128)
#print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


# Preparing pipeline
pipe = transformers.pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    torch_dtype=torch.float16
)

llm = langchain.llms.HuggingFacePipeline(
    pipeline=pipe
)

# Preparing prompt
template = """Tell me a sad joke about {subject}."""
#template="""Where is {subject}?"""
#template="""Who creates the {subject}?"""
prompt = PromptTemplate(template=template, input_variables=["subject"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

#subject = "Japan"
#subject = "iPhone"
subject = "dog"
result=llm_chain.predict(subject=subject)
print(result)

