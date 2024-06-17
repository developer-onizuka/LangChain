import transformers
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM 
import langchain.llms
import langchain.prompts
from langchain import PromptTemplate, LLMChain
import langchain_community.document_loaders
import torch
import bitsandbytes
from langchain.document_loaders import PyMuPDFLoader
import langchain.text_splitter
import langchain.embeddings
import langchain.vectorstores


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
template="""
Please answer the last question using the following context.
{context} \n
{query} \n
"""

loader = PyMuPDFLoader("./myFirstPdf.pdf")
documents = loader.load()

text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
)
documents = text_splitter.split_documents(documents)

# Preparing Vectorization
embedding = langchain.embeddings.HuggingFaceEmbeddings(
    model_name=model_id
)


# Store the value to DB
vectorstore = langchain.vectorstores.Chroma.from_documents(
    documents=documents,
    embedding=embedding
)

# Search
#query = "Question: Who is developer-onizuka?"
query = "Question: Where does developer-onizuka live?"
docs = vectorstore.similarity_search(query=query, k=5)
context = "\n".join([f"Context:\n{doc.page_content}" for doc in docs])

#for index, doc in enumerate(docs):
#    print("%d:" % (index + 1))
#    print(doc.page_content)

prompt = PromptTemplate(template=template, input_variables=["context","query"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Inference 
result=llm_chain.predict(query=query, context=context)
print(result)


