```
pip install transformers accelerate sentencepiece
pip install bitsandbytes
pip install langchain langchain_community
pip install intel-extension-for-transformers
pip install datasets
pip install sentence-transformers
pip install chromadb
pip install selenium
pip install unstructured
pip install pymupdf
pip install spacy
```

# 1. Transformer/Pipeline
The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the task summary for examples of use.

>https://huggingface.co/docs/transformers/v4.48.2/ja/main_classes/pipelines


>https://medium.com/@sharathhebbar24/text-generation-v-s-text2text-generation-3a2b235ac19b

Text Generation, also known as Causal Language Modeling, is the process of generating text that closely resembles human writing.
```
from transformers import pipeline
task = "text-generation"
model_name = "gpt2"
max_output_length = 30
num_of_return_sequences = 2
input_text = "Hello, "
text_generator = pipeline(
    task,
    model = model_name)

text_generator(
    input_text,
    max_length=max_output_length,
    num_return_sequences=num_of_return_sequences)
```

Text-to-Text Generation, also known as Sequence-to-Sequence Modeling, is the process of converting one piece of text into another.
```
from transformers import pipeline
task = "text2text-generation"
model_name = "google/flan-t5-base"
input_text = "Can you convert this text to French language: Hello, How are you"
text2text_generator = pipeline(
    task,
    model = model_name)

text2text_generator(input_text)
```


```
$ python sample.py
... snip ...
Question:  Tell me a sad joke about {subject}.
Subject:  Super Mario
Ans:  Mario never got a chance to play with luigi
```

```
$ python sample2.py
... snip ...
Question:  What languages can developer-onizuka speak besides python?
Ans:  Go
```

```
$ sudo docker run --net=host --device=/dev/dri --memory="32G" --shm-size="16g"  -it --rm intelanalytics/ipex-llm-xpu:latest
```
```
# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2024.17.5.0.08_160000.xmain-hotfix]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz OpenCL 3.0 (Build 0) [2024.17.5.0.08_160000.xmain-hotfix]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A310 Graphics OpenCL 3.0 NEO  [23.43.27642.50]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A310 Graphics 1.3 [1.3.27642]
```


