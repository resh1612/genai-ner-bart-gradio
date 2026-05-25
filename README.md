Name : RESHMITHAA B
Reg.no : 212224220080

## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

Named Entity Recognition (NER) is a fundamental Natural Language Processing (NLP) task that identifies and classifies entities such as names of people, organizations, locations, and miscellaneous entities in text.This project aims to create a simple, interactive web-based prototype that performs NER using a fine-tuned transformer model.The challenge lies in integrating a pre-trained transformer model with a user-friendly front-end to visualize entity recognition results efficiently.

### DESIGN STEPS:

#### STEP 1:

1.Import the required libraries — Transformers, Torch, and Gradio.

2.Load a fine-tuned transformer model (BERT/BART) for the NER task using the Hugging Face model hub.



#### STEP 2:

1.Create a pipeline using the pipeline() function from Hugging Face to handle tokenization and entity recognition automatically.

2.Define a function to process user input text and display recognized entities in a formatted structure.

#### STEP 3:

1.Design an interactive interface using Gradio with appropriate input and output components.

2.Launch the interface locally to test and visualize the model’s predictions on custom text inputs.

### PROGRAM:

```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
```
```
# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
```
```
API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Reshmithaa, I'm building DeepLearningAI and I live in India"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)
```
```
import gradio as gr
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Reshmithaa and I live in Chennai", "My name is Poli and work at HuggingFace"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```
```
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Reshmithaa, I'm building DeeplearningAI and I live in Chennai", "My name is Poli, I live in Vienna and work at HuggingFace"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```
```
gr.close_all()
```
### OUTPUT:

<img width="162" height="527" alt="image" src="https://github.com/user-attachments/assets/80515bae-1469-4999-abde-ebaf534605aa" />





<img width="640" height="342" alt="image" src="https://github.com/user-attachments/assets/072aef57-a4be-4386-a7ff-2f0b2147f7de" />

<img width="635" height="343" alt="image" src="https://github.com/user-attachments/assets/82c175af-c367-4460-a981-bb11233d9a90" />




### RESULT:

To develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation excuted successfully.

