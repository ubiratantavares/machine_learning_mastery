# How to Implement Named Entity Recognition with Hugging Face Transformers
Let's take a look at how we can perform NER using that Swiss army knife of NLP and LLM libraries, Hugging Face's Transformers.
By Matthew Mayo, KDnuggets Managing Editor on November 20, 2024 in Natural Language Processing
FacebookTwitterLinkedInRedditEmailCompartilhar

How to Implement Named Entity Recognition with Hugging Face Transformers
Image by Author | Ideogram

 
Named entity recognition (NER) is a fundamental natural language processing (NLP) task, involving the identification and classification of named entities within text into predefined categories. These categories could be person names, organizations, locations, dates, and more. This is a useful capability in a variety of real world scenarios, from information extraction, to text summarization, to Q&A, and beyond. In reality, for any situation in which the understanding and categorizing of specific elements of text is a goal, NER can help.

Let's take a look at how we can perform NER using that Swiss army knife of NLP and LLM libraries, Hugging Face's Transformers.

The new way to Cloud
Migrate to Big Query

 

Preparing Our Environment
 
We will assume you already have a Python environment setup. We can then being by installing the necessary libraries, which in our case are Transformers and PyTorch. We can use pip to do so:

pip install transformers torch
 

Now, let's do our imports:

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
 

Next we need a model. Hugging Face offers a range of pre-trained models suitable for NER, and for this tutorial we will use the dbmdz/bert-large-cased-finetuned-conll03-english model, which has been fine-tuned on the CoNLL-03 dataset for English NER tasks. You can load this model with the code below:

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = AutoModelForTokenClassification.from_pretrained(model_name)
 

Here is some text we can use for our testing:

KDnuggets is a leading website and online community focused on data science, machine learning, artificial intelligence, and analytics. Founded in 1997 near Boston, Massachussetts by Gregory Piatetsky-Shapiro, it has become one of the most prominent resources for professionals and enthusiasts in these fields. The site features articles, tutorials, news, and educational content contributed by industry experts and practitioners. The website's name originated from "Knowledge Discovery Nuggets," and began life as an email summarizing the proceedings of the knowledge discovery (data mining) industry's original conference, the KDD Conference, reflecting its mission to share valuable bits of knowledge in the field of data mining and analytics. It has played a significant role in building and supporting the data mining and data science communities over the past decades.

 

Before we can actually get to recognizing named entities, we first have to tokenize our text, given that the resulting tokens will become the model input. You can do so with the following:

text = """KDnuggets is a leading website and online community focused on data science, machine learning, artificial intelligence, and analytics. Founded in 1997 near Boston, Massachusetts by Gregory Piatetsky-Shapiro, it has become one of the most prominent resources for professionals and enthusiasts in these fields. The site features articles, tutorials, news, and educational content contributed by industry experts and practitioners. The website's name originated from "Knowledge Discovery Nuggets," a spin-off email summarizing the proceedings of the knowledge discovery industry's original conference, reflecting its mission to share valuable bits of knowledge in the field of data mining and analytics. It has played a significant role in building and supporting the data mining and data science communities over the past decades."""

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
 


Performing Named Entity Recognition
 
With the tokenized text, we can now run our model to perform NER. The model will output logits, which are the raw predictions for each token. Here’s the code to obtain the logits:

with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
 

The tokens need to be mapped to their corresponding entity labels for meaningful output. We can do this by converting the token IDs back to tokens and mapping the predictions to human-readable labels:

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
token_labels = [config.id2label[p.item()] for p in predictions[0]]
 

Next, we'll process the tokens and labels to handle subwords and format them into readable output:

# Process results, handling special tokens and subwords
results = []
current_entity = []
current_label = None

for token, label in zip(tokens[1:-1], token_labels[1:-1]):  # Skip [CLS] and [SEP]
    # Handle subwords
    if token.startswith("##"):
        if current_entity:
            current_entity[-1] += token[2:]
        continue
        
    # Handle entity continuation
    if label.startswith("B-") or label == "O":
        if current_entity:
            results.append((" ".join(current_entity), current_label))
            current_entity = []
        if label != "O":
            current_entity = [token]
            current_label = label[2:]  # Remove B- prefix
    elif label.startswith("I-"):
        if not current_entity:
            current_entity = [token]
            current_label = label[2:]
        else:
            current_entity.append(token)

# Add final entity if exists
if current_entity:
    results.append((" ".join(current_entity), current_label))

# Print results
for entity, label in results:
    if label:  # Only print actual entities
        print(f"{entity}: {label}")
 

This code handles subword tokens (like "##ing"), tracks entity boundaries using B- (beginning) and I- (inside) prefixes, and prints only the identified named entities with their labels.

 


Full Implementation
 
Finally, let's put it all together:

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

def perform_ner(text, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # Get tokens and their predictions
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_labels = [config.id2label[p.item()] for p in predictions[0]]
        
        # Process results, handling special tokens and subwords
        results = []
        current_entity = []
        current_label = None
        
        # Skip [CLS] and [SEP]
        for token, label in zip(tokens[1:-1], token_labels[1:-1]):
            # Handle subwords
            if token.startswith("##"):
                if current_entity:
                    current_entity[-1] += token[2:]
                continue
                
            # Handle entity continuation
            if label.startswith("B-") or label == "O":
                if current_entity:
                    results.append((" ".join(current_entity), current_label))
                    current_entity = []
                # Remove B- prefix
                if label != "O":
                    current_entity = [token]
                    current_label = label[2:]
            elif label.startswith("I-"):
                if not current_entity:
                    current_entity = [token]
                    current_label = label[2:]
                else:
                    current_entity.append(token)
        
        # Add final entity if exists
        if current_entity:
            results.append((" ".join(current_entity), current_label))
            
        return results
        
    except Exception as e:
        print(f"Error performing NER: {str(e)}")
        return []

text = """KDnuggets is a leading website and online community focused on data science, machine learning, artificial intelligence, and analytics. Founded in 1997 near Boston, Massachussetts by Gregory Piatetsky-Shapiro, it has become one of the most prominent resources for professionals and enthusiasts in these fields. The site features articles, tutorials, news, and educational content contributed by industry experts and practitioners. The website's name originated from "Knowledge Discovery Nuggets," a spin-off email summarizing the proceedings of the knowledge discovery industry's original conference, reflecting its mission to share valuable bits of knowledge in the field of data mining and analytics. It has played a significant role in building and supporting the data mining and data science communities over the past decades."""

results = perform_ner(text)
print(results)
 

And here are the results:

[('KDnuggets', 'ORG'), ('Boston', 'LOC'), ('Massachusetts', 'LOC'), ('Gregory Piatetsky - Shapiro', 'PER'), ('Knowledge Discovery Nuggets', 'ORG')]
 

There you have it, the named entities we would have expected!

 


Wrapping Up
 
Implementing NER using Hugging Face Transformers is both powerful and straightforward. By leveraging pre-trained models and tokenizers, you can quickly set up and perform NER tasks on various text data. Experimenting with different models and datasets can further enhance your NER capabilities and provide valuable insights for your data science projects.

This tutorial has provided a step-by-step guide to setting up your environment, processing text, performing NER, interpreting outputs, and visualizing results. Feel free to explore and adapt this approach to fit your specific needs and applications.


