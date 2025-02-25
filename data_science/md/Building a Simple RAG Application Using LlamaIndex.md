
[Link](https://machinelearningmastery.com/building-a-simple-rag-application-using-llamaindex/)

In this tutorial, we will explore Retrieval-Augmented Generation (RAG) and the LlamaIndex AI framework. We will learn how to use LlamaIndex to build a RAG-based application for Q&A over the private documents and enhance the application by incorporating a memory buffer. This will enable the LLM to generate the response using the context from both the document and previous interactions.

## What is RAG in LLMs?

Retrieval-Augmented Generation (RAG) is an advanced methodology designed to enhance the performance of large language models (LLMs) by integrating external knowledge sources into the generation process. 

RAG involves two main phases: retrieval and content generation. Initially, relevant documents or data are retrieved from external databases, which are then used to provide context for the LLM, ensuring that responses are based on the most current and domain-specific information available.

## What is LlamaIndex?

LlamaIndex is an advanced AI framework that is designed to enhance the capabilities of large language models (LLMs) by facilitating seamless integration with diverse data sources. It supports the retrieval of data from over 160 different formats, including APIs, PDFs, and SQL databases, making it highly versatile for building advanced AI applications. 

We can even build a complete multimodal and multistep AI application and then deploy it to a server to provide highly accurate, domain-specific responses. Compared to other frameworks like LangChain, LlamaIndex offers a simpler solution with built-in functions tailored for various types of LLM applications.

## Building RAG Applications using LlamaIndex

In this section, we will build an AI application that loads Microsoft Word files from a folder, converts them into embeddings, indexes them into the vector store, and builds a simple query engine. After that, we will build a proper RAG chatbot with history using vector store as a retriever, LLM, and the memory buffer.

### Setting up

Install all the necessary Python packages to load the data and for OpenAI API. 

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5|!pip install llama-index<br><br>!pip install llama-index-embeddings-openai<br><br>!pip install llama-index-llms-openai<br><br>!pip install llama-index-readers-file<br><br>!pip install docx2txt|

Initiate LLM and embedding model using OpenAI functions. We will use the latest “GPT-4o” and “text-embedding-3-small” models. 

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8|from llama_index.llms.openai import OpenAI<br><br>from llama_index.embeddings.openai import OpenAIEmbedding<br><br># initialize the LLM<br><br>llm=OpenAI(model="gpt-4o")<br><br># initialize the embedding<br><br>embed_model=OpenAIEmbedding(model="text-embedding-3-small")|

Set both LLM and embedding model to global so that when we invoke a function that requires LLM or embeddings, it will automatically use these settings. 

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5|from llama_index.core import Settings<br><br># global settings<br><br>Settings.llm=llm<br><br>Settings.embed_model=embed_model|

### Loading and Indexing the Documents

Load the data from the folder, convert it into the embedding, and store it into the vector store. 

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7|from llama_index.core import VectorStoreIndex,SimpleDirectoryReader<br><br># load documents<br><br>data=SimpleDirectoryReader(input_dir="/work/data/",required_exts=[".docx"]).load_data()<br><br># indexing documents using vector store<br><br>index=VectorStoreIndex.from_documents(data)|

### Building Query Engine 

Please convert the vector store to a query engine and begin asking questions about the documents. The documents consist of the blogs published in June on Machine Learning Mastery by the author Abid Ali Awan. 

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8|from llama_index.core<b>import</b>VectorStoreIndex<br><br># converting vector store to query engine<br><br>query_engine=index.as_query_engine(similarity_top_k=3)<br><br># generating query response<br><br>response=query_engine.query("What are the common themes of the blogs?")<br><br>print(response)|

And the answer is accurate. 

> The common themes of the blogs are centered around enhancing knowledge **and** skills **in** machine learning. They focus on providing resources such **as** free books, platforms **for** collaboration, **and** datasets to help individuals deepen their understanding of machine learning algorithms, collaborate effectively on projects, **and** gain practical experience through real-world data. These resources are aimed at both beginners **and** professionals looking to build a strong foundation **and** advance their careers **in** the field of machine learning.

### Building RAG Application with Memory Buffer

The previous app was simple; let’s create a more advanced chatbot with a history feature.

We will build the chatbot using a retriever, a chat memory buffer, and a GPT-4o model.

Afterward, we will test our chatbot by asking questions about one of the blog posts.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18|from llama_index.core.memory import ChatMemoryBuffer<br><br>from llama_index.core.chat_engine import CondensePlusContextChatEngine<br><br># creating chat memory buffer<br><br>memory=ChatMemoryBuffer.from_defaults(token_limit=4500)<br><br># creating chat engine<br><br>chat_engine=CondensePlusContextChatEngine.from_defaults(  <br><br> index.as_retriever(),  <br><br> memory=memory,  <br><br> llm=llm<br><br>)<br><br># generating chat response<br><br>response=chat_engine.chat(  <br><br> "What is the one best course for mastering Reinforcement Learning?"<br><br>)<br><br>print(str(response))|

It is highly accurate and to the point. 

> Based on the provided documents, the “Deep RL Course” by Hugging Face **is** highly recommended **for** mastering Reinforcement Learning. This course **is** particularly suitable **for** beginners **and** covers both the basics **and** advanced techniques of reinforcement learning. It includes topics such **as** Q-learning, deep Q-learning, policy gradients, ML agents, actor-critic methods, multi-agent systems, **and** advanced topics like RLHF (Reinforcement Learning **from** Human Feedback), Decision Transformers, **and** MineRL. The course **is** designed to be completed within a month **and** offers practical experimentation **with** models, strategies to improve scores, **and** a leaderboard to track progress.

Let’s ask follow-up questions and understand more about the course. 

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4|response=chat_engine.chat(<br><br>  "Tell me more about the course"<br><br>)<br><br>print(str(response))|

If you are having trouble running the above code, please refer to the Deepnote Notebook: [Building RAG Application using LlamaIndex](https://12ft.io/proxy?q=https%3A%2F%2Fdeepnote.com%2Fworkspace%2Fabid-5efa63e7-7029-4c3e-996f-40e8f1acba6f%2Fproject%2FBuilding-a-Simple-RAG-Application-using-LlamaIndex-5ef68174-c5cd-435e-882d-c0e112257391%2Fnotebook%2FNotebook%25201-2912a70b918b49549f1b333b8778212c).

## Conclusion

Building and deploying AI applications has been made easy by LlamaIndex. You just have to write a few lines of code and that’s it. 

The next step in your learning journey will be to build a proper Chatbot application using Gradio and deploy it on the server. To simplify your life even more, you can also check out Llama Cloud.

In this tutorial, we learned about LlamaIndex and how to build an RAG application that lets you ask questions from your private documentation. Then, we built a proper RAG chatbot that generates responses using private documents and previous chat interactions.







[[Data Science]]

