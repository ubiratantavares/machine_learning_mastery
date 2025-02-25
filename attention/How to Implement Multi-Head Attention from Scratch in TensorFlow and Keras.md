[Link](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/)

We have already familiarized ourselves with the theory behind the [Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F) and its [attention mechanism](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-attention-mechanism%2F). We have already started our journey of implementing a complete model by seeing how to [implement the scaled-dot product attention](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras). We shall now progress one step further into our journey by encapsulating the scaled-dot product attention into a multi-head attention mechanism, which is a core component. Our end goal remains to apply the complete model to Natural Language Processing (NLP).

In this tutorial, you will discover how to implement multi-head attention from scratch in TensorFlow and Keras. 

After completing this tutorial, you will know:

- The layers that form part of the multi-head attention mechanism.
- How to implement the multi-head attention mechanism from scratch.   

## **Tutorial Overview**

This tutorial is divided into three parts; they are:

- Recap of the Transformer Architecture
    - The Transformer Multi-Head Attention
- Implementing Multi-Head Attention From Scratch
- Testing Out the Code

## **Prerequisites**

For this tutorial, we assume that you are already familiar with:

- [The concept of attention](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwhat-is-attention%2F)
- [The Transfomer attention mechanism](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-attention-mechanism)
- [The Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F)
- [The scaled dot-product attention](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras)

## **Recap of the Transformer Architecture**

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F) having seen that the Transformer architecture follows an encoder-decoder structure. The encoder, on the left-hand side, is tasked with mapping an input sequence to a sequence of continuous representations; the decoder, on the right-hand side, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.

[![alt](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-727x1024.png)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwp-content%2Fuploads%2F2021%2F08%2Fattention_research_1.png)

The encoder-decoder structure of the Transformer architecture  
Taken from “[Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762)“

In generating an output sequence, the Transformer does not rely on recurrence and convolutions.

You have seen that the decoder part of the Transformer shares many similarities in its architecture with the encoder. One of the core mechanisms that both the encoder and decoder share is the _multi-head attention_ mechanism. 

### **The Transformer Multi-Head Attention**

Each multi-head attention block is made up of four consecutive levels:

- On the first level, three linear (dense) layers that each receive the queries, keys, or values 
- On the second level, a scaled dot-product attention function. The operations performed on both the first and second levels are repeated _h_ times and performed in parallel, according to the number of heads composing the multi-head attention block. 
- On the third level, a concatenation operation that joins the outputs of the different heads
- On the fourth level, a final linear (dense) layer that produces the output

[![alt](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_4-823x1024.png)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwp-content%2Fuploads%2F2021%2F09%2Ftour_4.png)

Multi-head attention  
Taken from “[Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762)“

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-attention-mechanism%2F) as well the important components that will serve as building blocks for your implementation of the multi-head attention:

- The **queries**, **keys,** and **values**: These are the inputs to each multi-head attention block. In the encoder stage, they each carry the same input sequence after this has been embedded and augmented by positional information. Similarly, on the decoder side, the queries, keys, and values fed into the first attention block represent the same target sequence after this would have also been embedded and augmented by positional information. The second attention block of the decoder receives the encoder output in the form of keys and values, and the normalized output of the first decoder attention block as the queries. The dimensionality of the queries and keys is denoted by $d_k$, whereas the dimensionality of the values is denoted by $d_v$.

- The **projection matrices**: When applied to the queries, keys, and values, these projection matrices generate different subspace representations of each. Each attention _head_ then works on one of these projected versions of the queries, keys, and values. An additional projection matrix is also applied to the output of the multi-head attention block after the outputs of each individual head would have been concatenated together. The projection matrices are learned during training.

Let’s now see how to implement the multi-head attention from scratch in TensorFlow and Keras.

**Implementing Multi-Head Attention from Scratch**

Let’s start by creating the class, `MultiHeadAttention`, which inherits from the `Layer` base class in Keras and initialize several instance attributes that you shall be working with (attribute descriptions may be found in the comments):

**ENTER CODE**

Here note that an instance of the `DotProductAttention` class that was implemented earlier has been created, and its output was assigned to the variable `attention`. [Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras) that you implemented the `DotProductAttention` class as follows:

**ENTER CODE**


Next, you will be reshaping the _linearly projected_ queries, keys, and values in such a manner as to allow the attention heads to be computed in parallel. 

The queries, keys, and values will be fed as input into the multi-head attention block having a shape of (_batch size_, _sequence length_, _model dimensionality_), where the _batch size_ is a hyperparameter of the training process, the _sequence length_ defines the maximum length of the input/output phrases, and the _model dimensionality_ is the dimensionality of the outputs produced by all sub-layers of the model. They are then passed through the respective dense layer to be linearly projected to a shape of (_batch size_, _sequence length_, _queries_/_keys_/_values dimensionality_).

The linearly projected queries, keys, and values will be rearranged into (_batch size_, _number of heads_, _sequence length_, _depth_), by first reshaping them into (_batch size_, _sequence length_, _number of heads_, _depth_) and then transposing the second and third dimensions. For this purpose, you will create the class method, `reshape_tensor`, as follows:

**ENTER CODE**

The `reshape_tensor` method receives the linearly projected queries, keys, or values as input (while setting the flag to `True`) to be rearranged as previously explained. Once the multi-head attention output has been generated, this is also fed into the same function (this time setting the flag to `False`) to perform a reverse operation, effectively concatenating the results of all heads together. 

Hence, the next step is to feed the linearly projected queries, keys, and values into the `reshape_tensor` method to be rearranged, then feed them into the scaled dot-product attention function. In order to do so, let’s create another class method, `call`, as follows:



**ENTER CODE**



Note that the `reshape_tensor` method can also receive a mask (whose value defaults to `None`) as input, in addition to the queries, keys, and values. 

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F) that the Transformer model introduces a _look-ahead mask_ to prevent the decoder from attending to succeeding words, such that the prediction for a particular word can only depend on known outputs for the words that come before it. Furthermore, since the word embeddings are zero-padded to a specific sequence length, a _padding mask_ also needs to be introduced to prevent the zero values from being processed along with the input. These look-ahead and padding masks can be passed on to the scaled-dot product attention through the `mask` argument.  

Once you have generated the multi-head attention output from all the attention heads, the final steps are to concatenate back all outputs together into a tensor of shape (_batch size_, _sequence length_, _values dimensionality_) and passing the result through one final dense layer. For this purpose, you will add the next two lines of code to the `call` method. 


**ENTER CODE**


Putting everything together, you have the following implementation of the multi-head attention:

**ENTER CODE**

## **Testing Out the Code**

You will be working with the parameter values specified in the paper, [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), by Vaswani et al. (2017):

**ENTER CODE**


As for the sequence length and the queries, keys, and values, you will be working with dummy data for the time being until you arrive at the stage of [training the complete Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftraining-the-transformer-model) in a separate tutorial, at which point you will be using actual sentences:

**ENTER CODE**


In the complete Transformer model, values for the sequence length and the queries, keys, and values will be obtained through a process of word tokenization and embedding. We will be covering this in a separate tutorial. 

Returning to the testing procedure, the next step is to create a new instance of the `MultiHeadAttention` class, assigning its output to the `multihead_attention` variable:

**ENTER CODE**


Since the `MultiHeadAttention` class inherits from the `Layer` base class, the `call()` method of the former will be automatically invoked by the magic `__call()__` method of the latter. The final step is to pass in the input arguments and print the result:

**ENTER CODE**


Tying everything together produces the following code listing:

**ENTER CODE**


Running this code produces an output of shape (_batch size_, _sequence length_, _model dimensionality_). Note that you will likely see a different output due to the random initialization of the queries, keys, and values and the parameter values of the dense layers.

## **Further Reading**

This section provides more resources on the topic if you are looking to go deeper.

### **Books**

- [Advanced Deep Learning with Python](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FAdvanced-Deep-Learning-Python-next-generation%2Fdp%2F178995617X), 2019
- [Transformers for Natural Language Processing](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FTransformers-Natural-Language-Processing-architectures%2Fdp%2F1800565798), 2021

### **Papers**

- [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), 2017

## **Summary**

In this tutorial, you discovered how to implement multi-head attention from scratch in TensorFlow and Keras. 

Specifically, you learned:

- The layers that form part of the multi-head attention mechanism
- How to implement the multi-head attention mechanism from scratch 

Do you have any questions?  
Ask your questions in the comments below, and I will do my best to answer.

