[Link](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)

We have arrived at a point where we have implemented and tested the Transformer [encoder](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras) and [decoder](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras) separately, and we may now join the two together into a complete model. We will also see how to create padding and look-ahead masks by which we will suppress the input values that will not be considered in the encoder or decoder computations. Our end goal remains to apply the complete model to Natural Language Processing (NLP).

In this tutorial, you will discover how to implement the complete Transformer model and create padding and look-ahead masks. 

After completing this tutorial, you will know:

- How to create a padding mask for the encoder and decoder
- How to create a look-ahead mask for the decoder
- How to join the Transformer encoder and decoder into a single model
- How to print out a summary of the encoder and decoder layers

## **Tutorial Overview**

This tutorial is divided into four parts; they are:

- Recap of the Transformer Architecture
- Masking
    - Creating a Padding Mask
    - Creating a Look-Ahead Mask
- Joining the Transformer Encoder and Decoder
- Creating an Instance of the Transformer Model
    - Printing Out a Summary of the Encoder and Decoder Layers

## **Prerequisites**

For this tutorial, we assume that you are already familiar with:

- [The Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F)
- [The Transformer encoder](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)
- [The Transformer decoder](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras)

## **Recap of the Transformer Architecture**

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F) having seen that the Transformer architecture follows an encoder-decoder structure. The encoder, on the left-hand side, is tasked with mapping an input sequence to a sequence of continuous representations; the decoder, on the right-hand side, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.

[![alt](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-727x1024.png)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwp-content%2Fuploads%2F2021%2F08%2Fattention_research_1.png)

The encoder-decoder structure of the Transformer architecture  
Taken from “[Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762)“

In generating an output sequence, the Transformer does not rely on recurrence and convolutions.

You have seen how to implement the Transformer encoder and decoder separately. In this tutorial, you will join the two into a complete Transformer model and apply padding and look-ahead masking to the input values.  

Let’s start first by discovering how to apply masking. 

**Kick-start your project** with my book [Building Transformer Models with Attention](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftransformer-models-with-attention%2F). It provides **self-study tutorials** with **working code** to guide you into building a fully-working transformer model that can  
_translate sentences from one language to another_...

## **Masking**

### **Creating a Padding Mask**

You should already be familiar with the importance of masking the input values before feeding them into the encoder and decoder. 

As you will see when you proceed to [train the Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftraining-the-transformer-model), the input sequences fed into the encoder and decoder will first be zero-padded up to a specific sequence length. The importance of having a padding mask is to make sure that these zero values are not processed along with the actual input values by both the encoder and decoder. 

Let’s create the following function to generate a padding mask for both the encoder and decoder:

**ENTER CODE**

Upon receiving an input, this function will generate a tensor that marks by a value of _one_ wherever the input contains a value of _zero_.  

Hence, if you input the following array:

**ENTER CODE**


Then the output of the `padding_mask` function would be the following:

**ENTER CODE**


### **Creating a Look-Ahead Mask**

A look-ahead mask is required to prevent the decoder from attending to succeeding words, such that the prediction for a particular word can only depend on known outputs for the words that come before it.

For this purpose, let’s create the following function to generate a look-ahead mask for the decoder:

**ENTER CODE**



You will pass to it the length of the decoder input. Let’s make this length equal to 5, as an example:

**ENTER CODE**


Then the output that the `lookahead_mask` function returns is the following:

**ENTER CODE**


Again, the _one_ values mask out the entries that should not be used. In this manner, the prediction of every word only depends on those that come before it.   

## **Joining the Transformer Encoder and Decoder**

Let’s start by creating the class, `TransformerModel`, which inherits from the `Model` base class in Keras:

**ENTER CODE**


Our first step in creating the `TransformerModel` class is to initialize instances of the `Encoder` and `Decoder` classes implemented earlier and assign their outputs to the variables, `encoder` and `decoder`, respectively. If you saved these classes in separate Python scripts, do not forget to import them. I saved my code in the Python scripts _encoder.py_ and _decoder.py_, so I need to import them accordingly. 

You will also include one final dense layer that produces the final output, as in the Transformer architecture of [Vaswani et al. (2017)](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762). 

Next, you shall create the class method, `call()`, to feed the relevant inputs into the encoder and decoder.

A padding mask is first generated to mask the encoder input, as well as the encoder output, when this is fed into the second self-attention block of the decoder:

**ENTER CODE**



A padding mask and a look-ahead mask are then generated to mask the decoder input. These are combined together through an element-wise `maximum` operation:

**ENTER CODE**


Next, the relevant inputs are fed into the encoder and decoder, and the Transformer model output is generated by feeding the decoder output into one final dense layer:

**ENTER CODE**


Combining all the steps gives us the following complete code listing:

**ENTER CODE**


Note that you have performed a small change to the output that is returned by the `padding_mask` function. Its shape is made broadcastable to the shape of the attention weight tensor that it will mask when you train the Transformer model. 

## **Creating an Instance of the Transformer Model**

You will work with the parameter values specified in the paper, [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), by Vaswani et al. (2017):

**ENTER CODE**



As for the input-related parameters, you will work with dummy values for now until you arrive at the stage of [training the complete Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftraining-the-transformer-model). At that point, you will use actual sentences:

**ENTER CODE**



You can now create an instance of the `TransformerModel` class as follows:

**ENTER CODE**


The complete code listing is as follows:

**ENTER CODE**


### **Printing Out a Summary of the Encoder and Decoder Layers**

You may also print out a summary of the encoder and decoder blocks of the Transformer model. The choice to print them out separately will allow you to be able to see the details of their individual sub-layers. In order to do so, add the following line of code to the `__init__()` method of both the `EncoderLayer` and `DecoderLayer` classes:

**ENTER CODE**


Then you need to add the following method to the `EncoderLayer` class:

**ENTER CODE**



And the following method to the `DecoderLayer` class:

**ENTER CODE**



This results in the `EncoderLayer` class being modified as follows (the three dots under the `call()` method mean that this remains the same as the one that was implemented [here](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)):

**ENTER CODE**


Similar changes can be made to the `DecoderLayer` class too.

Once you have the necessary changes in place, you can proceed to create instances of the `EncoderLayer` and `DecoderLayer` classes and print out their summaries as follows:

**ENTER CODE**


The resulting summary for the encoder is the following:

**ENTER CODE**


While the resulting summary for the decoder is the following:

**ENTER CODE**

## **Further Reading**

This section provides more resources on the topic if you are looking to go deeper.

### **Books**

- [Advanced Deep Learning with Python](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FAdvanced-Deep-Learning-Python-next-generation%2Fdp%2F178995617X), 2019
- [Transformers for Natural Language Processing](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FTransformers-Natural-Language-Processing-architectures%2Fdp%2F1800565798), 2021

### **Papers**

- [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), 2017

## **Summary**

In this tutorial, you discovered how to implement the complete Transformer model and create padding and look-ahead masks.

Specifically, you learned:

- How to create a padding mask for the encoder and decoder
- How to create a look-ahead mask for the decoder
- How to join the Transformer encoder and decoder into a single model
- How to print out a summary of the encoder and decoder layers

Do you have any questions?  
Ask your questions in the comments below and I will do my best to answer.