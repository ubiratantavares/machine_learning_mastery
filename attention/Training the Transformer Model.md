[Link](https://machinelearningmastery.com/training-the-transformer-model/)

We have put together the [complete Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fjoining-the-transformer-encoder-and-decoder-and-masking), and now we are ready to train it for neural machine translation. We shall use a training dataset for this purpose, which contains short English and German sentence pairs. We will also revisit the role of masking in computing the accuracy and loss metrics during the training process. 

In this tutorial, you will discover how to train the Transformer model for neural machine translation. 

After completing this tutorial, you will know:

- How to prepare the training dataset
- How to apply a padding mask to the loss and accuracy computations
- How to train the Transformer model

## **Tutorial Overview**

This tutorial is divided into four parts; they are:

- Recap of the Transformer Architecture
- Preparing the Training Dataset
- Applying a Padding Mask to the Loss and Accuracy Computations
- Training the Transformer Model

## **Prerequisites**

For this tutorial, we assume that you are already familiar with:

- [The theory behind the Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F)
- [An implementation of the Transformer model](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fjoining-the-transformer-encoder-and-decoder-and-masking)

## **Recap of the Transformer Architecture**

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F) having seen that the Transformer architecture follows an encoder-decoder structure. The encoder, on the left-hand side, is tasked with mapping an input sequence to a sequence of continuous representations; the decoder, on the right-hand side, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.

[![alt](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-727x1024.png)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwp-content%2Fuploads%2F2021%2F08%2Fattention_research_1.png)

The encoder-decoder structure of the Transformer architecture  
Taken from “[Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762)“

In generating an output sequence, the Transformer does not rely on recurrence and convolutions.

You have seen how to implement the complete Transformer model, so you can now proceed to train it for neural machine translation. 

Let’s start first by preparing the dataset for training.   

### Want to Get Started With Building Transformer Models with Attention?

Take my free 12-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

Download Your FREE Mini-Course

## **Preparing the Training Dataset**

For this purpose, you can refer to a previous tutorial that covers material about [preparing the text data](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdevelop-neural-machine-translation-system-keras%2F) for training. 

You will also use a dataset that contains short English and German sentence pairs, which you may download [here](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2FRishav09%2FNeural-Machine-Translation-System%2Fblob%2Fmaster%2Fenglish-german-both.pkl). This particular dataset has already been cleaned by removing non-printable and non-alphabetic characters and punctuation characters, further normalizing all Unicode characters to ASCII, and changing all uppercase letters to lowercase ones. Hence, you can skip the cleaning step, which is typically part of the data preparation process. However, if you use a dataset that does not come readily cleaned, you can refer to this [this previous tutorial](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdevelop-neural-machine-translation-system-keras%2F) to learn how to do so. 

Let’s proceed by creating the `PrepareDataset` class that implements the following steps:

- Loads the dataset from a specified filename. 

## ENTER CODE




- Selects the number of sentences to use from the dataset. Since the dataset is large, you will reduce its size to limit the training time. However, you may explore using the full dataset as an extension to this tutorial.

## ENTER CODE

- Appends start (<START>) and end-of-string (<EOS>) tokens to each sentence. For example, the English sentence, `i like to run`, now becomes, `<START> i like to run <EOS>`. This also applies to its corresponding translation in German, `ich gehe gerne joggen`, which now becomes, `<START> ich gehe gerne joggen <EOS>`.


## ENTER CODE

- Shuffles the dataset randomly. 

## ENTER CODE

- Splits the shuffled dataset based on a pre-defined ratio.

## ENTER CODE

- Creates and trains a tokenizer on the text sequences that will be fed into the encoder and finds the length of the longest sequence as well as the vocabulary size. 

## ENTER CODE

- Tokenizes the sequences of text that will be fed into the encoder by creating a vocabulary of words and replacing each word with its corresponding vocabulary index. The <START> and <EOS> tokens will also form part of this vocabulary. Each sequence is also padded to the maximum phrase length.  

## ENTER CODE

- Creates and trains a tokenizer on the text sequences that will be fed into the decoder, and finds the length of the longest sequence as well as the vocabulary size.

## ENTER CODE

- Repeats a similar tokenization and padding procedure for the sequences of text that will be fed into the decoder.

## ENTER CODE

The complete code listing is as follows (refer to [this previous tutorial](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdevelop-neural-machine-translation-system-keras%2F) for further details):

## ENTER CODE

Before moving on to train the Transformer model, let’s first have a look at the output of the `PrepareDataset` class corresponding to the first sentence in the training dataset:

## ENTER CODE

(Note: Since the dataset has been randomly shuffled, you will likely see a different output.)

You can see that, originally, you had a three-word sentence (_did tom tell you_) to which you appended the start and end-of-string tokens. Then you proceeded to vectorize (you may notice that the <START> and <EOS> tokens are assigned the vocabulary indices 1 and 2, respectively). The vectorized text was also padded with zeros, such that the length of the end result matches the maximum sequence length of the encoder:

## ENTER CODE

You can similarly check out the corresponding target data that is fed into the decoder:


## ENTER CODE

Here, the length of the end result matches the maximum sequence length of the decoder:

## ENTER CODE

## **Applying a Padding Mask to the Loss and Accuracy Computations**

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras) seeing that the importance of having a padding mask at the encoder and decoder is to make sure that the zero values that we have just appended to the vectorized inputs are not processed along with the actual input values. 

This also holds true for the training process, where a padding mask is required so that the zero padding values in the target data are not considered in the computation of the loss and accuracy.

Let’s have a look at the computation of loss first. 

This will be computed using a sparse categorical cross-entropy loss function between the target and predicted values and subsequently multiplied by a padding mask so that only the valid non-zero values are considered. The returned loss is the mean of the unmasked values:

## ENTER CODE

For the computation of accuracy, the predicted and target values are first compared. The predicted output is a tensor of size (_batch_size_, _dec_seq_length_, _dec_vocab_size_) and contains probability values (generated by the softmax function on the decoder side) for the tokens in the output. In order to be able to perform the comparison with the target values, only each token with the highest probability value is considered, with its dictionary index being retrieved through the operation: `argmax(prediction, axis=2)`. Following the application of a padding mask, the returned accuracy is the mean of the unmasked values:

## ENTER CODE

## **Training the Transformer Model**

Let’s first define the model and training parameters as specified by [Vaswani et al. (2017)](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762):

## ENTER CODE

(Note: Only consider two epochs to limit the training time. However, you may explore training the model further as an extension to this tutorial.)

You also need to implement a learning rate scheduler that initially increases the learning rate linearly for the first _warmup_steps_ and then decreases it proportionally to the inverse square root of the step number. Vaswani et al. express this by the following formula: 

$$\text{learning_rate} = \text{d_model}^{−0.5} \cdot \text{min}(\text{step}^{−0.5}, \text{step} \cdot \text{warmup_steps}^{−1.5})$$

## ENTER CODE

An instance of the `LRScheduler` class is subsequently passed on as the `learning_rate` argument of the Adam optimizer:

## ENTER CODE

Next,  split the dataset into batches in preparation for training:

## ENTER CODE

This is followed by the creation of a model instance:

## ENTER CODE

In training the Transformer model, you will write your own training loop, which incorporates the loss and accuracy functions that were implemented earlier. 

The default runtime in Tensorflow 2.0 is _eager execution_, which means that operations execute immediately one after the other. Eager execution is simple and intuitive, making debugging easier. Its downside, however, is that it cannot take advantage of the global performance optimizations that run the code using the _graph execution_. In graph execution, a graph is first built before the tensor computations can be executed, which gives rise to a computational overhead. For this reason, the use of graph execution is mostly recommended for large model training rather than for small model training, where eager execution may be more suited to perform simpler operations. Since the Transformer model is sufficiently large, apply the graph execution to train it. 

In order to do so, you will use the `@function` decorator as follows:

## ENTER CODE

With the addition of the `@function` decorator, a function that takes tensors as input will be compiled into a graph. If the `@function` decorator is commented out, the function is, alternatively, run with eager execution. 

The next step is implementing the training loop that will call the `train_step` function above. The training loop will iterate over the specified number of epochs and the dataset batches. For each batch, the `train_step` function computes the training loss and accuracy measures and applies the optimizer to update the trainable model parameters. A checkpoint manager is also included to save a checkpoint after every five epochs:

## ENTER CODE

An important point to keep in mind is that the input to the decoder is offset by one position to the right with respect to the encoder input. The idea behind this offset, combined with a look-ahead mask in the first multi-head attention block of the decoder, is to ensure that the prediction for the current token can only depend on the previous tokens. 

> _This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i._
> 
> _–_ [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), 2017. 

It is for this reason that the encoder and decoder inputs are fed into the Transformer model in the following manner:

`encoder_input = train_batchX[:, 1:]`

`decoder_input = train_batchY[:, :-1]`

Putting together the complete code listing produces the following:

## ENTER CODE

Running the code produces a similar output to the following (you will likely see different loss and accuracy values because the training is from scratch, whereas the training time depends on the computational resources that you have available for training):

## ENTER CODE

It takes 155.13s for the code to run using eager execution alone on the same platform that is making use of only a CPU, which shows the benefit of using graph execution. 

## **Further Reading**

This section provides more resources on the topic if you are looking to go deeper.

### **Books**

- [Advanced Deep Learning with Python](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FAdvanced-Deep-Learning-Python-next-generation%2Fdp%2F178995617X), 2019
- [Transformers for Natural Language Processing](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FTransformers-Natural-Language-Processing-architectures%2Fdp%2F1800565798), 2021

### **Papers**

- [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), 2017

### **Websites**

- Writing a training loop from scratch in Keras: [https://keras.io/guides/writing_a_training_loop_from_scratch/](https://12ft.io/proxy?q=https%3A%2F%2Fkeras.io%2Fguides%2Fwriting_a_training_loop_from_scratch%2F)

## **Summary**

In this tutorial, you discovered how to train the Transformer model for neural machine translation.

Specifically, you learned:

- How to prepare the training dataset
- How to apply a padding mask to the loss and accuracy computations
- How to train the Transformer model

Do you have any questions?  
Ask your questions in the comments below, and I will do my best to answer.