# Inferencing the Transformer Model
By Stefania Cristina on January 6, 2023 in Attention 17
 Post Share
We have seen how to train the Transformer model on a dataset of English and German sentence pairs and how to plot the training and validation loss curves to diagnose the model’s learning performance and decide at which epoch to run inference on the trained model. We are now ready to run inference on the trained Transformer model to translate an input sentence.

In this tutorial, you will discover how to run inference on the trained Transformer model for neural machine translation. 

After completing this tutorial, you will know:

How to run inference on the trained Transformer model
How to generate text translations
Kick-start your project with my book Building Transformer Models with Attention. It provides self-study tutorials with working code to guide you into building a fully-working transformer model that can
translate sentences from one language to another...

Let’s get started. 


Inferencing the Transformer model
Photo by Karsten Würth, some rights reserved.

Tutorial Overview
This tutorial is divided into three parts; they are:

Recap of the Transformer Architecture
Inferencing the Transformer Model
Testing Out the Code

Prerequisites
For this tutorial, we assume that you are already familiar with:

The theory behind the Transformer model
An implementation of the Transformer model
Training the Transformer model
Plotting the training and validation loss curves for the Transformer model
Recap of the Transformer Architecture
Recall having seen that the Transformer architecture follows an encoder-decoder structure. The encoder, on the left-hand side, is tasked with mapping an input sequence to a sequence of continuous representations; the decoder, on the right-hand side, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.


The encoder-decoder structure of the Transformer architecture
Taken from “Attention Is All You Need“

In generating an output sequence, the Transformer does not rely on recurrence and convolutions.

You have seen how to implement the complete Transformer model and subsequently train it on a dataset of English and German sentence pairs. Let’s now proceed to run inference on the trained model for neural machine translation. 


Inferencing the Transformer Model
Let’s start by creating a new instance of the TransformerModel class that was previously implemented in this tutorial. 

You will feed into it the relevant input arguments as specified in the paper of Vaswani et al. (2017) and the relevant information about the dataset in use: 

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
 
# Define the dataset parameters
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2405  # Encoder vocabulary size
dec_vocab_size = 3858  # Decoder vocabulary size
 
# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)
Here, note that the last input being fed into the TransformerModel corresponded to the dropout rate for each of the Dropout layers in the Transformer model. These Dropout layers will not be used during model inferencing (you will eventually set the training argument to False), so you may safely set the dropout rate to 0.

Furthermore, the TransformerModel class was already saved into a separate script named model.py. Hence, to be able to use the TransformerModel class, you need to include from model import TransformerModel.

Next, let’s create a class, Translate, that inherits from the Module base class in Keras and assign the initialized inferencing model to the variable transformer:

class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.transformer = inferencing_model
        ...
When you trained the Transformer model, you saw that you first needed to tokenize the sequences of text that were to be fed into both the encoder and decoder. You achieved this by creating a vocabulary of words and replacing each word with its corresponding vocabulary index. 

You will need to implement a similar process during the inferencing stage before feeding the sequence of text to be translated into the Transformer model. 

For this purpose, you will include within the class the following load_tokenizer method, which will serve to load the encoder and decoder tokenizers that you would have generated and saved during the training stage:

def load_tokenizer(self, name):
    with open(name, 'rb') as handle:
        return load(handle)
It is important that you tokenize the input text at the inferencing stage using the same tokenizers generated at the training stage of the Transformer model since these tokenizers would have already been trained on text sequences similar to your testing data. 

The next step is to create the class method, call(), that will take care to:

Append the start (<START>) and end-of-string (<EOS>) tokens to the input sentence:
def __call__(self, sentence):
    sentence[0] = "<START> " + sentence[0] + " <EOS>"
Load the encoder and decoder tokenizers (in this case, saved in the enc_tokenizer.pkl and dec_tokenizer.pkl pickle files, respectively):
enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
Prepare the input sentence by tokenizing it first, then padding it to the maximum phrase length, and subsequently converting it to a tensor:
encoder_input = enc_tokenizer.texts_to_sequences(sentence)
encoder_input = pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
encoder_input = convert_to_tensor(encoder_input, dtype=int64)
Repeat a similar tokenization and tensor conversion procedure for the <START> and <EOS> tokens at the output:
output_start = dec_tokenizer.texts_to_sequences(["<START>"])
output_start = convert_to_tensor(output_start[0], dtype=int64)
 
output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
output_end = convert_to_tensor(output_end[0], dtype=int64)
Prepare the output array that will contain the translated text. Since you do not know the length of the translated sentence in advance, you will initialize the size of the output array to 0, but set its dynamic_size parameter to True so that it may grow past its initial size. You will then set the first value in this output array to the <START> token:
decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
decoder_output = decoder_output.write(0, output_start)
Iterate, up to the decoder sequence length, each time calling the Transformer model to predict an output token. Here, the training input, which is then passed on to each of the Transformer’s Dropout layers, is set to False so that no values are dropped during inference. The prediction with the highest score is then selected and written at the next available index of the output array. The for loop is terminated with a break statement as soon as an <EOS> token is predicted:
for i in range(dec_seq_length):
 
    prediction = self.transformer(encoder_input, transpose(decoder_output.stack()), training=False)
 
    prediction = prediction[:, -1, :]
 
    predicted_id = argmax(prediction, axis=-1)
    predicted_id = predicted_id[0][newaxis]
 
    decoder_output = decoder_output.write(i + 1, predicted_id)
 
    if predicted_id == output_end:
        break
Decode the predicted tokens into an output list and return it:
output = transpose(decoder_output.stack())[0]
output = output.numpy()
 
output_str = []
 
# Decode the predicted tokens into an output list
for i in range(output.shape[0]):
 
   key = output[i]
   translation = dec_tokenizer.index_word[key]
   output_str.append(translation)
 
return output_str
The complete code listing, so far, is as follows:

from pickle import load
from tensorflow import Module
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from model import TransformerModel
 
# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
 
# Define the dataset parameters
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2405  # Encoder vocabulary size
dec_vocab_size = 3858  # Decoder vocabulary size
 
# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)
 
 
class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.transformer = inferencing_model
 
    def load_tokenizer(self, name):
        with open(name, 'rb') as handle:
            return load(handle)
 
    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"
 
        # Load encoder and decoder tokenizers
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
 
        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)
 
        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)
 
        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)
 
        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)
 
        for i in range(dec_seq_length):
 
            # Predict an output token
            prediction = self.transformer(encoder_input, transpose(decoder_output.stack()), training=False)
 
            prediction = prediction[:, -1, :]
 
            # Select the prediction with the highest score
            predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]
 
            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)
 
            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break
 
        output = transpose(decoder_output.stack())[0]
        output = output.numpy()
 
        output_str = []
 
        # Decode the predicted tokens into an output string
        for i in range(output.shape[0]):
 
            key = output[i]
            print(dec_tokenizer.index_word[key])
 
        return output_str
Want to Get Started With Building Transformer Models with Attention?
Take my free 12-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

Download Your FREE Mini-Course


Testing Out the Code
In order to test out the code, let’s have a look at the test_dataset.txt file that you would have saved when preparing the dataset for training. This text file contains a set of English-German sentence pairs that have been reserved for testing, from which you can select a couple of sentences to test.

Let’s start with the first sentence:

# Sentence to translate
sentence = ['im thirsty']
The corresponding ground truth translation in German for this sentence, including the <START> and <EOS> decoder tokens, should be: <START> ich bin durstig <EOS>.

If you have a look at the plotted training and validation loss curves for this model (here, you are training for 20 epochs), you may notice that the validation loss curve slows down considerably and starts plateauing at around epoch 16. 

So let’s proceed to load the saved model’s weights at the 16th epoch and check out the prediction that is generated by the model:

# Load the trained model's weights at the specified epoch
inferencing_model.load_weights('weights/wghts16.ckpt')
 
# Create a new instance of the 'Translate' class
translator = Translate(inferencing_model)
 
# Translate the input sentence
print(translator(sentence))
Running the lines of code above produces the following translated list of words:

['start', 'ich', 'bin', 'durstig', ‘eos']
Which is equivalent to the ground truth German sentence that was expected (always keep in mind that since you are training the Transformer model from scratch, you may arrive at different results depending on the random initialization of the model weights). 

Let’s check out what would have happened if you had, instead, loaded a set of weights corresponding to a much earlier epoch, such as the 4th epoch. In this case, the generated translation is the following:

['start', 'ich', 'bin', 'nicht', 'nicht', 'eos']
In English, this translates to: I in not not, which is clearly far off from the input English sentence, but which is expected since, at this epoch, the learning process of the Transformer model is still at the very early stages. 

Let’s try again with a second sentence from the test dataset:

# Sentence to translate
sentence = ['are we done']
The corresponding ground truth translation in German for this sentence, including the <START> and <EOS> decoder tokens, should be: <START> sind wir dann durch <EOS>.

The model’s translation for this sentence, using the weights saved at epoch 16, is:

['start', 'ich', 'war', 'fertig', 'eos']
Which, instead, translates to: I was ready. While this is also not equal to the ground truth, it is close to its meaning. 

What the last test suggests, however, is that the Transformer model might have required many more data samples to train effectively. This is also corroborated by the validation loss at which the validation loss curve plateaus remain relatively high. 

Indeed, Transformer models are notorious for being very data hungry. Vaswani et al. (2017), for example, trained their English-to-German translation model using a dataset containing around 4.5 million sentence pairs. 

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs…For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences…

– Attention Is All You Need, 2017.

They reported that it took them 3.5 days on 8 P100 GPUs to train the English-to-German translation model. 

In comparison, you have only trained on a dataset comprising 10,000 data samples here, split between training, validation, and test sets. 

So the next task is actually for you. If you have the computational resources available, try to train the Transformer model on a much larger set of sentence pairs and see if you can obtain better results than the translations obtained here with a limited amount of data. 


Further Reading
This section provides more resources on the topic if you are looking to go deeper.

Books
Advanced Deep Learning with Python, 2019
Transformers for Natural Language Processing, 2021
Papers
Attention Is All You Need, 2017
Summary
In this tutorial, you discovered how to run inference on the trained Transformer model for neural machine translation.

Specifically, you learned:

How to run inference on the trained Transformer model
How to generate text translations
Do you have any questions?
Ask your questions in the comments below, and I will do my best to answer.
