# Training the Transformer Model
By Stefania Cristina on January 6, 2023 in Attention 44
 Post Share
We have put together the complete Transformer model, and now we are ready to train it for neural machine translation. We shall use a training dataset for this purpose, which contains short English and German sentence pairs. We will also revisit the role of masking in computing the accuracy and loss metrics during the training process. 

In this tutorial, you will discover how to train the Transformer model for neural machine translation. 

After completing this tutorial, you will know:

How to prepare the training dataset
How to apply a padding mask to the loss and accuracy computations
How to train the Transformer model
Kick-start your project with my book Building Transformer Models with Attention. It provides self-study tutorials with working code to guide you into building a fully-working transformer model that can
translate sentences from one language to another...

Let’s get started. 


Training the transformer model
Photo by v2osk, some rights reserved.

Tutorial Overview
This tutorial is divided into four parts; they are:

Recap of the Transformer Architecture
Preparing the Training Dataset
Applying a Padding Mask to the Loss and Accuracy Computations
Training the Transformer Model

Prerequisites
For this tutorial, we assume that you are already familiar with:

The theory behind the Transformer model
An implementation of the Transformer model
Recap of the Transformer Architecture
Recall having seen that the Transformer architecture follows an encoder-decoder structure. The encoder, on the left-hand side, is tasked with mapping an input sequence to a sequence of continuous representations; the decoder, on the right-hand side, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.


The encoder-decoder structure of the Transformer architecture
Taken from “Attention Is All You Need“

In generating an output sequence, the Transformer does not rely on recurrence and convolutions.

You have seen how to implement the complete Transformer model, so you can now proceed to train it for neural machine translation. 

Let’s start first by preparing the dataset for training. 

Want to Get Started With Building Transformer Models with Attention?
Take my free 12-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

Download Your FREE Mini-Course


Preparing the Training Dataset
For this purpose, you can refer to a previous tutorial that covers material about preparing the text data for training. 

You will also use a dataset that contains short English and German sentence pairs, which you may download here. This particular dataset has already been cleaned by removing non-printable and non-alphabetic characters and punctuation characters, further normalizing all Unicode characters to ASCII, and changing all uppercase letters to lowercase ones. Hence, you can skip the cleaning step, which is typically part of the data preparation process. However, if you use a dataset that does not come readily cleaned, you can refer to this this previous tutorial to learn how to do so. 

Let’s proceed by creating the PrepareDataset class that implements the following steps:

Loads the dataset from a specified filename. 
clean_dataset = load(open(filename, 'rb'))
Selects the number of sentences to use from the dataset. Since the dataset is large, you will reduce its size to limit the training time. However, you may explore using the full dataset as an extension to this tutorial.
dataset = clean_dataset[:self.n_sentences, :]
Appends start (<START>) and end-of-string (<EOS>) tokens to each sentence. For example, the English sentence, i like to run, now becomes, <START> i like to run <EOS>. This also applies to its corresponding translation in German, ich gehe gerne joggen, which now becomes, <START> ich gehe gerne joggen <EOS>.
for i in range(dataset[:, 0].size):
	dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
	dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"
Shuffles the dataset randomly. 
shuffle(dataset)
Splits the shuffled dataset based on a pre-defined ratio.
train = dataset[:int(self.n_sentences * self.train_split)]
Creates and trains a tokenizer on the text sequences that will be fed into the encoder and finds the length of the longest sequence as well as the vocabulary size. 
enc_tokenizer = self.create_tokenizer(train[:, 0])
enc_seq_length = self.find_seq_length(train[:, 0])
enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])
Tokenizes the sequences of text that will be fed into the encoder by creating a vocabulary of words and replacing each word with its corresponding vocabulary index. The <START> and <EOS> tokens will also form part of this vocabulary. Each sequence is also padded to the maximum phrase length.  
trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
trainX = convert_to_tensor(trainX, dtype=int64)
Creates and trains a tokenizer on the text sequences that will be fed into the decoder, and finds the length of the longest sequence as well as the vocabulary size.
dec_tokenizer = self.create_tokenizer(train[:, 1])
dec_seq_length = self.find_seq_length(train[:, 1])
dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])
Repeats a similar tokenization and padding procedure for the sequences of text that will be fed into the decoder.
trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
trainY = convert_to_tensor(trainY, dtype=int64)
The complete code listing is as follows (refer to this previous tutorial for further details):

from pickle import load
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64
 
 
class PrepareDataset:
	def __init__(self, **kwargs):
		super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = 10000  # Number of sentences to include in the dataset
		self.train_split = 0.9  # Ratio of the training data split
 
	# Fit a tokenizer
	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(dataset)
 
		return tokenizer
 
	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)
 
	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)
 
		return len(tokenizer.word_index) + 1
 
	def __call__(self, filename, **kwargs):
		# Load a clean dataset
		clean_dataset = load(open(filename, 'rb'))
 
		# Reduce dataset size
		dataset = clean_dataset[:self.n_sentences, :]
 
		# Include start and end of string tokens
		for i in range(dataset[:, 0].size):
			dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
			dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"
 
		# Random shuffle the dataset
		shuffle(dataset)
 
		# Split the dataset
		train = dataset[:int(self.n_sentences * self.train_split)]
 
		# Prepare tokenizer for the encoder input
		enc_tokenizer = self.create_tokenizer(train[:, 0])
		enc_seq_length = self.find_seq_length(train[:, 0])
		enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])
 
		# Encode and pad the input sequences
		trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
		trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
		trainX = convert_to_tensor(trainX, dtype=int64)
 
		# Prepare tokenizer for the decoder input
		dec_tokenizer = self.create_tokenizer(train[:, 1])
		dec_seq_length = self.find_seq_length(train[:, 1])
		dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])
 
		# Encode and pad the input sequences
		trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
		trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
		trainY = convert_to_tensor(trainY, dtype=int64)
 
		return trainX, trainY, train, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size
Before moving on to train the Transformer model, let’s first have a look at the output of the PrepareDataset class corresponding to the first sentence in the training dataset:

# Prepare the training data
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('english-german-both.pkl')
 
print(train_orig[0, 0], '\n', trainX[0, :])
<START> did tom tell you <EOS> 
 tf.Tensor([ 1 25  4 97  5  2  0], shape=(7,), dtype=int64)
(Note: Since the dataset has been randomly shuffled, you will likely see a different output.)

You can see that, originally, you had a three-word sentence (did tom tell you) to which you appended the start and end-of-string tokens. Then you proceeded to vectorize (you may notice that the <START> and <EOS> tokens are assigned the vocabulary indices 1 and 2, respectively). The vectorized text was also padded with zeros, such that the length of the end result matches the maximum sequence length of the encoder:

print('Encoder sequence length:', enc_seq_length)
Encoder sequence length: 7
You can similarly check out the corresponding target data that is fed into the decoder:

print(train_orig[0, 1], '\n', trainY[0, :])
<START> hat tom es dir gesagt <EOS> 
 tf.Tensor([  1  14   5   7  42 162   2   0   0   0   0   0], shape=(12,), dtype=int64)
Here, the length of the end result matches the maximum sequence length of the decoder:

print('Decoder sequence length:', dec_seq_length)
Decoder sequence length: 12

Applying a Padding Mask to the Loss and Accuracy Computations
Recall seeing that the importance of having a padding mask at the encoder and decoder is to make sure that the zero values that we have just appended to the vectorized inputs are not processed along with the actual input values. 

This also holds true for the training process, where a padding mask is required so that the zero padding values in the target data are not considered in the computation of the loss and accuracy.

Let’s have a look at the computation of loss first. 

This will be computed using a sparse categorical cross-entropy loss function between the target and predicted values and subsequently multiplied by a padding mask so that only the valid non-zero values are considered. The returned loss is the mean of the unmasked values:

def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)
 
    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)
For the computation of accuracy, the predicted and target values are first compared. The predicted output is a tensor of size (batch_size, dec_seq_length, dec_vocab_size) and contains probability values (generated by the softmax function on the decoder side) for the tokens in the output. In order to be able to perform the comparison with the target values, only each token with the highest probability value is considered, with its dictionary index being retrieved through the operation: argmax(prediction, axis=2). Following the application of a padding mask, the returned accuracy is the mean of the unmasked values:

def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(math.equal(target, 0))
 
    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)
 
    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)
 
    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)
Training the Transformer Model
Let’s first define the model and training parameters as specified by Vaswani et al. (2017):

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
 
# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1
(Note: Only consider two epochs to limit the training time. However, you may explore training the model further as an extension to this tutorial.)

You also need to implement a learning rate scheduler that initially increases the learning rate linearly for the first warmup_steps and then decreases it proportionally to the inverse square root of the step number. Vaswani et al. express this by the following formula: 


 

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
 
        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps
 
    def __call__(self, step_num):
 
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
 
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
An instance of the LRScheduler class is subsequently passed on as the learning_rate argument of the Adam optimizer:

optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
Next,  split the dataset into batches in preparation for training:

train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)
This is followed by the creation of a model instance:

training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
In training the Transformer model, you will write your own training loop, which incorporates the loss and accuracy functions that were implemented earlier. 

The default runtime in Tensorflow 2.0 is eager execution, which means that operations execute immediately one after the other. Eager execution is simple and intuitive, making debugging easier. Its downside, however, is that it cannot take advantage of the global performance optimizations that run the code using the graph execution. In graph execution, a graph is first built before the tensor computations can be executed, which gives rise to a computational overhead. For this reason, the use of graph execution is mostly recommended for large model training rather than for small model training, where eager execution may be more suited to perform simpler operations. Since the Transformer model is sufficiently large, apply the graph execution to train it. 

In order to do so, you will use the @function decorator as follows:

@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
 
        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)
 
        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)
 
        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)
 
    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)
 
    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
 
    train_loss(loss)
    train_accuracy(accuracy)
With the addition of the @function decorator, a function that takes tensors as input will be compiled into a graph. If the @function decorator is commented out, the function is, alternatively, run with eager execution. 

The next step is implementing the training loop that will call the train_step function above. The training loop will iterate over the specified number of epochs and the dataset batches. For each batch, the train_step function computes the training loss and accuracy measures and applies the optimizer to update the trainable model parameters. A checkpoint manager is also included to save a checkpoint after every five epochs:

train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
 
# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)
 
for epoch in range(epochs):
 
    train_loss.reset_states()
    train_accuracy.reset_states()
 
    print("\nStart of epoch %d" % (epoch + 1))
 
    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
 
        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
 
        train_step(encoder_input, decoder_input, decoder_output)
 
        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
           
    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
 
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
An important point to keep in mind is that the input to the decoder is offset by one position to the right with respect to the encoder input. The idea behind this offset, combined with a look-ahead mask in the first multi-head attention block of the decoder, is to ensure that the prediction for the current token can only depend on the previous tokens. 

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

– Attention Is All You Need, 2017. 

It is for this reason that the encoder and decoder inputs are fed into the Transformer model in the following manner:

encoder_input = train_batchX[:, 1:]

decoder_input = train_batchY[:, :-1]

Putting together the complete code listing produces the following:

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from prepare_dataset import PrepareDataset
from time import time
 
 
# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
 
# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1
 
 
# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
 
        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps
 
    def __call__(self, step_num):
 
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
 
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
 
 
# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
 
# Prepare the training and test splits of the dataset
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('english-german-both.pkl')
 
# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)
 
# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
 
 
# Defining the loss function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)
 
    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)
 
 
# Defining the accuracy function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))
 
    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)
 
    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)
 
    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)
 
 
# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
 
# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)
 
# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
 
        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)
 
        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)
 
        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)
 
    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)
 
    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
 
    train_loss(loss)
    train_accuracy(accuracy)
 
 
for epoch in range(epochs):
 
    train_loss.reset_states()
    train_accuracy.reset_states()
 
    print("\nStart of epoch %d" % (epoch + 1))
 
    start_time = time()
 
    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
 
        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
 
        train_step(encoder_input, decoder_input, decoder_output)
 
        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            # print("Samples so far: %s" % ((step + 1) * batch_size))
 
    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
 
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
 
print("Total time taken: %.2fs" % (time() - start_time))
Running the code produces a similar output to the following (you will likely see different loss and accuracy values because the training is from scratch, whereas the training time depends on the computational resources that you have available for training):

Start of epoch 1
Epoch 1 Step 0 Loss 8.4525 Accuracy 0.0000
Epoch 1 Step 50 Loss 7.6768 Accuracy 0.1234
Epoch 1 Step 100 Loss 7.0360 Accuracy 0.1713
Epoch 1: Training Loss 6.7109, Training Accuracy 0.1924
 
Start of epoch 2
Epoch 2 Step 0 Loss 5.7323 Accuracy 0.2628
Epoch 2 Step 50 Loss 5.4360 Accuracy 0.2756
Epoch 2 Step 100 Loss 5.2638 Accuracy 0.2839
Epoch 2: Training Loss 5.1468, Training Accuracy 0.2908
Total time taken: 87.98s
It takes 155.13s for the code to run using eager execution alone on the same platform that is making use of only a CPU, which shows the benefit of using graph execution. 

Further Reading
This section provides more resources on the topic if you are looking to go deeper.

Books
Advanced Deep Learning with Python, 2019
Transformers for Natural Language Processing, 2021
Papers
Attention Is All You Need, 2017

Websites
Writing a training loop from scratch in Keras: https://keras.io/guides/writing_a_training_loop_from_scratch/
Summary
In this tutorial, you discovered how to train the Transformer model for neural machine translation.

Specifically, you learned:

How to prepare the training dataset
How to apply a padding mask to the loss and accuracy computations
How to train the Transformer model
Do you have any questions?
Ask your questions in the comments below, and I will do my best to answer.
