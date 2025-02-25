# One-Dimensional Tensors in Pytorch

[Referência](https://machinelearningmastery.com/one-dimensional-tensors-in-pytorch/)

# Tensores unidimensionais em Pytorch

PyTorch é um *framework* de *deep learning* de código aberto baseado na linguagem Python. 

Ele permite que você construa, treine e implante modelos de *deep learning*, oferecendo muita versatilidade e eficiência.

O PyTorch é focado principalmente em operações de tensores, enquanto um tensor pode ser um número, uma matriz ou uma matriz multidimensional.

Neste tutorial, realizaremos algumas operações básicas em tensores unidimensionais, pois eles são objetos matemáticos complexos e uma parte essencial da biblioteca PyTorch. 

Portanto, antes de entrar em detalhes e conceitos mais avançados, é preciso saber o básico.

Depois de concluir este tutorial, você irá:

* Entender os conceitos básicos das operações tensoriais unidimensionais no PyTorch;
* Conhecer os tipos e formas de tensores e execute operações de fatiamento e indexação de tensores;
* Ser capaz de aplicar alguns métodos em objetos tensores, como média, desvio padrão, adição, multiplicação e muito mais.

## Tipos e formas de tensores unidimensionais

Primeiro, vamos importar algumas bibliotecas que usaremos neste tutorial.

```Python
import torch
import numpy as np 
import pandas as pd
```

Se você tem experiência em outras linguagens de programação, a maneira mais fácil de entender um tensor é considerá-lo como um array multidimensional. 

Portanto, um tensor unidimensional é simplesmente um array unidimensional, ou um vetor. 

Para converter uma lista de inteiros em tensor, aplique o construtor **torch.tensor()**. 

Por exemplo, pegaremos uma lista de inteiros e a converteremos em vários objetos tensores.

```Python
int_to_tensor = torch.tensor([10, 11, 12, 13])
print("Tensor object type after conversion: ", int_to_tensor.dtype)
print("Tensor object type after conversion: ", int_to_tensor.type())
```

Além disso, você pode aplicar o mesmo método **torch.tensor()** para converter uma lista de float em um tensor de float.

```Python
float_to_tensor = torch.tensor([10.0, 11.0, 12.0, 13.0])
print("Tensor object type after conversion: ", float_to_tensor.dtype)
print("Tensor object type after conversion: ", float_to_tensor.type())
```

Note que elementos de uma lista que precisam ser convertidos em um tensor devem ter o mesmo tipo. 

Além disso, se você quiser converter uma lista para um certo tipo de tensor, o torch também permite que você faça isso. 

As linhas de código abaixo, por exemplo, converterão uma lista de inteiros para um tensor float.

```Python
int_list_to_float_tensor = torch.FloatTensor([10, 11, 12, 13])
int_list_to_float_tensor.type()
print("Tensor  type after conversion: ", int_list_to_float_tensor.type())
```

Da mesma forma, os métodos **size()** e **ndimension()** permitem que você encontre o tamanho e as dimensões de um objeto tensor.

```Python
print("Size of the int_list_to_float_tensor: ", int_list_to_float_tensor.size())
print("Dimensions of the int_list_to_float_tensor: ",int_list_to_float_tensor.ndimension())
```

Para remodelar um objeto tensor, o método **view()** pode ser aplicado. 

Ele recebe como argumentos o **rows** e **columns** como argumentos. 

Como exemplo, vamos usar este método para remodelar int_list_to_float_tensor.

```Python
reshaped_tensor = int_list_to_float_tensor.view(4, 1)
print("Original Size of the tensor: ", reshaped_tensor)
print("New size of the tensor: ", reshaped_tensor)
```

Como você pode ver, o método **view()** método alterou o tamanho do tensor para torch.Size([4, 1]), com 4 linhas e 1 coluna.

Embora o número de elementos em um objeto tensor deva permanecer constante após a aplicação do método **view()** , você pode usar  -1 (como  **reshaped_tensor.view(-1, 1)**) para remodelar um tensor de tamanho dinâmico.

## Convertendo matrizes Numpy em tensores

O Pytorch também permite que você converta arrays NumPy em tensores. 

Você pode usar **torch.from_numpy** para esta operação. Vamos pegar um array NumPy e aplicar a operação.

```Python
numpy_arr = np.array([10.0, 11.0, 12.0, 13.0])
from_numpy_to_tensor = torch.from_numpy(numpy_arr)

print("dtype of the tensor: ", from_numpy_to_tensor.dtype)
print("type of the tensor: ", from_numpy_to_tensor.type())
```

Similarmente, você pode converter o objeto tensor de volta para um array NumPy. 

Vamos usar o exemplo anterior para mostrar como isso é feito.

```Python
tensor_to_numpy = from_numpy_to_tensor.numpy()
print("back to numpy from tensor: ", tensor_to_numpy)
print("dtype of converted numpy array: ", tensor_to_numpy.dtype)
```

## Convertendo Séries Pandas em Tensores

Você também pode converter uma série pandas para um tensor. 

Para isso, primeiro você precisará armazenar a série pandas com a função **values()** usando um array NumPy.

```Python
pandas_series=pd.Series([1, 0.2, 3, 13.1])
store_with_numpy=torch.from_numpy(pandas_series.values)
print("Stored tensor in numpy array: ", store_with_numpy)
print("dtype of stored tensor: ", store_with_numpy.dtype)
print("type of stored tensor: ", store_with_numpy.type())
```

Além disso, o *framework* Pytorch nos permite fazer muitas coisas com tensores, como por exemplo, seu método **item()** que retorna um número Python de um tensor e método **tolist()** que retorna uma lista.

```Python
new_tensor=torch.tensor([10, 11, 12, 13]) 
print("the second item is",new_tensor[1].item())
tensor_to_list=new_tensor.tolist()
print('tensor:', new_tensor,"\nlist:",tensor_to_list)
```

## Indexação e fatiamento em tensores unidimensionais

As operações de indexação e fatiamento são quase as mesmas em Pytorch e em python. 

Portanto, o primeiro índice sempre começa em 0 e o último índice é menor que o comprimento total do tensor. Use colchetes para acessar qualquer número em um tensor.

```Python
tensor_index = torch.tensor([0, 1, 2, 3])
print("Check value at index 0:",tensor_index[0])
print("Check value at index 3:",tensor_index[3])
```

Vamos dar um exemplo para verificar como essas operações podem ser aplicadas.

```Python
example_tensor = torch.tensor([50, 11, 22, 33, 44])
sclicing_tensor = example_tensor[1:4]
print("example tensor : ", example_tensor)
print("subset of example tensor:", sclicing_tensor)
```

Agora, vamos alterar o valor no índice 3 de example_tensor:

```Python
print("value at index 3 of example tensor:", example_tensor[3])
example_tensor[3] = 0
print("new tensor:", example_tensor)
```

## Algumas funções para aplicar em tensores unidimensionais

Nesta seção, revisaremos alguns métodos estatísticos que podem ser aplicados em objetos tensores.

### Funções Min e Max

Esses dois métodos úteis são empregados para encontrar o valor mínimo e máximo em um tensor. Eis como eles funcionam.

Usaremos como exemplo um sample_tensor para aplicar esses métodos.


```Python
sample_tensor = torch.tensor([5, 4, 3, 2, 1])
min_value = sample_tensor.min()
max_value = sample_tensor.max()
print("check minimum value in the tensor: ", min_value)
print("check maximum value in the tensor: ", max_value)
```
## Média e desvio padrão

Média e desvio padrão são frequentemente usados ​​ao fazer operações estatísticas em tensores. 

Você pode aplicar essas duas métricas usando as funções .mean()e .std()no Pytorch.

Vamos usar um exemplo para ver como essas duas métricas são calculadas.

```Python
mean_std_tensor = torch.tensor([-1.0, 2.0, 1, -2])
Mean = mean_std_tensor.mean()
print("mean of mean_std_tensor: ", Mean)
std_dev = mean_std_tensor.std()
print("standard deviation of mean_std_tensor: ", std_dev)
```
## Operações simples de adição e multiplicação em tensores unidimensionais

Operações de adição e multiplicação podem ser facilmente aplicadas em tensores no Pytorch. 

Nesta seção, criaremos dois tensores unidimensionais para demonstrar como essas operações podem ser usadas.

```Python
a = torch.tensor([1, 1])
b = torch.tensor([2, 2])

add = a + b
multiply = a * b

print("addition of two tensors: ", add)
print("multiplication of two tensors: ", multiply)
```

Para sua conveniência, abaixo estão todos os exemplos acima reunidos para que você possa experimentá-los de uma só vez:

```Python
import torch
import numpy as np
import pandas as pd

int_to_tensor = torch.tensor([10, 11, 12, 13])
print("Tensor object type after conversion: ", int_to_tensor.dtype)
print("Tensor object type after conversion: ", int_to_tensor.type())

float_to_tensor = torch.tensor([10.0, 11.0, 12.0, 13.0])
print("Tensor object type after conversion: ", float_to_tensor.dtype)
print("Tensor object type after conversion: ", float_to_tensor.type())

int_list_to_float_tensor = torch.FloatTensor([10, 11, 12, 13])
int_list_to_float_tensor.type()
print("Tensor  type after conversion: ", int_list_to_float_tensor.type())

print("Size of the int_list_to_float_tensor: ", int_list_to_float_tensor.size())
print("Dimensions of the int_list_to_float_tensor: ",int_list_to_float_tensor.ndimension())

reshaped_tensor = int_list_to_float_tensor.view(4, 1)
print("Original Size of the tensor: ", reshaped_tensor)
print("New size of the tensor: ", reshaped_tensor)

numpy_arr = np.array([10.0, 11.0, 12.0, 13.0])
from_numpy_to_tensor = torch.from_numpy(numpy_arr)
print("dtype of the tensor: ", from_numpy_to_tensor.dtype)
print("type of the tensor: ", from_numpy_to_tensor.type())

tensor_to_numpy = from_numpy_to_tensor.numpy()
print("back to numpy from tensor: ", tensor_to_numpy)
print("dtype of converted numpy array: ", tensor_to_numpy.dtype)

pandas_series=pd.Series([1, 0.2, 3, 13.1])
store_with_numpy=torch.from_numpy(pandas_series.values)
print("Stored tensor in numpy array: ", store_with_numpy)
print("dtype of stored tensor: ", store_with_numpy.dtype)
print("type of stored tensor: ", store_with_numpy.type())

new_tensor=torch.tensor([10, 11, 12, 13]) 
print("the second item is",new_tensor[1].item())
tensor_to_list=new_tensor.tolist()
print('tensor:', new_tensor,"\nlist:",tensor_to_list)

tensor_index = torch.tensor([0, 1, 2, 3])
print("Check value at index 0:",tensor_index[0])
print("Check value at index 3:",tensor_index[3])

example_tensor = torch.tensor([50, 11, 22, 33, 44])
sclicing_tensor = example_tensor[1:4]
print("example tensor : ", example_tensor)
print("subset of example tensor:", sclicing_tensor)

print("value at index 3 of example tensor:", example_tensor[3])
example_tensor[3] = 0
print("new tensor:", example_tensor)

sample_tensor = torch.tensor([5, 4, 3, 2, 1])
min_value = sample_tensor.min()
max_value = sample_tensor.max()
print("check minimum value in the tensor: ", min_value)
print("check maximum value in the tensor: ", max_value)

mean_std_tensor = torch.tensor([-1.0, 2.0, 1, -2])
Mean = mean_std_tensor.mean()
print("mean of mean_std_tensor: ", Mean)
std_dev = mean_std_tensor.std()
print("standard deviation of mean_std_tensor: ", std_dev)

a = torch.tensor([1, 1])
b = torch.tensor([2, 2])
add = a + b
multiply = a * b
print("addition of two tensors: ", add)
print("multiplication of two tensors: ", multiply)
```

## Leitura Adicional

Desenvolvido ao mesmo tempo que o TensorFlow, o PyTorch costumava ter uma sintaxe mais simples até o TensorFlow adotar o Keras em sua versão 2.x. 

Para aprender o básico do PyTorch, você pode querer ler os tutoriais do PyTorch:

https://pytorch.org/tutorials/

Especialmente os conceitos básicos do tensor PyTorch podem ser encontrados na página do tutorial do Tensor:

https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

Há também alguns livros sobre PyTorch que são adequados para iniciantes. 

Um livro publicado mais recentemente deve ser recomendado, pois as ferramentas e a sintaxe estão evoluindo ativamente. 

Um exemplo é o livro "Aprendizado profundo com PyTorch" por Eli Stevens, Luca Antiga e Thomas Viehmann, 2020.





