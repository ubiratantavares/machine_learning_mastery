# Two-Dimensional Tensors in Pytorch

[Referência](https://machinelearningmastery.com/two-dimensional-tensors-in-pytorch/)

# Tensores bidimensionais em Pytorch

Os tensores bidimensionais são análogos às matrizes bidimensionais. Como uma matriz bidimensional, um tensor bidimensional também tem
número de linhas e colunas.

Vamos pegar uma imagem em escala de cinza como exemplo, que é uma matriz bidimensional de valores numéricos, comumente conhecidos como pixels. 

Variando de '0' a '255', cada número representa um valor de intensidade de pixel. 

Aqui, o número de menor intensidade (que é '0') representa regiões pretas na imagem, enquanto o número de maior intensidade (que é '255') representa regiões brancas na imagem. 

Usando a estrutura PyTorch, essa imagem ou matriz bidimensional pode ser convertida em um tensor bidimensional.

No tutorial anterior, aprendemos sobre tensores unidimensionais no PyTorch ([One-Dimensional Tensors in Pytorch](https://github.com/ubiratantavares/machine_learning_mastery/blob/main/deep_learning_with_pytorch/md/two_dimensional_tensors_in_pytorch.md) - script01.py
) e aplicamos algumas operações de tensores úteis. 

Neste tutorial, aplicaremos essas operações a tensores bidimensionais usando a biblioteca PyTorch. 

Especificamente, aprenderemos:

* Como criar tensores bidimensionais no PyTorch e explorar seus tipos e formas.

* Sobre operações de fatiamento e indexação em tensores bidimensionais em detalhes.

* Aplicar vários métodos a tensores, como adição de tensores, multiplicação e muito mais.

## Tipos e formas de tensores bidimensionais

Vamos primeiro importar algumas bibliotecas necessárias que usaremos neste tutorial.

```Python
import numpy as np 
import pandas as pd
import torch
```

Para verificar os tipos e formas dos tensores bidimensionais, usaremos os mesmos métodos do PyTorch, introduzidos anteriormente para tensores unidimensionais. 

Mas, deveria funcionar da mesma forma que funcionou para os tensores unidimensionais?

Vamos demonstrar convertendo uma lista 2D de inteiros para um objeto tensor 2D. Como exemplo, criaremos uma lista 2D e aplicaremos **torch.tensor()** para conversão.

```Python
example_2d_list = [[5, 10, 15, 20],
                   [25, 30, 35, 40],
                   [45, 50, 55, 00]]

list_to_tensor = torch.tensor(example_2d_list)

print(list_to_tensor)
```

Como você pode ver, o método **torch.tensor()** também funciona bem para tensores bidimensionais. 

Agora, vamos usar os métodos **shape()**, **size()**, e **ndimension()** para retornar a forma, o tamanho e as dimensões de um objeto tensor.

```Python
print(list_to_tensor.shape)
print(list_to_tensor.size())
print(list_to_tensor.ndimension())
print(list_to_tensor.type())
print("\n")
```

## Convertendo tensores bidimensionais em matrizes NumPy

PyTorch nos permite converter um tensor bidimensional para um array NumPy e então de volta para um tensor.

```Python
twoD_tensor_to_numpy = list_to_tensor.numpy()

print(twoD_tensor_to_numpy)
print(twoD_tensor_to_numpy.dtype)
print("\n")

back_to_tensor = torch.from_numpy(twoD_tensor_to_numpy)

print(back_to_tensor)
print(back_to_tensor.dtype)
print("\n")
```

## Convertendo séries de pandas em tensores bidimensionais

Da mesma forma, também podemos converter um DataFrame do pandas em um tensor. 

Assim como os tensores unidimensionais, usaremos os mesmos passos para a conversão. 

Usando o atributo values, obteremos o array NumPy e, em seguida, usaremos **torch.from_numpy** isso para converter um DataFrame do pandas em um tensor.

```Python
# Converting Pandas Dataframe to a Tensor
dataframe = pd.DataFrame({'x':[22,24,26],'y':[42,52,62]})

print(dataframe.values)
print(dataframe.values.dtype)
print("\n")

pandas_to_tensor = torch.from_numpy(dataframe.values)

print(pandas_to_tensor)
print(pandas_to_tensor.dtype)
print("\n")
```

## Operações de indexação e fatiamento em tensores bidimensionais

Para operações de indexação, diferentes elementos em um objeto tensor podem ser acessados​usando colchetes. 

Você pode simplesmente colocar índices correspondentes em colchetes para acessar os elementos desejados em um tensor.

No exemplo abaixo, criaremos um tensor e acessaremos certos elementos usando dois métodos diferentes. 

Observe que o valor do índice deve ser sempre um a menos do que onde o elemento está localizado em um tensor bidimensional.

```Python
# Indexing and Slicing Operations on Two-Dimensional Tensors
example_tensor = torch.tensor([[10, 20, 30, 40],
                               [50, 60, 70, 80],
                               [90, 100, 110, 120]])
                               
print(example_tensor[1,1])
print(example_tensor[1][1])
print("\n")

print(example_tensor[2,3])
print(example_tensor[2][3])
```

E se precisarmos acessar dois ou mais elementos ao mesmo tempo? 

É aí que o fatiamento tensor entra em cena. 

Vamos usar o exemplo anterior para acessar os dois primeiros elementos da segunda linha e os três primeiros elementos da terceira linha.

```Python
print(example_tensor[1, 0:2])
print(example_tensor[1][0:2])
print("\n")

print(example_tensor[2, 0:3])
print(example_tensor[2][0:3])
print("\n")
```

## Operações em tensores bidimensionais

Embora existam muitas operações que você pode aplicar em tensores bidimensionais usando a estrutura PyTorch, aqui apresentaremos a adição de tensores e a multiplicação de escalares e matrizes.

### Adicionando tensores bidimensionais

Adicionar dois tensores é semelhante à adição de matrizes. 

É um processo bem direto, pois você só precisa de um operador de adição (+) para executar a operação. 

Vamos adicionar dois tensores no exemplo abaixo.

```Python
# Adding Two-Dimensional Tensors
A = torch.tensor([[5, 10],
                  [50, 60], 
                  [100, 200]]) 
B = torch.tensor([[10, 20], 
                  [60, 70], 
                  [200, 300]])
add = A + B
print(add)
print("\n")
```


### Multiplicação escalar e matricial de tensores bidimensionais

A multiplicação escalar em tensores bidimensionais também é idêntica à multiplicação escalar em matrizes. 

Por exemplo, ao multiplicar um tensor por um escalar, digamos um escalar 4, você estará multiplicando cada elemento em um tensor por 4.

```Python
# Scalar Multiplication of Two-Dimensional Tensors
new_tensor = torch.tensor([[1, 2, 3], 
                           [4, 5, 6]]) 
mul_scalar = 4 * new_tensor

print(mul_scalar)
print("\n")
```

Chegando à multiplicação dos tensores bidimensionais, **torch.mm()** no PyTorch torna as coisas mais fáceis para nós. 

Semelhante à multiplicação de matrizes na álgebra linear, o número de colunas no objeto tensor A (ou seja, 2×3) deve ser igual ao número de linhas no objeto tensor B (ou seja, 3×2).

```Python
# Matrix Multiplication of Two-Dimensional Tensors
A = torch.tensor([[3, 2, 1], 
                  [1, 2, 1]])
B = torch.tensor([[3, 2], 
                  [1, 1], 
                  [2, 1]])
A_mult_B = torch.mm(A, B)

print("multiplying A with B: ", A_mult_B)
print("\n")
```

## Leitura Adicional

Desenvolvido ao mesmo tempo que o TensorFlow, o PyTorch costumava ter uma sintaxe mais simples até o TensorFlow adotar o Keras em sua versão 2.x. 

Para aprender o básico do PyTorch, você pode querer ler os tutoriais do PyTorch:

https://pytorch.org/tutorials/

Especialmente os conceitos básicos do tensor PyTorch podem ser encontrados na página do tutorial do Tensor:

https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

Há também alguns livros sobre PyTorch que são adequados para iniciantes. 

Um livro publicado mais recentemente deve ser recomendado, pois as ferramentas e a sintaxe estão evoluindo ativamente. 

Um exemplo é o livro "Deep Learning with PyTorch", Eli Stevens, Luca Antiga e  Thomas Viehmann, 2020.
