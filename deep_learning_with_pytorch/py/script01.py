# Importação das bibliotecas necessárias
import numpy as np  # Biblioteca para manipulação de arrays multidimensionais
import pandas as pd  # Biblioteca para manipulação de dados em formato tabular
import torch  # Biblioteca para computação em tensores (usada no aprendizado de máquina)

# Definindo uma lista de inteiros
lista_int = [10, 11, 12, 13]

# Convertendo a lista de inteiros para um tensor do PyTorch
int_to_tensor = torch.tensor(lista_int)

# Imprimindo o tipo do tensor
print(type(int_to_tensor))
# Imprimindo o tipo de dados do tensor (dtype)
print(int_to_tensor.dtype)
# Imprimindo o tipo do tensor (informação do tipo de dados do tensor)
print(int_to_tensor.type())
# Imprimindo as dimensões do tensor
print(int_to_tensor.size())
# Imprimindo o número de dimensões do tensor
print(int_to_tensor.ndimension())
print("\n")

# Convertendo a lista de inteiros para uma lista de floats
lista_float = [float(num) for num in lista_int]

# Convertendo a lista de floats para um tensor do PyTorch
float_to_tensor = torch.tensor(lista_float)

# Imprimindo informações sobre o novo tensor
print(type(float_to_tensor))
print(float_to_tensor.dtype)
print(float_to_tensor.type())
print(float_to_tensor.size())
print(float_to_tensor.ndimension())
print("\n")

# Usando o construtor específico para tensores com tipo float
int_list_to_float_tensor = torch.FloatTensor(lista_int)

# Imprimindo informações sobre o tensor criado
print(type(int_list_to_float_tensor))
print(int_list_to_float_tensor.dtype)
print(int_list_to_float_tensor.type())
print(int_list_to_float_tensor.size())
print(int_list_to_float_tensor.ndimension())
print("\n")

# Alterando a forma (reshape) do tensor, criando um tensor com 4 linhas e 1 coluna
reshaped_tensor = int_list_to_float_tensor.view(4, 1)

# Imprimindo o tensor com nova forma
print(reshaped_tensor)
print("\n")

# Alterando a forma (reshape) do tensor para 2 linhas e 2 colunas
reshaped_tensor = int_list_to_float_tensor.view(2, 2)

# Imprimindo o tensor com nova forma
print(reshaped_tensor)
print("\n")

# Convertendo uma lista de floats para um array numpy
numpy_arr = np.array(lista_float)
# Convertendo o array numpy para um tensor do PyTorch
from_numpy_to_tensor = torch.from_numpy(numpy_arr)

# Imprimindo informações sobre o tensor gerado a partir do numpy
print(type(from_numpy_to_tensor))
print(from_numpy_to_tensor.dtype)
print(from_numpy_to_tensor.type())
print(from_numpy_to_tensor.size())
print(from_numpy_to_tensor.ndimension())
print("\n")

# Convertendo o tensor de volta para um array numpy
tensor_to_numpy = from_numpy_to_tensor.numpy()

# Imprimindo o tipo e dtype do array numpy gerado a partir do tensor
print(type(tensor_to_numpy))
print(tensor_to_numpy.dtype)
print("\n")

# Definindo uma lista de números mistos (inteiros e floats)
lista = [1, 0.2, 3, 13.1]

# Criando um pandas Series a partir da lista
pandas_series = pd.Series(lista)
# Convertendo os valores do pandas Series para um tensor do PyTorch
store_with_numpy = torch.from_numpy(pandas_series.values)

# Imprimindo o tensor gerado e suas propriedades
print(store_with_numpy)
print(type(store_with_numpy))
print(store_with_numpy.dtype)
print(store_with_numpy.type())
print(store_with_numpy.size())
print(store_with_numpy.ndimension())
print("\n")

# Criando um novo tensor a partir da lista de inteiros
new_tensor = torch.tensor(lista_int)
# Acessando e imprimindo um valor específico do tensor usando indexação
print(new_tensor[1].item())
# Convertendo o tensor para uma lista
tensor_to_list = new_tensor.tolist()
# Imprimindo o tensor e a lista
print('tensor:', new_tensor,"\nlist:", tensor_to_list)
print("\n")

# Definindo uma lista de inteiros para indexação
lista2 = [0, 1, 2, 3]

# Criando um tensor a partir da lista
tensor_index = torch.tensor(lista2)
# Acessando e imprimindo elementos específicos do tensor
print(tensor_index[0])
print(tensor_index[3])
print("\n")

# Definindo uma lista de inteiros para demonstrar slicing
lista3 = [50, 11, 22, 33, 44]

# Criando um tensor a partir da lista
example_tensor = torch.tensor(lista3)

# Usando slicing para acessar uma parte do tensor
slicing_tensor = example_tensor[1:4]

# Imprimindo o tensor original e o tensor após o slicing
print(example_tensor)
print(slicing_tensor)
print("\n")

# Acessando e imprimindo um elemento específico do tensor
print(example_tensor[3])
print("\n")

# Alterando um valor específico do tensor
example_tensor[3] = 0

# Imprimindo o tensor após a modificação
print(example_tensor)
print("\n")

# Definindo uma lista para demonstrar funções de valor mínimo e máximo
lista4 = [5, 4, 3, 2, 1]

# Criando um tensor a partir da lista
sample_tensor = torch.tensor(lista4)

# Calculando o valor mínimo e máximo do tensor
min_value  = sample_tensor.min()
max_value = sample_tensor.max()

# Imprimindo os valores mínimo e máximo
print(min_value)
print(max_value)
print("\n")

# Definindo uma lista para calcular a média e o desvio padrão
lista5 = [-1.0, 2.0, 1, -2]

# Criando um tensor a partir da lista
mean_std_tensor = torch.tensor(lista5)

# Calculando a média e o desvio padrão do tensor
mean = mean_std_tensor.mean()
std_dev = mean_std_tensor.std()

# Imprimindo a média e o desvio padrão
print('mean:', mean, "std:", std_dev)
print("\n")

# Criando dois tensores para demonstrar operações aritméticas
a = torch.tensor([1, 1])
b = torch.tensor([2, 2])

# Realizando operações de adição e multiplicação
add = a + b
multiply = a * b

# Imprimindo os resultados das operações
print(a)
print(b)
print('add:', add, "multiply:", multiply)
print("\n")
