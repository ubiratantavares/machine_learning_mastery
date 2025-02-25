[Link](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)

Chegamos a um ponto em que implementamos e testamos o Transformer [codificador](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras) e [decodificador](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras) separadamente, e agora podemos juntar os dois em um modelo completo. Também veremos como criar máscaras de preenchimento e look-ahead pelas quais suprimiremos os valores de entrada que não serão considerados nos cálculos do codificador ou decodificador. Nosso objetivo final continua sendo aplicar o modelo completo ao Processamento de Linguagem Natural (PLN).

Neste tutorial, você descobrirá como implementar o modelo completo do Transformer e criar máscaras de preenchimento e de previsão.

Depois de concluir este tutorial, você saberá:

- Como criar uma máscara de preenchimento para o codificador e decodificador
- Como criar uma máscara de lookahead para o decodificador
- Como unir o codificador e o decodificador do Transformer em um único modelo
- Como imprimir um resumo das camadas do codificador e do decodificador

## **Visão geral do tutorial**

Este tutorial é dividido em quatro partes; são elas:

- Recapitulação da Arquitetura do Transformador
- Mascaramento
    - Criando uma máscara de preenchimento
    - Criando uma máscara de look-ahead
- Juntando o codificador e o decodificador do transformador
- Criando uma instância do modelo Transformer
    - Imprimindo um Resumo das Camadas do Codificador e do Decodificador

## **Pré-requisitos**

Para este tutorial, presumimos que você já esteja familiarizado com:

- [O modelo do transformador](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F)
- [O codificador Transformer](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementando-o-codificador-transformer-do-zero-em-tensorflow-e-keras)
- [O decodificador Transformer](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementando-o-decodificador-transformer-do-zero-em-tensorflow-e-keras)

## **Recapitulação da arquitetura do transformador**

[Recall](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fthe-transformer-model%2F) tendo visto que a arquitetura do Transformer segue uma estrutura codificador-decodificador. O codificador, no lado esquerdo, é encarregado de mapear uma sequência de entrada para uma sequência de representações contínuas; o decodificador, no lado direito, recebe a saída do codificador junto com a saída do decodificador no passo de tempo anterior para gerar uma sequência de saída.

[![alt](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-727x1024.png)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwp-content%2Fuploads%2F2021%2F08%2Fattention_research_1.png)

A estrutura codificador-decodificador da arquitetura Transformer  
Retirado de “[Atenção é tudo o que você precisa](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762)“

Ao gerar uma sequência de saída, o Transformer não depende de recorrência e convoluções.

Você viu como implementar o codificador e o decodificador do Transformer separadamente. Neste tutorial, você unirá os dois em um modelo Transformer completo e aplicará padding e look-ahead masking aos valores de entrada.

Vamos começar descobrindo como aplicar a máscara.

**Dê o pontapé inicial no seu projeto** com meu livro [Building Transformer Models with Attention](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftransformer-models-with-attention%2F). Ele fornece **tutoriais de autoestudo** com **código funcional** para orientá-lo na construção de um modelo de transformador totalmente funcional que pode  
_traduzir frases de um idioma para outro_...

## **Mascaramento**

### **Criando uma máscara de preenchimento**

Você já deve estar familiarizado com a importância de mascarar os valores de entrada antes de alimentá-los no codificador e no decodificador.

Como você verá quando prosseguir para [treinar o modelo Transformer](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftraining-the-transformer-model), as sequências de entrada alimentadas no codificador e decodificador serão primeiro preenchidas com zeros até um comprimento de sequência específico. A importância de ter uma máscara de preenchimento é garantir que esses valores zero não sejam processados junto com os valores de entrada reais pelo codificador e decodificador.

Vamos criar a seguinte função para gerar uma máscara de preenchimento para o codificador e o decodificador:

**DIGITE O CÓDIGO**

Ao receber uma entrada, esta função irá gerar um tensor que marca pelo valor _um_ sempre que a entrada contiver um valor _zero_.

Portanto, se você inserir a seguinte matriz:

**DIGITE O CÓDIGO**


Então a saída da função `padding_mask` seria a seguinte:

**DIGITE O CÓDIGO**


### **Criando uma máscara de previsão**

Uma máscara de previsão é necessária para evitar que o decodificador atenda às palavras subsequentes, de modo que a previsão de uma palavra específica possa depender apenas de saídas conhecidas para as palavras que vêm antes dela.

Para isso, vamos criar a seguinte função para gerar uma máscara de previsão para o decodificador:

**DIGITE O CÓDIGO**



Você passará para ele o comprimento da entrada do decodificador. Vamos fazer esse comprimento igual a 5, como exemplo:

**DIGITE O CÓDIGO**


Então a saída que a função `lookahead_mask` retorna é a seguinte:

**DIGITE O CÓDIGO**


Novamente, os valores _one_ mascaram as entradas que não devem ser usadas. Dessa maneira, a previsão de cada palavra depende somente daquelas que vêm antes dela.  

## **Juntando o codificador e o decodificador do transformador**

Vamos começar criando a classe `TransformerModel`, que herda da classe base `Model` no Keras:

**DIGITE O CÓDIGO**


Nosso primeiro passo na criação da classe `TransformerModel` é inicializar instâncias das classes `Encoder` e `Decoder` implementadas anteriormente e atribuir suas saídas às variáveis, `encoder` e `decoder`, respectivamente. Se você salvou essas classes em scripts Python separados, não se esqueça de importá-las. Salvei meu código nos scripts Python _encoder.py_ e _decoder.py_, então preciso importá-los adequadamente.

Você também incluirá uma camada densa final que produz a saída final, como na arquitetura Transformer de [Vaswani et al. (2017)](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762).

Em seguida, você deve criar o método de classe `call()`, para alimentar as entradas relevantes no codificador e no decodificador.

Uma máscara de preenchimento é gerada primeiro para mascarar a entrada do codificador, bem como a saída do codificador, quando esta é alimentada no segundo bloco de autoatenção do decodificador:

**DIGITE O CÓDIGO**



Uma máscara de preenchimento e uma máscara de look-ahead são então geradas para mascarar a entrada do decodificador. Elas são combinadas por meio de uma operação `máxima` elemento a elemento:

**DIGITE O CÓDIGO**


Em seguida, as entradas relevantes são alimentadas no codificador e no decodificador, e a saída do modelo do transformador é gerada alimentando a saída do decodificador em uma camada densa final:

**DIGITE O CÓDIGO**


A combinação de todos os passos nos dá a seguinte listagem de código completa:

**DIGITE O CÓDIGO**


Note que você realizou uma pequena alteração na saída que é retornada pela função `padding_mask`. Sua forma é tornada transmissível para a forma do tensor de peso de atenção que ele mascarará quando você treinar o modelo Transformer.

## **Criando uma instância do modelo Transformer**

Você trabalhará com os valores de parâmetros especificados no artigo [Attention Is All You Need](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), de Vaswani et al. (2017):

**DIGITE O CÓDIGO**



Quanto aos parâmetros relacionados à entrada, você trabalhará com valores fictícios por enquanto até chegar ao estágio de [treinar o modelo Transformer completo](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftraining-the-transformer-model). Nesse ponto, você usará frases reais:

**DIGITE O CÓDIGO**



Agora você pode criar uma instância da classe `TransformerModel` da seguinte maneira:

**DIGITE O CÓDIGO**


A listagem completa do código é a seguinte:

**DIGITE O CÓDIGO**


### **Imprimindo um resumo das camadas do codificador e do decodificador**

Você também pode imprimir um resumo dos blocos codificadores e decodificadores do modelo Transformer. A escolha de imprimi-los separadamente permitirá que você veja os detalhes de suas subcamadas individuais. Para fazer isso, adicione a seguinte linha de código ao método `__init__()` das classes `EncoderLayer` e `DecoderLayer`:

**DIGITE O CÓDIGO**


Então você precisa adicionar o seguinte método à classe `EncoderLayer`:

**DIGITE O CÓDIGO**



E o seguinte método para a classe `DecoderLayer`:

**DIGITE O CÓDIGO**



Isso resulta na classe `EncoderLayer` sendo modificada da seguinte forma (os três pontos sob o método `call()` significam que ele permanece o mesmo que foi implementado [aqui](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fimplementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)):

**DIGITE O CÓDIGO**


Alterações semelhantes também podem ser feitas na classe `DecoderLayer`.

Depois de fazer as alterações necessárias, você pode prosseguir para criar instâncias das classes `EncoderLayer` e `DecoderLayer` e imprimir seus resumos da seguinte maneira:

**DIGITE O CÓDIGO**


O resumo resultante para o codificador é o seguinte:

**DIGITE O CÓDIGO**


Enquanto o resumo resultante para o decodificador é o seguinte:

**DIGITE O CÓDIGO**

## **Leitura adicional**

Esta seção fornece mais recursos sobre o tópico caso você queira se aprofundar mais.

### **Livros**

- [Aprendizado profundo avançado com Python](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FAdvanced-Deep-Learning-Python-next-generation%2Fdp%2F178995617X), 2019
- [Transformadores para processamento de linguagem natural](https://12ft.io/proxy?q=https%3A%2F%2Fwww.amazon.com%2FTransformers-Natural-Language-Processing-architectures%2Fdp%2F1800565798), 2021

### **Documentos**

- [Atenção é tudo o que você precisa](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762), 2017

## **Resumo**

Neste tutorial, você descobriu como implementar o modelo completo do Transformer e criar máscaras de preenchimento e de previsão.

Especificamente, você aprendeu:

- Como criar uma máscara de preenchimento para o codificador e decodificador
- Como criar uma máscara de lookahead para o decodificador
- Como unir o codificador e o decodificador do Transformer em um único modelo
- Como imprimir um resumo das camadas do codificador e do decodificador

Você tem alguma dúvida?  
Faça suas perguntas nos comentários abaixo e farei o melhor para responder.