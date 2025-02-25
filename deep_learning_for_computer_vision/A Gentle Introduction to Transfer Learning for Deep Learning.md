# Uma introdução suave à aprendizagem de transferência para aprendizagem profunda

A aprendizagem por transferência é um método de aprendizado de máquina em que um modelo desenvolvido para uma tarefa é reutilizado como ponto de partida para um modelo em uma segunda tarefa.
É uma abordagem popular em aprendizado profundo, na qual modelos pré-treinados são usados ​​como ponto de partida em tarefas de visão computacional e processamento de linguagem natural, dados os vastos recursos de computação e tempo necessários para desenvolver modelos de redes neurais nesses problemas e os enormes avanços em habilidade que eles proporcionam em problemas relacionados.
## O que é aprendizagem por transferência?

A aprendizagem por transferência é uma técnica de aprendizado de máquina em que um modelo treinado em uma tarefa é reutilizado em uma segunda tarefa relacionada.

> A aprendizagem de transferência e a adaptação de domínio referem-se à situação em que o que foi aprendido num cenário… é explorado para melhorar a generalização noutro cenário

— Página 526, [Deep Learning](https://12ft.io/proxy?q=http%3A%2F%2Famzn.to%2F2fwdoKR) , 2016.

A aprendizagem por transferência é uma otimização que permite um progresso rápido ou um melhor desempenho ao modelar a segunda tarefa.

> A aprendizagem por transferência é a melhoria da aprendizagem em uma nova tarefa por meio da transferência de conhecimento de uma tarefa relacionada que já foi aprendida.

— [Capítulo 11: Transferência de Aprendizagem](https://12ft.io/proxy?q=ftp%3A%2F%2Fftp.cs.wisc.edu%2Fmachine-learning%2Fshavlik-group%2Ftorrey.handbook09.pdf) , [Manual de Pesquisa em Aplicações de Aprendizado de Máquina](https://12ft.io/proxy?q=http%3A%2F%2Famzn.to%2F2fgeVro) , 2009.

A aprendizagem por transferência está relacionada a problemas como aprendizagem multitarefa e desvio de conceitos e não é uma área de estudo exclusiva para aprendizagem profunda.

No entanto, a aprendizagem por transferência é popular na aprendizagem profunda, dados os enormes recursos necessários para treinar modelos de aprendizagem profunda ou os grandes e desafiadores conjuntos de dados nos quais os modelos de aprendizagem profunda são treinados.

A aprendizagem por transferência só funciona no aprendizado profundo se os recursos do modelo aprendidos na primeira tarefa forem gerais.

> Na aprendizagem por transferência, primeiro treinamos uma rede base em um conjunto de dados e tarefa base, e então reaproveitamos os recursos aprendidos, ou os transferimos, para uma segunda rede alvo a ser treinada em um conjunto de dados e tarefa alvo. Esse processo tenderá a funcionar se os recursos forem gerais, ou seja, adequados para tarefas base e alvo, em vez de específicos para a tarefa base.

— [Quão transferíveis são os recursos em redes neurais profundas?](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1411.1792)

Essa forma de transferência de aprendizado usada em aprendizado profundo é chamada de transferência indutiva. É onde o escopo de modelos possíveis (viés do modelo) é estreitado de forma benéfica usando um ajuste de modelo em uma tarefa diferente, mas relacionada.

## Como usar o aprendizado por transferência?

Você pode usar a aprendizagem por transferência em seus próprios problemas [de modelagem preditiva](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fgentle-introduction-to-predictive-modeling%2F) .

Duas abordagens comuns são as seguintes:

1. Desenvolver abordagem de modelo
2. Abordagem de modelo pré-treinado

### Desenvolver abordagem de modelo

1. **Selecione Tarefa de Origem** . Você deve selecionar um problema de modelagem preditiva relacionado com uma abundância de dados onde haja alguma relação nos dados de entrada, dados de saída e/ou conceitos aprendidos durante o mapeamento dos dados de entrada para os dados de saída.
2. **Desenvolver modelo de origem** . Em seguida, você deve desenvolver um modelo habilidoso para esta primeira tarefa. O modelo deve ser melhor do que um modelo ingênuo para garantir que algum aprendizado de recurso tenha sido realizado.
3. **Reutilizar modelo** . O ajuste do modelo na tarefa de origem pode então ser usado como ponto de partida para um modelo na segunda tarefa de interesse. Isso pode envolver o uso de todo ou partes do modelo, dependendo da técnica de modelagem usada.
4. **Ajustar modelo** . Opcionalmente, o modelo pode precisar ser adaptado ou refinado nos dados do par de entrada-saída disponíveis para a tarefa de interesse.

### Abordagem de modelo pré-treinado

1. **Selecione o modelo de origem** . Um modelo de origem pré-treinado é escolhido entre os modelos disponíveis. Muitas instituições de pesquisa liberam modelos em conjuntos de dados grandes e desafiadores que podem ser incluídos no conjunto de modelos candidatos para escolher.
2. **Reutilizar modelo** . O modelo pré-treinado pode então ser usado como ponto de partida para um modelo na segunda tarefa de interesse. Isso pode envolver o uso de todo ou partes do modelo, dependendo da técnica de modelagem usada.
3. **Ajustar modelo** . Opcionalmente, o modelo pode precisar ser adaptado ou refinado nos dados do par de entrada-saída disponíveis para a tarefa de interesse.

Esse segundo tipo de transferência de aprendizagem é comum no campo do aprendizado profundo.

## Exemplos de Aprendizagem de Transferência com Aprendizagem Profunda

Vamos concretizar isso com dois exemplos comuns de aprendizagem por transferência com modelos de aprendizagem profunda.

### Transferência de Aprendizagem com Dados de Imagem

É comum realizar transferência de aprendizado com problemas de modelagem preditiva que usam dados de imagem como entrada.

Esta pode ser uma tarefa de previsão que usa fotografias ou dados de vídeo como entrada.

Para esses tipos de problemas, é comum usar um modelo de aprendizado profundo pré-treinado para uma tarefa grande e desafiadora de classificação de imagens, como a competição de classificação de fotografias de 1.000 classes [do ImageNet .](https://12ft.io/proxy?q=http%3A%2F%2Fwww.image-net.org%2F)

As organizações de pesquisa que desenvolvem modelos para esta competição e se saem bem frequentemente liberam seu modelo final sob uma licença permissiva para reutilização. Esses modelos podem levar dias ou semanas para treinar em hardware moderno.

Esses modelos podem ser baixados e incorporados diretamente em novos modelos que esperam dados de imagem como entrada.

Três exemplos de modelos deste tipo incluem:

- [Modelo Oxford VGG](https://12ft.io/proxy?q=http%3A%2F%2Fwww.robots.ox.ac.uk%2F~vgg%2Fresearch%2Fvery_deep%2F)
- [Modelo de criação do Google](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2Ftensorflow%2Fmodels%2Ftree%2Fmaster%2Fresearch%2Finception)
- [Modelo Microsoft ResNet](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2FKaimingHe%2Fdeep-residual-networks)

Para mais exemplos, veja o [Caffe Model Zoo](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2FBVLC%2Fcaffe%2Fwiki%2FModel-Zoo) , onde mais modelos pré-treinados são compartilhados.

Essa abordagem é eficaz porque as imagens foram treinadas em um grande corpus de fotografias e exigem que o modelo faça previsões em um número relativamente grande de classes, exigindo, por sua vez, que o modelo aprenda a extrair características de fotografias de forma eficiente para ter um bom desempenho no problema.

Em seu curso de Stanford sobre Redes Neurais Convolucionais para Reconhecimento Visual, os autores alertam para escolher cuidadosamente quanto do modelo pré-treinado usar em seu novo modelo.

> [Redes Neurais Convolucionais] Os recursos são mais genéricos nas camadas iniciais e mais específicos do conjunto de dados original nas camadas posteriores

— Aprendizagem de Transferência, [CS231n Redes Neurais Convolucionais para Reconhecimento Visual](https://12ft.io/proxy?q=http%3A%2F%2Fcs231n.github.io%2Ftransfer-learning%2F)

### Transferência de Aprendizagem com Dados de Linguagem

É comum realizar transferência de aprendizagem com problemas de processamento de linguagem natural que usam texto como entrada ou saída.

Para esses tipos de problemas, é usada uma incorporação de palavras, que é um mapeamento de palavras para um espaço vetorial contínuo de alta dimensão, onde palavras diferentes com um significado semelhante têm uma representação vetorial semelhante.

Existem algoritmos eficientes para aprender essas representações de palavras distribuídas e é comum que organizações de pesquisa liberem modelos pré-treinados em grandes conjuntos de documentos de texto sob uma licença permissiva.

Dois exemplos de modelos deste tipo incluem:

- [Modelo word2vec do Google](https://12ft.io/proxy?q=https%3A%2F%2Fcode.google.com%2Farchive%2Fp%2Fword2vec%2F)
- [Modelo GloVe de Stanford](https://12ft.io/proxy?q=https%3A%2F%2Fnlp.stanford.edu%2Fprojects%2Fglove%2F)

Esses modelos de representação de palavras distribuídas podem ser baixados e incorporados em modelos de linguagem de aprendizado profundo, tanto na interpretação de palavras como entrada quanto na geração de palavras como saída do modelo.

Em seu livro sobre Aprendizado Profundo para Processamento de Linguagem Natural, Yoav Goldberg adverte:

> … pode-se baixar vetores de palavras pré-treinados que foram treinados em grandes quantidades de texto […] diferenças nos regimes de treinamento e corpora subjacentes têm uma forte influência nas representações resultantes, e que as representações pré-treinadas disponíveis podem não ser a melhor escolha para [seu] caso de uso específico.

— Página 135, [Métodos de Redes Neurais em Processamento de Linguagem Natural](https://12ft.io/proxy?q=http%3A%2F%2Famzn.to%2F2fwTPCn) , 2017.

## Quando usar a aprendizagem por transferência?

A aprendizagem por transferência é uma otimização, um atalho para economizar tempo ou obter melhor desempenho.

Em geral, não é óbvio que haverá algum benefício em usar a aprendizagem por transferência no domínio até que o modelo tenha sido desenvolvido e avaliado.

Lisa Torrey e Jude Shavlik, em [seu capítulo sobre aprendizagem por transferência,](https://12ft.io/proxy?q=http%3A%2F%2Famzn.to%2F2fgeVro) descrevem três possíveis benefícios a serem procurados ao usar a aprendizagem por transferência:

1. **Início mais alto** . A habilidade inicial (antes de refinar o modelo) no modelo de origem é mais alta do que seria de outra forma.
2. **Inclinação mais alta** . A taxa de melhoria de habilidade durante o treinamento do modelo de origem é mais íngreme do que seria de outra forma.
3. **Assíntota mais alta** . A habilidade convergente do modelo treinado é melhor do que seria de outra forma.

O ideal seria que você visse todos os três benefícios de uma aplicação bem-sucedida da aprendizagem por transferência.

É uma abordagem a ser tentada se você puder identificar uma tarefa relacionada com dados abundantes e tiver recursos para desenvolver um modelo para essa tarefa e reutilizá-lo em seu próprio problema, ou se houver um modelo pré-treinado disponível que você pode usar como ponto de partida para seu próprio modelo.

Em alguns problemas nos quais você pode não ter muitos dados, a aprendizagem por transferência pode permitir que você desenvolva modelos habilidosos que você simplesmente não conseguiria desenvolver na ausência da aprendizagem por transferência.

A escolha dos dados de origem ou do modelo de origem é um problema em aberto e pode exigir conhecimento de domínio e/ou intuição desenvolvida por meio da experiência.

