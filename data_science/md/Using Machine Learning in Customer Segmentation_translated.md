[Link](https://machinelearningmastery.com/using-machine-learning-in-customer-segmentation/)

No passado, as empresas agrupavam os clientes com base em coisas simples como idade ou gênero. Agora, o aprendizado de máquina mudou esse processo. Algoritmos de aprendizado de máquina podem analisar grandes quantidades de dados. Neste artigo, exploraremos como o aprendizado de máquina melhora a segmentação de clientes.
## Introdução à Segmentação de Clientes

A segmentação de clientes divide os clientes em diferentes grupos. Esses grupos são baseados em características ou comportamentos semelhantes. O objetivo principal é entender melhor cada grupo. Isso ajuda as empresas a criar estratégias de marketing e produtos que se ajustem às necessidades específicas de cada grupo.

Os clientes podem ser divididos em grupos com base em vários critérios:

1. **Segmentação demográfica**: Com base em fatores como idade, sexo e ocupação.
2. **Segmentação psicográfica**: foco na fidelidade à marca e na frequência de uso.
4. **Segmentação geográfica**: divide os clientes com base em sua localização geográfica.

A segmentação de clientes oferece diversas vantagens para as empresas:

- **Marketing personalizado**: as empresas podem enviar mensagens específicas para cada grupo de clientes.
- **Melhoria na retenção de clientes**: As organizações podem identificar as preferências dos clientes e torná-los clientes fiéis.
- **Desenvolvimento de produtos aprimorado**: a segmentação ajuda a entender quais produtos os clientes desejam.

## Algoritmos de aprendizado de máquina para segmentação de clientes

O aprendizado de máquina usa vários algoritmos para categorizar clientes com base em suas características. Alguns algoritmos comumente usados incluem:

1. **Agrupamento K-means**: Divide os clientes em grupos com base em características semelhantes.
2. **Agrupamento hierárquico**: organiza os clientes em uma hierarquia de clusters em forma de árvore.
3. **DBSCAN**: Identifica clusters com base na densidade de pontos no espaço de dados.
4. **Análise de Componentes Principais (ACP)**: Reduz a dimensionalidade dos dados e preserva informações importantes.
5. **Árvores de decisão**: Divide os clientes com base em uma série de decisões hierárquicas.
6. **Redes Neurais**: Aprenda padrões complexos em dados por meio de camadas interconectadas de nós.

Usaremos o algoritmo K-means para segmentar clientes em vários grupos.

## Implementando o algoritmo de agrupamento K-means

O clustering K-means é um algoritmo não supervisionado. Ele opera sem rótulos predefinidos ou exemplos de treinamento. Este algoritmo é usado para agrupar pontos de dados semelhantes em um conjunto de dados. O objetivo é dividir os dados em clusters. Cada cluster contém pontos de dados semelhantes. Vamos ver como este algoritmo funciona.

1. **Inicialização**: Escolha o número de clusters (k). Inicialize k pontos aleatoriamente como centroides.
2. **Atribuição**: Atribua cada ponto de dados ao centróide mais próximo e forme os clusters.
3. **Atualizar Centroides**: Calcula a média de todos os pontos de dados atribuídos a cada centroide. Mova o centroide para esta posição média.

Repita os passos 2 e 3 até a convergência.

Nas seções a seguir, implementaremos o algoritmo de agrupamento K-means para agrupar clientes em clusters de acordo com diferentes características.

## Preparação de dados

Vamos explorar o conjunto de dados do cliente. Nosso conjunto de dados tem cerca de 5.00.000 pontos de dados.

![Conjunto de dados do cliente](https://www.kdnuggets.com/wp-content/uploads/Screenshot-84-1.png)

Conjunto de dados do cliente  

Os valores ausentes e duplicados são removidos e três recursos (&#39;Quantidade&#39;, &#39;Preço unitário&#39;, &#39;ID do cliente&#39;) são selecionados para agrupamento.

```Python
importar pandas como pd
de sklearn.preprocessing importar StandardScaler

# Carregue o conjunto de dados (substitua &#39;data.csv&#39; pelo caminho real do arquivo)
df = pd.read_csv(&#39;dados do usuário.csv&#39;)

# Limpeza de dados: remover duplicatas e manipular valores ausentes
df = df.drop_duplicates()
df = df.dropna()

# Seleção de recursos: seleção de recursos relevantes para agrupamento
selected_features = [&#39;Quantidade&#39;, &#39;Preço unitário&#39;, &#39;ID do cliente&#39;]
X = df[características_selecionadas]

# Normalização (padronização)
escalador = StandardScaler()
X_escalado = scaler.fit_transform(X)

```

## Ajuste de hiperparâmetros

Um desafio no agrupamento K-means é descobrir o número ótimo de clusters. O método do cotovelo nos ajuda a fazer isso. Ele plota a soma das distâncias quadradas de cada ponto ao seu centroide de cluster atribuído (inércia) contra K. T Procure o ponto onde a inércia não diminui mais significativamente com o aumento de K. Este ponto é chamado de cotovelo do modelo de agrupamento. Ele sugere um valor de K adequado.

```Píton
# Determine o número ideal de clusters usando o Método do Cotovelo
def determine_optimal_clusters(X_scaled, max_clusters=10):
    distâncias = []

    para n no intervalo(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n, estado_aleatório=42)
        kmeans.fit(X_escala)
        distâncias.append(kmeans.inertia_)

    plt.figure(tamanhodafigura=(7, 5))
    plt.plot(intervalo(2, max_clusters+1), distâncias, marcador=&#39;o&#39;)
    plt.title(&#39;Método Elbow&#39;)
    plt.xlabel(&#39;Número de clusters&#39;)
    plt.ylabel(&#39;Soma das distâncias ao quadrado&#39;)
    plt.xticks(intervalo(2, max_clusters+1))
    plt.grid(Verdadeiro)
    plt.mostrar()

    distâncias de retorno

distâncias = determine_optimal_clusters(X_scaled)
```

Podemos gerar um gráfico de inércia versus número de clusters usando o código acima.

![Método do cotovelo](https://www.kdnuggets.com/wp-content/uploads/Elbow-method-1.png)

Método do cotovelo  

Em K=1, a inércia está no máximo. De K=1 a K=5, a inércia diminui abruptamente. Entre K=5 a K=7, a curva diminui gradualmente. Finalmente, em K=7, ela se torna estável, então o valor ótimo de K é 7.

## Visualizando resultados de segmentação

Vamos implementar o algoritmo de agrupamento K-means e visualizar os resultados.

```Píton
# Aplicar agrupamento K-means com o número escolhido de clusters
grupos_escolhidos = 7  
kmeans = KMeans(n_clusters=clusters_escolhidos, estado_aleatório=42)
kmeans.fit(X_escala)

# Obter rótulos de cluster
cluster_labels = kmeans.labels_

# Adicione os rótulos do cluster ao dataframe original
df[&#39;Cluster&#39;] = rótulos_do_cluster

# Visualize os clusters em 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projeção=&#39;3d&#39;)

# Diagrama de dispersão para cada cluster
para cluster no intervalo (chosen_clusters):
    cluster_data = df[df[&#39;Cluster&#39;] == cluster]
    ax.scatter(cluster_data[&#39;Quantidade&#39;], cluster_data[&#39;PreçoUnitário&#39;], cluster_data[&#39;IDCliente&#39;],
               rótulo=f&#39;Cluster {cluster}&#39;, s=50)

ax.set_xlabel(&#39;Quantidade&#39;)
ax.set_ylabel(&#39;PreçoUnitário&#39;)
ax.set_zlabel(&#39;ID do cliente&#39;)

# Adicionar uma legenda
machado.legenda()
plt.mostrar()
```

![Gráfico de dispersão](https://www.kdnuggets.com/wp-content/uploads/Visualization-1.png)

O gráfico de dispersão 3D visualiza os clusters com base em &#39;Quantidade&#39;, &#39;PreçoUnitário&#39; e &#39;ID do Cliente&#39;. Cada cluster é diferenciado por cor e rotulado de acordo.

## Conclusão

Discutimos a segmentação de clientes usando machine learning e seus benefícios. Além disso, mostramos como implementar o algoritmo K-means para segmentar clientes em diferentes grupos. Primeiro, encontramos um número adequado de clusters usando o método elbow. Então, implementamos o algoritmo K-means e visualizamos os resultados usando um gráfico de dispersão. Por meio dessas etapas, as empresas podem segmentar clientes em grupos de forma eficiente.
