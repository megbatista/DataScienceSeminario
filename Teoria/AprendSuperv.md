# Aprendizado Supervisionado

## K-Nearest Neighbors e Linear Models

### I. Conceito

   K-Nearest Neighbors (KNN), é um dos algoritmos de machine learning mais fáceis de se aprender, pode ser usado para classificação e regressão, sendo a técnica mais usada em problemas de classificação. É utilizado em áreas como:
  * Microbiologia - para classificação de células
  * Marketing - para segmentação de clientes
  * Reconhecimento Facial
  * Mineração de texto
  * Sistemas de recomendação
  
  A classificação é feita a partir do cálculo da distância dos vizinhos mais próximos de cada ponto: a um ponto de consulta é atribuída a classe de dados que tem mais representantes dentro dos vizinhos mais próximos do ponto.
  Observando o gráfico abaixo, em que o eixo y representa a altura e o eixo x, a idade, identifica-se 3 tipos de grupos de pessoas, em verde, laranja e azul. O ponto preto é o ponto a ser classificado, intuitivamente pode-se inferir que o esse ponto pertence ao grupo laranja, pois está mais próximo desse conjunto, portanto, as demais características desse dado devem ser semelhantes aos destes outros pontos.
  
![image](https://user-images.githubusercontent.com/77736052/155669688-fee42120-6475-4ae5-925c-a25759c2e0cc.png)
![image](https://user-images.githubusercontent.com/77736052/155669627-9e8ea38d-9758-4bfd-ba3d-7cf452434b33.png)

   O algoritmo prevê, por meio de comparação entre as instâncias, valores de quaisquer novos pontos de dados. O novo ponto recebe um valor baseado em quão próximo ele se parece dos pontos no conjunto de treinamento. 
   O KNN utiliza aprendizagem baseada em instâncias, ou seja, não existe um modelo, regra ou função construído a partir de uma etapa de treinamento. Esse método, armazena todos os dados de treinamento, e a cada nova instância que se queira classificar, são realizados cálculos entre esta instância e os dados armazenados anteriormente.

### II. Classes de Problemas com melhores resultados
   * Quando se tem dados devidamente rotulados. Por exemplo, se desejamos prever se alguém tem diabetes ou não, o rótulo final pode ser 1 ou 0. Não pode ser NaN ou -1.
   * Problemas com dados são livres de ruído. Para o conjunto de dados de diabetes não se pode ter um nível de glicose como 0 ou 10.000.
   * Conjunto de dados pequeno.

### III. Definição Teórica e Modelagem Matemática
   A abordagem de KNN é relativamente simples e completamente não paramétrica. Dado um ponto x0 que desejamos classificar em um dos K grupos, encontramos os k pontos de dados observados mais próximos de x0. A regra de classificação é atribuir x0 à população que possui os pontos de dados mais observados dos k vizinhos mais próximos. Os pontos para os quais não há maioria são classificados aleatoriamente para uma das populações majoritárias ou deixados sem classificação.
   
Os métodos mais utilizados para o cálculo da distância entre o novo ponto e o ponto de treinamento são:
   -  Euclidiano
      A distância é calculada como a raiz quadrada da soma das diferenças quadráticas entre um novo ponto (x) e um ponto existente (y).
      Se p e q são pontos de R^n, a distância euclidiana de p a q é o número
      
      ![image](https://user-images.githubusercontent.com/77736052/155674750-193b977a-295b-4ee9-84e3-1e7fab85624a.png)
      
      ![image](https://user-images.githubusercontent.com/77736052/155674916-ab91910b-a3fd-4bf5-bf0d-c4718bd7a7b6.png)

   - Manhattan (para contínuo) 
       Distância entre vetores reais usando a soma de sua diferença absoluta.
       
      ![image](https://user-images.githubusercontent.com/77736052/155675642-31d1a5f1-ed0f-48af-8824-32c30529d1fe.png)

   - Distância Hamming
      É usada quando se tem variáveis categóricas. Se o valor (x) e o valor (y) forem iguais, a distância D será igual a 0. Caso contrário, D = 1.
         ![image](https://user-images.githubusercontent.com/77736052/155675824-7cacb7ea-1c76-4723-af5e-5b71dd598fe1.png)

### IV. Vantagens e Desvantagens
   * Vantagens
      - Simples, fácil de implementar.
      - Não há necessidade de construir um modelo, ajustar vários parâmetros ou fazer suposições adicionais
      - Possui um bom desempenho quando os dados apresentam relacionamentos entre características complexos.
      - Pode ser usado para classificação, regressão e pesquisa.
   * Desvantagens
      - A principal desvantagem do KNN é se tornar significativamente mais lento à medida que o volume de dados aumenta, o que o torna impraticável em cenários onde as previsões precisam ser feitas rapidamente. Além disso, existem algoritmos mais rápidos que podem produzir resultados de classificação e regressão mais precisos.
      - Definir um valor para K é normalmente uma tarefa manual, experimental e não trivial.
      - Como a métrica utilizada pelo, é o cálculo da distância entre pontos, exige uma verificação antes dos dados serem processados, pois as variáveis categóricas precisam ser tansformadas.

### V. Exemplo de uma aplicação em Python

   [Classificação de pontos](https://github.com/megbatista/DataScienceSeminario/blob/75d76d340fa4c2a6245c91dc33d039368a1dd0c1/Aplica%C3%A7%C3%B5esPython/KNN_plot_classification.ipynb)

## Naive Bayes Classifiers

### I. Conceito

O classificador Naive Bayes é um simples algoritmo de classificação realiza predições em aprendizagem de máquina. O termo “naive” (ingênuo) diz respeito à forma como o algoritmo analisa as características de uma base de dados, desconsiderando as correlações entre as features. 

O teorema de Bayes trata sobre probabilidade condicional, que é a probabilidade de o evento A ocorrer, dado o evento B. Ele também assume que as variáveis features são igualmente importantes para o resultado.

O funcionamento do algoritmo pode ser facilmente descrito em termos estatísticos: para calcular a predição, define-se primeiramente, uma tabela de probabilidades, em que consta a frequência dos preditores com relação às variáveis de saída. Assim, o resultado é calculado a partir da maior probabilidade para oferecer uma solução.



### II. Classes de Problemas com melhores resultados

Naive Bayes deve ser utilizado quando:
* Há disponibilidade de um conjunto de treinamento grande ou moderado.
* Os atributos que descrevem as instâncias forem condicionalmente independentes dada a classe.

A utilização do Naive Bayes é muito eficiente em categorização de texto, filtragem de SPAM e análise de sentimento em redes sociais.
Possui aplicações bem sucedidas em diagnóstico médico.
Além disso, o algoritmo é muito robusto para previsões em tempo real, ainda mais por precisar de poucos dados para realizar a classificação. Entretanto, caso haja necessidade de correlacionar fatores, o Naive Bayes tende a falhar na predição.

### III. Definição Teórica e Modelagem Matemática

Utiliza o Teorema de Bayes com "naive", supondo a condição de independência entre cada par de características dado o valor da classe da variável.

![image](https://user-images.githubusercontent.com/77736052/156112459-65a63587-7068-44d4-bccd-8715a89792e0.png)

Onde:

P(A|B): probabilidade da classe A dado o vetor B

P(B|A): probabilidade do vetor B dada a classe A, também chamado de probabilidade condicional

P(A): probabilidade a priori da classe A

P(B): probabilidade a priori do vetor de treinamento B

P(A|B) é chamada de probabilidade a posteriori de A porque ela reflete nossa confiança que A se mantenha após termos observado o vetor de treinamento B.

P(A|B) reflete a influência do vetor de treinamento B, enquanto que a probabilidade a priori P(A) é independente de B.

Geralmente queremos encontrar a classe mais provável A, sendo fornecidos os exemplos de treinamento B.

O classificador Naive Bayes é baseado na suposição de que os valores dos atributos são condicionalmente independentes dado o valor alvo. A probabilidade de observar a conjunção de atributos a1, a2,..., an é somente o produto das probabilidades para os atributos individuais:

![image](https://user-images.githubusercontent.com/77736052/156113756-af390b5c-5591-4ecb-8529-c88543a13f30.png)

Portanto, o classificador pode ser definido como:

![image](https://user-images.githubusercontent.com/77736052/156113858-c03b90a2-f5ed-4771-abd6-328a0190e423.png)

CNB indica o valor alvo fornecido pelo algoritmo.
Os termos P(cj) e P(ai|cj) são estimados baseados nas suas frequências no conjunto de treinamento, estas probabilidades “aprendidas” são utilizadas para classificar uma nova instância aplicando a equação anterior (CNB).

### IV. Vantagens e Desvantagens

 * Vantagens
      - Simples, fácil de implementar.
      - Não requer uma grande quantidade de dados de treinamento.
      - Pode lidar com dados discretos e contínuos.
      - É rápido e pode ser usado para fazer predições reais.
      - É mais adequado para variáveis de entrada categóricas do que para variáveis numéricas.
 * Desvantagens
      - Esse algoritmo enfrenta o 'problema de frequência zero', onde atribui probabilidade zero a uma variável categórica cuja categoria no conjunto de dados de teste não estava disponível no conjunto de dados de treinamento. Seria melhor se você usasse uma técnica de suavização para superar esse problema.
      - Assume que todos os preditores são independentes, o que raramente acontece na vida real e acaba limitando a aplicabilidade desse algoritmo em casos de uso do mundo real.
      - As estimativas podem estar erradas em alguns casos, portanto deve-se levar as saídas muito a sério.

### V. Exemplo de uma aplicação em Python
[Naives Bayes usando conjunto de dados Iris](https://github.com/megbatista/DataScienceSeminario/blob/9fda802249bf5b6d244ef4867351e78b68976d0a/Aplica%C3%A7%C3%B5esPython/NaiveBayes.ipynb)

## Decision Trees e Ensembles of Decision Trees

### I. Conceito

Decision Tree é um modelo muito poderoso que ajuda a classificar dados rotulados e a fazer previsões. Ele também esclarece com muitas informações sobre os dados e é fácil de interpretar. Uma árvore de decisão pode ser considerada um “classificador if-else aninhado”.

Dos classificadores que se tem no mundo do aprendizado de máquina, a Árvore de decisão, é um dos classificadores que se assemelha muito ao raciocínio humano.
É possível conceituar este algoritmo por meio de um exemplo, suponha que temos um conjunto de dados de 50 frutas, das quais poucas são limões e poucas são maçãs, e recebemos a cor da fruta e o tamanho da fruta como feature de entrada (portanto, esses são nossas 2 features independentes).
A árvore de decisão para este problema poderia ser assim:

![image](https://user-images.githubusercontent.com/77736052/156300017-04a496fc-ed7a-45a6-85fe-4a16c169402b.png)

Algumas terminologias-chave para trabalhar com a Árvore de Decisão são:

* NODE: Cada vez que faz-se uma pergunta ou quando se toma uma decisão, um NODE é representado.
* ROOT: O nó mais alto da árvore (raiz) com o qual começa-se a questionar.
* INTERIOR: Qualquer nó, exceto o nó ROOT, onde novamente faz-se uma pergunta.
* LEAF: Quando se chega a um ponto em que não faz-se uma pergunta, mas toma-se uma decisão, chamamos de nó LEAF.

Um exemplo geral da estrutura DT seria:

![image](https://user-images.githubusercontent.com/77736052/156301266-9e244804-bab9-4745-8f85-e7f6aa7a0829.png)

Existem algumas suposições que precisam ser consideradas antes de criar a árvore:

* No início, todos os dados são considerados como a raiz, depois, usamos os algoritmos para fazer uma divisão ou dividir a raiz em subárvores.
* Os valores das features são considerados categóricos. Se os valores forem contínuos, eles serão separados antes da construção do modelo.
* Os registros são distribuídos recursivamente com base nos valores dos atributos.
* A ordenação dos atributos como raiz ou nó interno da árvore é feita por meio de uma abordagem estatística.

### II. Classes de Problemas com melhores resultados

Decision Tree é um dos algoritmos básicos e amplamente utilizados nas áreas de Aprendizado de Máquina. É usado em diferentes áreas na modelagem de classificação e regressão. Devido à sua capacidade de representar a saída visualizada, pode-se facilmente extrair insights do fluxo do processo de modelagem. Alguns exemplos em que a Árvore de Decisão pode ser usada são:

* Gestão de negócios
* Gestão de Relacionamento com o Cliente
* Detecção de declaração fraudulenta
* Consumo de energia
* Gestão de Saúde
* Diagnóstico de falhas

### III. Definição Teórica e Modelagem Matemática

### IV. Vantagens e Desvantagens

 * Vantagens
      - Levam pouco tempo para processar os dados quando comparadas a outros algoritmos.
      - Poucas etapas de pré-processamento, como normalização, transformação e dimensionamento dos dados, podem ser ignoradas.
      - Embora existam valores ausentes no conjunto de dados, o desempenho do modelo não será afetado.
      - É intuitivo e fácil de explicar para as equipes técnicas e partes interessadas, e pode ser implementado em várias organizações.
 * Desvantagens
      - Pequenas mudanças nos dados podem causar uma grande mudança na estrutura da árvore de decisão que, por sua vez, leva à instabilidade.
      - O tempo de treinamento aumenta drasticamente, proporcional ao tamanho do conjunto de dados.
      - Em alguns casos, os cálculos podem se tornar complexos em comparação com os outros algoritmos tradicionais.

### V. Exemplo de uma aplicação em Python
[Decision Trees](https://github.com/megbatista/DataScienceSeminario/blob/9fda802249bf5b6d244ef4867351e78b68976d0a/Aplica%C3%A7%C3%B5esPython/NaiveBayes.ipynb)

