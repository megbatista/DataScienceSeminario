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
   

## Naive Bayes Classifiers

O classificador Naive Bayes é um simples algoritmo de classificação realiza predições em aprendizagem de máquina. O termo “naive” (ingênuo) diz respeito à forma como o algoritmo analisa as características de uma base de dados: ele assume que as features são independentes entre si. 

Além disso, ele também assume que as variáveis features são todas igualmente importantes para o resultado. Em cenários em que isso não ocorre, essa técnica deixa de ser a opção ideal. Discutiremos adiante sobre as aplicações.

Como Bayes é um nome famoso na estatística, é fácil concluir que o seu algoritmo tem uma forte base dessa área, reforçando a relação entre estatística e inteligência artificial.

Inclusive, o seu funcionamento pode ser facilmente descrito em termos estatísticos: para calcular a predição, o algoritmo define, primeiramente, uma tabela de probabilidades, em que consta a frequência dos preditores com relação às variáveis de saída. Então, o cálculo final leva em conta a probabilidade maior para oferecer uma solução.
, se provou muito eficiente em categorização de texto

Utiliza o teorema de Bayes com "naive", supondo a condição de independência entre cada par de características dado o valor da classe da variável.

Dados a classe de variáveis *y* e o vetor de características dependentes *x1* até *xn*, o teorema de Bayes afirma que:
![image](https://user-images.githubusercontent.com/77736052/155605749-3f3fc0c8-3b4d-4d88-9a6d-1a9307dd99f8.png)
