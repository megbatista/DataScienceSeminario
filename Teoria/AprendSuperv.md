# Aprendizado Supervisionado

## K-Nearest Neighbors e Linear Models

I. Conceito (O que é? Pra que serve?)

   K-Nearest Neighbors (KNN), é um dos algoritmos de machine learning mais fáceis de se aprender, pode ser usado para classificação e regressão, sendo a técnica mais usada em problemas de classificação. É utilizado em agumas áreas como:
  * Microbiologia - para classificação de células
  * Marketing - para segmentação de clientes
  
  A classificação é feita a partir do cálculo da distância dos vizinhos mais próximos de cada ponto: a um ponto de consulta é atribuída a classe de dados que tem mais representantes dentro dos vizinhos mais próximos do ponto.
  Observando o gráfico abaixo, em que o eixo y representa a altura e o eixo x, a idade, identifica-se 3 tipos de grupos de pessoas, em verde, laranja e azul. O ponto preto é o ponto a ser classificado, intuitivamente pode-se inferir que o esse ponto pertence ao grupo laranja, pois está mais próximo desse conjunto, portanto, as demais características desse dado devem ser semelhantes aos destes outros pontos.
  
![image](https://user-images.githubusercontent.com/77736052/155669688-fee42120-6475-4ae5-925c-a25759c2e0cc.png)
![image](https://user-images.githubusercontent.com/77736052/155669627-9e8ea38d-9758-4bfd-ba3d-7cf452434b33.png)

   O algoritmo prevê, por meio de comparação entre as instâncias, valores de quaisquer novos pontos de dados. O novo ponto recebe um valor baseado em quão próximo ele se parece dos pontos no conjunto de treinamento. 
   O KNN utiliza aprendizagem baseada em instâncias, ou seja, não existe um modelo, regra ou função construído a partir de uma etapa de treinamento. Esse método, armazena todos os dados de treinamento, e a cada nova instância que se queira classificar, são realizados cálculos entre esta e os dados armazenados anteriormente.


II. Classes de Problemas com melhores resultados

III. Definição Teórica e Modelagem Matemática

IV. Vantagens e Desvantagens (limitações)

V. Exemplo de uma aplicação em Python

B
## Naive Bayes Classifiers

O classificador Naive Bayes é um simples algoritmo de classificação realiza predições em aprendizagem de máquina. O termo “naive” (ingênuo) diz respeito à forma como o algoritmo analisa as características de uma base de dados: ele assume que as features são independentes entre si. 

Além disso, ele também assume que as variáveis features são todas igualmente importantes para o resultado. Em cenários em que isso não ocorre, essa técnica deixa de ser a opção ideal. Discutiremos adiante sobre as aplicações.

Como Bayes é um nome famoso na estatística, é fácil concluir que o seu algoritmo tem uma forte base dessa área, reforçando a relação entre estatística e inteligência artificial.

Inclusive, o seu funcionamento pode ser facilmente descrito em termos estatísticos: para calcular a predição, o algoritmo define, primeiramente, uma tabela de probabilidades, em que consta a frequência dos preditores com relação às variáveis de saída. Então, o cálculo final leva em conta a probabilidade maior para oferecer uma solução.
, se provou muito eficiente em categorização de texto

Utiliza o teorema de Bayes com "naive", supondo a condição de independência entre cada par de características dado o valor da classe da variável.

Dados a classe de variáveis *y* e o vetor de características dependentes *x1* até *xn*, o teorema de Bayes afirma que:
![image](https://user-images.githubusercontent.com/77736052/155605749-3f3fc0c8-3b4d-4d88-9a6d-1a9307dd99f8.png)
