# Aprendizado Supervisionado

## K-Nearest Neighbors e Linear Models

I. Conceito (O que é? Pra que serve?)
  
Pode ser usado para problemas de classificação e regressão, porém é mais usado em problemas de classificação
é utilizado em agumas áareas como:
Microbiologia - para classificação de células
Marketing - para segmentação de clientes



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
