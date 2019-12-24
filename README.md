# Projeto
Projeto desenvolvido com o intutio de criar uma rede neural 
capaz de jogar uma versão simplificada do jogo Pong

## Como Funciona?

O projeto possui três arquivos:

- data.csv: Arquivo com os dados utilizados para 
treinar a rede neural
- neural.py: Script com o código da rede neural 
- game.py: Script com o código do jogo

O jogo, como já informado, trata-se de uma versão 
simplificada do jogo Pong. Onde, ao invés de ser um jogo 
de ping-pong, a barra, localizada na parte inferior da tela, 
deve impedir que as bolas passem por ela.

### Rodando o projeto

Assim que realizar o download ou clonar o projeto, 
basta rodar o comando:

```bash
$ python game.py
```

Após isso, uma nova janela com o jogo se abrirá. 
O score indica quantos pontos a rede neural fez 
e a geração indica quantas vezes a rede neural foi 
apresentada ao conjunto de dados presentes em data.csv

## Fontes
[IA Aprende A Jogar "Ping-Pong" Usando Rede Neural](https://www.youtube.com/watch?v=ETn61j8kIaU)
[Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
