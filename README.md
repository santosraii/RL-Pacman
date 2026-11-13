# RL-Pacman

Ambiente de Pac-Man com observacao parcial para a disciplina de aprendizado por reforco.

## Modelo

Este projeto adapta um MDP discreto de Pac-Man para um agente com observacao
parcial. O estado interno completo contem a posicao do Pac-Man, as posicoes
dos fantasmas, paredes, pastilhas, pilulas de poder e o temporizador. O
agente, porem, recebe somente uma janela egocentrica `5x5` ao redor do
Pac-Man e o temporizador da pilula de poder. Os valores da grade sao:

| Valor | Celula |
| ---: | --- |
| 0 | parede |
| 1 | vazio |
| 2 | pastilha |
| 3 | pilula de poder |
| 4 | fantasma |
| 5 | fantasma vulneravel |
| 6 | Pac-Man |

As acoes sao `cima`, `baixo`, `esquerda`, `direita` e `parar`. A acao escolhida tem
probabilidade 0.8; cada direcao perpendicular tem probabilidade 0.1. Os
fantasmas se movem apos o Pac-Man, perseguindo-o com probabilidade 0.6 e
escolhendo aleatoriamente uma posicao valida nos demais casos. Uma posicao
valida e uma celula livre, que nao e uma parede.

O episodio termina com vitoria quando todas as pastilhas e pilulas de poder
sao coletadas. Nesse caso, o agente recebe uma recompensa adicional de `+1000`
e `info["motivo_termino"]` recebe o valor `"vitoria"`. Se houver uma colisao
fatal no mesmo passo, a derrota tem prioridade.

O ambiente conhece o estado completo do jogo: posicoes, paredes, pastilhas,
fantasmas e temporizador. Por isso, sua dinamica interna pode ser descrita
como um MDP. O agente, porem, enxerga apenas a janela local e o temporizador.
Como ele nao ve todo o mapa nem todos os fantasmas, nao consegue determinar
com certeza o estado real do jogo. Para o agente, portanto, o problema e um
POMDP. Essa e a principal diferenca em relacao ao Pac-Man totalmente
observavel.

## Executar

O ambiente funciona com a biblioteca padrao do Python. Para obter a interface
completa do Gymnasium e observacoes como `numpy.ndarray`, instale as
dependencias opcionais:

```bash
python -m pip install numpy gymnasium
```

```bash
python -m unittest -v
```

Exemplo:

```python
from pacman_env import AcoesPacman, PacmanEnv

env = PacmanEnv(seed=42)
observation, info = env.reset()
observacao, recompensa, terminou, truncado, info = env.step(AcoesPacman.DIREITA)
```
