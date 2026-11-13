"""Ambiente de Pac-Man com observacao parcial."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import random
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
    from gymnasium import Env, spaces
except ImportError:
    np = None

    class Env:
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            return None

    spaces = None

Pos = Tuple[int, int]
Acao = str


class AcoesPacman:
    CIMA = "cima"
    BAIXO = "baixo"
    ESQUERDA = "esquerda"
    DIREITA = "direita"
    PARAR = "parar"
    TODAS = (CIMA, BAIXO, ESQUERDA, DIREITA, PARAR)


movimentos: Dict[Acao, Pos] = {
    AcoesPacman.CIMA: (-1, 0),
    AcoesPacman.BAIXO: (1, 0),
    AcoesPacman.ESQUERDA: (0, -1),
    AcoesPacman.DIREITA: (0, 1),
    AcoesPacman.PARAR: (0, 0),
}


@dataclass(frozen=True)
class Estado:
    visao: Tuple[Tuple[int, ...], ...]
    timer_poder: int


class PacmanEnv(Env):
    """MDP compacto de Pac-Man com janela local 5x5."""

    PAREDE = 0
    VAZIO = 1
    PASTILHA = 2
    PILULA_PODER = 3
    FANTASMA = 4
    FANTASMA_VULNERAVEL = 5
    PACMAN = 6

    RECOMPENSAS = {
        "passo": -0.04,
        "pastilha": 10.0,
        "poder": 100.0,
        "fantasma": 200.0,
        "morte": -1000.0,
        "vitoria": 1000.0,
    }
    MAPA_TESTE = (
        "#######",
        "#P..F.#",
        "#.###.#",
        "#...*.#",
        "#######",
    )

    def __init__(self, observation_size: int = 5, ghost_count: int = 1, seed: Optional[int] = None) -> None:
        if observation_size % 2 == 0 or observation_size < 3:
            raise ValueError("observation_size deve ser um numero impar >= 3")
        if ghost_count < 1:
            raise ValueError("ghost_count deve ser pelo menos 1")
        self.tamanho_visao = observation_size
        self.n_fantasmas = ghost_count
        self.random = random.Random(seed)
        self.layout = self.MAPA_TESTE
        self.action_space = spaces.Discrete(5) if spaces else None
        self.observation_space = (
            spaces.Dict({
                "visao_local": spaces.Box(
                    low=0,
                    high=self.PACMAN,
                    shape=(self.tamanho_visao, self.tamanho_visao),
                    dtype=np.int8,
                ),
                "timer_poder": spaces.Discrete(21),
            })
            if spaces
            else None
        )
        self._inicializar_mapa()

    def _inicializar_mapa(self) -> None:
        self.walls = {pos for pos, valor in self._posicoes_mapa() if valor == "#"}
        self.pastilhas = {pos for pos, valor in self._posicoes_mapa() if valor == "."}
        self.pilulas = {
            pos for pos, valor in self._posicoes_mapa() if valor in ("o", "*")
        }
        self.pacman = next(pos for pos, valor in self._posicoes_mapa() if valor == "P")
        self.ghosts = [pos for pos, valor in self._posicoes_mapa() if valor == "F"]
        celulas_livres = [pos for pos, valor in self._posicoes_mapa() if valor != "#"]
        candidatas = [
            pos for pos in celulas_livres if pos != self.pacman and pos not in self.ghosts
        ]
        while len(self.ghosts) < self.n_fantasmas:
            self.ghosts.append(candidatas[len(self.ghosts) % len(candidatas)])
        self.timer = 0
        self._pilula_coletada = False
        self.terminou = False
        self.passos = 0

    def _posicoes_mapa(self) -> Iterable[Tuple[Pos, str]]:
        for linha, texto in enumerate(self.layout):
            for coluna, valor in enumerate(texto):
                yield (linha, coluna), valor

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.random.seed(seed)
        self._inicializar_mapa()
        return self._observar(), {}

    def step(self, acao: Acao):
        if isinstance(acao, Integral):
            acao = AcoesPacman.TODAS[acao] if 0 <= acao < len(AcoesPacman.TODAS) else acao
        if acao not in AcoesPacman.TODAS:
            raise ValueError(f"acao desconhecida: {acao!r}")
        if self.terminou:
            raise RuntimeError("o episodio terminou; chame reset() antes de step()")

        acao_executada = self._sortear_acao(acao)
        self._pilula_coletada = False
        self.pacman = self._mover(self.pacman, acao_executada)
        recompensa = self.RECOMPENSAS["passo"] + self._coletar_item()
        info: Dict[str, object] = {"acao_executada": acao_executada}
        self._mover_fantasmas()
        colisao = self._resolver_colisao()
        if colisao == "morte":
            recompensa = self.RECOMPENSAS["morte"]
            self.terminou = True
            info["motivo_termino"] = "capturado"
        elif colisao == "fantasma":
            recompensa += self.RECOMPENSAS["fantasma"]
            info["fantasma_capturado"] = True
        if not self.terminou and not self.pastilhas and not self.pilulas:
            recompensa += self.RECOMPENSAS["vitoria"]
            self.terminou = True
            info["motivo_termino"] = "vitoria"
        if self.timer > 0 and not self._pilula_coletada:
            self.timer = max(0, self.timer - 1)
        self.passos += 1
        info["passos"] = self.passos
        return self._observar(), recompensa, self.terminou, False, info

    def _sortear_acao(self, acao: Acao) -> Acao:
        if acao == AcoesPacman.PARAR:
            return acao
        perpendiculares = {
            AcoesPacman.CIMA: (AcoesPacman.ESQUERDA, AcoesPacman.DIREITA),
            AcoesPacman.BAIXO: (AcoesPacman.ESQUERDA, AcoesPacman.DIREITA),
            AcoesPacman.ESQUERDA: (AcoesPacman.CIMA, AcoesPacman.BAIXO),
            AcoesPacman.DIREITA: (AcoesPacman.CIMA, AcoesPacman.BAIXO),
        }[acao]
        sorteio = self.random.random()
        if sorteio < 0.8:
            return acao
        return perpendiculares[0] if sorteio < 0.9 else perpendiculares[1]

    def _mover(self, pos: Pos, acao: Acao) -> Pos:
        delta = movimentos[acao]
        nova_pos = (pos[0] + delta[0], pos[1] + delta[1])
        return pos if nova_pos in self.walls else nova_pos

    def _coletar_item(self) -> float:
        if self.pacman in self.pastilhas:
            self.pastilhas.remove(self.pacman)
            return self.RECOMPENSAS["pastilha"]
        if self.pacman in self.pilulas:
            self.pilulas.remove(self.pacman)
            self.timer = 20
            self._pilula_coletada = True
            return self.RECOMPENSAS["poder"]
        return 0.0

    def _mover_fantasmas(self) -> None:
        novas_posicoes: List[Pos] = []
        for fantasma in self.ghosts:
            possiveis = [self._mover(fantasma, acao) for acao in movimentos]
            possiveis = [pos for pos in possiveis if pos not in self.walls]
            if not possiveis:
                novas_posicoes.append(fantasma)
            elif self.random.random() < 0.6:
                novas_posicoes.append(min(possiveis, key=lambda pos: self._distancia(pos, self.pacman)))
            else:
                novas_posicoes.append(self.random.choice(possiveis))
            self.ghosts = novas_posicoes

    def _resolver_colisao(self) -> Optional[str]:
        if self.pacman not in self.ghosts:
            return None
        if self.timer > 0:
            self.ghosts = [fantasma for fantasma in self.ghosts if fantasma != self.pacman]
            return "fantasma"
        return "morte"

    @staticmethod
    def _distancia(primeira: Pos, segunda: Pos) -> int:
        return abs(primeira[0] - segunda[0]) + abs(primeira[1] - segunda[1])

    def _observar(self) -> Dict[str, object]:
        raio = self.tamanho_visao // 2
        visao: List[Tuple[int, ...]] = []
        for linha in range(self.pacman[0] - raio, self.pacman[0] + raio + 1):
            valores: List[int] = []
            for coluna in range(self.pacman[1] - raio, self.pacman[1] + raio + 1):
                pos = (linha, coluna)
                if (
                    linha < 0
                    or coluna < 0
                    or linha >= len(self.layout)
                    or coluna >= len(self.layout[0])
                    or pos in self.walls
                ):
                    valor = self.PAREDE
                elif pos in self.ghosts:
                    valor = self.FANTASMA_VULNERAVEL if self.timer > 0 else self.FANTASMA
                elif pos == self.pacman:
                    valor = self.PACMAN
                elif pos in self.pilulas:
                    valor = self.PILULA_PODER
                elif pos in self.pastilhas:
                    valor = self.PASTILHA
                else:
                    valor = self.VAZIO
                valores.append(valor)
            visao.append(tuple(valores))
        visao_local = np.asarray(visao, dtype=np.int8) if np is not None else tuple(visao)
        return {
            "visao_local": visao_local,
            "timer_poder": self.timer,
        }

    def render(self) -> None:
        tela = [list(linha) for linha in self.layout]
        for linha, coluna in self.pastilhas:
            tela[linha][coluna] = "."
        for linha, coluna in self.pilulas:
            tela[linha][coluna] = "*"
        for linha, coluna in self.ghosts:
            tela[linha][coluna] = "F"
        linha, coluna = self.pacman
        tela[linha][coluna] = "P"
        print("\n".join("".join(texto) for texto in tela))
        print("=" * 15)

    def estado(self) -> Estado:
        observacao = self._observar()
        return Estado(tuple(map(tuple, observacao["visao_local"])), observacao["timer_poder"])