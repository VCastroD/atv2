"""Testes estatisticos sobre as predicoes: intervalo de confianca e comparacao.

Duas perguntas, dois instrumentos:

* "Quanto vale a metrica, com que incerteza?" -> :func:`ic_bootstrap`, percentil
  sobre reamostragem com reposicao das LINHAS do dataset.
* "A v2 e melhor que a v1, ou a diferenca cabe no ruido amostral?" ->
  :func:`comparar_versoes`, que combina bootstrap PAREADO (intervalo para a
  diferenca) com o teste exato de McNemar (p-valor).

Decisoes de projeto (justificadas em RELATORIO.md, secao 3):

1. Bootstrap nao parametrico por percentil. Nao assumimos normalidade da metrica;
   so assumimos que as linhas sao i.i.d. -- que e como o dataset foi gerado.
   BCa daria correcao de vies/assimetria, mas com n=4000 e metricas que sao
   medias de indicadoras a distribuicao ja e quase simetrica; o ganho nao paga a
   complexidade.
2. A comparacao entre versoes e PAREADA: as duas versoes preveem as MESMAS
   linhas, entao reamostrar as versoes de forma independente jogaria fora a
   correlacao entre elas e inflaria o intervalo. Cada reamostra sorteia linhas e
   avalia as duas versoes nas mesmas linhas sorteadas.
3. McNemar exato (binomial com p=0.5 sobre os pares discordantes) em vez da
   aproximacao qui-quadrado: e exato para qualquer contagem e nao precisa de
   correcao de continuidade.
4. Reamostras em que a metrica e indefinida (ex.: TPR sem nenhum positivo no
   grupo sorteado) viram ``nan`` e sao descartadas, com a contagem reportada em
   :attr:`IntervaloConfianca.descartadas`. Descartar em silencio seria mentir
   sobre o tamanho efetivo da amostra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

__all__ = [
    "SEMENTE",
    "N_REAMOSTRAS",
    "ALFA",
    "acuracia",
    "taxa_selecao",
    "taxa_verdadeiro_positivo",
    "precisao",
    "IntervaloConfianca",
    "ic_bootstrap",
    "ComparacaoVersoes",
    "comparar_versoes",
    "mcnemar_exato",
    "teste_permutacao",
]

SEMENTE = 20260910
N_REAMOSTRAS = 5000
ALFA = 0.05


# --------------------------------------------------------------------- metricas


def acuracia(y: np.ndarray, pred: np.ndarray) -> float:
    """Fracao de acertos."""
    return float(np.mean(pred == y))


def taxa_selecao(pred: np.ndarray) -> float:
    """Fracao de decisoes positivas (aqui: credito aprovado). Base da paridade."""
    return float(np.mean(pred))


def taxa_verdadeiro_positivo(y: np.ndarray, pred: np.ndarray) -> float:
    """TPR / recall: entre quem de fato e bom pagador, quantos foram aprovados.

    Indefinida (``nan``) se nao ha nenhum positivo real -- ver decisao (4).
    """
    positivos = y == 1
    if not positivos.any():
        return float("nan")
    return float(np.mean(pred[positivos]))


def precisao(y: np.ndarray, pred: np.ndarray) -> float:
    """Entre os aprovados, quantos eram de fato bons pagadores."""
    aprovados = pred == 1
    if not aprovados.any():
        return float("nan")
    return float(np.mean(y[aprovados] == 1))


# ------------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class IntervaloConfianca:
    """Estimativa pontual mais os limites do intervalo percentil."""

    ponto: float
    inferior: float
    superior: float
    confianca: float
    n_reamostras: int
    descartadas: int = 0

    @property
    def largura(self) -> float:
        return self.superior - self.inferior

    def contem(self, valor: float) -> bool:
        return self.inferior <= valor <= self.superior

    def __str__(self) -> str:
        pct = int(round(self.confianca * 100))
        aviso = f" ({self.descartadas} reamostras indefinidas)" if self.descartadas else ""
        return (
            f"{self.ponto:+.4f}  IC{pct}% [{self.inferior:+.4f}, {self.superior:+.4f}]"
            f"{aviso}"
        )


def ic_bootstrap(
    metrica: Callable[..., float],
    dados: Mapping[str, np.ndarray],
    n_reamostras: int = N_REAMOSTRAS,
    confianca: float = 1 - ALFA,
    semente: int = SEMENTE,
) -> IntervaloConfianca:
    """Intervalo percentil por bootstrap nao parametrico sobre as linhas.

    ``dados`` mapeia os nomes dos parametros de ``metrica`` para vetores de mesmo
    comprimento; cada reamostra sorteia N indices com reposicao e aplica esses
    MESMOS indices a todos os vetores, o que preserva o pareamento entre eles
    (rotulo, predicao da v1, predicao da v2, grupo...).

    :raises ValueError: se os vetores tiverem comprimentos diferentes.
    """
    vetores = {nome: np.asarray(v) for nome, v in dados.items()}
    tamanhos = {len(v) for v in vetores.values()}
    if len(tamanhos) != 1:
        raise ValueError(f"vetores de comprimentos diferentes: {tamanhos}")
    n = tamanhos.pop()
    if n == 0:
        raise ValueError("nao da para reamostrar um dataset vazio")

    rng = np.random.default_rng(semente)
    amostras = np.empty(n_reamostras, dtype=float)
    for k in range(n_reamostras):
        indices = rng.integers(0, n, size=n)
        amostras[k] = metrica(**{nome: v[indices] for nome, v in vetores.items()})

    validas = amostras[~np.isnan(amostras)]
    if validas.size == 0:
        raise ValueError("todas as reamostras produziram metrica indefinida")

    meio = (1 - confianca) / 2 * 100
    inferior, superior = np.percentile(validas, [meio, 100 - meio])
    return IntervaloConfianca(
        ponto=float(metrica(**vetores)),
        inferior=float(inferior),
        superior=float(superior),
        confianca=confianca,
        n_reamostras=n_reamostras,
        descartadas=int(n_reamostras - validas.size),
    )


# ---------------------------------------------------- comparacao entre versoes


def mcnemar_exato(b: int, c: int) -> float:
    """p-valor bilateral do teste de McNemar exato.

    ``b`` = casos em que so a versao A acertou, ``c`` = so a versao B acertou.
    Sob H0 ("as duas versoes erram igual"), cada par discordante e um cara-ou-coroa
    justo, entao ``b ~ Binomial(b + c, 0.5)``.
    """
    total = b + c
    if total == 0:
        return 1.0  # nenhuma discordancia: nada a testar
    menor = min(b, c)
    cauda = sum(math.comb(total, k) for k in range(menor + 1)) / (2 ** total)
    return float(min(1.0, 2 * cauda))


@dataclass(frozen=True)
class ComparacaoVersoes:
    """Resultado de comparar duas versoes de modelo nas mesmas linhas."""

    metrica_a: float
    metrica_b: float
    diferenca: IntervaloConfianca
    so_a_acertou: int
    so_b_acertou: int
    p_mcnemar: float
    alfa: float = ALFA

    @property
    def significativa(self) -> bool:
        """Verdadeiro se o IC da diferenca exclui zero E o p-valor < alfa."""
        return not self.diferenca.contem(0.0) and self.p_mcnemar < self.alfa

    @property
    def veredito(self) -> str:
        if self.significativa:
            direcao = "B > A" if self.diferenca.ponto > 0 else "A > B"
            return f"diferenca significativa ({direcao})"
        return "diferenca compativel com ruido amostral"

    def __str__(self) -> str:
        return (
            f"  metrica A         : {self.metrica_a:.4f}\n"
            f"  metrica B         : {self.metrica_b:.4f}\n"
            f"  diferenca (B - A) : {self.diferenca}\n"
            f"  pares discordantes: so A acertou={self.so_a_acertou}, "
            f"so B acertou={self.so_b_acertou}\n"
            f"  p (McNemar exato) : {self.p_mcnemar:.2e}\n"
            f"  veredito          : {self.veredito}"
        )


def comparar_versoes(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_reamostras: int = N_REAMOSTRAS,
    confianca: float = 1 - ALFA,
    semente: int = SEMENTE,
    alfa: float = ALFA,
) -> ComparacaoVersoes:
    """Compara a acuracia de duas versoes avaliadas nas MESMAS linhas.

    Devolve o intervalo bootstrap pareado para ``acuracia(B) - acuracia(A)`` e o
    p-valor exato de McNemar. Os dois concordarem e o que autoriza a conclusao:
    o intervalo diz o TAMANHO do efeito, o p-valor diz se ele existe.
    """
    y = np.asarray(y)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    def diferenca_acuracia(y, pred_a, pred_b):
        return acuracia(y, pred_b) - acuracia(y, pred_a)

    ic = ic_bootstrap(
        diferenca_acuracia,
        {"y": y, "pred_a": pred_a, "pred_b": pred_b},
        n_reamostras=n_reamostras,
        confianca=confianca,
        semente=semente,
    )

    acertou_a = pred_a == y
    acertou_b = pred_b == y
    so_a = int(np.sum(acertou_a & ~acertou_b))
    so_b = int(np.sum(~acertou_a & acertou_b))

    return ComparacaoVersoes(
        metrica_a=acuracia(y, pred_a),
        metrica_b=acuracia(y, pred_b),
        diferenca=ic,
        so_a_acertou=so_a,
        so_b_acertou=so_b,
        p_mcnemar=mcnemar_exato(so_a, so_b),
        alfa=alfa,
    )


# ----------------------------------------------------------------- permutacao


def teste_permutacao(
    estatistica: Callable[[np.ndarray], float],
    rotulos: np.ndarray,
    n_permutacoes: int = N_REAMOSTRAS,
    semente: int = SEMENTE,
) -> float:
    """p-valor bilateral por permutacao dos ``rotulos``.

    Usado pelo modulo de fairness: sob H0 ("a decisao independe do grupo"), o
    vetor de grupos e permutavel, entao embaralha-lo nao deveria mudar o tamanho
    da disparidade. O p-valor e a fracao de permutacoes que produzem disparidade
    tao extrema quanto a observada.

    Usa a correcao (+1)/(B+1), que evita p=0 -- com B permutacoes nao da para
    afirmar mais do que ``p < 1/(B+1)``.
    """
    rotulos = np.asarray(rotulos)
    observada = abs(estatistica(rotulos))
    rng = np.random.default_rng(semente)
    extremas = 0
    for _ in range(n_permutacoes):
        valor = estatistica(rng.permutation(rotulos))
        if not math.isnan(valor) and abs(valor) >= observada:
            extremas += 1
    return (extremas + 1) / (n_permutacoes + 1)
