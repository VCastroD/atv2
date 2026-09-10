"""Auditoria de equidade entre grupos de um atributo sensivel.

Metricas calculadas, e por que estas:

* **Paridade demografica** -- diferenca entre as taxas de aprovacao dos grupos,
  ``P(pred=1 | G=a) - P(pred=1 | G=b)``. E a metrica principal aqui porque a
  auditoria comeca verificando que a **taxa-base real** dos grupos
  (``P(y_real=1 | G)``) e estatisticamente indistinguivel: quando o risco real e
  o mesmo, qualquer diferenca sistematica na taxa de aprovacao vem do modelo, e
  nao do mundo. Se as taxas-base fossem diferentes, exigir paridade seria exigir
  que o modelo ignorasse informacao legitima -- por isso a checagem vem antes.
* **Razao de impacto desproporcional** (``menor/maior``) -- a leitura da "regra
  dos 4/5", tradicional em auditoria de credito e de contratacao: razao abaixo de
  0.8 e tida como indicio de impacto desproporcional.
* **Igualdade de oportunidade** -- diferenca de TPR, ``P(pred=1 | y_real=1, G)``.
  Condiciona em quem de fato e bom pagador, entao continua valendo mesmo se as
  taxas-base diferirem. Serve de contraprova da paridade.

Cada diferenca vem com **intervalo de confianca bootstrap** (tamanho do efeito) e
**p-valor de permutacao** (existencia do efeito). Um numero sozinho nao decide:
um gap de 1 p.p. pode ser gigante em amostra grande e irrelevante em amostra
pequena -- e a incerteza que diz qual dos dois e o caso.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from estatistica import (
    ALFA,
    N_REAMOSTRAS,
    SEMENTE,
    IntervaloConfianca,
    ic_bootstrap,
    taxa_selecao,
    taxa_verdadeiro_positivo,
    teste_permutacao,
)

__all__ = [
    "LIMIAR_REGRA_QUATRO_QUINTOS",
    "MetricasGrupo",
    "AuditoriaFairness",
    "metricas_por_grupo",
    "diferenca_paridade",
    "razao_impacto",
    "diferenca_oportunidade",
    "auditar",
]

#: Abaixo disso a razao entre taxas de aprovacao e indicio de impacto
#: desproporcional (regra dos 4/5, usada por auditorias desde os anos 1970).
LIMIAR_REGRA_QUATRO_QUINTOS = 0.8


@dataclass(frozen=True)
class MetricasGrupo:
    """Retrato de um grupo: quantos sao, quantos foram aprovados, quantos deviam."""

    grupo: str
    n: int
    taxa_selecao: float
    taxa_base_real: float
    tpr: float

    def __str__(self) -> str:
        return (
            f"{self.grupo:>12} | n={self.n:>5} | aprovados={self.taxa_selecao:6.2%} "
            f"| bons pagadores reais={self.taxa_base_real:6.2%} | TPR={self.tpr:6.2%}"
        )


@dataclass(frozen=True)
class AuditoriaFairness:
    """Resultado da auditoria de uma versao de modelo sobre um atributo sensivel."""

    versao: str
    atributo: str
    grupo_a: str
    grupo_b: str
    por_grupo: Mapping[str, MetricasGrupo]
    diferenca_base_real: IntervaloConfianca
    paridade: IntervaloConfianca
    p_paridade: float
    razao_impacto: float
    oportunidade: IntervaloConfianca
    p_oportunidade: float
    alfa: float = ALFA

    @property
    def taxas_base_comparaveis(self) -> bool:
        """As taxas-base reais dos dois grupos sao indistinguiveis?

        Se sim, paridade demografica e um criterio defensavel para este dataset.
        """
        return self.diferenca_base_real.contem(0.0)

    @property
    def paridade_violada(self) -> bool:
        return not self.paridade.contem(0.0) and self.p_paridade < self.alfa

    @property
    def oportunidade_violada(self) -> bool:
        return not self.oportunidade.contem(0.0) and self.p_oportunidade < self.alfa

    @property
    def passa_regra_quatro_quintos(self) -> bool:
        return self.razao_impacto >= LIMIAR_REGRA_QUATRO_QUINTOS

    @property
    def veredito(self) -> str:
        if self.paridade_violada or self.oportunidade_violada:
            quais = []
            if self.paridade_violada:
                quais.append("paridade demografica")
            if self.oportunidade_violada:
                quais.append("igualdade de oportunidade")
            return f"REPROVADO: {' e '.join(quais)} violada(s) de forma significativa"
        return "APROVADO: nenhuma disparidade estatisticamente distinguivel de zero"

    def __str__(self) -> str:
        linhas = [f"== {self.versao} | atributo sensivel: {self.atributo} =="]
        linhas += [f"  {m}" for m in self.por_grupo.values()]
        rotulo = f"{self.grupo_a} - {self.grupo_b}"
        base = "comparaveis" if self.taxas_base_comparaveis else "DIFERENTES"
        linhas += [
            f"  taxa-base real  ({rotulo}): {self.diferenca_base_real}  -> {base}",
            f"  paridade demog. ({rotulo}): {self.paridade}  p={self.p_paridade:.4f}",
            f"  razao de impacto (menor/maior): {self.razao_impacto:.4f} "
            f"({'passa' if self.passa_regra_quatro_quintos else 'FALHA'} na regra dos 4/5)",
            f"  oportunidade    ({rotulo}): {self.oportunidade}  p={self.p_oportunidade:.4f}",
            f"  -> {self.veredito}",
        ]
        return "\n".join(linhas)


# ---------------------------------------------------------------- estatisticas


def metricas_por_grupo(
    grupo: np.ndarray, pred: np.ndarray, y: np.ndarray
) -> dict:
    """Taxa de aprovacao, taxa-base real e TPR de cada grupo."""
    grupo = np.asarray(grupo)
    pred = np.asarray(pred)
    y = np.asarray(y)
    resultado = {}
    for nome in sorted(set(grupo.tolist())):
        dentro = grupo == nome
        resultado[nome] = MetricasGrupo(
            grupo=str(nome),
            n=int(dentro.sum()),
            taxa_selecao=taxa_selecao(pred[dentro]),
            taxa_base_real=float(np.mean(y[dentro] == 1)),
            tpr=taxa_verdadeiro_positivo(y[dentro], pred[dentro]),
        )
    return resultado


def _taxa_no_grupo(grupo: np.ndarray, valores: np.ndarray, nome: str) -> float:
    dentro = grupo == nome
    if not dentro.any():
        return float("nan")
    return float(np.mean(valores[dentro]))


def diferenca_paridade(
    grupo: np.ndarray, pred: np.ndarray, grupo_a: str, grupo_b: str
) -> float:
    """``P(pred=1 | G=a) - P(pred=1 | G=b)``. Zero e paridade perfeita."""
    grupo = np.asarray(grupo)
    pred = np.asarray(pred)
    return _taxa_no_grupo(grupo, pred, grupo_a) - _taxa_no_grupo(grupo, pred, grupo_b)


def razao_impacto(grupo: np.ndarray, pred: np.ndarray) -> float:
    """Razao entre a menor e a maior taxa de aprovacao entre os grupos.

    Fica em [0, 1]; 1.0 e paridade perfeita. Se a maior taxa for zero (ninguem
    aprovado em lugar nenhum), a razao e indefinida.
    """
    grupo = np.asarray(grupo)
    pred = np.asarray(pred)
    taxas = [
        _taxa_no_grupo(grupo, pred, nome) for nome in sorted(set(grupo.tolist()))
    ]
    maior = max(taxas)
    if maior == 0:
        return float("nan")
    return float(min(taxas) / maior)


def diferenca_oportunidade(
    grupo: np.ndarray, pred: np.ndarray, y: np.ndarray, grupo_a: str, grupo_b: str
) -> float:
    """Diferenca de TPR entre os grupos: ``P(pred=1|y=1,a) - P(pred=1|y=1,b)``."""
    grupo = np.asarray(grupo)
    pred = np.asarray(pred)
    y = np.asarray(y)
    positivos = y == 1
    return _taxa_no_grupo(grupo[positivos], pred[positivos], grupo_a) - _taxa_no_grupo(
        grupo[positivos], pred[positivos], grupo_b
    )


# ------------------------------------------------------------------ auditoria


def auditar(
    grupo: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    versao: str = "modelo",
    atributo: str = "atributo sensivel",
    grupo_a: Optional[str] = None,
    grupo_b: Optional[str] = None,
    n_reamostras: int = N_REAMOSTRAS,
    semente: int = SEMENTE,
    alfa: float = ALFA,
) -> AuditoriaFairness:
    """Auditoria completa de uma versao de modelo sobre dois grupos.

    Quando ``grupo_a``/``grupo_b`` nao sao informados, usa os dois primeiros
    valores do atributo em ordem alfabetica -- explicitar e melhor, porque o
    SINAL de todas as diferencas depende de quem e A e quem e B.

    :raises ValueError: se algum dos grupos pedidos nao existir nos dados.
    """
    grupo = np.asarray(grupo)
    pred = np.asarray(pred)
    y = np.asarray(y)

    presentes = sorted(set(grupo.tolist()))
    if grupo_a is None or grupo_b is None:
        if len(presentes) < 2:
            raise ValueError("a auditoria precisa de pelo menos dois grupos")
        grupo_a, grupo_b = presentes[0], presentes[1]
    for nome in (grupo_a, grupo_b):
        if nome not in presentes:
            raise ValueError(f"grupo {nome!r} ausente; presentes: {presentes}")

    ic_base = ic_bootstrap(
        lambda grupo, y: diferenca_paridade(grupo, (y == 1).astype(int), grupo_a, grupo_b),
        {"grupo": grupo, "y": y},
        n_reamostras=n_reamostras,
        semente=semente,
    )
    ic_paridade = ic_bootstrap(
        lambda grupo, pred: diferenca_paridade(grupo, pred, grupo_a, grupo_b),
        {"grupo": grupo, "pred": pred},
        n_reamostras=n_reamostras,
        semente=semente,
    )
    ic_oportunidade = ic_bootstrap(
        lambda grupo, pred, y: diferenca_oportunidade(grupo, pred, y, grupo_a, grupo_b),
        {"grupo": grupo, "pred": pred, "y": y},
        n_reamostras=n_reamostras,
        semente=semente,
    )

    # Permutacao: sob H0 a decisao independe do grupo, entao embaralhar os
    # rotulos de grupo nao deveria mudar o tamanho da disparidade.
    p_paridade = teste_permutacao(
        lambda g: diferenca_paridade(g, pred, grupo_a, grupo_b),
        grupo,
        n_permutacoes=n_reamostras,
        semente=semente,
    )
    positivos = y == 1
    p_oportunidade = teste_permutacao(
        lambda g: diferenca_paridade(g, pred[positivos], grupo_a, grupo_b),
        grupo[positivos],
        n_permutacoes=n_reamostras,
        semente=semente,
    )

    return AuditoriaFairness(
        versao=versao,
        atributo=atributo,
        grupo_a=grupo_a,
        grupo_b=grupo_b,
        por_grupo=metricas_por_grupo(grupo, pred, y),
        diferenca_base_real=ic_base,
        paridade=ic_paridade,
        p_paridade=p_paridade,
        razao_impacto=razao_impacto(grupo, pred),
        oportunidade=ic_oportunidade,
        p_oportunidade=p_oportunidade,
        alfa=alfa,
    )
