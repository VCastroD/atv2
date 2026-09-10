"""Validacao de dados: schema, completude, faixa/dominio e unicidade.

O contrato do dataset e declarado como dado (:data:`ESQUEMA_CREDITO`), nao como
codigo espalhado: cada coluna diz o proprio tipo, obrigatoriedade, faixa
numerica, dominio categorico e se e chave. As quatro familias de regra leem esse
contrato, entao acrescentar uma coluna e acrescentar uma linha na tupla.

Decisoes de projeto (justificadas em RELATORIO.md, secao 2):

1. O CSV e lido como TEXTO PURO (``dtype=str``, sem NA automatico). Se deixassemos
   o pandas inferir, uma unica celula ``"indisponivel"`` transformaria a coluna
   inteira em ``object`` -- ou, pior, viraria ``NaN`` e o erro de tipo se
   disfarcaria de valor faltante. Lendo como texto, a checagem de tipo e por
   celula e o defeito aparece exatamente onde ele esta.
2. Uma celula vazia e reportada UMA vez, como falha de completude. Ela nao e
   recontada como erro de tipo nem de faixa: um defeito, uma violacao.
3. Dominios categoricos sao sensiveis a caixa e a espaco. ``"f"`` e ``"Sudeste "``
   sao defeitos tipicos de ingestao, nao sinonimos aceitaveis.
4. ``"1.0"`` e aceito onde se espera inteiro (produtores serializam de formas
   diferentes); ``"1.5"`` nao e.
5. As linhas sao reportadas na numeracao do arquivo CSV (cabecalho = linha 1),
   que e a numeracao que a pessoa ve ao abrir o arquivo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

__all__ = [
    "Coluna",
    "Violacao",
    "Relatorio",
    "DadosInvalidosError",
    "ESQUEMA_CREDITO",
    "ler_bruto",
    "validar",
    "validar_schema",
    "validar_completude",
    "validar_faixa",
    "validar_unicidade",
    "converter",
    "carregar",
]

#: Quantas linhas de exemplo cada violacao carrega no relatorio.
MAX_EXEMPLOS = 5

#: Deslocamento entre o indice do DataFrame (0-based) e a linha do CSV.
_OFFSET_CSV = 2


class DadosInvalidosError(ValueError):
    """Erguido por :func:`carregar` quando o dataset viola o contrato."""

    def __init__(self, relatorio: "Relatorio") -> None:
        super().__init__(f"{len(relatorio.violacoes)} violacao(oes) de contrato")
        self.relatorio = relatorio


@dataclass(frozen=True)
class Coluna:
    """Contrato de uma coluna do dataset."""

    nome: str
    tipo: str  # "inteiro" | "decimal" | "texto"
    obrigatoria: bool = True
    minimo: Optional[float] = None
    maximo: Optional[float] = None
    categorias: Optional[frozenset] = None
    unica: bool = False

    def __post_init__(self) -> None:
        if self.tipo not in {"inteiro", "decimal", "texto"}:
            raise ValueError(f"tipo desconhecido em {self.nome!r}: {self.tipo!r}")


@dataclass(frozen=True)
class Violacao:
    """Uma regra quebrada: qual, onde e quantas vezes."""

    regra: str  # "schema" | "completude" | "faixa" | "unicidade"
    coluna: str
    detalhe: str
    ocorrencias: int = 1
    linhas: tuple = ()

    def __str__(self) -> str:
        quantas = f" [{self.ocorrencias}x]" if self.ocorrencias > 1 else ""
        onde = ""
        if self.linhas:
            reticencias = "..." if self.ocorrencias > len(self.linhas) else ""
            onde = f" (linha {', '.join(str(n) for n in self.linhas)}{reticencias})"
        return f"[{self.regra}] {self.coluna}: {self.detalhe}{quantas}{onde}"


@dataclass
class Relatorio:
    """Resultado da validacao: lista vazia significa dataset aprovado."""

    violacoes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violacoes

    def por_regra(self, regra: str) -> list:
        return [v for v in self.violacoes if v.regra == regra]

    def __str__(self) -> str:
        if self.ok:
            return "VALIDACAO OK - nenhuma violacao de contrato"
        linhas = [f"VALIDACAO FALHOU - {len(self.violacoes)} violacao(oes)"]
        linhas += [f"  {v}" for v in self.violacoes]
        return "\n".join(linhas)


# ------------------------------------------------------------------ contrato

ESQUEMA_CREDITO: tuple = (
    Coluna("id_cliente", "texto", unica=True),
    Coluna("genero", "texto", categorias=frozenset({"F", "M"})),
    Coluna(
        "regiao",
        "texto",
        categorias=frozenset({"Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"}),
    ),
    Coluna(
        "faixa_etaria",
        "texto",
        categorias=frozenset({"18-25", "26-40", "41-60", "60+"}),
    ),
    Coluna("renda_mensal", "decimal", minimo=1000.0, maximo=40000.0),
    Coluna("score_bureau", "inteiro", minimo=300, maximo=1000),
    Coluna("tempo_emprego_meses", "inteiro", minimo=0, maximo=480),
    Coluna("divida_ativa", "inteiro", minimo=0, maximo=1),
    Coluna("y_real", "inteiro", minimo=0, maximo=1),
    Coluna("score_modelo_v1", "decimal", minimo=0.0, maximo=1.0),
    Coluna("pred_v1", "inteiro", minimo=0, maximo=1),
    Coluna("score_modelo_v2", "decimal", minimo=0.0, maximo=1.0),
    Coluna("pred_v2", "inteiro", minimo=0, maximo=1),
)


# -------------------------------------------------------------------- leitura


def ler_bruto(caminho) -> pd.DataFrame:
    """Le o CSV sem nenhuma inferencia de tipo: tudo texto, nenhum ``NaN``.

    E a leitura que uma etapa de ingestao deveria fazer antes de confiar no
    arquivo -- ver decisao (1) no topo do modulo.
    """
    return pd.read_csv(Path(caminho), dtype=str, keep_default_na=False, na_filter=False)


# --------------------------------------------------------------------- regras


def _vazio(valor) -> bool:
    return valor is None or str(valor).strip() == ""


def _parseavel(valor, tipo: str) -> bool:
    """O texto da celula representa um valor do tipo declarado?"""
    if tipo == "texto":
        return True
    try:
        numero = float(str(valor).strip())
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numero):
        return False
    if tipo == "inteiro":
        return numero.is_integer()  # decisao (4)
    return True


def _violacao(regra: str, coluna: str, detalhe: str, indices: list) -> Violacao:
    linhas = tuple(int(i) + _OFFSET_CSV for i in indices[:MAX_EXEMPLOS])
    return Violacao(regra, coluna, detalhe, len(indices), linhas)


def validar_schema(
    bruto: pd.DataFrame, esquema: Sequence[Coluna] = ESQUEMA_CREDITO
) -> list:
    """Colunas esperadas presentes, nenhuma coluna a mais, valores parseaveis."""
    violacoes: list = []
    presentes = list(bruto.columns)
    declaradas = {c.nome for c in esquema}

    for coluna in esquema:
        if coluna.nome not in presentes:
            violacoes.append(Violacao("schema", coluna.nome, "coluna obrigatoria ausente"))
    for nome in presentes:
        if nome not in declaradas:
            violacoes.append(Violacao("schema", nome, "coluna nao declarada no contrato"))

    for coluna in esquema:
        if coluna.nome not in presentes or coluna.tipo == "texto":
            continue
        serie = bruto[coluna.nome]
        # Celula vazia e problema de completude, nao de tipo -- decisao (2).
        ruins = [
            i for i, v in enumerate(serie) if not _vazio(v) and not _parseavel(v, coluna.tipo)
        ]
        if ruins:
            violacoes.append(
                _violacao(
                    "schema",
                    coluna.nome,
                    f"valor nao conversivel para {coluna.tipo} "
                    f"(ex.: {serie.iloc[ruins[0]]!r})",
                    ruins,
                )
            )
    return violacoes


def validar_completude(
    bruto: pd.DataFrame, esquema: Sequence[Coluna] = ESQUEMA_CREDITO
) -> list:
    """Nenhuma celula vazia nas colunas declaradas obrigatorias."""
    violacoes: list = []
    for coluna in esquema:
        if coluna.nome not in bruto.columns or not coluna.obrigatoria:
            continue
        faltantes = [i for i, v in enumerate(bruto[coluna.nome]) if _vazio(v)]
        if faltantes:
            violacoes.append(
                _violacao("completude", coluna.nome, "valor obrigatorio ausente", faltantes)
            )
    return violacoes


def validar_faixa(
    bruto: pd.DataFrame, esquema: Sequence[Coluna] = ESQUEMA_CREDITO
) -> list:
    """Numericos dentro de [minimo, maximo] e categoricos dentro do dominio."""
    violacoes: list = []
    for coluna in esquema:
        if coluna.nome not in bruto.columns:
            continue
        serie = bruto[coluna.nome]

        if coluna.categorias is not None:
            fora = [
                i
                for i, v in enumerate(serie)
                if not _vazio(v) and v not in coluna.categorias
            ]
            if fora:
                violacoes.append(
                    _violacao(
                        "faixa",
                        coluna.nome,
                        f"valor fora do dominio {sorted(coluna.categorias)} "
                        f"(ex.: {serie.iloc[fora[0]]!r})",
                        fora,
                    )
                )

        if coluna.minimo is None and coluna.maximo is None:
            continue
        fora = []
        for i, v in enumerate(serie):
            if _vazio(v) or not _parseavel(v, coluna.tipo):
                continue  # ja reportado por completude/schema -- decisao (2)
            numero = float(v)
            abaixo = coluna.minimo is not None and numero < coluna.minimo
            acima = coluna.maximo is not None and numero > coluna.maximo
            if abaixo or acima:
                fora.append(i)
        if fora:
            violacoes.append(
                _violacao(
                    "faixa",
                    coluna.nome,
                    f"valor fora de [{coluna.minimo}, {coluna.maximo}] "
                    f"(ex.: {serie.iloc[fora[0]]!r})",
                    fora,
                )
            )
    return violacoes


def validar_unicidade(
    bruto: pd.DataFrame, esquema: Sequence[Coluna] = ESQUEMA_CREDITO
) -> list:
    """Colunas marcadas como chave nao podem repetir valor."""
    violacoes: list = []
    for coluna in esquema:
        if coluna.nome not in bruto.columns or not coluna.unica:
            continue
        serie = bruto[coluna.nome]
        repetidas = [i for i, dup in enumerate(serie.duplicated(keep="first")) if dup]
        if repetidas:
            violacoes.append(
                _violacao(
                    "unicidade",
                    coluna.nome,
                    f"valor repetido (ex.: {serie.iloc[repetidas[0]]!r})",
                    repetidas,
                )
            )
    return violacoes


def validar(bruto: pd.DataFrame, esquema: Sequence[Coluna] = ESQUEMA_CREDITO) -> Relatorio:
    """Roda as quatro familias de regra e devolve o relatorio consolidado."""
    return Relatorio(
        validar_schema(bruto, esquema)
        + validar_completude(bruto, esquema)
        + validar_faixa(bruto, esquema)
        + validar_unicidade(bruto, esquema)
    )


# ------------------------------------------------------------------ conversao


def converter(
    bruto: pd.DataFrame, esquema: Sequence[Coluna] = ESQUEMA_CREDITO
) -> pd.DataFrame:
    """Aplica os tipos do contrato. So faz sentido depois de :func:`validar`."""
    tipado = {}
    for coluna in esquema:
        serie = bruto[coluna.nome]
        if coluna.tipo == "inteiro":
            tipado[coluna.nome] = serie.astype(float).astype(int)
        elif coluna.tipo == "decimal":
            tipado[coluna.nome] = serie.astype(float)
        else:
            tipado[coluna.nome] = serie.astype(str)
    return pd.DataFrame(tipado)


def carregar(caminho, esquema: Sequence[Coluna] = ESQUEMA_CREDITO) -> pd.DataFrame:
    """Le, valida e tipa. Erguer e o comportamento certo: dado invalido nao passa.

    :raises DadosInvalidosError: se qualquer regra do contrato for violada.
    """
    bruto = ler_bruto(caminho)
    relatorio = validar(bruto, esquema)
    if not relatorio.ok:
        raise DadosInvalidosError(relatorio)
    return converter(bruto, esquema)
