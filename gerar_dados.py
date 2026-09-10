"""Gera o dataset sintetico de concessao de credito usado na ATV2.

Por que sintetico: o dataset precisa ter, ao mesmo tempo, (a) um atributo
sensivel, (b) o rotulo verdadeiro ``y_real`` e (c) as predicoes de DUAS versoes
de modelo sobre as MESMAS linhas -- so assim da para comparar versoes de forma
pareada e medir fairness. Alem disso, com dados sinteticos o mecanismo gerador
e conhecido, entao sabemos qual e a resposta certa que a suite deveria achar.

Mecanismo (documentado em RELATORIO.md, secao 1):

* ``solvencia`` e uma variavel latente N(0,1): e o risco real do cliente.
* ``score_bureau``, ``renda_mensal`` e ``divida_ativa`` sao consequencias ruidosas
  da solvencia -- sao os preditores legitimos.
* ``tempo_emprego_meses`` tambem depende da solvencia, mas leva um desconto fixo
  para o grupo F (interrupcao de carreira / trabalho nao remunerado). Ou seja:
  e um PROXY do atributo sensivel disfarcado de feature neutra.
* ``y_real`` (bom pagador) depende da solvencia e SO MARGINALMENTE do tempo de
  emprego. Nao ha termo de genero: por construcao, o risco real nao difere
  entre os grupos.
* O modelo v1 (o que foi para producao) pesa demais o proxy; o v2 e o mesmo
  modelo com o peso do proxy trazido de volta ao seu valor preditivo real.

Uso:
    python gerar_dados.py        # reescreve dados/credito.csv e dados/credito_corrompido.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

SEMENTE = 20260910
N_CLIENTES = 4000
LIMIAR_APROVACAO = 0.5

REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
PESO_REGIOES = [0.09, 0.27, 0.42, 0.14, 0.08]
FAIXAS = ["18-25", "26-40", "41-60", "60+"]
PESO_FAIXAS = [0.18, 0.40, 0.32, 0.10]

RAIZ = Path(__file__).resolve().parent
ARQUIVO_LIMPO = RAIZ / "dados" / "credito.csv"
ARQUIVO_CORROMPIDO = RAIZ / "dados" / "credito_corrompido.csv"

#: Desconto medio, em meses, aplicado ao tempo de emprego do grupo F.
#: E a unica porta de entrada do vies no dataset.
PENALIDADE_PROXY_MESES = 26.0


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _padronizar(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def gerar(n: int = N_CLIENTES, semente: int = SEMENTE) -> pd.DataFrame:
    """Constroi o dataset limpo, com predicoes de duas versoes de modelo."""
    rng = np.random.default_rng(semente)

    genero = rng.choice(["F", "M"], size=n, p=[0.49, 0.51])
    regiao = rng.choice(REGIOES, size=n, p=PESO_REGIOES)
    faixa_etaria = rng.choice(FAIXAS, size=n, p=PESO_FAIXAS)

    solvencia = rng.standard_normal(n)

    score_bureau = np.clip(
        np.round(620 + 95 * solvencia + 25 * rng.standard_normal(n)), 300, 1000
    ).astype(int)

    base_renda = {"18-25": 7.45, "26-40": 8.05, "41-60": 8.35, "60+": 8.10}
    mu_renda = np.array([base_renda[f] for f in faixa_etaria])
    renda_mensal = np.round(
        np.clip(
            np.exp(mu_renda + 0.30 * solvencia + 0.38 * rng.standard_normal(n)),
            1000.0,
            40000.0,
        ),
        2,
    )

    divida_ativa = rng.binomial(1, _sigmoide(-1.25 - 0.90 * solvencia))

    base_tempo = {"18-25": 22.0, "26-40": 70.0, "41-60": 150.0, "60+": 210.0}
    mu_tempo = np.array([base_tempo[f] for f in faixa_etaria])
    tempo_emprego_meses = np.clip(
        np.round(
            mu_tempo
            + 18 * solvencia
            + 30 * rng.standard_normal(n)
            - PENALIDADE_PROXY_MESES * (genero == "F")
        ),
        0,
        480,
    ).astype(int)

    z_bureau = _padronizar(score_bureau.astype(float))
    z_renda = _padronizar(np.log(renda_mensal))
    z_tempo = _padronizar(tempo_emprego_meses.astype(float))

    # Rotulo verdadeiro: bom pagador = 1. Sem termo de genero.
    logito_real = (
        0.30
        + 1.05 * z_bureau
        + 0.55 * z_renda
        - 0.85 * divida_ativa
        + 0.12 * z_tempo
    )
    y_real = rng.binomial(1, _sigmoide(logito_real))

    # Residuo de modelagem: o mesmo para v1 e v2 (e o mesmo modelo, retreinado),
    # o que deixa a comparacao pareada sensivel apenas a mudanca de peso.
    residuo = 0.30 * rng.standard_normal(n)

    logito_v1 = (
        0.30 + 1.00 * z_bureau + 0.50 * z_renda - 0.80 * divida_ativa
        + 1.15 * z_tempo + residuo
    )
    logito_v2 = (
        0.30 + 1.05 * z_bureau + 0.55 * z_renda - 0.85 * divida_ativa
        + 0.12 * z_tempo + residuo
    )
    score_v1 = np.round(_sigmoide(logito_v1), 6)
    score_v2 = np.round(_sigmoide(logito_v2), 6)

    return pd.DataFrame(
        {
            "id_cliente": [f"C{i:05d}" for i in range(1, n + 1)],
            "genero": genero,
            "regiao": regiao,
            "faixa_etaria": faixa_etaria,
            "renda_mensal": renda_mensal,
            "score_bureau": score_bureau,
            "tempo_emprego_meses": tempo_emprego_meses,
            "divida_ativa": divida_ativa,
            "y_real": y_real,
            "score_modelo_v1": score_v1,
            "pred_v1": (score_v1 >= LIMIAR_APROVACAO).astype(int),
            "score_modelo_v2": score_v2,
            "pred_v2": (score_v2 >= LIMIAR_APROVACAO).astype(int),
        }
    )


def corromper(limpo: pd.DataFrame) -> pd.DataFrame:
    """Planta defeitos conhecidos no dataset -- o analogo do 'codigo bugado' da ATV1.

    Cada defeito exercita uma das quatro familias de regra da validacao. Se a
    suite passar nesse arquivo, a suite e que esta quebrada.
    """
    sujo = limpo.copy().astype(object)

    # (1) SCHEMA: coluna obrigatoria ausente.
    sujo = sujo.drop(columns=["divida_ativa"])
    # (2) SCHEMA: coluna nao declarada no contrato (campo interno vazado).
    sujo["obs_internas"] = ""
    # (3) SCHEMA: valor nao parseavel como inteiro.
    sujo.loc[10, "score_bureau"] = "indisponivel"
    # (4) COMPLETUDE: atributo sensivel em branco.
    sujo.loc[[20, 21, 22], "genero"] = ""
    sujo.loc[30, "renda_mensal"] = ""
    # (5) FAIXA: numerico fora do intervalo plausivel.
    sujo.loc[40, "renda_mensal"] = -1500.0
    sujo.loc[41, "score_bureau"] = 1500
    sujo.loc[42, "score_modelo_v1"] = 1.7
    # (6) FAIXA/CATEGORIA: valor fora do dominio declarado.
    sujo.loc[50, "genero"] = "f"          # caixa divergente
    sujo.loc[51, "regiao"] = "Sudeste "   # espaco no fim
    sujo.loc[52, "pred_v1"] = 2           # predicao que nao e 0/1
    # (7) UNICIDADE: chave primaria repetida.
    sujo.loc[60, "id_cliente"] = sujo.loc[59, "id_cliente"]
    return sujo


def _escrever(df: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def main() -> None:
    limpo = gerar()
    _escrever(limpo, ARQUIVO_LIMPO)
    _escrever(corromper(limpo), ARQUIVO_CORROMPIDO)
    print(f"{ARQUIVO_LIMPO.name}: {len(limpo)} linhas x {limpo.shape[1]} colunas")
    print(f"{ARQUIVO_CORROMPIDO.name}: defeitos plantados nas 4 familias de regra")


if __name__ == "__main__":
    main()
