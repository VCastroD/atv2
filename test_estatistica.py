"""Testes do modulo estatistico: bootstrap, McNemar e comparacao de versoes.

Um teste estatistico so serve se ele erra pouco nos DOIS sentidos. Por isso a
suite tem os dois lados:

* sensibilidade -- diferenca real entre v1 e v2 tem que ser detectada;
* calibracao ("placebo") -- duas versoes que so diferem por ruido NAO podem ser
  declaradas diferentes.
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import estatistica as E
import validacao as V

RAIZ = Path(__file__).resolve().parent
RAPIDO = 400  # reamostras suficientes para os testes; a suite final usa 5000


@pytest.fixture(scope="module")
def dados():
    df = V.carregar(RAIZ / "dados" / "credito.csv")
    return {
        "y": df["y_real"].to_numpy(),
        "v1": df["pred_v1"].to_numpy(),
        "v2": df["pred_v2"].to_numpy(),
    }


# ----------------------------------------------------------------- metricas


def test_metricas_em_caso_conhecido():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    assert E.acuracia(y, pred) == 0.5
    assert E.taxa_selecao(pred) == 0.5
    assert E.taxa_verdadeiro_positivo(y, pred) == 0.5
    assert E.precisao(y, pred) == 0.5


def test_metricas_condicionais_indefinidas_viram_nan():
    """Sem positivo real nao existe TPR; sem aprovado nao existe precisao."""
    assert np.isnan(E.taxa_verdadeiro_positivo(np.zeros(3), np.ones(3)))
    assert np.isnan(E.precisao(np.ones(3), np.zeros(3)))


# ---------------------------------------------------------------- bootstrap


def test_ic_de_metrica_degenerada_colapsa_no_ponto():
    """Se todo mundo acerta, toda reamostra acerta: o intervalo vira um ponto."""
    y = np.ones(50, dtype=int)
    ic = E.ic_bootstrap(E.acuracia, {"y": y, "pred": y}, n_reamostras=RAPIDO)
    assert (ic.ponto, ic.inferior, ic.superior) == (1.0, 1.0, 1.0)
    assert ic.largura == 0.0


def test_ic_da_acuracia_no_dataset_real(dados):
    ic = E.ic_bootstrap(
        E.acuracia, {"y": dados["y"], "pred": dados["v1"]}, n_reamostras=RAPIDO
    )
    assert ic.contem(ic.ponto)
    assert 0.0 <= ic.inferior < ic.superior <= 1.0
    assert ic.largura < 0.05  # com n=4000 o intervalo e estreito


def test_ic_estreita_com_mais_dados():
    """Mais dados, menos incerteza -- a propriedade que justifica o bootstrap."""
    rng = np.random.default_rng(7)
    larguras = []
    for n in (100, 10_000):
        y = rng.integers(0, 2, n)
        pred = rng.integers(0, 2, n)
        larguras.append(
            E.ic_bootstrap(
                E.acuracia, {"y": y, "pred": pred}, n_reamostras=RAPIDO
            ).largura
        )
    assert larguras[0] > 3 * larguras[1]


def test_ic_reporta_reamostras_indefinidas():
    """Grupo com um unico positivo: parte das reamostras nao tem positivo algum."""
    y = np.array([1] + [0] * 29)
    pred = np.ones(30, dtype=int)
    ic = E.ic_bootstrap(
        E.taxa_verdadeiro_positivo, {"y": y, "pred": pred}, n_reamostras=RAPIDO
    )
    assert ic.descartadas > 0
    assert "indefinidas" in str(ic)


def test_bootstrap_recusa_entradas_inconsistentes():
    with pytest.raises(ValueError, match="comprimentos diferentes"):
        E.ic_bootstrap(E.acuracia, {"y": np.ones(3), "pred": np.ones(4)})
    with pytest.raises(ValueError, match="vazio"):
        E.ic_bootstrap(E.acuracia, {"y": np.array([]), "pred": np.array([])})


# ------------------------------------------------------------------ McNemar


@pytest.mark.parametrize(
    "b, c, esperado",
    [
        (0, 0, 1.0),           # nenhuma discordancia: nada a testar
        (5, 5, 1.0),           # discordancias simetricas: H0 intacta
        (1, 9, 2 * 11 / 1024), # calculado a mao: 2 * P(X<=1), X~Bin(10, 0.5)
        (0, 10, 2 / 1024),
    ],
)
def test_mcnemar_em_casos_calculados_a_mao(b, c, esperado):
    assert E.mcnemar_exato(b, c) == pytest.approx(esperado)


# ------------------------------------------------- comparacao entre versoes


def test_versoes_identicas_nao_sao_declaradas_diferentes(dados):
    """Placebo perfeito: comparar a v1 com ela mesma nao pode acusar nada."""
    resultado = E.comparar_versoes(
        dados["y"], dados["v1"], dados["v1"], n_reamostras=RAPIDO
    )
    assert resultado.diferenca.ponto == 0.0
    assert resultado.p_mcnemar == 1.0
    assert not resultado.significativa
    assert "ruido amostral" in resultado.veredito


def test_taxa_de_falso_alarme_fica_perto_de_alfa():
    """Placebo realista: 20 pares de versoes de qualidade IDENTICA (75% cada),
    diferindo so pelo ruido de cada uma.

    Um teste com alfa=0.05 pode acusar diferenca em ~5% desses pares -- e o preco
    combinado. O que ele nao pode e acusar em muito mais do que isso, que e o que
    aconteceria se o intervalo estivesse estreito demais (bootstrap nao pareado,
    por exemplo). Os 20 pares sao gerados com sementes fixas: o teste e
    deterministico, nao um sorteio a cada execucao.
    """
    falsos_alarmes = 0
    for semente in range(20):
        rng = np.random.default_rng(semente)
        n = 1500
        y = rng.integers(0, 2, n)
        pred_a = np.where(rng.random(n) < 0.75, y, 1 - y)
        pred_b = np.where(rng.random(n) < 0.75, y, 1 - y)
        if E.comparar_versoes(y, pred_a, pred_b, n_reamostras=200).significativa:
            falsos_alarmes += 1
    assert falsos_alarmes <= 3, f"{falsos_alarmes}/20 falsos alarmes: teste mal calibrado"


def test_diferenca_real_entre_v1_e_v2_e_detectada(dados):
    resultado = E.comparar_versoes(
        dados["y"], dados["v1"], dados["v2"], n_reamostras=RAPIDO
    )
    assert resultado.significativa
    assert resultado.diferenca.ponto > 0  # v2 acerta mais
    assert not resultado.diferenca.contem(0.0)
    assert resultado.p_mcnemar < E.ALFA


# ----------------------------------------------------------------- permutacao


def test_permutacao_nao_acusa_estatistica_independente_do_rotulo():
    rng = np.random.default_rng(3)
    valores = rng.random(500)
    grupo = np.array(["A"] * 250 + ["B"] * 250)
    p = E.teste_permutacao(
        lambda g: valores[g == "A"].mean() - valores[g == "B"].mean(),
        grupo,
        n_permutacoes=RAPIDO,
    )
    assert p > E.ALFA


def test_permutacao_acusa_dependencia_forte():
    valores = np.array([0.0] * 250 + [1.0] * 250)
    grupo = np.array(["A"] * 250 + ["B"] * 250)
    p = E.teste_permutacao(
        lambda g: valores[g == "A"].mean() - valores[g == "B"].mean(),
        grupo,
        n_permutacoes=RAPIDO,
    )
    assert p == pytest.approx(1 / (RAPIDO + 1))  # o minimo que B permutacoes permitem


# ---------------------------------------------------- testes de propriedade

binarios = st.lists(st.integers(min_value=0, max_value=1), min_size=3, max_size=60)


@settings(max_examples=50, deadline=None)
@given(binarios, st.data())
def test_propriedade_ponto_e_a_metrica_nos_dados_completos(y, data):
    """P1: o bootstrap nunca reestima o ponto -- ele so mede a incerteza dele."""
    pred = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=1), min_size=len(y), max_size=len(y)
        )
    )
    y, pred = np.array(y), np.array(pred)
    ic = E.ic_bootstrap(E.acuracia, {"y": y, "pred": pred}, n_reamostras=100)
    assert ic.ponto == E.acuracia(y, pred)
    assert 0.0 <= ic.inferior <= ic.superior <= 1.0


@settings(max_examples=50, deadline=None)
@given(binarios, st.data())
def test_propriedade_mesma_semente_mesmo_intervalo(y, data):
    """P2: reprodutibilidade -- resultado aleatorio nao pode ser irreprodutivel."""
    pred = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=1), min_size=len(y), max_size=len(y)
        )
    )
    dados = {"y": np.array(y), "pred": np.array(pred)}
    primeiro = E.ic_bootstrap(E.acuracia, dados, n_reamostras=100, semente=99)
    segundo = E.ic_bootstrap(E.acuracia, dados, n_reamostras=100, semente=99)
    assert primeiro == segundo


@settings(max_examples=300)
@given(st.integers(min_value=0, max_value=60), st.integers(min_value=0, max_value=60))
def test_propriedade_mcnemar_simetrico_e_valido(b, c):
    """P3: p-valor vive em (0, 1] e nao depende de qual versao chamamos de A."""
    p = E.mcnemar_exato(b, c)
    assert 0.0 < p <= 1.0
    assert p == pytest.approx(E.mcnemar_exato(c, b))
    if b == c:
        assert p == 1.0


@settings(max_examples=50, deadline=None)
@given(binarios, st.data())
def test_propriedade_comparar_versao_consigo_mesma_e_nula(y, data):
    """P4: nenhuma versao e diferente de si mesma, para qualquer entrada."""
    pred = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=1), min_size=len(y), max_size=len(y)
        )
    )
    y, pred = np.array(y), np.array(pred)
    resultado = E.comparar_versoes(y, pred, pred, n_reamostras=100)
    assert resultado.diferenca.ponto == 0.0
    assert resultado.so_a_acertou == resultado.so_b_acertou == 0
    assert resultado.p_mcnemar == 1.0
    assert not resultado.significativa
