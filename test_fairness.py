"""Testes da auditoria de fairness.

Vale a mesma exigencia dos dois lados do modulo estatistico:

* o auditor tem que REPROVAR um preditor com vies plantado (senao ele nao serve
  para nada -- e o analogo da "prova do bug" da ATV1);
* o auditor tem que APROVAR um preditor construido para ser cego ao grupo (senao
  ele acusa qualquer coisa e tambem nao serve para nada).
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import estatistica as E
import fairness as F
import validacao as V

RAIZ = Path(__file__).resolve().parent
RAPIDO = 400  # reamostras/permutacoes nos testes; a suite final usa 5000


@pytest.fixture(scope="module")
def dados():
    df = V.carregar(RAIZ / "dados" / "credito.csv")
    return {
        "grupo": df["genero"].to_numpy(),
        "y": df["y_real"].to_numpy(),
        "v1": df["pred_v1"].to_numpy(),
        "v2": df["pred_v2"].to_numpy(),
    }


def _mundo(n=2000, gap=0.0, semente=5):
    """Gera grupo/rotulo/predicao com uma disparidade de aprovacao de ``gap``.

    O rotulo verdadeiro nao depende do grupo: por construcao, qualquer gap na
    predicao e do preditor, nao do mundo.
    """
    rng = np.random.default_rng(semente)
    grupo = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y = rng.integers(0, 2, n)
    base = 0.5 + gap / 2
    taxa = np.where(grupo == "A", base, base - gap)
    pred = (rng.random(n) < taxa).astype(int)
    return grupo, pred, y


# ------------------------------------------------------- metricas descritivas


def test_metricas_por_grupo_em_caso_conhecido():
    grupo = np.array(["A", "A", "B", "B"])
    pred = np.array([1, 0, 1, 1])
    y = np.array([1, 1, 1, 0])
    metricas = F.metricas_por_grupo(grupo, pred, y)
    assert metricas["A"].n == 2
    assert metricas["A"].taxa_selecao == 0.5
    assert metricas["A"].taxa_base_real == 1.0
    assert metricas["A"].tpr == 0.5
    assert metricas["B"].taxa_selecao == 1.0


def test_paridade_perfeita_zera_a_diferenca_e_crava_a_razao_em_um():
    grupo = np.array(["A", "A", "B", "B"])
    pred = np.array([1, 0, 1, 0])
    assert F.diferenca_paridade(grupo, pred, "A", "B") == 0.0
    assert F.razao_impacto(grupo, pred) == 1.0


def test_razao_de_impacto_no_pior_caso():
    """Um grupo inteiro reprovado: razao zero, o piso da metrica."""
    grupo = np.array(["A", "A", "B", "B"])
    pred = np.array([1, 1, 0, 0])
    assert F.razao_impacto(grupo, pred) == 0.0


def test_razao_de_impacto_indefinida_sem_nenhuma_aprovacao():
    grupo = np.array(["A", "A", "B", "B"])
    assert np.isnan(F.razao_impacto(grupo, np.zeros(4, dtype=int)))


def test_oportunidade_condiciona_nos_positivos_reais():
    """So as linhas com y_real=1 entram no calculo do TPR."""
    grupo = np.array(["A", "A", "B", "B"])
    y = np.array([1, 0, 1, 0])
    pred = np.array([1, 0, 0, 1])  # os y=0 sao ruido para esta metrica
    assert F.diferenca_oportunidade(grupo, pred, y, "A", "B") == 1.0


# -------------------------------------- o auditor pega o vies e nao inventa vies


def test_auditor_reprova_preditor_com_vies_plantado():
    """Prova de que a auditoria pega um problema real: 20 p.p. de gap plantados."""
    grupo, pred, y = _mundo(gap=0.20)
    auditoria = F.auditar(grupo, pred, y, grupo_a="A", grupo_b="B", n_reamostras=RAPIDO)
    assert auditoria.paridade_violada
    assert auditoria.paridade.ponto == pytest.approx(0.20, abs=0.05)
    assert not auditoria.paridade.contem(0.0)
    assert auditoria.p_paridade < E.ALFA
    assert auditoria.razao_impacto < F.LIMIAR_REGRA_QUATRO_QUINTOS
    assert "REPROVADO" in auditoria.veredito


def test_auditor_aprova_preditor_cego_ao_grupo():
    """Sem vies plantado, a auditoria nao pode acusar disparidade."""
    grupo, pred, y = _mundo(gap=0.0)
    auditoria = F.auditar(grupo, pred, y, grupo_a="A", grupo_b="B", n_reamostras=RAPIDO)
    assert not auditoria.paridade_violada
    assert not auditoria.oportunidade_violada
    assert auditoria.paridade.contem(0.0)
    assert auditoria.razao_impacto > F.LIMIAR_REGRA_QUATRO_QUINTOS
    assert "APROVADO" in auditoria.veredito


def test_taxa_de_falso_alarme_da_auditoria_fica_perto_de_alfa():
    """20 mundos sem vies nenhum, sementes fixas: quase nenhum pode ser reprovado."""
    reprovados = 0
    for semente in range(20):
        grupo, pred, y = _mundo(n=800, gap=0.0, semente=semente)
        auditoria = F.auditar(
            grupo, pred, y, grupo_a="A", grupo_b="B", n_reamostras=150
        )
        reprovados += auditoria.paridade_violada
    assert reprovados <= 3, f"{reprovados}/20 falsos alarmes: auditoria mal calibrada"


# --------------------------------------------- o dataset da atividade, de fato


def test_v1_e_reprovada_no_dataset_da_atividade(dados):
    auditoria = F.auditar(
        dados["grupo"], dados["v1"], dados["y"],
        versao="v1", atributo="genero", grupo_a="F", grupo_b="M",
        n_reamostras=RAPIDO,
    )
    assert auditoria.taxas_base_comparaveis  # risco real igual: paridade e criterio valido
    assert auditoria.paridade_violada
    assert auditoria.oportunidade_violada
    assert auditoria.paridade.ponto < 0  # o grupo F e o prejudicado


def test_v2_e_aprovada_no_dataset_da_atividade(dados):
    auditoria = F.auditar(
        dados["grupo"], dados["v2"], dados["y"],
        versao="v2", atributo="genero", grupo_a="F", grupo_b="M",
        n_reamostras=RAPIDO,
    )
    assert not auditoria.paridade_violada
    assert not auditoria.oportunidade_violada
    assert auditoria.razao_impacto > F.LIMIAR_REGRA_QUATRO_QUINTOS


def test_regra_dos_quatro_quintos_pode_passar_com_paridade_violada(dados):
    """O achado central da atividade: limiar fixo e teste estatistico discordam.

    A razao de impacto da v1 (0.88) passa na regra dos 4/5, mas o intervalo de
    confianca da diferenca exclui zero. Limiar de auditoria nao substitui teste.
    """
    auditoria = F.auditar(
        dados["grupo"], dados["v1"], dados["y"],
        grupo_a="F", grupo_b="M", n_reamostras=RAPIDO,
    )
    assert auditoria.passa_regra_quatro_quintos
    assert auditoria.paridade_violada


def test_auditar_recusa_grupo_inexistente(dados):
    with pytest.raises(ValueError, match="ausente"):
        F.auditar(dados["grupo"], dados["v1"], dados["y"], grupo_a="F", grupo_b="X")


def test_auditar_recusa_atributo_de_um_grupo_so():
    with pytest.raises(ValueError, match="pelo menos dois grupos"):
        F.auditar(np.array(["A"] * 10), np.ones(10, dtype=int), np.ones(10, dtype=int))


# ---------------------------------------------------- testes de propriedade

lote_binario = st.lists(st.integers(min_value=0, max_value=1), min_size=4, max_size=60)


@settings(max_examples=200)
@given(lote_binario, st.data())
def test_propriedade_paridade_e_antissimetrica(pred, data):
    """P1: trocar quem e A por quem e B apenas troca o sinal da diferenca."""
    grupo = np.array(
        data.draw(st.lists(st.sampled_from(["A", "B"]), min_size=len(pred), max_size=len(pred)))
    )
    pred = np.array(pred)
    assume(len(set(grupo.tolist())) == 2)
    ida = F.diferenca_paridade(grupo, pred, "A", "B")
    volta = F.diferenca_paridade(grupo, pred, "B", "A")
    assert ida == pytest.approx(-volta)


@settings(max_examples=200)
@given(lote_binario, st.data())
def test_propriedade_razao_de_impacto_vive_no_intervalo_unitario(pred, data):
    """P2: a razao menor/maior nunca escapa de [0, 1], para qualquer entrada."""
    grupo = np.array(
        data.draw(st.lists(st.sampled_from(["A", "B"]), min_size=len(pred), max_size=len(pred)))
    )
    razao = F.razao_impacto(grupo, np.array(pred))
    assert np.isnan(razao) or 0.0 <= razao <= 1.0


@settings(max_examples=200)
@given(lote_binario, st.data())
def test_propriedade_renomear_grupos_nao_muda_a_disparidade(pred, data):
    """P3: a metrica mede a disparidade, nao o nome do grupo."""
    grupo = np.array(
        data.draw(st.lists(st.sampled_from(["A", "B"]), min_size=len(pred), max_size=len(pred)))
    )
    pred = np.array(pred)
    renomeado = np.where(grupo == "A", "alfa", "beta")
    assert F.diferenca_paridade(grupo, pred, "A", "B") == pytest.approx(
        F.diferenca_paridade(renomeado, pred, "alfa", "beta"), nan_ok=True
    )
    assert F.razao_impacto(grupo, pred) == pytest.approx(
        F.razao_impacto(renomeado, pred), nan_ok=True
    )


@settings(max_examples=100)
@given(st.integers(min_value=2, max_value=40))
def test_propriedade_decisao_uniforme_nunca_e_disparidade(k):
    """P4: aprovar todo mundo (ou reprovar todo mundo) e sempre paridade perfeita."""
    grupo = np.array(["A"] * k + ["B"] * k)
    for constante in (0, 1):
        pred = np.full(2 * k, constante)
        assert F.diferenca_paridade(grupo, pred, "A", "B") == 0.0
        razao = F.razao_impacto(grupo, pred)
        assert razao == 1.0 or np.isnan(razao)
