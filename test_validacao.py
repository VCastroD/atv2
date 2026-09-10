"""Testes da validacao de dados: schema, completude, faixa e unicidade.

A prova de que a validacao vale alguma coisa e o par de arquivos
``dados/credito.csv`` (limpo, tem que passar) e ``dados/credito_corrompido.csv``
(defeitos plantados, tem que reprovar defeito por defeito) -- mesma ideia da
versao "bugada" da ATV1.
"""

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import validacao as V

RAIZ = Path(__file__).resolve().parent
LIMPO = RAIZ / "dados" / "credito.csv"
CORROMPIDO = RAIZ / "dados" / "credito_corrompido.csv"


@pytest.fixture(scope="module")
def bruto_limpo():
    return V.ler_bruto(LIMPO)


@pytest.fixture(scope="module")
def bruto_corrompido():
    return V.ler_bruto(CORROMPIDO)


@pytest.fixture(scope="module")
def relatorio_corrompido(bruto_corrompido):
    return V.validar(bruto_corrompido)


# ------------------------------------------------------- o arquivo limpo passa


def test_dataset_limpo_passa_em_todas_as_regras(bruto_limpo):
    relatorio = V.validar(bruto_limpo)
    assert relatorio.ok, str(relatorio)


def test_carregar_devolve_dataframe_tipado():
    df = V.carregar(LIMPO)
    assert len(df) == 4000
    assert list(df.columns) == [c.nome for c in V.ESQUEMA_CREDITO]
    assert df["score_bureau"].dtype.kind == "i"
    assert df["renda_mensal"].dtype.kind == "f"
    assert set(df["genero"].unique()) == {"F", "M"}


def test_carregar_recusa_dataset_corrompido():
    with pytest.raises(V.DadosInvalidosError) as erro:
        V.carregar(CORROMPIDO)
    assert not erro.value.relatorio.ok


# --------------------------------------- cada defeito plantado e mesmo pego


@pytest.mark.parametrize(
    "regra, coluna, trecho",
    [
        ("schema", "divida_ativa", "ausente"),
        ("schema", "obs_internas", "nao declarada"),
        ("schema", "score_bureau", "nao conversivel"),
        ("completude", "genero", "obrigatorio ausente"),
        ("completude", "renda_mensal", "obrigatorio ausente"),
        ("faixa", "genero", "fora do dominio"),
        ("faixa", "regiao", "fora do dominio"),
        ("faixa", "renda_mensal", "fora de"),
        ("faixa", "score_bureau", "fora de"),
        ("faixa", "score_modelo_v1", "fora de"),
        ("faixa", "pred_v1", "fora de"),
        ("unicidade", "id_cliente", "repetido"),
    ],
)
def test_defeito_plantado_e_reportado(relatorio_corrompido, regra, coluna, trecho):
    achadas = [
        v
        for v in relatorio_corrompido.por_regra(regra)
        if v.coluna == coluna and trecho in v.detalhe
    ]
    assert achadas, f"nenhuma violacao [{regra}] {coluna} ~ {trecho!r}"


def test_violacao_aponta_a_linha_do_csv(relatorio_corrompido):
    """As linhas sao as do arquivo (cabecalho = 1), nao o indice do DataFrame."""
    (dup,) = relatorio_corrompido.por_regra("unicidade")
    assert dup.linhas == (62,)  # id duplicado plantado no indice 60


def test_completude_conta_todas_as_ocorrencias(relatorio_corrompido):
    faltantes = [
        v for v in relatorio_corrompido.por_regra("completude") if v.coluna == "genero"
    ]
    assert faltantes[0].ocorrencias == 3


def test_celula_vazia_e_reportada_uma_unica_vez(relatorio_corrompido):
    """Decisao (2): vazio e falha de completude, e nao tambem de tipo/faixa.

    A linha 32 tem ``renda_mensal`` em branco: deve aparecer em completude e em
    nenhuma outra regra.
    """
    onde = {
        v.regra
        for v in relatorio_corrompido.violacoes
        if v.coluna == "renda_mensal" and 32 in v.linhas
    }
    assert onde == {"completude"}


# -------------------------------------------------- regras isoladas, em miniatura

ESQUEMA_MINI = (
    V.Coluna("chave", "texto", unica=True),
    V.Coluna("nota", "inteiro", minimo=0, maximo=10),
    V.Coluna("grupo", "texto", categorias=frozenset({"A", "B"})),
)


def _mini(chaves, notas, grupos):
    return pd.DataFrame(
        {"chave": list(chaves), "nota": [str(n) for n in notas], "grupo": list(grupos)}
    )


def test_schema_aceita_inteiro_serializado_como_decimal():
    """Decisao (4): '7.0' e um 7 escrito de outro jeito; '7.5' nao e inteiro."""
    ok = _mini(["a"], ["7.0"], ["A"])
    assert V.validar_schema(ok, ESQUEMA_MINI) == []
    ruim = _mini(["a"], ["7.5"], ["A"])
    assert len(V.validar_schema(ruim, ESQUEMA_MINI)) == 1


def test_dominio_categorico_e_sensivel_a_caixa_e_espaco():
    """Decisao (3): 'a' e 'A ' sao defeitos de ingestao, nao sinonimos."""
    for valor in ("a", "A ", "", "C"):
        quadro = _mini(["k"], [5], [valor])
        violacoes = V.validar_faixa(quadro, ESQUEMA_MINI) + V.validar_completude(
            quadro, ESQUEMA_MINI
        )
        assert violacoes, f"{valor!r} passou como categoria valida"


def test_coluna_opcional_pode_vir_vazia():
    esquema = (V.Coluna("obs", "texto", obrigatoria=False),)
    quadro = pd.DataFrame({"obs": ["", "algo", ""]})
    assert V.validar_completude(quadro, esquema) == []


def test_tipo_de_coluna_desconhecido_e_erro_de_programacao():
    with pytest.raises(ValueError):
        V.Coluna("x", "booleano")


# ---------------------------------------------------- testes de propriedade

nota_valida = st.integers(min_value=0, max_value=10)
nota_invalida = st.integers().filter(lambda n: n < 0 or n > 10)


@settings(max_examples=200)
@given(st.lists(nota_valida, min_size=1, max_size=30))
def test_propriedade_dado_dentro_do_contrato_nunca_viola(notas):
    """P1: qualquer lote que respeita o contrato passa em todas as regras."""
    quadro = _mini([f"k{i}" for i in range(len(notas))], notas, ["A"] * len(notas))
    assert V.validar(quadro, ESQUEMA_MINI).ok


@settings(max_examples=200)
@given(
    st.lists(nota_valida, min_size=1, max_size=30),
    nota_invalida,
    st.data(),
)
def test_propriedade_valor_fora_da_faixa_sempre_aparece(notas, intrusa, data):
    """P2: uma nota fora de [0, 10] em QUALQUER posicao e sempre reportada."""
    posicao = data.draw(st.integers(min_value=0, max_value=len(notas)))
    notas = list(notas)
    notas.insert(posicao, intrusa)
    quadro = _mini([f"k{i}" for i in range(len(notas))], notas, ["A"] * len(notas))

    violacoes = V.validar_faixa(quadro, ESQUEMA_MINI)
    assert len(violacoes) == 1
    assert violacoes[0].coluna == "nota"
    assert posicao + 2 in violacoes[0].linhas or violacoes[0].ocorrencias > V.MAX_EXEMPLOS


@settings(max_examples=200)
@given(st.lists(st.text(min_size=1, max_size=4), min_size=1, max_size=20))
def test_propriedade_unicidade_pega_qualquer_repeticao(chaves):
    """P3: repetir uma chave existente sempre acusa; sem repeticao, nunca acusa."""
    unicas = list(dict.fromkeys(chaves))
    n = len(unicas)
    limpo = _mini(unicas, [1] * n, ["A"] * n)
    assert V.validar_unicidade(limpo, ESQUEMA_MINI) == []

    repetido = _mini(unicas + [unicas[0]], [1] * (n + 1), ["A"] * (n + 1))
    violacoes = V.validar_unicidade(repetido, ESQUEMA_MINI)
    assert len(violacoes) == 1
    assert violacoes[0].linhas == (n + 2,)
