"""Roda a suite completa sobre o dataset e grava as evidencias.

    python suite.py

Ordem das etapas -- e a ordem importa:

1. **Validacao** do arquivo limpo e do arquivo corrompido. A validacao vem antes
   de tudo porque metrica calculada sobre dado invalido e numero bonito e errado:
   uma coluna ``genero`` com celulas em branco silenciosamente encolhe um grupo e
   enviesa toda a analise de fairness que vier depois.
2. **Estatistica**: intervalos de confianca das metricas e comparacao v1 x v2.
3. **Fairness**: auditoria das duas versoes sobre ``genero`` (atributo sensivel) e
   um controle negativo sobre ``regiao``.

A leitura dos resultados esta em RELATORIO.md, secao 5.
"""

from __future__ import annotations

from pathlib import Path

import estatistica as E
import fairness as F
import validacao as V

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dados"
EVIDENCIAS = RAIZ / "evidencias"

ATRIBUTO_SENSIVEL = "genero"
GRUPO_A, GRUPO_B = "F", "M"


class Secao:
    """Acumula as linhas de uma etapa, imprime na tela e grava a evidencia."""

    def __init__(self, titulo: str, arquivo: str) -> None:
        self.titulo = titulo
        self.arquivo = EVIDENCIAS / arquivo
        self.linhas: list = []

    def __enter__(self) -> "Secao":
        self.escrever("=" * 78)
        self.escrever(self.titulo)
        self.escrever("=" * 78)
        return self

    def escrever(self, texto: str = "") -> None:
        print(texto)
        self.linhas.append(texto)

    def __exit__(self, *_) -> None:
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        self.arquivo.write_text("\n".join(self.linhas) + "\n", encoding="utf-8")
        print()


# ----------------------------------------------------------------- 1. validacao


def etapa_validacao():
    with Secao(
        "1. VALIDACAO DE DADOS - arquivo de producao (dados/credito.csv)",
        "01_validacao_limpo.txt",
    ) as secao:
        bruto = V.ler_bruto(DADOS / "credito.csv")
        secao.escrever(f"linhas: {len(bruto)}   colunas: {len(bruto.columns)}")
        secao.escrever(f"contrato: {len(V.ESQUEMA_CREDITO)} colunas declaradas")
        secao.escrever("")
        for coluna in V.ESQUEMA_CREDITO:
            restricao = []
            if coluna.minimo is not None or coluna.maximo is not None:
                restricao.append(f"faixa [{coluna.minimo}, {coluna.maximo}]")
            if coluna.categorias:
                restricao.append(f"dominio {sorted(coluna.categorias)}")
            if coluna.unica:
                restricao.append("unica")
            secao.escrever(
                f"  {coluna.nome:<22} {coluna.tipo:<8} "
                f"{'obrigatoria' if coluna.obrigatoria else 'opcional':<12} "
                f"{'; '.join(restricao)}"
            )
        secao.escrever("")
        secao.escrever(str(V.validar(bruto)))

    with Secao(
        "2. VALIDACAO DE DADOS - arquivo com defeitos plantados "
        "(dados/credito_corrompido.csv)",
        "02_validacao_corrompido.txt",
    ) as secao:
        secao.escrever(
            "Controle positivo da validacao: se a suite passar aqui, a suite e que"
        )
        secao.escrever("esta quebrada. Sao 12 defeitos, das 4 familias de regra.")
        secao.escrever("")
        relatorio = V.validar(V.ler_bruto(DADOS / "credito_corrompido.csv"))
        secao.escrever(str(relatorio))
        secao.escrever("")
        for regra in ("schema", "completude", "faixa", "unicidade"):
            secao.escrever(f"  {regra:<12}: {len(relatorio.por_regra(regra))} violacao(oes)")
        secao.escrever("")
        try:
            V.carregar(DADOS / "credito_corrompido.csv")
        except V.DadosInvalidosError as erro:
            secao.escrever(f"carregar() abortou o pipeline: {type(erro).__name__}: {erro}")

    return V.carregar(DADOS / "credito.csv")


# --------------------------------------------------------------- 2. estatistica


def etapa_estatistica(df):
    y = df["y_real"].to_numpy()
    v1 = df["pred_v1"].to_numpy()
    v2 = df["pred_v2"].to_numpy()

    with Secao(
        "3. TESTE ESTATISTICO - intervalos de confianca (bootstrap) e comparacao",
        "03_estatistica.txt",
    ) as secao:
        secao.escrever(
            f"bootstrap percentil, {E.N_REAMOSTRAS} reamostras, "
            f"confianca {int((1 - E.ALFA) * 100)}%, semente {E.SEMENTE}"
        )
        secao.escrever("")
        secao.escrever("-- Intervalos de confianca por versao " + "-" * 39)
        for nome, pred in (("v1", v1), ("v2", v2)):
            for rotulo, metrica, dados in (
                ("acuracia     ", E.acuracia, {"y": y, "pred": pred}),
                ("precisao     ", E.precisao, {"y": y, "pred": pred}),
                ("taxa aprovac.", E.taxa_selecao, {"pred": pred}),
                ("TPR (recall) ", E.taxa_verdadeiro_positivo, {"y": y, "pred": pred}),
            ):
                ic = E.ic_bootstrap(metrica, dados)
                secao.escrever(f"  {nome} {rotulo}: {ic}")
            secao.escrever("")

        secao.escrever("-- v1 x v2: a diferenca de acuracia e real? " + "-" * 32)
        comparacao = E.comparar_versoes(y, v1, v2)
        secao.escrever("  (A = v1, B = v2, avaliadas nas MESMAS 4000 linhas)")
        secao.escrever(str(comparacao))
        secao.escrever("")
        secao.escrever("-- Controle: v1 comparada com ela mesma " + "-" * 36)
        placebo = E.comparar_versoes(y, v1, v1)
        secao.escrever(
            f"  diferenca={placebo.diferenca.ponto:+.4f}  p={placebo.p_mcnemar:.4f}"
            f"  -> {placebo.veredito}"
        )

    return comparacao


# ------------------------------------------------------------------ 3. fairness


def etapa_fairness(df):
    y = df["y_real"].to_numpy()
    genero = df[ATRIBUTO_SENSIVEL].to_numpy()

    with Secao(
        "4. TESTE DE FAIRNESS - paridade demografica e igualdade de oportunidade",
        "04_fairness.txt",
    ) as secao:
        secao.escrever(
            f"bootstrap + permutacao, {E.N_REAMOSTRAS} repeticoes, alfa={E.ALFA}"
        )
        secao.escrever(
            f"diferencas no sentido ({GRUPO_A} - {GRUPO_B}): negativo = {GRUPO_A} prejudicado"
        )
        secao.escrever("")
        auditorias = {}
        for versao in ("v1", "v2"):
            auditorias[versao] = F.auditar(
                genero,
                df[f"pred_{versao}"].to_numpy(),
                y,
                versao=f"modelo {versao}",
                atributo=ATRIBUTO_SENSIVEL,
                grupo_a=GRUPO_A,
                grupo_b=GRUPO_B,
            )
            secao.escrever(str(auditorias[versao]))
            secao.escrever("")

        secao.escrever("-- Controle negativo: atributo nao usado como proxy " + "-" * 24)
        secao.escrever(
            "  A regiao nao entra no modelo. Se a auditoria acusasse disparidade"
        )
        secao.escrever("  aqui, ela estaria acusando ruido como se fosse vies.")
        secao.escrever("")
        controle = F.auditar(
            df["regiao"].to_numpy(),
            df["pred_v1"].to_numpy(),
            y,
            versao="modelo v1",
            atributo="regiao",
            grupo_a="Norte",
            grupo_b="Sudeste",
        )
        secao.escrever(str(controle))

    return auditorias


# ----------------------------------------------------------------------- painel


def painel(comparacao, auditorias):
    with Secao("5. PAINEL FINAL", "05_painel.txt") as secao:
        for versao, auditoria in auditorias.items():
            metrica = comparacao.metrica_a if versao == "v1" else comparacao.metrica_b
            secao.escrever(
                f"  {versao}: acuracia={metrica:.4f} | "
                f"paridade={auditoria.paridade.ponto:+.4f} "
                f"(p={auditoria.p_paridade:.4f}) | "
                f"razao de impacto={auditoria.razao_impacto:.3f} | "
                f"{'REPROVADO' if auditoria.paridade_violada or auditoria.oportunidade_violada else 'APROVADO'}"
            )
        secao.escrever("")
        secao.escrever(f"  v1 -> v2: {comparacao.veredito}")
        secao.escrever(
            f"  ganho de acuracia: {comparacao.diferenca}  p={comparacao.p_mcnemar:.2e}"
        )
        secao.escrever("")
        secao.escrever(f"  evidencias gravadas em {EVIDENCIAS}")


def main() -> None:
    df = etapa_validacao()
    comparacao = etapa_estatistica(df)
    auditorias = etapa_fairness(df)
    painel(comparacao, auditorias)


if __name__ == "__main__":
    main()
