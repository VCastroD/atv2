# ATV2 — Validação de dados + testes estatísticos e de fairness

Suíte que audita um modelo de concessão de crédito combinando **validação de
dados** (schema, completude, faixa, unicidade) com **testes estatísticos**
(intervalo de confiança bootstrap e comparação entre versões) e **testes de
fairness** (paridade demográfica, regra dos 4/5, igualdade de oportunidade).

> **A interpretação completa está em [RELATORIO.md](RELATORIO.md)** — este README é
> só o mapa.

## O caso

4000 clientes, atributo sensível `genero` (F/M), rótulo `y_real` (bom pagador) e
duas versões de modelo avaliadas **nas mesmas linhas**. O dataset é sintético de
propósito: com o mecanismo gerador conhecido, dá para verificar se a suíte
encontra o que foi plantado, em vez de só confiar nela.

O gancho do caso: `tempo_emprego_meses` carrega um desconto fixo para o grupo F —
é um **proxy do atributo sensível disfarçado de feature neutra**. `y_real` não tem
termo de gênero, e **nenhuma das duas versões de modelo recebe a coluna `genero`**.

## Resultados

```
v1: acuracia=0.7107 | paridade=-0.0650 (p=0.0002) | razao de impacto=0.881 | REPROVADO
v2: acuracia=0.7320 | paridade=+0.0101 (p=0.5233) | razao de impacto=0.981 | APROVADO

v1 -> v2: diferenca significativa (B > A)
ganho de acuracia: +0.0212  IC95% [+0.0092, +0.0335]  p=7.97e-04
```

- A **v1 discrimina sem nunca ver o atributo sensível** — a falha do *fairness
  through unawareness*, visível só ao medir o resultado por grupo.
- A **regra dos 4/5 teria absolvido a v1** (0,881 > 0,8) enquanto o teste
  estatístico rejeita paridade. Limiar fixo não substitui teste.
- Corrigir o viés **rendeu** 2,1 p.p. de acurácia em vez de custar: o trade-off
  justiça × desempenho não é lei da natureza.

## Como rodar

```bash
pip install -r requirements.txt
python gerar_dados.py    # opcional: os CSVs já estão versionados (gerador determinístico)
python suite.py          # ~30 s, grava evidencias/01..05
python -m pytest -q      # 62 passed
```

## Mapa dos arquivos

| Arquivo | Papel |
|---|---|
| [`ATV2.MD`](ATV2.MD) | Enunciado da atividade |
| [`RELATORIO.md`](RELATORIO.md) | Relatório completo, com a interpretação dos resultados |
| [`gerar_dados.py`](gerar_dados.py) | Gera `dados/credito.csv` e a versão com defeitos plantados |
| [`validacao.py`](validacao.py) | Entregável 1 — schema, completude, faixa/domínio, unicidade |
| [`estatistica.py`](estatistica.py) | Entregável 2 — IC bootstrap e comparação v1 × v2 (McNemar exato) |
| [`fairness.py`](fairness.py) | Entregável 3 — paridade demográfica, regra dos 4/5, oportunidade igual |
| [`suite.py`](suite.py) | Roda tudo em ordem e grava as evidências |
| [`test_*.py`](.) | 62 testes: exemplos, bordas, calibração e 11 testes de propriedade |
| [`evidencias/`](evidencias/) | Saídas brutas de cada etapa e do `pytest` |
