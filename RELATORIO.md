# ATV2 — Suíte de validação de dados + testes estatísticos e de fairness

**Caso analisado:** concessão de crédito. Um modelo de risco (`v1`) que já está em
produção, uma versão corrigida (`v2`), e a pergunta que a suíte responde:
*a v1 trata os grupos de forma diferente, e a v2 é mesmo melhor ou é só sorte da
amostra?*

**Atributo sensível:** `genero` (grupos `F` e `M`), com `regiao` usada como
controle negativo. **Rótulo:** `y_real` (1 = bom pagador). **Predições:**
`pred_v1` e `pred_v2` (1 = crédito aprovado), sobre as **mesmas 4000 linhas**.

| Arquivo | Papel |
|---|---|
| [`gerar_dados.py`](gerar_dados.py) | Gera `dados/credito.csv` e a versão com defeitos plantados |
| [`validacao.py`](validacao.py) | Entregável 1 — schema, completude, faixa/domínio, unicidade |
| [`estatistica.py`](estatistica.py) | Entregável 2 — IC bootstrap e comparação v1 × v2 (McNemar) |
| [`fairness.py`](fairness.py) | Entregável 3 — paridade demográfica, regra dos 4/5, oportunidade igual |
| [`suite.py`](suite.py) | Roda tudo em ordem e grava as evidências |
| [`test_validacao.py`](test_validacao.py) · [`test_estatistica.py`](test_estatistica.py) · [`test_fairness.py`](test_fairness.py) | 62 testes (exemplos, bordas, calibração e 11 testes de propriedade) |
| [`evidencias/`](evidencias/) | Saídas brutas de cada etapa e do `pytest` |

```bash
pip install -r requirements.txt
python gerar_dados.py    # opcional: o CSV já está versionado
python suite.py          # ~30 s, grava evidencias/01..05
python -m pytest -q      # 62 passed
```

---

## 1. O dataset e por que ele é sintético

O dataset precisa de três coisas ao mesmo tempo: um atributo sensível, o rótulo
verdadeiro **e** as predições de duas versões de modelo sobre as **mesmas linhas**
(sem isso não existe comparação pareada). Nenhum dataset público que eu tinha à
mão entrega as três, e há uma vantagem decisiva no sintético: **o mecanismo
gerador é conhecido**, então sabemos qual é a resposta certa e dá para verificar
se a suíte a encontra — em vez de só confiar nela.

O mecanismo ([`gerar_dados.py`](gerar_dados.py), semente `20260910`, n = 4000):

- `solvencia` é uma variável latente `N(0,1)` — o risco real do cliente.
- `score_bureau`, `renda_mensal` e `divida_ativa` são consequências ruidosas da
  solvência: são os preditores legítimos.
- `tempo_emprego_meses` também depende da solvência, **mas leva um desconto fixo
  de 26 meses para o grupo F** (interrupção de carreira / trabalho não
  remunerado). Na amostra isso vira 77,1 meses de média em F contra 98,7 em M.
  É um **proxy do atributo sensível disfarçado de feature neutra**.
- `y_real` depende da solvência e só marginalmente do tempo de emprego. **Não há
  termo de gênero:** por construção, o risco real é o mesmo nos dois grupos.
- `v1` é um modelo logístico que pesa o proxy muito acima do valor preditivo dele
  (peso 1,15 contra os 0,12 reais); `v2` é o mesmo modelo com esse peso trazido de
  volta ao valor real. **Nenhuma das duas versões usa a coluna `genero`** — as
  duas são "cegas ao gênero" no sentido literal.

Note o que isso monta: um modelo que nunca vê o atributo sensível e ainda assim
pode discriminar através dele. É esse cenário que a suíte precisa detectar.

---

## 2. Entregável 1 — Validação de dados

O contrato é declarado como **dado**, não espalhado pelo código
([`ESQUEMA_CREDITO`](validacao.py)): cada coluna diz o próprio tipo,
obrigatoriedade, faixa numérica, domínio categórico e se é chave. Quatro famílias
de regra leem esse contrato:

| Regra | O que verifica | Exemplo no contrato |
|---|---|---|
| **schema** | colunas esperadas presentes, nenhuma coluna a mais, valores conversíveis para o tipo declarado | `score_bureau` é `inteiro` |
| **completude** | nenhuma célula vazia em coluna obrigatória | `genero` não pode faltar |
| **faixa** | numéricos em `[mínimo, máximo]`, categóricos dentro do domínio | `score_modelo_v1 ∈ [0,1]`, `regiao ∈ {Norte, …}` |
| **unicidade** | colunas-chave não repetem | `id_cliente` |

### Decisões de projeto

1. **O CSV é lido como texto puro** (`dtype=str`, sem NA automático). Se
   deixássemos o pandas inferir, uma única célula `"indisponivel"` transformaria a
   coluna inteira em `object` — ou, pior, viraria `NaN` e o **erro de tipo se
   disfarçaria de valor faltante**. Lendo como texto, a checagem de tipo é por
   célula e o defeito aparece exatamente onde ele está.
2. **Um defeito, uma violação.** Célula vazia é reportada só como falha de
   completude; ela não é recontada como erro de tipo nem de faixa. Sem isso, um
   campo em branco apareceria três vezes no relatório e inflaria a contagem.
3. **Domínio categórico é sensível a caixa e a espaço.** `"f"` e `"Sudeste "` são
   defeitos típicos de ingestão, não sinônimos aceitáveis — normalizar em silêncio
   esconderia o problema na origem.
4. **`"1.0"` é aceito onde se espera inteiro** (produtores serializam de formas
   diferentes); `"1.5"` não é.
5. **As linhas são reportadas na numeração do arquivo** (cabeçalho = linha 1), que
   é a que a pessoa vê ao abrir o CSV — e não o índice do DataFrame.
6. **`carregar()` ergue exceção** em vez de devolver dados parciais. Métrica
   calculada sobre dado inválido é número bonito e errado.

### A prova de que a validação funciona

Do mesmo jeito que a ATV1 provou o teste de propriedade contra uma função
propositalmente bugada, aqui existe um `dados/credito_corrompido.csv` com **12
defeitos plantados**, um para cada regra. O arquivo limpo passa; o corrompido é
reprovado defeito por defeito ([`evidencias/02_validacao_corrompido.txt`](evidencias/02_validacao_corrompido.txt)):

```
VALIDACAO FALHOU - 12 violacao(oes)
  [schema] divida_ativa: coluna obrigatoria ausente
  [schema] obs_internas: coluna nao declarada no contrato
  [schema] score_bureau: valor nao conversivel para inteiro (ex.: 'indisponivel') (linha 12)
  [completude] genero: valor obrigatorio ausente [3x] (linha 22, 23, 24)
  [completude] renda_mensal: valor obrigatorio ausente (linha 32)
  [faixa] genero: valor fora do dominio ['F', 'M'] (ex.: 'f') (linha 52)
  [faixa] regiao: valor fora do dominio ['Centro-Oeste', ...] (ex.: 'Sudeste ') (linha 53)
  [faixa] renda_mensal: valor fora de [1000.0, 40000.0] (ex.: '-1500.0') (linha 42)
  [faixa] score_bureau: valor fora de [300, 1000] (ex.: '1500') (linha 43)
  [faixa] score_modelo_v1: valor fora de [0.0, 1.0] (ex.: '1.7') (linha 44)
  [faixa] pred_v1: valor fora de [0, 1] (ex.: '2') (linha 54)
  [unicidade] id_cliente: valor repetido (ex.: 'C00060') (linha 62)
```

**Por que a validação vem antes de tudo:** as três células de `genero` em branco
encolheriam o grupo F em silêncio, e a análise de fairness rodaria sem reclamar,
sobre um grupo que não é mais o grupo. Validação não é burocracia de entrada — é
a pré-condição para que os números da Aula 4 signifiquem alguma coisa.

---

## 3. Entregável 2 — Teste estatístico

Foram feitos **os dois** itens do enunciado.

### 3.1 Intervalo de confiança por bootstrap

Bootstrap percentil não paramétrico, 5000 reamostras, 95%, semente fixa. Cada
reamostra sorteia 4000 **linhas** com reposição e aplica os mesmos índices a todos
os vetores — o que preserva o pareamento entre rótulo, grupo e as duas predições.

```
  v1 acuracia     : +0.7107  IC95% [+0.6973, +0.7248]
  v1 precisao     : +0.7251  IC95% [+0.7056, +0.7442]
  v1 taxa aprovac.: +0.5148  IC95% [+0.4993, +0.5305]
  v1 TPR (recall) : +0.7164  IC95% [+0.6967, +0.7359]

  v2 acuracia     : +0.7320  IC95% [+0.7183, +0.7458]
  v2 precisao     : +0.7389  IC95% [+0.7199, +0.7576]
  v2 taxa aprovac.: +0.5295  IC95% [+0.5142, +0.5450]
  v2 TPR (recall) : +0.7510  IC95% [+0.7320, +0.7695]
```

Decisões: percentil em vez de **BCa** — com n = 4000 e métricas que são médias de
indicadoras, a distribuição bootstrap já é praticamente simétrica e a correção de
viés não pagaria a complexidade. Reamostras em que a métrica é indefinida (TPR num
sorteio sem nenhum positivo) viram `nan`, são descartadas e **a contagem de
descartes é reportada** — descartar em silêncio seria mentir sobre o tamanho
efetivo da amostra.

### 3.2 A v2 é mesmo melhor que a v1?

Duas ferramentas, porque elas respondem coisas diferentes: o **intervalo** diz o
tamanho do efeito, o **p-valor** diz se ele existe.

```
  metrica A (v1)    : 0.7107
  metrica B (v2)    : 0.7320
  diferenca (B - A) : +0.0212  IC95% [+0.0092, +0.0335]
  pares discordantes: so A acertou=272, so B acertou=357
  p (McNemar exato) : 7.97e-04
  veredito          : diferenca significativa (B > A)
```

- A comparação é **pareada**: as duas versões preveem as mesmas linhas, e
  reamostrá-las de forma independente jogaria fora a correlação entre elas e
  inflaria o intervalo.
- **McNemar exato** (binomial com p = 0,5 sobre os 629 pares discordantes) em vez
  da aproximação qui-quadrado: é exato para qualquer contagem e dispensa correção
  de continuidade.
- Só é declarada significativa a diferença em que **o IC exclui zero e o p-valor
  fica abaixo de α = 0,05**. Aqui os dois concordam: +2,12 p.p. de acurácia, com o
  intervalo inteiro acima de zero.

**Resposta ao enunciado:** não é ruído de amostra. A chance de ver uma assimetria
de 272 × 357 se as duas versões errassem igual é de 8 em 10 000.

### 3.3 O teste não acusa qualquer coisa

Um teste que sempre diz "significativo" não informa nada. Dois controles, em
[`test_estatistica.py`](test_estatistica.py):

- comparar a v1 **com ela mesma** dá diferença exata 0 e p = 1;
- 20 pares de versões de **qualidade idêntica** (75% de acurácia cada, diferindo só
  pelo próprio ruído, sementes fixas) produziram **0 falsos alarmes em 20** — com
  α = 0,05 o esperado é ~1. O teste está calibrado, e conservador.

---

## 4. Entregável 3 — Teste de fairness

### Quais métricas, e por quê

**Paridade demográfica** — `P(aprovado | G=F) − P(aprovado | G=M)` — é a métrica
principal, **mas só depois de uma verificação**: a auditoria começa medindo a
diferença de **taxa-base real** entre os grupos, `P(y_real=1 | G)`. Se o risco real
fosse diferente, exigir paridade seria exigir que o modelo ignorasse informação
legítima. No dataset:

```
  taxa-base real  (F - M): +0.0018  IC95% [-0.0292, +0.0335]  -> comparaveis
```

O intervalo contém zero com folga: os dois grupos são igualmente bons pagadores.
**Com risco real igual, qualquer diferença sistemática de aprovação vem do modelo,
não do mundo** — e a paridade demográfica passa a ser um critério defensável aqui.
Essa checagem é o que separa "usei paridade demográfica" de "usei paridade
demográfica porque neste dataset ela é o critério certo".

Além dela:

- **Razão de impacto desproporcional** (`menor/maior`), a leitura da **regra dos
  4/5** — tradicional em auditoria de crédito e contratação, onde razão < 0,8 é
  indício de impacto desproporcional.
- **Igualdade de oportunidade** — diferença de TPR, `P(aprovado | y_real=1, G)`.
  Condiciona em quem de fato é bom pagador, então continua valendo mesmo se as
  taxas-base diferissem. Serve de contraprova da paridade.

Cada diferença vem com **IC bootstrap** (tamanho) e **p-valor de permutação**
(existência). No teste de permutação, sob H₀ a decisão independe do grupo, então
os rótulos de grupo são permutáveis: embaralhá-los não deveria mudar o tamanho da
disparidade. Uso a correção `(extremas+1)/(B+1)`, que evita p = 0 — com 5000
permutações não dá para afirmar mais do que `p < 1/5001`.

### Resultados ([`evidencias/04_fairness.txt`](evidencias/04_fairness.txt))

```
== modelo v1 | atributo sensivel: genero ==
             F | n= 1964 | aprovados=48.17% | bons pagadores reais=52.19% | TPR=67.90%
             M | n= 2036 | aprovados=54.67% | bons pagadores reais=52.01% | TPR=75.26%
  paridade demog. (F - M): -0.0650  IC95% [-0.0958, -0.0335]  p=0.0002
  razao de impacto (menor/maior): 0.8811 (passa na regra dos 4/5)
  oportunidade    (F - M): -0.0736  IC95% [-0.1120, -0.0356]  p=0.0004
  -> REPROVADO: paridade demografica e igualdade de oportunidade violadas

== modelo v2 | atributo sensivel: genero ==
             F | n= 1964 | aprovados=53.46% | bons pagadores reais=52.19% | TPR=75.32%
             M | n= 2036 | aprovados=52.46% | bons pagadores reais=52.01% | TPR=74.88%
  paridade demog. (F - M): +0.0101  IC95% [-0.0208, +0.0403]  p=0.5233
  razao de impacto (menor/maior): 0.9812 (passa na regra dos 4/5)
  oportunidade    (F - M): +0.0044  IC95% [-0.0329, +0.0411]  p=0.8342
  -> APROVADO: nenhuma disparidade estatisticamente distinguivel de zero
```

**Controle negativo:** a mesma auditoria sobre `regiao` (Norte × Sudeste), atributo
que não entra em nenhum dos modelos, dá +0,24 p.p. com p = 0,94 — não acusa nada.
A auditoria não é um carimbo de "reprovado".

---

## 5. Entregável 4 — Interpretação

### O que os resultados dizem

**1. A v1 discrimina, e a diferença não é ruído.** O grupo F é aprovado 6,5 p.p.
menos que o M (IC95% [−9,6; −3,4], p = 0,0002) enquanto as taxas de bom pagador
dos dois grupos são estatisticamente indistinguíveis. Entre os clientes que **de
fato pagariam**, a v1 aprova 67,9% das mulheres contra 75,3% dos homens: 7,4 p.p.
de crédito bom negado a quem merecia, por causa do grupo. Não é um efeito de
composição da amostra — é decisão do modelo.

**2. Modelo cego ao atributo sensível não é modelo justo.** Nem a v1 nem a v2
recebem a coluna `genero`. A v1 discrimina mesmo assim, porque
`tempo_emprego_meses` carrega o gênero embutido (77 meses contra 99). É a falha do
*fairness through unawareness*: apagar a coluna sensível não apaga a informação
dela, só torna o problema invisível para quem só lê a lista de features. **Foi
preciso medir o resultado por grupo para enxergar.** Este é o achado central da
atividade.

**3. A regra dos 4/5 teria absolvido a v1.** A razão de impacto da v1 é 0,881 —
acima do limiar de 0,8, "passa". Ao mesmo tempo, o IC da diferença exclui zero e o
teste de permutação rejeita H₀. Um limiar fixo não sabe quantas linhas você tem;
com n = 4000 uma disparidade de 6,5 p.p. é perfeitamente resolvível, e chamá-la de
aceitável é uma decisão de política, não uma conclusão estatística. **Limiar de
auditoria não substitui teste** — e o caminho contrário também existe: em amostra
pequena, uma razão de 0,7 pode ser puro ruído. Este ponto está travado em teste
([`test_fairness.py`](test_fairness.py), `test_regra_dos_quatro_quintos_pode_passar_com_paridade_violada`).

**4. Corrigir o viés não custou acurácia — rendeu acurácia.** A v2 elimina a
disparidade (paridade +1,0 p.p., p = 0,52; oportunidade +0,4 p.p., p = 0,83) **e**
acerta 2,12 p.p. a mais (IC95% [+0,9; +3,3], p = 8·10⁻⁴). O trade-off
"justiça × desempenho" não é uma lei da natureza: aqui o viés vinha de um proxy
com pouco poder preditivo real, então tirar peso dele **melhorou as duas coisas ao
mesmo tempo**. O trade-off aparece quando a feature enviesada é de fato preditiva
— não era o caso. Diagnosticar de onde vem o viés é o que decide se existe preço a
pagar.

**5. As duas metades da suíte dependem uma da outra.** Três células de `genero` em
branco bastariam para a auditoria rodar sobre um grupo F menor do que o real, sem
erro nenhum na tela. A validação da Aula 3 é o que dá sentido aos números da Aula
4; e o teste estatístico é o que impede que o número da Aula 4 seja lido como
verdade absoluta.

### O que a suíte **não** prova

- **"Aprovado" significa "não distingo de zero", não "provei que é igual".** Com
  n = 4000 o IC da paridade tem ~6 p.p. de largura: uma disparidade real de 2 p.p.
  passaria despercebida. Ausência de evidência não é evidência de ausência, e o
  relatório reporta a largura do intervalo justamente para deixar isso visível.
- **A auditoria mede associação, não causalidade.** Aqui eu afirmo que o proxy
  *causa* a disparidade porque **eu escrevi o gerador**. Em dados reais, o mesmo
  resultado numérico exigiria investigação adicional para atribuir a causa.
- **Paridade demográfica não é o critério universal.** Ela é defensável neste
  dataset porque as taxas-base foram verificadas como iguais. Quando as taxas-base
  diferem, paridade demográfica, igualdade de oportunidade e calibração são
  **matematicamente incompatíveis** — escolher entre elas vira decisão de política,
  e a suíte só pode explicitar o que se está escolhendo.
- **O limiar de decisão é fixo em 0,5** e a auditoria é sobre a decisão binária.
  Uma análise mais completa varreria o limiar, inclusive limiares por grupo — que
  resolveriam a paridade, mas ao custo de tratar pessoas com o mesmo score de
  formas diferentes, o que é outra discussão.
- **Os dados são sintéticos.** O que a suíte demonstra é que ela encontra o que foi
  plantado e não inventa o que não foi. Sobre dados reais, os mesmos instrumentos
  se aplicam; as conclusões seriam sobre o mundo, não sobre o meu gerador.

---

## 6. Testes: 62 casos, incluindo 11 de propriedade

Seguindo a ATV1, a suíte não se limita a exemplos.

| Arquivo | Cobre |
|---|---|
| `test_validacao.py` (25) | os 12 defeitos plantados, um a um; decisões (2), (3) e (4); 3 propriedades |
| `test_estatistica.py` (20) | McNemar em casos calculados à mão; IC degenerado; IC estreita com mais dados; calibração em 20 réplicas; 4 propriedades |
| `test_fairness.py` (17) | viés plantado é reprovado; preditor cego é aprovado; taxa de falso alarme; v1/v2 do dataset; 4 propriedades |

Propriedades que valem para **qualquer** entrada válida — por exemplo: o ponto do
bootstrap é sempre a métrica nos dados completos; a mesma semente sempre dá o mesmo
intervalo; McNemar é simétrico em `(b, c)` e vive em `(0, 1]`; a diferença de
paridade é antissimétrica em relação a quem é A e quem é B; a razão de impacto
nunca escapa de `[0, 1]`; renomear os grupos não muda disparidade nenhuma;
aprovar todo mundo é sempre paridade perfeita.

```
62 passed in 12.03s
```
