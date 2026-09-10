Aplicar sobre um dataset com um atributo sensível (ex: gênero, região, faixa etária, ou outro grupo relevante ao seu contexto) uma suíte que combine validação de dados (Aula 3) com testes estatísticos e de fairness (Aula 4). Pode ser usado um dataset público, sintético ou de um projeto seu, desde que tenha pelo menos uma coluna que sirva como atributo de grupo (para a análise de fairness) e uma predição/rótulo associado.

Entregáveis

1. Validação de dados, cobrindo pelo menos:
   • Schema (colunas e tipos esperados)
   • Completude (ausência de valores faltantes onde não deveria)
   • Faixa de valores e/ou unicidade, conforme fizer sentido pro seu dataset
2. Teste estatístico, cobrindo pelo menos um dos dois:
   • Intervalo de confiança (bootstrap) para uma métrica do seu modelo/predição
   • Comparação estatística entre duas versões/variações (a diferença é significativa, ou pode ser ruído de amostra?)
3. Teste de fairness, verificando paridade demográfica (ou outra métrica de equidade à sua escolha, desde que justificada) entre pelo menos dois grupos do seu atributo sensível.
4. Interpretação: É preciso explicar o que os resultados dizem sobre o dataset/modelo analisado.
