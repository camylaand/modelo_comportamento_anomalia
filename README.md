# 🛡️ Projeto: Mapeamento de Comportamento e Detecção de Anomalias em Transações

Este projeto tem como objetivo identificar comportamentos transacionais suspeitos em contas bancárias, por meio de aprendizado não supervisionado (Autoencoder + KMeans) e classificação supervisionada (XGBoost). O pipeline permite mapear o comportamento padrão de cada conta, detectar desvios e gerar alertas operacionais.

---

## ⚙️ Etapas Desenvolvidas – Mapeamento Comportamental

### 1. Pré-processamento e Enriquecimento dos Dados
- Enriquecimento com variáveis temporais derivadas da data:
  - Dia da semana
  - Faixa horária (madrugada, manhã, tarde, noite)
  - Flag de fim de semana
- Codificação:
  - Variáveis categóricas → OneHotEncoder
  - Variáveis numéricas → RobustScaler

### 2. Representação Comportamental com Autoencoder
- **Encoder**: reduz os dados a um vetor latente de 3 dimensões.
- **Decoder**: reconstrói a transação original.
- **Erro de reconstrução**:
  - Erros baixos: comportamento comum.
  - Erros altos: comportamento incomum/anômalo.

### 3. Clusterização com KMeans
- Aplicação de KMeans (2 clusters) sobre os vetores latentes.
- Atribuição da variável `cluster_autoencoder` por transação.
- Validação estatística com:
  - **ANOVA** (p < 0.001 nas variáveis-chave)
  - **GLM** associando erro de reconstrução com variáveis comportamentais

### 4. Classificação de Suspeita (Heurística)
Criação da flag `suspeita` com base no erro de reconstrução:
| Nível de Suspeita | Critério                          |
|-------------------|-----------------------------------|
| Alta              | Erro > 95º percentil              |
| Média             | Erro > 90º percentil              |
| Baixa             | Erro > 75º percentil              |
| Nenhuma           | Caso contrário                    |

### 5. Geração de Perfis Comportamentais por Conta
Cada conta recebe um perfil médio contendo:
- Média de valor transacionado
- Frequência por tipo de transação (PIX, saque, etc.)
- Frequência em finais de semana e faixas de horário
- Dia e horário mais comuns

Esses perfis permitem:
- Comparar o comportamento atual vs. padrão
- Detectar mudanças bruscas
- Priorizar investigações

---

## 🧠 Etapas Desenvolvidas – Detecção Supervisionada (Anomalias)

### 1. Construção do Rótulo `anomalia_confirmada`
Como a base não tinha rótulos, foi criado um rótulo sintético com base em:

- **Valor alto**: acima da média da conta + 3 desvios
- **Horário suspeito**: transações de madrugada
- **Alta frequência**: intervalo < 60 segundos
- **Erro de reconstrução e cluster**: valores extremos

Transações com **pontuação ≥ 3** foram marcadas como `anomalia_confirmada = 1`.

> Um ruído artificial de 0,5% foi adicionado para simular incertezas.

### 2. Uso do Autoencoder + KMeans como Variáveis de Entrada
- O erro de reconstrução e a distância ao centróide foram usados como features explicativas no modelo supervisionado.

### 3. Modelo Supervisionado: XGBoost
- **Balanceamento** com SMOTE.
- Métrica alvo: `anomalia_confirmada`.
- Métricas avaliadas:
  - F1-score
  - Recall por grupo sensível
  - Equality of Opportunity (EQOP)

### 4. Avaliação com Múltiplos Thresholds

| Threshold | F1-score | Falsos Negativos | Falsos Positivos | EQOP    |
|-----------|----------|------------------|------------------|---------|
| 0.40      | 0.9057   | 85.916           | 3.288            | 0.0006  |
| 0.50      | 0.9045   | 87.191           | 2.999            | 0.0006  |
| 0.55      | 0.9040   | 87.657           | 2.964            | 0.0006  |
| 0.60      | 0.9035   | 88.051           | 2.962            | 0.0005  |
| 0.70      | 0.9025   | 88.903           | 2.924            | 0.0005  |
| 0.80      | 0.9019   | 89.451           | 2.922            | 0.0004  |

---

## 🧾 Interpretação dos Resultados

- Thresholds baixos (ex: 0.40) → maior sensibilidade, mais alertas falsos.
- Thresholds altos (ex: 0.70) → menos falsos positivos, mas mais anomalias passam despercebidas.
- O **threshold de 0.60** foi escolhido por equilibrar bem:
  - F1-score alto
  - Baixo viés entre grupos
  - Volume de alertas gerenciável pela área de controle

---

## 🚀 Conclusão

O sistema desenvolvido é capaz de:
- Mapear o comportamento padrão de cada conta
- Detectar desvios em tempo quase real
- Utilizar aprendizado não supervisionado e supervisionado de forma integrada
- Gerar explicações e priorizações para apoiar decisões da área antifraude

---
