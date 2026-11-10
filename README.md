# Brasileirão Predictor AI ⚽🤖

> **Predição de resultados da Série A do Campeonato Brasileiro com fusão de dados estatísticos e extração estruturada de contexto esportivo por LLM.**

Este repositório implementa um sistema experimental rigoroso e reproduzível para avaliar a predição probabilística de partidas de futebol (1X2 e distribuição de placares), comparando baselines puramente estatísticos contra modelos enriquecidos com contexto esportivo (lesões confirmadas, trocas de técnico, protestos/crises e rodízio de titulares) extraídos de notícias auditadas por LLMs estruturados.

---

## 📌 Decisão Central de Arquitetura

1. **O LLM não prevê o vencedor da partida:** Sua responsabilidade restringe-se a analisar textos pré-jogo autorizados e convertê-los em variáveis numéricas e categóricas ancoradas em evidências citáveis sob um JSON Schema estrito (Apêndice A).
2. **Modelos preditivos dedicados:** Classificadores tabulares (LightGBM) e modelos probabilísticos de contagem de gols (Poisson / Dixon-Coles) geram as probabilidades e a matriz de placares.
3. **Prevenção estrita de Data Leakage:** Ponto de corte temporal fixado em $t^* = \text{kickoff} - 24\text{h}$ (ou $48\text{h}$). Nenhuma informação pós-corte pode entrar nas features ou na extração do LLM.
4. **Governança da Hipótese Nula ($H_0$):** O modelo contextual ($M_1$) só é promovido a padrão de produção se demonstrar uma redução relativa mínima de 2% em Log Loss sobre o baseline numérico ($B_1$) com consistência temporal. Caso contrário, $B_1$ é mantido e o resultado negativo é documentado como conclusão científica válida.

---

## 🏗️ Família de Modelos

| Modelo | Insumos | Papel |
| :--- | :--- | :--- |
| **B0 — Prior Simples** | Frequências históricas de classes (Mandante, Empate, Visitante) | Sanidade básica |
| **B1 — Numérico** | Elo rating pré-jogo, forma recente (5 jogos), gols, descanso, mando | Baseline estatístico principal |
| **B2 — Conteúdo Simples** | B1 + contagem e cobertura de notícias | Controle de volume de cobertura |
| **M1 — Contextual LLM** | B1 + scores ancorados (crise, desfalques, rodízio) + missing indicators | Modelo de teste da hipótese |
| **G1 — Poisson** | Força de ataque/defesa e mando de campo | Distribuição de placares e 1X2 derivado |

---

## ⚖️ Conformidade e Licenciamento de Dados

- **Resultados e Jogos:** Acervo do [OpenFootball](https://github.com/openfootball/south-america) (licença CC0 1.0).
- **Notícias:** Armazenamento estrito de metadados, links e hashes SHA-256 (`METADATA_ONLY`). Nenhum scraping integral indiscriminado de portais de notícias protegidos por copyright.
- **Fixtures Sintéticas:** Fornecido dataset sintético anotado para testes determinísticos e reprodução sem riscos jurídicos.

---

## 🚀 Quickstart

### Pré-requisitos
- Python 3.12+
- Node.js LTS (para o frontend)
- Docker Engine / Compose (opcional para PostgreSQL e contêineres)

### 1. Configuração do Ambiente
```bash
cp .env.example .env
pip install -e ".[dev]"
```

### 2. Inicialização do Banco de Dados
```bash
python -m src.database
```

### 3. Ingestão de Dados das Partidas
```bash
python -m src.ingestion.openfootball --seasons 2021,2022,2023,2024
```

### 4. Validação e Construção de Snapshots Tabulares
```bash
python -m src.data_quality.auditor
python -m src.features.snapshot_builder --as-of-hours 24
```

### 5. Execução do Experimento e Avaliação
```bash
python -m src.evaluation.run_experiment --config configs/experiment_v1.yaml
```

### 6. Execução da API e Interface Web
```bash
# Backend (FastAPI na porta 8000)
uvicorn apps.api.main:app --reload --port 8000

# Frontend (React/Vite na porta 5173 / 3000)
cd apps/web && npm install && npm run dev
```

---

## 🧪 Suíte de Testes Automatizados
```bash
pytest tests/ -v
```
