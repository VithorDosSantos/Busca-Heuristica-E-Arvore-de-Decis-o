# Trabalho AV1 - IA Computacional

Projeto completo para a disciplina **Inteligencia Artificial e Computacional (0700M8)**.

Este pacote atende a estrutura da lauda com:
- **Parte 1 (OPCAO A): Busca Heuristica**
- **Parte 2 (Obrigatoria): Arvores de Decisao**

## Estrutura

- `src/heuristic_search.py`: modelagem do problema, Greedy Best-First, A*, Weighted A* (variante), comparacao de heuristicas.
- `src/decision_tree_part.py`: treinamento/validacao/teste de arvores de decisao, comparacao de configuracoes e visualizacao.
- `src/main.py`: executa tudo e gera evidencias automaticamente.
- `docs/relatorio_tecnico.md`: relatorio tecnico em Markdown.
- `docs/relatorio_tecnico.pdf`: relatorio tecnico em PDF para submissao.
- `slides/apresentacao.md`: roteiro de slides.
- `docs/roteiro_oral.md`: fala sugerida para a apresentacao ao vivo.
- `docs/anexo_uso_ia.md`: anexo de uso de IA.
- `outputs/`: evidencias geradas durante a execucao.

## Como executar

1. Criar ambiente virtual (opcional):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
py -m pip install -r requirements.txt
```

3. Executar o projeto:

```powershell
py src/main.py
```

## Evidencias geradas

- `outputs/heuristic_search/heuristic_metrics.csv`
- `outputs/heuristic_search/heuristic_paths_and_expansions.json`
- `outputs/heuristic_search/heuristic_validity_checks.json`
- `outputs/heuristic_search/*.png` (visualizacoes da exploracao)
- `outputs/decision_tree/decision_tree_metrics.csv`
- `outputs/decision_tree/decision_tree_summary.json`
- `outputs/decision_tree/decision_tree_classification_report.txt`
- `outputs/decision_tree/decision_tree_rules.txt`
- `outputs/decision_tree/decision_tree_visualization.png`
- `outputs/execution_summary.json`

## Conversao do relatorio para PDF

Use qualquer editor (Word/Google Docs/VS Code + extensao) para exportar `docs/relatorio_tecnico.md` em PDF.
