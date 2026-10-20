# Detecção de fadiga em corrida com IMUs

Projeto de Machine Learning (Mestrado, Mackenzie) sobre o dataset público
**4TU — physical fatigue detection in running using IMUs**
([doi:10.4121/14307743.v1](https://doi.org/10.4121/14307743.v1);
Marotta et al., *Sensors* 2021;21(10):3451,
[texto completo](https://pmc.ncbi.nlm.nih.gov/articles/PMC8156769/)).

**Pergunta:** a partir dos sinais de sensores inerciais (IMU) de uma passada, o
corredor está sem fadiga (`none`), com fadiga moderada (`mild`) ou intensa
(`heavy`)?

O dataset tem 8 corredores, 3 condições por corredor, 31 gravações brutas
(240 Hz, ~5,6 GB) e uma tabela de 12.513 passadas com 157 features já extraídas
pelos autores (`TableFeats`).

## Estrutura

| caminho | conteúdo |
|---|---|
| `src/` | camada de dados: leitor das gravações e passadas, decodificador da `TableFeats.mat`, validação do dataset, CLI |
| `notebooks/` | os três notebooks de análise (abaixo) |
| `data/raw/` | `.mat` originais — **não versionados**, baixe do 4TU |
| `data/preprocessed/` | `TableFeats.csv`, gerada por `python -m src export-tablefeats` |
| `data/models/` | modelo treinado pelo notebook 03 (ignorado pelo git) |
| `NOTES.md` | notas de leitura do dataset: erros do README original, armadilhas, significado das colunas |
| `docs/` | formulários da disciplina |

## Notebooks

Executar em ordem; todos rodam a partir da raiz ou de `notebooks/`.

1. **[01_exploracao_inicial](notebooks/01_exploracao_inicial.ipynb)** —
   inventário dos arquivos, glossário dos rótulos, canais e das 165 colunas da
   `TableFeats`, sinal bruto e passadas. Conclusão central: a `TableFeats` é
   *separável demais*, porque foi z-scorada por sujeito usando as três
   condições (normalização transdutiva).
2. **[02_arvores_de_decisao](notebooks/02_arvores_de_decisao.ipynb)** —
   árvores de decisão: critério de Gini calculado à mão, árvore interpretável,
   sobreajuste × profundidade, validação aninhada por sujeito, comparação com
   floresta aleatória, estabilidade das features, efeito de `HR_norm`.
3. **[03_processo_classificacao](notebooks/03_processo_classificacao.ipynb)** —
   processo completo de classificação: definição do problema, auditoria,
   validação *leave-one-subject-out* aninhada, 7 algoritmos comparados com
   bootstrap por sujeito, análise de erros e confiança, modelo final salvo e
   função de inferência.

## Principais resultados

Avaliação *leave-one-subject-out* (o teste é sempre um corredor que o modelo
não viu), 157 features, sem `HR_norm`:

| modelo (notebook 03) | acurácia | F1-macro |
|---|---|---|
| floresta aleatória | 0,877 | 0,878 |
| árvore de decisão | 0,876 | 0,877 |
| gradient boosting | 0,871 | 0,869 |
| SVM (RBF) | 0,808 | 0,811 |
| regressão logística | 0,724 | 0,727 |
| kNN | 0,617 | 0,619 |
| acaso | 0,333 | 0,167 |

- Os três primeiros são indistinguíveis dentro da incerteza (IC95 % do F1-macro
  de ~15 pontos de largura, por haver só 8 corredores).
- Um split aleatório dá ≈ 1,00 com a mesma árvore: mede memorização de passadas
  vizinhas, não detecção de fadiga. Por isso toda a validação é por sujeito.
- Erros concentram-se entre classes vizinhas; `none`↔`heavy` é ~0,3 % na
  floresta.

**Ressalva importante:** como a normalização por sujeito é transdutiva, esses
números valem para um corredor com linha de base calibrada, não para um
corredor totalmente novo. Para o baseline por janelas do sinal bruto, ver
`NOTES.md`.

## Como rodar

Python 3.9+ (testado em 3.13). Na raiz do projeto:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # versões exatas: requirements.lock.txt
```

1. Baixe o dataset do [4TU](https://doi.org/10.4121/14307743.v1) e coloque a
   pasta em `data/raw/` (ou aponte `FATIGUE_DATA` para ela). O caminho é
   resolvido pelo conteúdo, então o nome da pasta não importa.
2. Use a CLI da camada de dados:

```bash
.venv/bin/python -m src inspect              # gravações, canais, nº de passadas
.venv/bin/python -m src validate --full      # confere os dados contra o README e as hipóteses do projeto
.venv/bin/python -m src export-tablefeats    # -> data/preprocessed/TableFeats.csv
```

3. Abra os notebooks no Jupyter/VS Code, ou reexecute-os headless:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/03_processo_classificacao.ipynb
```

Os notebooks 02 e 03 só precisam de `data/preprocessed/TableFeats.csv`
(versionado) e do `scikit-learn`; o 01 também lê alguns `.mat` brutos. O
notebook 03 leva ~3 min e o 02 ~4 min.

Como biblioteca:

```python
import src

for rec in src.iter_recordings():
    acc  = rec.acc("PEL")        # (N, 3) m/s²
    knee = rec.angle("RKNE")     # (N, 3) GRAUS (o README original diz radianos, errado)
    keep = rec.paper_mask()      # voltas 2-4 / 8-10 / 11-13, as usadas no artigo
```

## Armadilhas do dataset

Documentadas e verificadas em `NOTES.md`; as principais:

- `joint.*.angle` está em **graus**, não radianos.
- `p007` não tem a gravação pós-fadiga (são 31 arquivos, não 32).
- O token `CW`/`CCW` do nome do arquivo não é posicional (em `p003`, `p006` e
  `p008` a ordem é invertida).
- Os ids da `TableFeats` (`sub` ∈ {1, 5, 6, 13, 15, 16, 17, 18}) não são os dos
  arquivos (`p001`…`p008`); o mapeamento foi estabelecido por estatura.
- A segunda corrida do protocolo (a que induz a fadiga) não é distribuída.
- As passadas normalizadas (`pXXX_strides.mat`) não trazem rótulo.

## Licença

MIT (ver `LICENSE`). Os dados do 4TU têm licença própria; consulte o DOI acima.
