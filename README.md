# Assignment 2 — Object Recognition and Image Retrieval

Classificação e retrieval de imagens em cinco categorias (aeroplane, motorbike,
person, horse, car) contra imagens de background. Comparamos representações
handcrafted (BoVW, VLAD, Fisher Vectors sobre dense SIFT) com representações
pré-treinadas congeladas (ResNet-18, ViT-B/32, DINOv2, CLIP), com um SVM linear
ou, na Parte 5, sem classificador.

Este README descreve o nosso código e como reproduzir os resultados do
relatório. O [code/README.md](code/README.md) é o README original fornecido com
o enunciado e ficou como estava. Algumas partes dele já não se aplicam: os
notebooks `1_handcraft.ipynb` e `2_deep_models.ipynb` foram substituídos pelos
scripts de `code/runs/`.

## Estrutura

```text
.
├── assignment_2.pdf          # Enunciado.
├── README.md                 # Este ficheiro.
├── report/                   # Secções LaTeX do relatório (intro, method, part1–5, discussion, ...).
├── answers/                  # Respostas às perguntas do enunciado, por parte (rascunhos do relatório).
├── wheel/                    # Wheel do cyvlfeat para Windows (fornecido, não versionado).
└── code/
    ├── README.md             # README original fornecido (inalterado).
    ├── env.yml, env-win.yml  # Ambiente conda (Python 3.11 + cyvlfeat 0.7.1).
    ├── requirements.txt      # Pacotes pip.
    ├── data/                 # Imagens, splits, encoders e encodings (não versionado, ver "Dados").
    ├── src/                  # Biblioteca: SIFT, encoders, extração, splits, SVM.
    ├── runs/                 # Um script por parte/stage; escrevem CSVs e figuras em results/.
    └── results/              # CSVs e figuras usados no relatório.
```

### Código fornecido e código nosso

| Ficheiro | Origem |
|---|---|
| `src/features.py` | Fornecido: dense SIFT (`vl_phow`, implementação Python do `vl_phow` do VLFeat sobre o `dsift` do cyvlfeat). |
| `src/encoders.py`, `src/train.py`, `src/extract.py` | Fornecidos: BoVW / VLAD / FV, treino dos codebooks e extração. Alterações mínimas (o `train.py` guarda o tempo de ajuste em `fit_time_s`). |
| `src/deep_models.py`, `src/utils.py`, `src/visualization.py` | Fornecidos. O `utils.train_svm` é o SVM usado em todas as partes. |
| `src/splits.py` | Nosso: split train/val/test do protocolo. |
| `src/extract_deep.py` | Nosso: extração das features pré-treinadas (com `--flip`) e `meta.json` com custos e versões. |
| `runs/*.py` | Nossos: todas as experiências, tabelas e figuras. |

## Ambiente

Testado em Windows 11, CPU AMD (AMD64 Family 25 Model 80), **sem GPU**: todos
os tempos do relatório são em CPU. Versões usadas:

| Pacote | Versão |
|---|---|
| Python | 3.11.9 |
| cyvlfeat | 0.7.1 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| torch / torchvision | 2.14.0 / 0.29.0 (builds CPU) |
| CLIP | pacote oficial `openai/CLIP`, commit `d05afc4` |
| DINOv2 | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` |

Instalação, a partir de `code/`:

```bash
# Linux
conda env create -f env.yml
# Windows: copiar a pasta wheel/ (materiais da disciplina) para code/ e usar
conda env create -f env-win.yml

conda activate cvc-a2
python -m pip install -r requirements.txt
```

O `requirements.txt` já está configurado para torch/torchvision de CPU. Para
CUDA 12.6, descomentar as linhas `+cu126` e comentar as de CPU. Os pesos
pré-treinados (torchvision, DINOv2 via Torch Hub, CLIP) são descarregados
automaticamente no primeiro uso e não estão incluídos.

## Dados

O dataset é o fornecido com o enunciado: 4946 imagens de um subconjunto do
PASCAL VOC 2007 (trainval), com listas `<categoria>_train.txt` e
`<categoria>_val.txt` para as cinco categorias e para o background. Colocar em:

```text
code/data/images/   # *.jpg
code/data/splits/   # <categoria>_{train,val}.txt fornecidos
```

`code/data/` não está no git (é grande e é regenerado pelos passos abaixo).

## Protocolo

- **Splits** (`data/splits_protocol`): o `val` fornecido é o nosso **test**. Do
  `train` fornecido separamos 20 % para **validação**, por imagem (as imagens
  com várias categorias ficam no mesmo subconjunto), estratificado pela
  categoria mais rara, seed 42.
- **Tarefa**: um classificador binário por categoria, positivos contra o
  background. AP = `sklearn.metrics.average_precision_score`, com a categoria
  como classe positiva. mAP = média sobre as 5 categorias.
- **Seleção**: todas as escolhas (K, tiles, normalização, C, prompts) são
  feitas no AP de validação. O test só é avaliado na configuração escolhida. A
  exceção é a Parte 3, que reporta todas as condições no test, como o
  enunciado pede.
- **SVM**: `LinearSVC(C=10, loss="hinge", dual=True, max_iter=100, tol=1e-3,
  random_state=42)`, sem `class_weight` (`src/utils.py`). O C só varia no
  sweep da Parte 4, Stage D.
- **Seeds**:

  | Onde | Seeds |
  |---|---|
  | Split train/val, encoders (amostragem, PCA, k-means/GMM), SVM | 42 |
  | Parte 1, Stage G (frações de treino) | 0–4 |
  | Parte 3 (subamostras de 90 % do treino) | 0–2 |
  | Parte 4, few-shot (Stage C/D) | 0–4 |
  | Parte 5 (queries e PCA: 0; subconjuntos do corpus: 0–2) | 0, 0–2 |
  | Erro de quantização (`runs.quantization`) | 0 |

## Reproduzir os resultados

Todos os comandos correm a partir de `code/`. Os scripts de `runs/` escrevem em
`results/` (opção `--out`). Por omissão, a extração não recalcula o que já
existe (`-f` força).

### 1. Splits do protocolo

```bash
python -m src.splits            # data/splits -> data/splits_protocol
```

### 2. Encoders handcrafted

O dense SIFT é recalculado com `vl_phow` (passo 4 px, bins 4/6/8/10 px,
imagem em cinzento com altura máxima 480 px), em vez de usar os descritores
fornecidos. O BoVW usa o SIFT de 128-d. A VLAD e o FV usam PCA para 80-d. Os
codebooks são ajustados só nas imagens de treino do protocolo: 733 imagens (até
150 por classe) e 400 descritores por imagem.

```bash
for K in 32 64 128 256 512 1024; do
  python -m src.train -e bovw -k $K -c 0 --data-dir data/splits_protocol --cache-dir cache/
done
for K in 16 32 64; do
  python -m src.train -e vlad -k $K -c 80 --data-dir data/splits_protocol --cache-dir cache/
  python -m src.train -e fv   -k $K -c 80 --data-dir data/splits_protocol --cache-dir cache/
done
```

`--data-dir data/splits_protocol` é necessário. Com o valor por omissão
(`data/splits`), o codebook veria também as imagens de validação.

### 3. Encodings por imagem

```bash
# Handcrafted: um encoding por encoder, para todas as categorias.
for enc in data/encoders/*.npz; do
  python -m src.extract -e $enc --catg background motorbike aeroplane person horse car --cache-dir cache/
done

# Pré-treinados congelados (center crop 224, pré-processamento oficial), com as imagens espelhadas.
python -m src.extract_deep -m resnet18 vit_b_32 dinov2_vits14 clip_vit_b_32 --flip
```

A extração usa as listas fornecidas (`data/splits`). Os scripts de `runs/`
aplicam o protocolo ao carregar. Cada pasta de encodings deep tem um
`meta.json` com o modelo, os pesos, as versões, o hardware e o tempo de
extração.

### 4. Experiências

Correr pela ordem indicada: alguns scripts leem CSVs de partes anteriores.

| Parte | Comando | Saídas principais em `results/` |
|---|---|---|
| 1 (Stages A/C–G) | `python -m runs.run_part1` | `part1_grid.csv`, `part1_fractions.csv/.png` |
| 1 (Stages A–D, figuras) | `python -m runs.run_part1_inspect` | `part1_samples.png`, `part1_ranked_*.png`, `part1_words.png`, `part1_pr_curves.png` |
| 1 (erro de quantização) | `python -m runs.quantization` | `quantization.csv/.png` |
| 2 | `python -m runs.run_part2` | `part2_grid.csv`, `part2_encodings.csv`, `part2_pr_curves.png`, `part2_pr_points.csv` |
| 3 | `python -m runs.run_part3` | `part3_grid.csv`, `part3_summary.csv`, `part3_conditions.csv`, `part3_per_class.csv`, `part3_map_vs_dim.png` |
| 4, Stage A (ResNet-18) | `python -m runs.run_part4_resnet` | `part4_resnet_grid.csv`, `part4_resnet_encodings.csv` (precisa de `part2_encodings.csv`) |
| 4, Stage B (DINOv2) | `python -m runs.run_part4_dino` | `part4_dino_grid.csv`, `part4_dino_confusion_*.csv/.png` |
| 4, Stages C–D (few-shot, flip, C) | `python -m runs.run_part4_fewshot` | `part4_fewshot.csv/.png`, `part4_tta.csv`, `part4_svm_c*.csv/.png` |
| 4, Stage E (CLIP zero-shot) | `python -m runs.run_part4_clip` | `part4_clip_*.csv`, `part4_clip_confusion.png` |
| 4, Stage F (exemplos e mapas) | `python -m runs.run_part4_visual` | `part4_examples_*.png`, `part4_examples.csv` (precisa de `part4_clip_prompts.csv`) |
| 5 | `python -m runs.run_part5` | `part5_*.csv` (lê `part2_encodings.csv` e `part3_summary.csv` para os tempos) |
| 5 (figuras) | `python -m runs.run_part5_visual` | `part5_*.png` |
| Resumo (Partes 1–4) | `python -m runs.report` | `report_part<N>.md`, `report_setup.md` |

Cada script tem uma docstring com o que mede e as opções (`--help`). O
`runs.report` não recalcula nada: junta as tabelas e as figuras de cada parte
em Markdown a partir dos CSVs.

### Notas sobre custos

- Extração handcrafted: SIFT + encoding medidos em 5 imagens.
- Extração deep: forward em CPU sobre as 4946 imagens (original + flip),
  incluindo o carregamento do modelo, a descodificação JPEG e o
  pré-processamento.
- Inferência: mapa de features + SVM por imagem de test, com as features já
  extraídas.
- Armazenamento: dimensão × 4 B (float32), para que as representações sejam
  comparáveis. A Parte 2 reporta os arrays como estão guardados (BoVW int64,
  FV float64).

## Referências de código e modelos reutilizados

- Código base da disciplina (`code/src/`, `code/README.md`), incluindo a
  implementação Python do `vl_phow`.
- [VLFeat](https://www.vlfeat.org/) / [cyvlfeat](https://github.com/menpo/cyvlfeat): dense SIFT.
- [scikit-learn](https://scikit-learn.org/): k-means, GMM, PCA, `LinearSVC`, métricas.
- [torchvision](https://github.com/pytorch/vision): ResNet-18 e ViT-B/32 (`IMAGENET1K_V1`).
- [DINOv2](https://github.com/facebookresearch/dinov2): `dinov2_vits14` via Torch Hub.
- [OpenAI CLIP](https://github.com/openai/CLIP): `ViT-B/32`.
- Dataset: subconjunto do [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) fornecido com o enunciado.
