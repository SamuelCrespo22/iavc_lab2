## Parte 5 — Retrieval por query

**Setup.** Categoria *horse*. Corpus = test do protocolo, 2474 imagens, 148 relevantes (prevalência 0.060 = AP de um ranking aleatório). Os irrelevantes são 1077 cenas de background (com vacas, ovelhas, cães) e 1249 imagens das outras categorias. 12 queries do val, 6 com pessoa e 6 sem; 3 candidatas foram excluídas por serem quase-duplicadas do corpus (cosseno DINOv2 > 0.9). Ranking por cosseno, sem classificador e sem escolher nada no corpus.

#### Que estratégia dá os melhores resultados?

O **DINOv2 (CLS)**: mAP 0.806, P@5/10/20 = 0.98 / 0.98 / 0.96. O CLIP texto-imagem ("a photo of a horse.") fica igual (AP 0.803, P@10 = 1.00) sem precisar de uma imagem de query.

| Método | mAP | P@5 | P@10 | P@20 |
|---|---|---|---|---|
| BoVW K512 | 0.105 | 0.20 | 0.22 | 0.16 |
| VLAD K32 | 0.119 | 0.20 | 0.17 | 0.18 |
| FV K32 | 0.083 | 0.12 | 0.09 | 0.13 |
| ResNet-18 | 0.517 | 0.73 | 0.75 | 0.71 |
| ViT-B/32 | 0.592 | 0.72 | 0.74 | 0.77 |
| DINOv2 CLS | **0.806** | 0.98 | 0.98 | 0.96 |
| DINOv2 média patches | 0.757 | 0.95 | 0.93 | 0.88 |
| CLIP imagem | 0.648 | 0.97 | 0.97 | 0.92 |
| CLIP texto | 0.803 | 1.00 | 1.00 | 0.95 |

Nos exemplos:
- Na query mais fácil (004339, cavaleiro num campo), todos os métodos profundos têm 8/8 relevantes no top-8, o BoVW e a VLAD 4/8 e o FV 1/8.
- Na mais difícil (006134, pónei branco num estábulo, sem pessoa), o DINOv2 mantém 8/8, o CLIP imagem tem 4/8, a ResNet e o ViT 1/8, e os locais 0/8. Os erros são vacas, ovelhas e pessoas.

#### Padrões locais (BoVW / VLAD) vs. embeddings globais profundos?

Os locais ficam perto do acaso (mAP 0.08–0.12, P@10 ≤ 0.22), e os profundos ficam 4–10× acima. As mesmas features davam AP 0.66–0.76 em horse com o SVM da Parte 4, por isso a informação existe. Sem SVM, porém, o cosseno pesa todas as palavras por igual e o fundo domina (relva, céu, textura). Na query fácil, o BoVW devolve grupos de motards e cenas de campo. Os embeddings profundos já agrupam por objeto.

Os profundos são também mais baratos: 18–52 ms de extração, 1.5–3 KB por imagem e 0.15–0.20 ms por query, contra 161–221 ms, 8–80 KB e 1.3–17 ms nos locais.

#### Um vocabulário maior ajuda? O PCA mantém o desempenho?

Quase nada. O BoVW global com Hellinger passa de 0.086 (K=32) para 0.105 (K=1024), e o mesmo acontece com intersection (0.088 → 0.103). Com o cosseno em L2 o mAP até desce (0.085 → 0.076). Todas as variantes ficam entre 0.07 e 0.11.

O PCA, ajustado em 1000 imagens de treino, mantém o desempenho e às vezes melhora-o:
- BoVW K1024 → 16-d: 0.104 (era 0.105);
- VLAD → 128-d: 0.117 (era 0.119, 80× menor);
- FV → 64-d: 0.123 (era 0.083).

O tempo por query cai de 8 ms (VLAD) e 17 ms (FV) para ≤ 0.12 ms com d=128. O whitening só ajuda com d pequeno (FV d=32: 0.134, o melhor handcrafted) e piora com d ≥ 128 (BoVW K1024 d=512: 0.078).

À mesma dimensão, K grande + PCA supera K pequeno, como na Parte 3 (d=32: 0.106 vs. 0.086; d=128: 0.106 vs. 0.094), mas a diferença absoluta é pequena.

#### Como afeta a diversidade das queries?

Muito. Nos métodos profundos, o desvio do AP entre queries vai de 0.13 (DINOv2) a 0.24 (ViT); no DINOv2 o AP vai de 0.52 a 0.94 e no CLIP imagem de 0.27 a 0.88.

As queries com pessoa são mais fáceis para todos os métodos (DINOv2 0.879 vs. 0.733; CLIP imagem 0.776 vs. 0.519; ResNet 0.620 vs. 0.414), porque 116 dos 148 relevantes têm cavaleiro. Uma query sem pessoa é atípica para o conjunto relevante.

A ordem dos métodos também muda de query para query. Na 006134 o DINOv2 tem 0.80 e o CLIP 0.27; na 002042 o CLIP tem 0.71 e o DINOv2 0.52. Uma só query não chega para comparar métodos.

#### CLIP: o wording da query importa? Mais detalhe ajuda?

Importa: o AP vai de 0.718 a 0.888 entre 7 formulações.

| Query | AP | P@10 | Sobreposição do top-20 com "photo" |
|---|---|---|---|
| "horse" | 0.888 | 0.90 | 0.75 |
| "a photo of a horse." | 0.803 | 1.00 | 1.00 |
| 8 templates (média) | 0.807 | 1.00 | 1.00 |
| sinónimos (horse, pony) | 0.806 | 1.00 | 0.85 |
| descrição (patas, crina, cauda) | 0.883 | 1.00 | 0.70 |
| cena ("a person riding a horse in a field") | 0.718 | 0.70 | 0.05 |
| sobre-específica ("a white horse jumping over a fence…") | 0.732 | 1.00 | 0.00 |

Mais detalhe só ajuda se descrever o objeto. Descrever a cena traz paisagens sem cavalos. Uma descrição sobre-específica acerta no top-10 (saltos de obstáculos) mas empurra os outros cavalos para baixo. Estas duas mudam completamente o ranking (Spearman 0.58–0.61 com "photo", contra ≥ 0.82 nas outras). É um único corpus e uma categoria, por isso as diferenças pequenas (0.80 vs. 0.81) não são significativas.

#### Limitações de um corpus pequeno

- **Tamanho e desequilíbrio:** o AP depende da prevalência. Com só 10% dos irrelevantes (381 imagens, prevalência 0.39), o BoVW sobe de 0.105 para 0.484 e o DINOv2 de 0.806 para 0.962. O AP absoluto não se compara entre corpora; a ordem dos métodos mantém-se.
- **Hard negatives:** o que torna o corpus difícil é o background. Sem as outras categorias, o CLIP texto fica em 0.811; sem o background, sobe para 0.971.
- **Ground truth:** a relevância vem dos labels VOC ao nível da imagem. Cavalos pequenos ou não anotados contam como irrelevantes, e com 148 relevantes o R@20 nunca passa de 0.135.
- **Quase-duplicados:** há 31 pares no corpus com cosseno DINOv2 > 0.9, 15 deles entre relevantes (sequências do mesmo evento) e 2 cópias exatas. Uma query que acerta num destes grupos tem o P@K inflacionado.
- **Só 12 queries:** o mAP tem uma incerteza grande (desvio do AP 0.13–0.24 nos métodos profundos).
