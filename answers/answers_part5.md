## Parte 5 — Retrieval por query

**Setup.** Categoria *horse*. Corpus = test do protocolo, 2474 imagens, 148 relevantes (prevalência 0.060 = AP de um ranking aleatório). Os irrelevantes são 1077 cenas de background (com vacas, ovelhas, cães) e 1249 imagens das outras categorias. 12 queries do val, 6 com pessoa e 6 sem; 3 candidatas foram excluídas por serem quase-duplicadas do corpus (cosseno DINOv2 > 0.9). Ranking por cosseno, sem classificador e sem escolher nada no corpus.

#### Que estratégia dá os melhores resultados?

O **DINOv2 (CLS)**: mAP 0.806, P@5/10/20 = 0.98 / 0.98 / 0.96. O CLIP texto-imagem ("a photo of a horse.") fica igual (AP 0.803, P@10 = 1.00) sem precisar de uma imagem de query.

| Método | mAP | P@5 | P@10 | P@20 | R@20 |
|---|---|---|---|---|---|
| BoVW K512 | 0.105 | 0.20 | 0.22 | 0.16 | 0.022 |
| VLAD K32 | 0.119 | 0.20 | 0.17 | 0.18 | 0.024 |
| ResNet-18 | 0.517 | 0.73 | 0.75 | 0.71 | 0.096 |
| DINOv2 CLS | **0.806** | 0.98 | 0.98 | 0.96 | 0.130 |
| CLIP texto | 0.803 | 1.00 | 1.00 | 0.95 | 0.128 |

Na query mais difícil (002042, cavalo visto da sela, num campo), o DINOv2 tem 7/8 relevantes no top-8, o BoVW 2/8 e a VLAD e a ResNet 1/8. Os handcrafted trazem sobretudo vacas, ovelhas e cães, e a ResNet paisagens com pessoas e motas.

#### Padrões locais (BoVW / VLAD) vs. embeddings globais profundos?

Os locais ficam perto do acaso (mAP 0.105 e 0.119, P@10 ≤ 0.22), e os profundos ficam 4–8× acima. As mesmas features davam AP 0.66 e 0.74 em horse com o SVM da Parte 4, por isso a informação existe. Sem SVM, porém, o cosseno pesa todas as palavras por igual e o fundo domina (relva, céu, textura). Os embeddings profundos já agrupam por objeto.

Os profundos são também mais baratos: 18–52 ms de extração, 1.5–2 KB por imagem e 0.16–0.23 ms por query, contra 224–254 ms, 8–40 KB e 1.3–8.7 ms nos locais.

#### Um vocabulário maior ajuda? O PCA mantém o desempenho?

Quase nada. O BoVW global com Hellinger passa de 0.086 (K=32) para 0.105 (K=1024). Na classificação (Parte 3), o mesmo aumento de K dava 0.494 → 0.681.

O PCA, ajustado em 1000 imagens de treino, mantém o desempenho:
- BoVW 2048 → 32-d: 0.105 (igual);
- VLAD 10240 → 64-d: 0.120 (era 0.119, 160× menor).

O tempo por query cai de 1.3 ms (BoVW) e 8 ms (VLAD) para 0.05–0.06 ms.

#### Como afeta a diversidade das queries?

Muito. O desvio do AP entre queries é 0.13 no DINOv2 e 0.17 na ResNet; no DINOv2 o AP vai de 0.52 a 0.94.

As queries com pessoa são mais fáceis (DINOv2 0.879 vs. 0.733; ResNet 0.620 vs. 0.414), porque 116 dos 148 relevantes têm cavaleiro. Uma query sem pessoa é atípica para o conjunto relevante.

A distância entre métodos também muda: na 006134 o DINOv2 tem 0.80 e a ResNet 0.21; na 006836, 0.76 e 0.69. Uma só query não chega para comparar métodos.

#### CLIP: o wording da query importa? Mais detalhe ajuda?

Importa: o AP vai de 0.718 a 0.888 entre 7 formulações. Como no val da Parte 4 (0.949 contra 0.885), o nome sozinho funciona melhor do que "a photo of a horse.".

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

- **Tamanho e desequilíbrio:** o AP depende da prevalência. Com só 10% dos irrelevantes (381 imagens, prevalência 0.39), o BoVW sobe de 0.105 para 0.484 e o DINOv2 de 0.806 para 0.962. O AP absoluto não se compara entre corpora; os locais continuam muito abaixo dos profundos.
- **Hard negatives:** o que torna o corpus difícil é o background. Sem as outras categorias, o CLIP texto fica em 0.811; sem o background, sobe para 0.971.
- **Ground truth:** a relevância vem dos labels VOC ao nível da imagem. Cavalos pequenos ou não anotados contam como irrelevantes, e com 148 relevantes o R@20 nunca passa de 0.135.
- **Quase-duplicados:** excluímos 3 candidatas a query com cosseno DINOv2 > 0.9 ao corpus, mas o limiar é arbitrário e usa um dos métodos avaliados. Dentro do corpus há várias fotos do mesmo evento, que inflacionam o P@K da query que as encontra.
- **Só 12 queries:** o mAP tem uma incerteza grande (desvio do AP 0.13–0.17 nos métodos profundos).
