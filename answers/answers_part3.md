#### Um codebook maior melhora o reconhecimento antes do PCA? Para que classes ou padrões visuais?

Sim. Sem PCA, o mAP do BoVW sobe sempre com K: 0.494 (K = 32), 0.632 (K = 256) e 0.681 (K = 1024). Os ganhos vão diminuindo. Cada duplicação de K acrescenta +0.061, +0.038, +0.039 e +0.035 até K = 512, e de 512 para 1024 só +0.014. A VLAD (0.709 → 0.766, de K = 16 para 64) e o FV (0.773 → 0.798) mostram a mesma tendência.

O erro de quantização relativo cai de 0.52 (K = 128) para 0.43 (K = 1024). É praticamente igual nos descritores de treino e de test (0.513 / 0.520 e 0.418 / 0.427). Ou seja, um vocabulário maior descreve os patches com mais detalhe sem sobreajustar.

O ganho é muito desigual entre classes. De K = 256 para 1024, quem mais ganha são o motorbike (+0.103) e o horse (+0.084). São classes identificadas por estruturas locais específicas (rodas, raios e quadro; pernas e crina), que um vocabulário grosseiro mistura com patches de outros objetos. O car ganha +0.039, o person +0.014 e o aeroplane quase nada (+0.007). Uma explicação plausível para o aeroplane é que é reconhecido sobretudo pelo contexto (céu, pista, grandes zonas uniformes), que poucas palavras já captam. Não é uma questão de nº de exemplos: o aeroplane tem tantos positivos de treino como o motorbike (90 contra 96).

Isto está de acordo com a literatura: o BoVW melhora com o tamanho do vocabulário e satura (Nowak et al., 2006; Chatfield et al., 2011), porque uma quantização mais fina reduz a ambiguidade das palavras visuais (van Gemert et al., 2010).

#### O PCA mantém o ganho do codebook maior? Quanta redução de dimensão é possível antes de o desempenho cair?

Em grande parte sim, e a perda é gradual, sem nenhum precipício. O BoVW K = 1024 (0.681) com PCA para 512 / 256 / 128 dimensões dá 0.663 / 0.659 / 0.652. Isto mantém 64%, 56% e 41% da vantagem de 0.049 sobre o K = 256 (0.632). Mesmo em d = 32, com 97% das dimensões removidas, o K = 1024 (0.636) ainda empata com o K = 256 sem PCA.

Com uma tolerância de −0.02 de mAP, o PCA consegue remover:

- BoVW: 50% das dimensões (K = 1024 → 512: −0.018).
- VLAD: 97.5% (K = 64, 20 480 → 512: 0.766 → 0.749).
- FV: 98.8% (K = 64, 41 216 → 512: 0.798 → 0.785).

O PCA nunca melhorou o resultado face ao mesmo codebook sem PCA. Funciona como compressão e não como regularizador. O PCA não usa labels: guarda as direções de maior variância, que não são necessariamente as mais discriminativas, e por isso perde-se alguma informação de classe com d pequeno. A maior parte do ganho sobrevive porque os histogramas são muito redundantes: as palavras do mesmo objeto aparecem juntas.

Há uma anomalia na VLAD K = 64: o mAP desce de d = 128 para 256 e recupera em 512 (0.732 → 0.715 → 0.749). Não é um problema de otimização, porque com 5000 iterações o SVM converge e dá os mesmos valores. Também não é causado pela normalização L2 depois do PCA, porque sem ela a quebra mantém-se. Na K = 16 e K = 32 a diferença está dentro do ruído. Como o PCA ordena as direções por variância e não por relevância para a classe, uma leitura plausível é que as componentes ~129–256 têm variação grande mas pouco discriminativa, que o SVM ajusta em parte, até que as componentes seguintes trazem informação útil suficiente. Não verificámos isto mais a fundo.

Isto está de acordo com os resultados de compressão de VLAD e FV (Jégou et al., 2010; Perronnin et al., 2010). Jégou & Chum (2012) obtêm ganhos com o PCA apenas quando o combinam com whitening, que aqui não aplicámos.

#### Para uma dimensão final semelhante, um codebook grande + PCA supera um codebook pequeno usado diretamente?

Sim até d = 256, e a vantagem cresce à medida que d diminui (BoVW, condição C contra D):

| d | D: K = d, sem PCA | C: K = 1024 + PCA | diferença |
|---|---|---|---|
| 32 | 0.494 | 0.636 | +0.142 |
| 64 | 0.555 | 0.636 | +0.081 |
| 128 | 0.593 | 0.652 | +0.059 |
| 256 | 0.632 | 0.659 | +0.028 |
| 512 | 0.667 | 0.663 | −0.003 (dentro do desvio, ±0.007) |

Para um d fixo, o mAP também sobe sempre com o K de origem (d = 32: 0.494 → 0.537 → 0.572 → 0.607 → 0.629 → 0.636, de K = 32 para K = 1024). A VLAD e o FV seguem o mesmo padrão (d = 128: VLAD 0.678 / 0.702 / 0.732 e FV 0.727 / 0.728 / 0.750, para K = 16 / 32 / 64).

A razão é esta. Um k-means pequeno junta descritores só pela distância no espaço SIFT, uma decisão "dura" tomada antes de ver qualquer imagem. O PCA sobre um histograma grande combina as palavras que aparecem juntas nas imagens, uma fusão "suave" aprendida com os dados, que perde menos informação por dimensão. A vantagem desaparece quando d se aproxima de K: em d = 512 o PCA só remove metade das dimensões do K = 1024, e o K = 512 direto já chega.

O preço é o custo de construção, que o PCA não elimina. O codebook K = 1024 demora 87 s a ajustar (19–31 s para K = 128–256). O encoding por imagem sobe para 152 ms (contra 41–56 ms para K = 128–256), porque cada descritor é comparado com 1024 palavras.

Isto está de acordo com Jégou et al. (2012), onde uma VLAD com mais centros reduzida por PCA supera uma mais pequena com a mesma dimensão, e com o argumento das co-ocorrências de Jégou & Chum (2012).

#### Qual é o compromisso entre accuracy/mAP, memória, tempo de extração e tempo de inferência?

A extração domina o custo total. O dense SIFT custa ~130 ms por imagem e o encoding mais 29–195 ms (no BoVW cresce com K: 29 ms com K = 32 e 152 ms com K = 1024; VLAD 66–77 ms; FV 127–195 ms). A inferência do SVM custa microssegundos. Os tempos foram medidos em 5 imagens, no CPU, e são indicativos. O PCA não reduz este custo, porque é preciso construir primeiro a representação completa, e o fit do codebook também cresce com K (BoVW: 31 s com K = 256, 87 s com K = 1024).

O que o PCA reduz é a memória e o treino do classificador. O FV K = 64 passa de 161 KB para 0.5 KB por imagem (322× menos, em float32) e o treino do SVM de 2.8 s para 0.01 s. O custo é −0.048 de mAP em d = 128, ou −0.013 em d = 512.

A inferência por imagem fica mais lenta com o PCA (BoVW K = 1024: 0.42 → 4.3 µs; FV K = 64: 12.5 → 74 µs). Projetar de D para d custa D·d operações, contra D do produto interno do SVM. O PCA só compensa na inferência quando a projeção é feita uma vez e guardada, como num índice de retrieval.

Para a mesma memória, a VLAD e o FV com PCA dominam o BoVW. A VLAD K = 64 + PCA 128 (0.5 KB, 0.732) supera o BoVW K = 1024 completo (4 KB, 0.681). O FV K = 64 + PCA 512 (2 KB, 0.785) supera a VLAD K = 64 completa (80 KB, 0.766).

A accuracy quase não varia (0.828–0.887), porque é dominada pelo grande número de imagens de background. O macro-F1 (0.602–0.802) e o mAP são as medidas informativas.

#### A resposta muda para BoVW vs VLAD / Fisher Vector? Porque é que a dimensão se comporta de forma diferente?

A conclusão principal não muda: também na VLAD e no FV, um K maior reduzido com PCA supera um K menor para o mesmo d (d = 128: VLAD 0.678 → 0.732 e FV 0.727 → 0.750, de K = 16 para 64).

O que muda é a escala. Para o mesmo K, a VLAD (4·K·80) e o FV (4·(2·K·80 + K)) são 320 a 644 vezes maiores do que o BoVW. Por isso nenhum codebook compacto se aproxima de uma saída pequena do PCA: a condição D não existe, porque a VLAD mais pequena já tem 5120 dimensões. O PCA passa a ser o único caminho para um descritor compacto.

Estas representações também se comprimem muito melhor. Reduzir para d = 128 remove 99.4–99.7% da VLAD e do FV K = 64, contra 87.5% do BoVW K = 1024, e mesmo assim mantém uma fração semelhante do mAP:

| | mAP mantido em d = 128 |
|---|---|
| BoVW | 95–97% |
| VLAD | 95–96% |
| FV | 93–94% |

Há duas razões para isto:

- As dimensões são muito correlacionadas: são resíduos de SIFT reduzido por PCA, repetidos nos quatro tiles.
- O nº de imagens limita as direções úteis. Com 815–1473 imagens de treino por seed, o vetor de pesos do SVM linear fica no espaço gerado pelos vetores de treino (teorema do representante). Assim, das 41 216 dimensões do FV, no máximo ~1500 direções podem de facto ser usadas. As dimensões nominais exageram a capacidade e não são comparáveis entre encodings.

O FV perde um pouco mais com d pequeno, plausivelmente porque as estatísticas de 2.ª ordem (variâncias) espalham a informação por mais componentes. Em todos os d a ordem FV > VLAD > BoVW mantém-se (0.750 / 0.732 / 0.652 em d = 128). Isto está de acordo com a literatura: FV e VLAD com K pequeno superam o BoVW com K grande porque guardam mais informação por palavra (Chatfield et al., 2011; Sánchez et al., 2013), e essa informação comprime-se bem (Jégou et al., 2010; Perronnin et al., 2010).
