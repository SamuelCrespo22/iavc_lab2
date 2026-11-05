#### Um codebook maior melhora o reconhecimento antes do PCA? Para que classes?

Sim, com ganhos decrescentes. O mAP do BoVW sem PCA sobe de 0.502 (K = 32) para 0.631 (K = 256) e 0.682 (K = 1024); de K = 512 para 1024 já só ganha +0.016. O erro de quantização cai de 0.52 para 0.43 (K = 128 → 1024) e é igual em treino e test, por isso o codebook maior não sobreajusta.

De K = 256 para 1024, quem mais ganha são o motorbike (+0.116) e o horse (+0.079): objetos com estruturas locais específicas (rodas e quadro; pernas e crina) que um vocabulário grosseiro mistura. O aeroplane quase não ganha (+0.007), plausivelmente porque é reconhecido pelo contexto (céu, pista), que poucas palavras já captam. VLAD e FV mostram a mesma tendência (Chatfield et al., 2011).

#### O PCA mantém o ganho do codebook maior? Quanto se pode reduzir antes de o desempenho cair?

Em grande parte sim, e a perda é gradual. O BoVW K = 1024 com PCA para 512 / 256 / 128 dimensões dá 0.669 / 0.660 / 0.654, mantendo 75 / 57 / 45% da vantagem sobre K = 256; mesmo com d = 32 (97% removido) ainda empata com ele. Com tolerância de −0.02, removem-se cerca de 50% das dimensões no BoVW, 95% na VLAD e 98.8% no FV K = 64 (41 216 → 512: 0.797 → 0.783).

O PCA nunca melhorou o mesmo codebook sem PCA: comprime, mas não regulariza, porque guarda as direções de maior variância e não as mais discriminativas. Há uma quebra na VLAD K = 32 e 64 entre d ≈ 192 e 256 (K = 64: 0.736 → 0.717, recupera para 0.746 em d = 512). Não é um problema de convergência do SVM nem da normalização L2, e não a investigámos mais.

#### Para a mesma dimensão final, codebook grande + PCA supera um codebook pequeno?

Sim, e a vantagem é maior quanto menor for d:

| d | 32 | 128 | 256 | 512 |
|---|---|---|---|---|
| K = d, sem PCA | 0.502 | 0.590 | 0.631 | 0.666 |
| K = 1024 + PCA | 0.633 | 0.654 | 0.660 | 0.669 |

Em d = 512 a diferença está dentro do ruído. Um k-means pequeno junta descritores só pela distância no espaço SIFT; o PCA sobre um histograma grande combina as palavras que co-ocorrem nas imagens, e perde menos informação por dimensão (Jégou et al., 2012). O preço é o custo de construção, que o PCA não remove: 87 s de fit para K = 1024 (19–31 s para K = 128–256) e 180 ms de encoding por imagem.

#### Qual é o compromisso entre mAP, memória, extração e inferência?

- **Extração:** domina o custo, com ~125 ms de SIFT e 25–180 ms de encoding por imagem, contra microssegundos de inferência. O PCA não a reduz, porque a representação completa tem de ser construída primeiro.
- **Memória e treino:** é o que o PCA reduz. O FV K = 64 passa de 161 para 0.5 KB por imagem e o treino do SVM de 3.1 s para 0.01 s, por −0.041 de mAP em d = 128.
- **Inferência:** fica mais lenta com o PCA (FV K = 64: 9.6 → 85 µs), porque projetar D → d custa D·d operações. Só compensa se a projeção for guardada, como num índice.
- **Para a mesma memória:** VLAD e FV com PCA dominam o BoVW. VLAD K = 64 + PCA 128 (0.5 KB, 0.730) supera o BoVW K = 1024 completo (4 KB, 0.682).
- **Métricas:** a accuracy quase não varia (0.83–0.89) por causa do background; mAP e macro-F1 são as medidas informativas.

#### A resposta muda para VLAD / FV? Porque é que a dimensão se comporta de outra forma?

A conclusão não muda: também na VLAD e no FV, um K maior com PCA supera um K menor para o mesmo d (d = 128: VLAD 0.682 → 0.730, FV 0.720 → 0.756 de K = 16 para 64). Muda a escala: para o mesmo K, VLAD e FV são 320–644× maiores do que o BoVW. Não existe condição D comparável (a VLAD mais pequena já tem 5120 dimensões), e o PCA passa a ser o único caminho para um descritor compacto.

Reduzir a d = 128 remove 99.4–99.7% da VLAD / FV K = 64, contra 87.5% do BoVW K = 1024, e mantém uma fração semelhante do mAP (94–97%). Há duas razões:
- as dimensões são muito correlacionadas;
- com ~900–1600 imagens de treino, o SVM só pode usar tantas direções quantas imagens (teorema do representante), por isso a dimensão nominal exagera a capacidade.

Em todos os d mantém-se FV > VLAD > BoVW (0.756 / 0.730 / 0.654 em d = 128).
