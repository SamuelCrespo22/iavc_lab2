## Stage H — VLAD

#### Ganho obtido só por mudar o encoding

Com os mesmos descritores, splits e SVM (C = 10) e a normalização escolhida na validação, a VLAD sobe o test mAP de 0.694 (BoVW K = 512) para 0.714 / 0.746 / 0.766 com K = 16 / 32 / 64, e melhora todas as classes. O ganho vem sobretudo da raiz quadrada com sinal: com K = 32, o val mAP passa de 0.715 (`l2`) para 0.790 (`ssr_l2`). É o mesmo efeito "bursty" do Hellinger no BoVW. A intra-normalização acrescenta um pouco com K = 32 e 64 (0.793 e 0.822).

#### Compare as dimensões nominais do BoVW (K) e da VLAD (K × D). O que muda na capacidade e no armazenamento?

O BoVW guarda um número por palavra (a contagem); a VLAD guarda D = 80 por palavra (a soma dos resíduos), ou seja, *onde* estão os descritores dentro da célula. Por isso bastam poucas palavras: a VLAD com K = 16 (5120 dimensões) já supera o BoVW com K = 512 (0.714 contra 0.694). A dimensão nominal exagera a capacidade, porque todas as configurações separam o treino na perfeição (train AP 1.000) e o ganho só se vê no test.

Em custos, o codebook é mais barato (1.9–8.1 s contra 64 s) e o encoding mais rápido (64–78 contra 102 ms por imagem). Em troca, o armazenamento é maior (20–80 KB por imagem contra 16 KB; 101–384 MB em disco contra 12 MB), e o SVM fica mais caro: treino 0.5–1.1 s contra 0.16 s, inferência 91–458 µs contra 20 µs.

## Stage I — Fisher Vector

#### Comparação FV vs VLAD vs BoVW

Com o GMM ajustado só em descritores de treino e o mesmo SVM e splits, o FV (2x2 tiles, `ssr_l2`) obtém 0.775 / 0.784 / 0.803 de test mAP com K = 16 / 32 / 64. É o melhor encoding em todos os K e em todas as classes; contra o BoVW, ganha mais no motorbike (+0.178) e no horse (+0.108). Para uma dimensão semelhante supera a VLAD (FV K = 16 com 10 304 dimensões: 0.775; VLAD K = 32 com 10 240: 0.746).

#### Que informação extra codifica o FV face à VLAD, e que custos introduz?

O FV acrescenta três coisas à VLAD:
- estatísticas de 2.ª ordem (desvios das variâncias, por componente e dimensão), além da média dos resíduos;
- atribuição suave, pesada pela probabilidade a posteriori de cada gaussiana, em vez de um único centro;
- resíduos normalizados pelo desvio padrão e pelo peso de cada componente.

Por isso a dimensão é 2·K·D (+ K) por tile, o dobro da VLAD para o mesmo K.

Os custos crescem em todo o lado:
- o EM do GMM demora 14.7–55.7 s, cerca de 7× o k-means;
- o encoding demora 99–175 ms por imagem, contra 64–78 ms, porque cada descritor exige a posteriori de todas as componentes;
- o armazenamento é 4× o da VLAD (dobro das dimensões, em float64: 80.5 KB contra 20 KB com K = 16, 1.7 GB em disco com K = 64);
- o SVM também fica mais caro (treino 1.2–3.3 s, inferência 205–807 µs).

Compensa: o FV com K = 16 já supera todas as VLAD.
