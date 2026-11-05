## Stage H

#### Ganho obtido só por mudar o encoding

Com os mesmos descritores SIFT, os mesmos splits, o mesmo SVM linear (C = 10) e a normalização escolhida na validação, trocar o BoVW pela VLAD sobe o mAP de test de 0.694 (BoVW K = 512) para:

VLAD mAP de test ganho sobre o BoVW
K = 16 0.714 +0.020
K = 32 0.746 +0.052
K = 64 0.766 +0.072
Com K = 64, todas as classes melhoram: motorbike 0.632 → 0.745, aeroplane 0.691 → 0.765, person 0.750 → 0.798, horse 0.662 → 0.717 e car 0.734 → 0.805.

Os 2x2 tiles são sempre melhores do que o vetor global (val mAP com K = 16: 0.740 contra 0.660).

A normalização é decisiva. A raiz quadrada com sinal (power normalization) é a que mais pesa: com K = 32, o val mAP passa de 0.715 (l2) para 0.790 (ssr_l2). A intra-normalização dá um pequeno ganho extra com K = 32 e 64 (0.793 e 0.822). É o mesmo efeito do Hellinger no BoVW: as texturas repetidas ("burstiness") acumulam resíduos grandes em poucas palavras, e a raiz reduz o seu peso.

O none e o l2 dão quase o mesmo resultado. Cada tile já vem normalizado em L2 pelo encoder, e por isso o vetor inteiro tem sempre norma 2. Dividir por 2 só muda a escala, o que com C fixo equivale a mudar ligeiramente a regularização.

#### Para K palavras e descritores de dimensão D, compare as dimensões nominais do BoVW (K) e da VLAD (K × D). O que muda na capacidade de representação e no armazenamento?

Capacidade. O BoVW guarda um número por palavra: quantos descritores caíram nela (estatística de ordem 0). A VLAD guarda D números por palavra: a soma dos resíduos x − cₖ dos descritores atribuídos a essa palavra (estatística de 1.ª ordem). Isso diz onde, dentro da célula de Voronoi, estão os descritores, e não só quantos são.

Cada palavra passa a ser muito mais informativa, por isso bastam poucas: a VLAD com K = 16 (5120 dimensões) já supera o BoVW com K = 512 (2048 dimensões) (0.714 contra 0.694). Para o mesmo K a dimensão é D = 80 vezes maior: com tiles, 4·K·80 contra 4·K.

A dimensão nominal exagera a capacidade real. Com ~900–1600 imagens de treino, o vetor de pesos do SVM linear fica no espaço gerado por essas imagens, e todas as configurações separam o treino na perfeição (train AP = 1.000). O ganho da VLAD vê-se no val e no test, porque a informação extra generaliza melhor, e não por ter mais dimensões.

#### Custos.

- Codebook mais barato: como bastam poucas palavras, o PCA + k-means demora 1.9–8.1 s, contra 64 s do BoVW K = 512.
- Encoding mais rápido: 64–78 ms por imagem contra 102 ms, porque cada descritor é comparado com 16–64 centros e não com 512.
- Armazenamento maior: 20–80 KB por imagem em float32, contra 16 KB do BoVW guardado em int64. Em disco são 101–384 MB contra 12 MB.
- SVM mais caro: o treino passa de 0.16 s para 0.50–1.05 s e a inferência de 20 µs para 91–458 µs por imagem.

O SIFT é igual para todos (~150 ms por imagem). Os tempos foram medidos em 5 imagens, no CPU, e são indicativos.

## Stage I

#### Comparação FV vs VLAD vs BoVW (mesmo classificador e mesmos splits)

O GMM (e o PCA para 80 dimensões) foi ajustado apenas com descritores das imagens de treino do protocolo: 733 imagens e 293 200 descritores, os mesmos do BoVW e da VLAD. A melhor configuração em todos os K é 2x2 tiles com ssr_l2, a "improved FV" sem o passo de normalização por componente.

mAP de test dimensão KB / imagem ajuste do codebook encoding / imagem
BoVW K = 512 0.694 2048 16 64 s 102 ms
VLAD K = 16 / 32 / 64 0.714 / 0.746 / 0.766 5120–20 480 20–80 1.9–8.1 s 64–78 ms
FV K = 16 / 32 / 64 0.775 / 0.784 / 0.803 10 304–41 216 80.5–322 14.7–55.7 s 99–175 ms
O FV é o melhor encoding em todos os K e em todas as classes. Em relação ao BoVW, o FV K = 64 ganha mais nas classes onde o BoVW era fraco:

classe BoVW → FV K = 64 ganho
motorbike 0.632 → 0.810 +0.178
horse 0.662 → 0.770 +0.108
aeroplane 0.691 → 0.788 +0.097
car 0.734 → 0.828 +0.094
person 0.750 → 0.820 +0.070
Para uma dimensão semelhante, o FV supera a VLAD: o FV K = 16 (10 304 dimensões, 0.775) bate a VLAD K = 32 (10 240 dimensões, 0.746). Até supera a VLAD K = 64 (0.766), que tem o dobro das dimensões.

#### Que informação extra codifica o Fisher Vector face à VLAD, e que custos computacionais introduz?

Informação extra. O FV acrescenta três coisas à VLAD:

Estatísticas de 2.ª ordem. Para cada componente e cada dimensão guarda também o desvio dos quadrados dos resíduos face à variância do GMM, ou seja, se os descritores dessa célula estão mais concentrados ou mais espalhados do que o modelo prevê. A VLAD só guarda a média dos resíduos (1.ª ordem).
Atribuição suave. Cada descritor contribui para todos os componentes, pesado pela probabilidade a posteriori γₖ(x). Isto reduz a ambiguidade de quantização de descritores perto das fronteiras entre células. A VLAD atribui cada descritor a um único centro.
Normalização pelo modelo. Os resíduos são divididos pelo desvio padrão de cada componente e pesados pelo prior do GMM, por isso as dimensões ficam numa escala comparável. A implementação usada também acrescenta K termos de ordem 0 (pesos da mistura).
O resultado é uma dimensão de 2·K·D + K por tile, o dobro da VLAD para o mesmo K.

Custos.

Treino do codebook: o EM do GMM demora 14.7–55.7 s, contra 1.9–8.1 s do k-means da VLAD (cerca de 7× mais).
Encoding: 99–175 ms por imagem contra 64–78 ms. Cada descritor exige a posteriori de todos os K gaussianos (N × K avaliações de densidades com exponenciais) e as estatísticas de 2.ª ordem, em vez de uma única distância mínima.
Armazenamento: dobro das dimensões e guardado em float64, logo 4× a VLAD para o mesmo K (80.5 KB contra 20 KB com K = 16). O FV K = 64 ocupa 322 KB por imagem e 1.7 GB em disco. Em float32 seria metade, sem perda prática.
SVM: o treino passa a 1.15–3.28 s e a inferência a 205–807 µs por imagem, proporcionais à dimensão.
O compromisso compensa aqui: com K = 16 o FV já supera todas as VLAD e o BoVW, com um encoding só ~1.5× mais lento do que a VLAD e um custo de treino que se paga uma vez.
