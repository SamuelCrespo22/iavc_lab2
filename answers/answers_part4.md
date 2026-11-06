Proposta de resposta para cada pergunta da Parte 4, com os números de onde vem, o raciocínio e a comparação com a literatura. Os números vêm de [report_part4.md](../code/results/report_part4.md), e de `part4_examples.csv`. As explicações estão em português; as respostas propostas estão em inglês, como o resto do relatório.

Contagens úteis (split de teste, positivos contra 1077 imagens de background): aeroplane 126, motorbike 125, horse 148, car 337, person 983. Positivos de treino: 90 / 96 / 111 / 298 / 821.

---

## Stage A — Em que diferem as features aprendidas dos descritores locais handcrafted? Mais dimensão dá necessariamente melhor AP?

**Os dados:**

| representação | dim | test mAP | extração (ms/img) | inferência (µs/img) |
|---|---|---|---|---|
| BoVW K512 (Hellinger) | 2048 | 0.694 | 254 | 34 |
| VLAD K32 | 10 240 | 0.746 | 224 | 336 |
| FV K32 | 20 608 | 0.784 | 278 | 463 |
| ResNet-18 layer2 | 128 | 0.781 | 18 | 0.8 |
| ResNet-18 layer3 | 256 | 0.903 | 18 | 1.7 |
| ResNet-18 layer4 | 512 | **0.964** | 18 | 4.3 |
| DINOv2 patchmean (Stage B) | 384 | **0.984** | 52 | 3.2 |

- O layer2 (128-d) já empata com o FV (20 608-d). O layer4 (512-d) ganha +0.18 mAP ao FV com 40× menos dimensões.
- Dentro da ResNet, o AP sobe com a profundidade (0.781 → 0.903 → 0.964). A dimensão também sobe, mas o ganho vem da profundidade: o layer3 (256-d) bate o FV com 20 608-d.
- O melhor resultado (DINOv2, 0.984) tem a menor dimensão de todas as features profundas.
- Os ganhos são maiores nas classes mais difíceis para o BoVW: motorbike passa de 0.632 a 0.974 e horse de 0.662 a 0.954.

**O raciocínio:**
- **Handcrafted:**
  - o SIFT é um histograma de gradientes fixo, de um patch pequeno;
  - o codebook é aprendido sem rótulos e o pooling ignora a ordem;
  - a única coisa aprendida com rótulos é o SVM linear.
- **ResNet:**
  - a hierarquia não-linear foi aprendida com rótulos em 1.28 M imagens do ImageNet, que tem classes muito próximas das nossas (airliner, vários tipos de carro, moped, sorrel...);
  - cada posição do layer4 vê quase a imagem toda e responde a partes e objetos, não a gradientes.
- **Porque mais dimensão não basta:** a dimensão mede o tamanho do vetor, não a informação nem a separabilidade linear. A Parte 3 já mostrou que o FV comprime de 41 216 para 512 dimensões quase sem perder AP.

**Literatura:** está de acordo.
- Features de CNN congeladas com um SVM linear superam BoVW e FV (Donahue et al., 2014; Razavian et al., 2014; Chatfield et al., 2014).
- Chatfield et al. (2014) mostram também que uma CNN reduzida a 128-d perde pouco.
- As camadas mais profundas são mais semânticas e transferem melhor para categorias próximas das de pré-treino (Zeiler & Fergus, 2014; Yosinski et al., 2014).

**Proposta de resposta:**
> Handcrafted pipelines fix the local descriptor (SIFT gradient histograms of small patches), learn the codebook without labels and pool the codes in an orderless way. The only part learned from labels is the linear SVM. A pretrained CNN learns a non-linear hierarchy end-to-end from 1.28 M labeled ImageNet images, which include classes very close to ours (airliners, car types, mopeds, horses). Its last-layer units have receptive fields covering most of the image and respond to object parts and objects rather than to gradients. The result is a large gap: ResNet-18 layer4 reaches 0.964 test mAP against 0.784 for FV, 0.746 for VLAD and 0.694 for BoVW. The gap is largest on the classes BoVW found hardest (motorbike 0.632 → 0.974, horse 0.662 → 0.954). Extraction is also cheaper on our CPU (18 ms vs. 220–280 ms per image), as is inference (4 µs vs. up to 460 µs). A higher dimension does not imply a better AP. The 512-d layer4 beats the 20 608-d FV by 0.18, layer2 with 128 dimensions already matches FV, and the best model, DINOv2 (Stage B), has the smallest feature (384-d, 0.984). Within ResNet, AP grows from layer2 to layer4 (0.781 → 0.903 → 0.964) together with the dimension, but the cause is depth: layer3 (256-d) beats FV with 80× fewer dimensions. Dimension measures size, not how linearly separable the classes are. This agrees with off-the-shelf CNN results (Donahue et al., 2014; Razavian et al., 2014; Chatfield et al., 2014) and with the increasing semantic level of deeper layers (Zeiler & Fergus, 2014; Yosinski et al., 2014).

---

## Stage B, Q1 — Porque é que o DINOv2 é útil com poucos exemplos, se não foi treinado como classificador destas categorias?

**Os dados:**
- O DINOv2 CLS dá 0.981 de test mAP e o patchmean 0.984, com 384 dimensões. É mais do que a ResNet supervisionada (0.964) e o ViT-B/32 supervisionado (0.961).
- **Stage C:** com 5 positivos por categoria, a val mAP é 0.945; com 1 positivo já é 0.78. O BoVW (Parte 1, Stage G) fica em 0.717 de val mAP com **todos** os positivos.

**O raciocínio:**
- O DINOv2 foi treinado sem rótulos em 142 M imagens curadas, com auto-destilação (DINO) e reconstrução de patches mascarados (iBOT).
- Isto obriga a que vistas diferentes da mesma imagem tenham o mesmo embedding, e o resultado é um espaço onde imagens semanticamente parecidas ficam juntas. Os grupos de "cavalos" ou "aviões" emergem sem rótulos.
- O SVM só precisa de encontrar uma direção que separe um grupo já formado do resto. Isso pede poucos exemplos, porque o difícil (a invariância à pose, à iluminação e ao fundo) já foi aprendido.
- Os rótulos do pré-treino não importam. O que importa é que as nossas categorias são visualmente comuns nos dados de pré-treino.

**Literatura:** está de acordo.
- Oquab et al. (2023) mostram que o DINOv2 congelado com uma sonda linear compete com modelos supervisionados.
- Caron et al. (2021) mostram que as features DINO já agrupam classes (k-NN forte).
- Tian et al. (2020) concluem que, em few-shot, "a good embedding is all you need".

**Proposta de resposta:**
> DINOv2 is pretrained without labels on 142 M curated images with self-distillation and masked-patch objectives, which force different views of the same image to share an embedding. The resulting space places semantically similar images close together, so categories such as horses or aeroplanes form compact, nearly linearly separable groups even though no class name was ever used. A linear SVM then only has to find a direction separating an existing group from the rest. That needs few examples, because the hard part (invariance to pose, illumination and background) is already in the features. Our results show this: frozen DINOv2 + linear SVM reaches 0.981 test mAP (CLS, 384-d), above the supervised ResNet-18 (0.964) and ViT-B/32 (0.961). With only 5 positives per category it reaches 0.945 validation mAP, and with a single positive 0.78. BoVW with all ~280 positives per category reaches 0.717 (Part 1, Stage G). This matches the linear-probe and k-NN results of DINO / DINOv2 (Caron et al., 2021; Oquab et al., 2023) and the view that few-shot performance is mostly a property of the embedding (Tian et al., 2020).

---

## Stage B, Q2 — Que informação se perde quando os patch tokens são substituídos por um único descritor global?

**Os dados** (matrizes de confusão de teste, imagens com uma só categoria):

| | CLS | patchmean |
|---|---|---|
| motorbike corretas | 8 / 37 (18 → car) | 21 / 37 |
| aeroplane corretas | 81 / 112 (24 → person) | 87 / 112 |
| background → person | 67 | 31 |
| test mAP | 0.981 | 0.984 |

- Nos mapas da Stage F, os erros FN são quase sempre objetos pequenos, distantes ou parcialmente visíveis: um cavalo minúsculo numa praia, uma mota vista só no espelho de uma placa.

**O raciocínio:**
1. Os 256 patch tokens (16×16) guardam **onde** está cada coisa. Um vetor global perde:
   - a disposição espacial;
   - quantos objetos existem e o tamanho relativo de cada um;
   - a separação entre objetos que coexistem (cavaleiro e cavalo, motociclista e mota).
2. **Média dos patches:** um objeto que ocupa 5 % da imagem contribui só com 5 % do vetor, ficando diluído pelo fundo.
3. **CLS:** é um resumo aprendido, dominado pelo conteúdo mais saliente, e por isso tende a descrever a cena principal.
4. **Os dados sugerem que o CLS perde mais nas categorias que coexistem com outras.** A motorbike é confundida com car, e o aeroplane com person. Estas confusões também vêm de os SVMs one-vs-background não estarem calibrados entre si, por isso a comparação é indicativa.
5. A média dos patches também perde a posição. Simplesmente, agrega de outra forma.

**Literatura:** está de acordo. Oquab et al. (2023) avaliam a sonda linear com CLS concatenado com a média dos patch tokens, porque é melhor do que o CLS sozinho. A perda de layout pelo pooling sem ordem é o argumento clássico do spatial pyramid (Lazebnik et al., 2006).

**Proposta de resposta:**
> The 256 patch tokens keep where things are. A single global vector loses the spatial layout, the number and relative size of objects, the separation between co-occurring objects (rider and horse, person and motorbike) and fine local detail. With mean pooling, a small object covering a few patches contributes only a small fraction of the vector and is diluted by the background. The CLS token is a learned summary dominated by the most salient content. Both global descriptors give similar mAP (CLS 0.981, patch mean 0.984), but the confusion matrices differ. With CLS only 8 of 37 single-label motorbike images are classified correctly (18 go to car), against 21 with the patch mean. Aeroplane improves from 81 to 87 of 112, and background images predicted as person drop from 67 to 31. This suggests the CLS summary loses more of the evidence for objects that share the image with other content, although the one-vs-background SVMs are not calibrated against each other, so the comparison is indicative. Neither descriptor keeps position: the Stage F maps, which can only be built from the patch tokens, show that this information exists before pooling. Most remaining misses are small, distant or partly visible objects, exactly the case where pooling dilutes the evidence most. This is consistent with DINOv2's own linear evaluation, which concatenates the CLS token with the average of the patch tokens (Oquab et al., 2023), and with the classic argument for spatial pooling (Lazebnik et al., 2006).

---

## Stage C, Q1 — Quanto é que a qualidade da representação pré-treinada reduz a dependência de exemplos rotulados?

**Os dados** (val mAP do DINOv2 CLS, sem augmentation):

| positivos | 1 | 5 | 10 | 20 | 50 | todos (90–821) |
|---|---|---|---|---|---|---|
| mAP | 0.779 ± 0.060 | 0.945 ± 0.013 | 0.950 | 0.966 | 0.974 | 0.980 |

- **A comparação principal:** o BoVW com Hellinger (Parte 1, Stage G) tem val mAP de 0.503 com 10 % dos dados (9–83 positivos) e de 0.717 com todos. **Um só positivo com DINOv2 (0.779) já supera o BoVW com todos os dados.**
- **Satura cedo:** de 20 positivos para todos (4 a 40× mais rótulos) ganha só +0.014.
- **Por classe:**
  - aeroplane chega a 1.000 com 20 positivos e horse a 0.999 com 5;
  - person (0.949 → 0.978) e car (0.905 → 0.945) continuam a ganhar até usarem todos os positivos.
- **A variância com 1 positivo é enorme** (motorbike ± 0.356): depende muito de qual imagem calha.

**O raciocínio:**
- A representação faz quase todo o trabalho, e os rótulos só servem para escolher a direção. Por isso a curva sobe muito até 5 positivos e depois satura.
- As classes que continuam a ganhar são as de maior variação intra-classe: pessoas em todas as escalas, poses e multidões; carros de todos os ângulos. Precisam de mais exemplos para cobrir os modos da classe.
- O aeroplane é homogéneo (céu, silhueta), e poucos exemplos chegam.

**Literatura:** está de acordo. Oquab et al. (2023) e Tian et al. (2020) mostram que embeddings fortes dão few-shot quase saturado. No BoVW, ao contrário, a curva da Parte 1 ainda não satura (Stage G).

**Proposta de resposta:**
> By more than an order of magnitude. With DINOv2 (CLS, L2, linear SVM), validation mAP is 0.779 ± 0.060 with a single positive per category, 0.945 with 5, 0.966 with 20 and 0.980 with all of them (90–821). One labeled positive with DINOv2 already beats BoVW + Hellinger trained on every positive (0.717 validation mAP, Part 1 Stage G), and 5 positives beat it by 0.23. The curve saturates early: going from 20 positives to all of them (4–40× more labels) adds only 0.014. The classes that keep improving are those with the largest intra-class variation, person (0.949 → 0.978) and car (0.905 → 0.945), whose many poses, scales and viewpoints need more examples. Homogeneous classes saturate almost immediately (aeroplane 1.000 at 20 positives, horse 0.999 at 5). Very small training sets remain unreliable, though: with one positive the result depends on which image is drawn (motorbike ± 0.36). This agrees with few-shot results on strong pretrained embeddings (Tian et al., 2020; Oquab et al., 2023): the representation does most of the work and the labels only orient the classifier.

---

## Stage C, Q2 — Quando é que o flip horizontal pode prejudicar? Fazer a média das features é equivalente a fazer a média dos scores?

**Os dados:**
- **Flip no treino:** a diferença é ≤ 0.005 em todos os tamanhos (1 positivo: 0.784 contra 0.779, ± 0.06; todos: 0.982 contra 0.980). Está dentro do desvio.
- **Flip no teste (TTA, medido no val):** com treino com flip, no_tta dá 0.9823, score_avg 0.9816 e feature_avg 0.9818. Não há ganho.

**O raciocínio:**
- **Quando prejudica:** quando o rótulo depende da orientação:
  - texto, dígitos e sinais de trânsito;
  - categorias quirais ou de lado definido (mão esquerda/direita, posição do volante, setas "virar à esquerda");
  - layouts de cena com lado fixo, quando se usa informação espacial.
  
  Nas nossas 5 classes, o flip preserva o rótulo, e por isso não prejudica.
- **Porque não ajuda aqui:** o DINOv2 foi treinado com flips aleatórios como augmentation. As features da imagem e da sua versão espelhada são quase iguais, logo o flip não acrescenta informação nova.
- **Média das features e média dos scores:**
  - para um score linear s(x) = w·φ(x) + b, a média dos scores é w·(φ(x) + φ(x'))/2 + b;
  - a média das features é w·φ((x + x')/2) + b;
  - as duas só são iguais se φ for linear. Nós usamos L2, que não é linear: φ((x + x')/2) = (x + x')/‖x + x'‖, enquanto a média dos scores usa x/‖x‖ e x'/‖x'‖.
  
  A diferença é um fator de escala por imagem (≈ 1/cos(θ/2), sendo θ o ângulo entre as duas features), e pode mudar o ranking. Aqui θ é pequeno, por isso os resultados quase coincidem (0.9816 contra 0.9818).
  
  Com um classificador não-linear (SVM com kernel, MLP) as duas médias deixam de ser equivalentes. A média das imagens também não equivale à média das features, porque a rede não é linear.

**Literatura:** está de acordo.
- Chatfield et al. (2014) relatam ganhos pequenos (~1 ponto) com augmentation por flip em features CNN.
- Oquab et al. (2023) incluem o flip horizontal nas augmentations do DINOv2, o que explica a quase invariância.
- O caso em que o flip prejudica é o argumento clássico para o MNIST e o texto (Simard et al., 2003).

**Proposta de resposta:**
> Flipping harms recognition when the label depends on left-right orientation: text, digits and traffic signs, chiral or side-specific categories (left vs. right hand, steering-wheel side, turn-left arrows), or scenes with a fixed layout when spatial information is used. For our five categories, flipping preserves the label, so no harm is expected, and none is observed. It also brings no gain: training with flips changes validation mAP by at most 0.005 (0.784 vs. 0.779 with one positive, within ± 0.06; 0.982 vs. 0.980 with all), and test-time flipping, measured on validation, does not help either (0.9823 without TTA, 0.9816 averaging scores, 0.9818 averaging features). DINOv2 was pretrained with random horizontal flips, so an image and its mirror image get almost the same embedding, and the flipped copy adds little information. Averaging features is not in general equivalent to averaging classifier scores. For a linear score s(x) = w·φ(x) + b the two coincide only if the feature map φ is linear. We L2-normalize, so feature averaging scores (x + x')/‖x + x'‖, while score averaging uses x/‖x‖ and x'/‖x'‖. The difference is a per-image scale factor (≈ 1/cos(θ/2), with θ the angle between the two features) that can change the ranking. Here θ is small, hence the nearly identical results. With non-linear classifiers (kernel SVM, MLP) or when averaging images before a non-linear network, the two are clearly different.

---

## Stage D — Como é que a regularização afeta o compromisso bias–variance quando há poucos positivos?

**Os dados** (mAP de treino e de validação):

| C | 5 pos: treino | 5 pos: val | todos: treino | todos: val |
|---|---|---|---|---|
| 0.1 | 1.000 | **0.955 ± 0.008** | 0.987 | 0.985 |
| 1 | 1.000 | 0.951 ± 0.010 | 0.993 | **0.989** |
| 10 | 1.000 | 0.945 ± 0.013 | 0.999 | 0.980 |
| 100 | 1.000 | 0.945 ± 0.013 | 1.000 | 0.972 |
| 1000 | 1.000 | 0.945 ± 0.013 | 1.000 | 0.972 |

- No teste, com o C escolhido na validação: 5 positivos (C = 0.1) dão 0.949 e todos os positivos (C = 1) dão 0.987. Isto é acima dos 0.981 com o C = 10 fixo da Stage B.

**O raciocínio:**
- **Com 5 positivos, o AP de treino é 1.000 para todos os C:** 5 pontos em 384-d são sempre separáveis. O treino não dá nenhum sinal, e só a validação diagnostica o overfitting.
- **A partir de C = 10 os resultados são idênticos:** todas as restrições já estão satisfeitas (hard margin), e aumentar C não muda nada.
- **C pequeno é mais regularização:**
  - o w fica mais curto e o SVM tolera violações da margem;
  - mais pontos contribuem para o w (com α limitado a C), em vez de um punhado de support vectors o definir;
  - isto dá mais bias e menos variância, que é o que compensa quando há poucos positivos: a val sobe de 0.945 para 0.955 e o desvio desce de 0.013 para 0.008.
- **Com todos os positivos aparece a curva clássica:**
  - C = 0.1 faz underfitting ligeiro (treino 0.987);
  - C ≥ 100 faz overfitting (treino 1.000, val 0.972);
  - o ótimo fica no meio (C = 1).
  
  Com mais dados, a variância é menor e o C ótimo é maior.

**Literatura:** está de acordo com a teoria da margem e da regularização (Vapnik, 1995; Hastie et al., 2009, cap. 12). A forma de diagnóstico (curvas de treino e de validação em função de C) é a de Hsu, Chang & Lin (2003).

**Proposta de resposta:**
> C weighs the training errors against the margin: a small C means strong regularization (short w, margin violations tolerated), a large C fits the training data as closely as possible. With 5 positives, training mAP is 1.000 for every C, because 5 points are always separable in 384 dimensions. The training curve therefore carries no information, and only validation can reveal overfitting. Validation mAP is best with the strongest regularization (0.955 ± 0.008 at C = 0.1) and falls to 0.945 ± 0.013 for C ≥ 10, where all constraints are already satisfied and the solution no longer changes. With so few positives, the unregularized solution is set by a handful of support vectors and depends strongly on which positives were drawn (high variance). A small C keeps w short and lets more examples contribute with bounded weight: more bias, but less variance, which pays off. With all positives the classic trade-off appears. C = 0.1 slightly underfits (train 0.987, val 0.985), C ≥ 100 overfits (train 1.000, val 0.972), and C = 1 is best (0.989), so the optimal C grows with the amount of data. Evaluated once on the test set, the selected values give 0.949 (5 positives, C = 0.1) and 0.987 (all positives, C = 1). That is above the 0.981 obtained with the fixed C = 10, which is therefore slightly too weakly regularized for these features. This matches standard margin and regularization theory (Vapnik, 1995; Hastie et al., 2009).

---

## Stage E, Q1 — Em que difere o zero-shot do CLIP de um SVM linear treinado nos embeddings de imagem do CLIP?

**Os dados:**

| | zero-shot ("photo") | SVM nos embeddings CLIP |
|---|---|---|
| test mAP | 0.826 | 0.991 |
| person AP | 0.583 | 0.991 |
| horse AP | 0.811 | 0.992 |
| aeroplane AP | 0.982 | 0.998 |

- **O person em zero-shot está perto do acaso:** um ranking aleatório teria AP igual à prevalência, 983 / 2060 = 0.477.
- **Classificação a 6 classes:** accuracy 0.602.
  - motorbike e horse têm recall alto (0.97) e precision baixa (0.38 e 0.20);
  - 106 imagens de background são classificadas como horse e 288 como person;
  - 239 imagens de person vão para background.
- **Os FP da Stage F** são conceitos vizinhos que não estão entre os prompts: bicicletas → motorbike, pássaro no céu → aeroplane, vacas e ovelhas → horse, autocarro → car.

**O raciocínio:**
- **O zero-shot é um classificador linear com w = embedding do texto**, sem bias e sem ver nenhuma imagem do dataset. Mede "quão bem esta imagem corresponde à frase", e não "o que distingue esta categoria do background deste dataset".
- **O SVM aprende o w e o b com os rótulos.** Usa a mesma geometria do CLIP, mas ajusta-a à distribuição real dos negativos e à definição de "person" deste dataset (multidões, cavaleiros, ciclistas, pessoas pequenas). Os FN do zero-shot em person são exatamente esses: o embedding descreve "uma corrida de bicicletas" ou "cavalos", e não "uma pessoa".
- **O background é um conjunto aberto e heterogéneo.** Um único prompt ("a photo of a background scene") modela-o mal, e daí os conceitos vizinhos ganharem.
- **O AP não usa o argmax,** mas sofre do mesmo problema: as semelhanças imagem–texto variam num intervalo estreito (0.16–0.32 na Stage F). Os embeddings de imagem e de texto ocupam regiões diferentes do espaço (o "modality gap"), por isso o prompt não é um bom hiperplano para este conjunto.

**Literatura:** está de acordo. Radford et al. (2021) mostram que uma sonda linear nos embeddings CLIP supera o zero-shot quando há exemplos rotulados. Liang et al. (2022) descrevem o modality gap. O zero-shot tem a vantagem de não precisar de rótulos e de aceitar categorias novas por texto.

**Proposta de resposta:**
> Zero-shot CLIP is a linear classifier whose weight vector is the text embedding of the class prompt, with no bias and no image from our dataset involved. It measures how well an image matches a sentence, not what separates this category from this dataset's background. The SVM uses the same image embeddings but learns w and b from labeled images. It therefore adapts to the actual negatives and to this dataset's definition of each category (e.g., "person" includes crowds, riders and cyclists). On the same embeddings the SVM reaches 0.991 test mAP against 0.826 for zero-shot. The gap is largest for person (0.991 vs. 0.583, barely above the 0.477 of a random ranking at this prevalence) and horse (0.992 vs. 0.811), and smallest for aeroplane (0.998 vs. 0.982). Its 6-way classification reaches 0.602 accuracy. Motorbike and horse have high recall (0.97) but low precision (0.38 and 0.20), because neighbouring concepts that are not among the prompts take their place: bicycles become motorbikes, cows and sheep horses, birds in the sky aeroplanes, buses cars (Stage F). A single "background scene" prompt also cannot model an open, heterogeneous negative class: 288 background images are labeled person and 239 person images background. The missed persons are typically small people in scenes whose main concept is something else (a bicycle race, horses), which the SVM has learned to accept from labeled examples. Zero-shot needs no labels and accepts new categories as text; the SVM needs labels but is far more accurate. This agrees with Radford et al. (2021), where a linear probe on CLIP features outperforms zero-shot once labeled data is available, and with the image–text modality gap (Liang et al., 2022).

---

## Stage E, Q2 — Quão sensíveis são as previsões à formulação do prompt? Que riscos tem escolher os prompts no teste?

**Os dados** (val AP por conjunto de prompts):

| prompt set | motorbike | aeroplane | person | horse | car | mAP |
|---|---|---|---|---|---|---|
| name | **0.946** | 0.981 | 0.596 | **0.949** | 0.791 | 0.853 |
| photo | 0.916 | 1.000 | **0.656** | 0.885 | 0.827 | **0.857** |
| templates (8) | 0.939 | 1.000 | 0.615 | 0.892 | **0.834** | 0.856 |
| synonyms | 0.921 | 1.000 | 0.610 | 0.884 | 0.803 | 0.844 |

- **O mAP quase não muda:** 0.844 a 0.857, uma amplitude de 0.013.
- **Mas cada classe muda muito, e em sentidos opostos:**
  - person 0.596 → 0.656 (+0.06 de "name" para "photo");
  - horse 0.949 → 0.885 (−0.065 no mesmo sentido).
  
  Nenhum conjunto é o melhor para todas as classes.
- **O ensemble de 8 templates não ganha ao prompt único "photo".**
- **Selecionado na validação:** "photo", com 0.857 de val. No teste dá 0.826, uma descida de 0.031, maior do que toda a diferença entre os conjuntos.

**O raciocínio:**
- O prompt é o próprio classificador. Mudar a frase muda o w, e as semelhanças variam num intervalo estreito, por isso pequenas mudanças trocam a ordem.
- As médias por classe compensam-se, e daí o mAP parecer estável.
- **Os riscos de escolher no teste:**
  1. o resultado passa a ser uma estimativa otimista, porque a escolha está a ajustar-se ao ruído do teste;
  2. com 4 conjuntos, que é uma pesquisa pequena, o ganho possível (≈ 0.01–0.06 por classe) já é da ordem das diferenças que queremos medir;
  3. deixa de ser zero-shot, porque os rótulos de teste foram usados para escolher;
  4. a comparação com os outros métodos deixa de ser justa, já que esses foram escolhidos na validação.
- **A própria validação é pequena** (22–204 positivos): a diferença entre "photo" e "templates" (0.001) não é significativa. Uma escolha por conhecimento prévio ("a photo of a {}", de Radford et al.) seria igualmente defensável.

**Literatura:**
- Radford et al. (2021) e Zhou et al. (2022, CoOp) mostram que a formulação do prompt muda bastante a accuracy por classe.
- Radford et al. obtêm +3–5 pontos com ensembles de templates no ImageNet. Aqui não há ganho, o que é plausível com só 5 classes e com o background como principal fonte de erro, que nenhum template resolve.
- O viés otimista da seleção no teste é o de Cawley & Talbot (2010).

**Proposta de resposta:**
> Per class, quite sensitive; on average, much less. Across our four prompt sets, validation mAP only ranges from 0.844 to 0.857, but individual classes move by up to 0.065 AP and in opposite directions. From the bare name to "a photo of a …", person improves from 0.596 to 0.656 while horse drops from 0.949 to 0.885, so no wording is best for every class. Averaging 8 templates (0.856) does not beat the single "photo" prompt (0.857). This contrasts with the 3–5 point gains of prompt ensembles on ImageNet (Radford et al., 2021), plausibly because our main error source is the heterogeneous background, which no template addresses. In zero-shot recognition the prompt is the classifier: changing the sentence changes the weight vector, and since image–text similarities lie in a narrow range, small changes reorder the ranking (Zhou et al., 2022). Selecting prompts on the test set would make the reported number an optimistic estimate fitted to test noise. Even our small search yields per-class differences (0.01–0.06) as large as the effects being measured. The method would also no longer be zero-shot, since test labels were used, and the comparison with the other methods, all selected on validation, would be unfair. The size of the risk is visible here: the prompt set chosen on validation (0.857) drops to 0.826 on test, more than the whole spread between prompt sets. With 22–204 validation positives per class, even validation selection is noisy: "photo" and "templates" differ by 0.001, so a choice from prior knowledge would have been equally defensible (Cawley & Talbot, 2010).

---

## Stage F, Q1 — As regiões realçadas correspondem ao objeto? O que se pode e não se pode concluir de uma visualização de saliência?

**O que as figuras mostram:**
- **No TP, a ResNet costuma realçar o objeto:** o corpo do carro, o cavalo com o cavaleiro, o avião, as caras e os corpos das pessoas.
- **O DINOv2 realça o objeto no person e no car, mas espalha-se pelo fundo no aeroplane e na motorbike.** No aeroplane, o céu à volta do avião fica tão "quente" como o próprio avião; na motorbike, a estrada também contribui.
- **Os FP mostram o uso de contexto e de formas parecidas.** As mesmas imagens aparecem nos FP da ResNet e do DINOv2:
  - car: um barco num atrelado numa rua, ruas com autocarros, uma estação;
  - aeroplane: um dirigível "T-Systems";
  - horse: vacas no campo (ResNet).
- **O CLIP confunde conceitos vizinhos:** bicicletas → motorbike, pássaro no céu → aeroplane, vacas → horse.

**O raciocínio — o que se pode concluir:**
- Os nossos mapas são uma **decomposição exata** do score do SVM: a média do mapa mais o bias é igual ao score, e o script verifica isto. Dizem com rigor quanto cada posição da feature map contribuiu para a decisão deste SVM.
- Permitem detetar dependência de contexto (céu, estrada) e comparar padrões entre modelos.

**O que não se pode concluir:**
1. **Que a evidência está no pixel realçado.** Cada posição do layer4 da ResNet tem um campo recetivo que cobre quase a imagem toda, e no ViT a self-attention mistura todos os patches. Uma posição "no avião" pode codificar o céu, e vice-versa.
2. **Causalidade.** O mapa é correlacional. Para saber se o modelo precisa do objeto, seria preciso intervir, por exemplo tapar o objeto ou o fundo e medir a mudança do score.
3. **A imagem completa da decisão.** Só mostramos a parte positiva, por isso a evidência contra fica escondida. A resolução é baixa (7×7, 16×16) e a escala de cor é por categoria.
4. **Uma conclusão geral.** São poucos exemplos, escolhidos nos extremos do ranking.
5. **Que o modelo "raciocina" como uma pessoa.** O mapa explica este SVM sobre estas features, não o modelo em geral.

**Literatura:**
- O mapa da ResNet é um CAM com os pesos do SVM (Zhou et al., 2016).
- Adebayo et al. (2018) mostram que vários métodos de saliência produzem mapas plausíveis mesmo com pesos aleatórios, e por isso exigem sanity checks.
- O uso de contexto é o "Clever Hans" de Lapuschkin et al. (2019) e o viés de contexto de Torralba & Efros (2011) e Beery et al. (2018).

**Proposta de resposta:**
> Often, but not always. For correctly detected images, the ResNet-18 maps mostly cover the object: the car body, horse and rider, the aeroplane, faces and bodies. DINOv2 is object-centred for person and car, but its aeroplane and motorbike maps spread over the background: the sky around a plane is as strong as the plane itself, and the road contributes to motorbikes. The false positives show the same reliance on context and similar shapes, and the same images fool ResNet and DINOv2: a boat on a trailer in a street, street scenes with buses and a station scored as car, a blimp in the sky as aeroplane, cows in a field as horse (ResNet). CLIP shows the semantic version: bicycles as motorbikes, a bird in the sky as an aeroplane, cows as horses. Our maps are exact decompositions of the SVM score (map mean + bias = score, checked by the script), so they show reliably how much each feature-map position contributed to this SVM's decision, and they can reveal context use. They cannot show that the evidence lies in the highlighted pixels: a layer4 position of ResNet sees almost the whole image, and ViT self-attention mixes all patches, so a position on the plane may encode the sky. They show correlation, not causation; that would require interventions such as occluding the object or the background. They also show only the positive part, at coarse resolution (7×7, 16×16), on a few examples selected at the extremes of the ranking, so they do not support general claims about the model. This follows the cautions of Adebayo et al. (2018); our ResNet maps are CAMs built from the SVM weights (Zhou et al., 2016), and the context effects are the "Clever Hans" and dataset-bias phenomena of Lapuschkin et al. (2019), Torralba & Efros (2011) and Beery et al. (2018).

---

## Stage F, Q2 — Que categorias e condições visuais continuam difíceis, e que evidência suporta a explicação?

**Os dados:**
- **Test AP mais baixos:**

  | método | classes mais difíceis |
  |---|---|
  | DINOv2 | car 0.967, motorbike 0.980 |
  | ResNet | horse e car, ambos 0.954 |
  | ViT | person 0.937 |
  | CLIP zero-shot | person 0.583, horse 0.811 |
  | handcrafted | motorbike e horse |

  O car e as categorias com objetos pequenos repetem-se em todos os métodos.
- **Confusões (DINOv2):** motorbike → car/person, car → person (37), person ↔ background (~50 em cada sentido).
- **Só 37 das 125 motorbikes de teste e 31 dos 148 cavalos têm uma única categoria.** As restantes têm também pessoas ou carros, ou seja, as categorias co-ocorrem.
- **As mesmas imagens falham em vários modelos** (`part4_examples.csv`):
  - motorbike 8742, um portal ornamentado sem mota visível no crop, falha em ResNet, DINOv2 e CLIP;
  - horse 3611, um cavaleiro minúsculo numa praia com palmeiras, falha nos três;
  - horse 3754 e car 6588 / 9254 (comboios com um carro pequeno) falham em ResNet e DINOv2.
- **Condições comuns aos FN:**
  - objeto pequeno ou distante (cavalo ao longe, carro junto a comboios);
  - objeto fora ou no bordo do crop central 224 (a mota vista só no espelho de uma placa);
  - contra-luz ou silhueta (cavalo ao pôr do sol);
  - cenas muito cheias (uma mota num quarto atulhado);
  - pessoa pequena ou cortada (gato em primeiro plano, bicicleta, mesa de jantar).

**O raciocínio:**
- **Quando modelos com treinos tão diferentes falham na mesma imagem, a dificuldade está na imagem, não no modelo:** o objeto ocupa poucos pixels, ou fica fora do crop.
- **O crop central** (o lado menor é redimensionado para 256 e depois corta-se 224×224) remove 12.5 % do lado menor e ~34 % do lado maior numa imagem 4:3. O global pooling dilui o que resta (Stage B, Q2).
- **O car e o person são difíceis** porque têm a maior variação intra-classe e aparecem muitas vezes como elementos secundários de cenas de rua. Os FP de car são ruas sem carro (contexto).
- **As confusões entre categorias** vêm da co-ocorrência (cavaleiros, motociclistas) e de os SVMs one-vs-background nunca terem visto as outras categorias como negativos.

**Literatura:** está de acordo. Hoiem et al. (2012) identificam o tamanho do objeto, a oclusão e o ponto de vista como as principais causas de erro em VOC. Torralba & Efros (2011) e Beery et al. (2018) descrevem a dependência de contexto. Lazebnik et al. (2006) e o próprio DINOv2 notam a perda de informação local pelo pooling global.

**Proposta de resposta:**
> The hardest categories are car and person, and, for weaker models, horse and motorbike. The lowest test APs are car (0.967) and motorbike (0.980) for DINOv2, horse and car (0.954) for ResNet-18, person (0.937) for ViT-B/32, and person (0.583) and horse (0.811) for CLIP zero-shot. Car and person have the largest intra-class variation and often appear as secondary elements of street scenes. Car false positives are often street scenes without a car (buses, a boat on a trailer), and DINOv2 assigns 37 single-label car images to person. The difficult conditions are small or distant objects, objects at the border of or outside the 224-pixel centre crop, back-lit silhouettes, heavy clutter and partly visible people. The strongest evidence is that models trained very differently fail on the same images. A tiny rider on a palm-lined beach (horse 3611) and a motorbike that is not visible in the centre crop (8742) are missed by ResNet, DINOv2 and CLIP. Distant horses and small cars next to trains are missed by both ResNet and DINOv2. When independent models agree on an error, the cause is in the image (few object pixels, cropping), not in one model's features: the centre crop removes 12.5 % of the short side and about a third of the long side of a 4:3 image, and global pooling dilutes what remains (Stage B). Co-occurrence is a second cause: only 37 of the 125 motorbike test images and 31 of the 148 horse images contain a single category, because riders appear with them. The one-vs-background SVMs never saw other categories as negatives, which explains the motorbike → car / person confusions. These causes (object size, occlusion, truncation, context) match the error analysis of Hoiem et al. (2012) and the context-bias literature (Torralba & Efros, 2011; Beery et al., 2018).

---

### Referências usadas
- Adebayo et al., *Sanity checks for saliency maps*, NeurIPS 2018.
- Beery, Van Horn & Perona, *Recognition in terra incognita*, ECCV 2018.
- Caron et al., *Emerging properties in self-supervised vision transformers* (DINO), ICCV 2021.
- Cawley & Talbot, *On over-fitting in model selection and subsequent selection bias in performance evaluation*, JMLR 2010.
- Chatfield, Simonyan, Vedaldi & Zisserman, *Return of the devil in the details: delving deep into convolutional nets*, BMVC 2014.
- Donahue et al., *DeCAF: a deep convolutional activation feature for generic visual recognition*, ICML 2014.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2.ª ed., 2009.
- Hoiem, Chodpathumwan & Dai, *Diagnosing error in object detectors*, ECCV 2012.
- Hsu, Chang & Lin, *A practical guide to support vector classification*, 2003.
- Lapuschkin et al., *Unmasking Clever Hans predictors and assessing what machines really learn*, Nature Communications 2019.
- Lazebnik, Schmid & Ponce, *Beyond bags of features: spatial pyramid matching*, CVPR 2006.
- Liang et al., *Mind the gap: understanding the modality gap in multi-modal contrastive representation learning*, NeurIPS 2022.
- Oquab et al., *DINOv2: learning robust visual features without supervision*, TMLR 2024 (arXiv 2023).
- Radford et al., *Learning transferable visual models from natural language supervision* (CLIP), ICML 2021.
- Razavian et al., *CNN features off-the-shelf: an astounding baseline for recognition*, CVPR Workshops 2014.
- Simard, Steinkraus & Platt, *Best practices for convolutional neural networks applied to visual document analysis*, ICDAR 2003.
- Tian et al., *Rethinking few-shot image classification: a good embedding is all you need?*, ECCV 2020.
- Torralba & Efros, *Unbiased look at dataset bias*, CVPR 2011.
- Vapnik, *The Nature of Statistical Learning Theory*, 1995.
- Yosinski et al., *How transferable are features in deep neural networks?*, NeurIPS 2014.
- Zeiler & Fergus, *Visualizing and understanding convolutional networks*, ECCV 2014.
- Zhou et al., *Learning deep features for discriminative localization* (CAM), CVPR 2016.
- Zhou et al., *Learning to prompt for vision-language models* (CoOp), IJCV 2022.

### A confirmar antes de usar
- **Hipóteses minhas, não provadas pelos dados:**
  - a explicação de o DINOv2 ser quase invariante ao flip porque foi treinado com flips;
  - a do modality gap;
  - a ideia de o CLS perder mais do que o patchmean (as matrizes de confusão também dependem da calibração dos SVMs).
- **A imagem car 3054 (uma sala com sofá) é FN em CLIP e DINOv2.** Não se vê nenhum carro no crop: pode ser um erro de anotação ou um carro fora do crop. Não a usei como argumento.
