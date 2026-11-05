## Stage A

#### Em que diferem as features aprendidas dos descritores locais handcrafted? Mais dimensão dá necessariamente melhor AP?

O pipeline handcrafted fixa o descritor local (histogramas de gradientes), aprende o codebook sem labels e agrega sem ordem; só o SVM usa labels. Uma CNN pré-treinada aprende uma hierarquia não-linear com 1.28 M imagens rotuladas do ImageNet, que inclui classes próximas das nossas, e as unidades do layer4 respondem a partes e objetos.

O resultado é um salto grande: a ResNet-18 layer4 obtém 0.964 de test mAP, contra 0.784 (FV), 0.746 (VLAD) e 0.694 (BoVW). Também extrai mais depressa no CPU (18 contra 210–300 ms por imagem).

Mais dimensão não implica melhor AP:
- o layer4 (512-d) bate o FV (20 608-d) por 0.18;
- o layer2 (128-d) já empata com o FV;
- o melhor modelo, o DINOv2, tem só 384-d (0.984).

Dentro da ResNet, o ganho vem da profundidade (layer2 → 3 → 4: 0.781 → 0.903 → 0.964) (Razavian et al., 2014).

## Stage B

#### Porque é que o DINOv2 é útil com poucos exemplos, sem ter sido treinado como classificador destas categorias?

O DINOv2 foi treinado sem labels em 142 M imagens, com objetivos que obrigam vistas diferentes da mesma imagem a ter o mesmo embedding. O resultado é um espaço onde imagens semelhantes ficam juntas, e categorias como "cavalo" ou "avião" formam grupos quase linearmente separáveis. O SVM só tem de encontrar a direção que separa um grupo já formado, o que pede poucos exemplos, porque as invariâncias (pose, luz, fundo) já estão nas features.

Com o DINOv2 congelado e um SVM linear obtém-se 0.981 de test mAP (CLS), acima da ResNet-18 (0.964). Com só 5 positivos por categoria chega a 0.945 de val mAP, quando o BoVW com todos os positivos fica em 0.717 (Oquab et al., 2023).

#### Que informação se perde ao substituir os patch tokens por um único descritor global?

Os 256 patch tokens guardam *onde* está cada coisa. Um vetor global perde a disposição espacial, o nº e tamanho relativo dos objetos e a separação entre objetos que coexistem (cavaleiro e cavalo). Com a média dos patches, um objeto pequeno pesa pouco e é diluído pelo fundo; o CLS é um resumo dominado pelo conteúdo mais saliente.

O mAP é parecido (CLS 0.981, média 0.984), mas as confusões diferem. Com o CLS só 8 de 37 motos isoladas são bem classificadas (18 vão para car), contra 21 com a média, e as imagens de background classificadas como person descem de 67 para 31. Como os SVMs one-vs-background não estão calibrados entre si, a comparação é indicativa. Os erros que sobram são quase sempre objetos pequenos ou parcialmente visíveis.

## Stage C

#### Quanto é que a qualidade da representação reduz a dependência de exemplos rotulados?

Em mais de uma ordem de grandeza. Com o DINOv2, o val mAP é 0.779 com 1 positivo por categoria, 0.945 com 5, 0.966 com 20 e 0.980 com todos (90–821). Um único positivo já supera o BoVW + Hellinger treinado com todos os positivos (0.717).

A curva satura cedo: de 20 positivos para todos (4–40× mais labels) ganha só 0.014. Continuam a melhorar as classes com mais variação intra-classe, person (0.949 → 0.978) e car (0.905 → 0.945); o aeroplane chega a 1.000 com 20. Com 1 positivo o resultado depende muito da imagem sorteada (motorbike ±0.36).

#### Quando é que o flip horizontal prejudica? Média das features é equivalente a média dos scores?

O flip prejudica quando o label depende da orientação: texto, dígitos, sinais de trânsito, objetos com lado definido. Nas nossas 5 classes preserva o label, e não prejudica nem ajuda:
- no treino muda o val mAP no máximo 0.005;
- no test (TTA) dá 0.9823 sem TTA, 0.9816 com média dos scores e 0.9818 com média das features.

Plausivelmente é porque o DINOv2 já foi treinado com flips, e as duas vistas têm quase o mesmo embedding.

As duas médias não são equivalentes em geral. Para s(x) = w·φ(x) + b só coincidem se φ for linear; nós normalizamos em L2, o que introduz um fator de escala por imagem (≈ 1/cos(θ/2), θ o ângulo entre as duas features) que pode mudar o ranking. Aqui θ é pequeno, e por isso os valores quase coincidem.

## Stage D

#### Como é que a regularização afeta o compromisso bias–variance com poucos positivos?

Um C pequeno é mais regularização: um w mais curto, que tolera violações da margem.
- **Com 5 positivos:** o train AP é 1.000 para todos os C (5 pontos em 384-d são sempre separáveis), por isso só a validação revela o sobreajuste. O melhor é a regularização mais forte (C = 0.1: 0.955 ± 0.008); a partir de C = 10 a solução já não muda (0.945 ± 0.013). Com tão poucos positivos, a solução sem regularização é definida por poucos support vectors e varia muito com a amostra; mais bias compensa a menor variância.
- **Com todos os positivos:** aparece a curva clássica. C = 0.1 sub-ajusta (val 0.985), C ≥ 100 sobreajusta (0.972) e o ótimo é C = 1 (0.989): o C ótimo cresce com os dados. No test, os C escolhidos dão 0.949 (5 positivos) e 0.987 (todos), acima dos 0.981 com o C = 10 fixo.

## Stage E

#### Em que difere o zero-shot do CLIP de um SVM linear nos embeddings de imagem do CLIP?

O zero-shot é um classificador linear cujo vetor de pesos é o embedding do texto do prompt, sem bias e sem ver imagens do dataset. Mede quão bem a imagem corresponde à frase, não o que a separa do background deste dataset. O SVM usa os mesmos embeddings, mas aprende w e b com labels, e adapta-se aos negativos reais e à definição de cada classe neste dataset ("person" inclui multidões e cavaleiros).

Nos mesmos embeddings, o SVM obtém 0.991 de test mAP contra 0.826 do zero-shot. A diferença é maior no person (0.991 contra 0.583, perto do aleatório de 0.477) e menor no aeroplane (0.998 contra 0.982). Os falsos positivos do zero-shot são conceitos vizinhos que não estão nos prompts: bicicletas → motorbike, vacas → horse, pássaros no céu → aeroplane. O zero-shot não precisa de labels; o SVM é muito mais preciso (Radford et al., 2021).

#### Quão sensíveis são as previsões ao prompt? Que riscos tem escolher os prompts no test?

Por classe, muito; em média, pouco. Entre os 4 conjuntos de prompts o val mAP só varia entre 0.844 e 0.857, mas as classes mudam até 0.065 e em sentidos opostos: de "nome" para "a photo of a …", o person sobe de 0.596 para 0.656 e o horse desce de 0.949 para 0.885. O ensemble de 8 templates (0.856) não supera o prompt único (0.857). No zero-shot o prompt *é* o classificador, e como as semelhanças imagem–texto variam num intervalo estreito, pequenas mudanças trocam a ordem.

Escolher no test daria uma estimativa otimista ajustada ao ruído, deixaria de ser zero-shot (os labels de test seriam usados) e tornaria injusta a comparação com os outros métodos, escolhidos na validação. O risco é real: o conjunto escolhido na validação (0.857) cai para 0.826 no test, mais do que toda a diferença entre conjuntos.

## Stage F

#### As regiões realçadas correspondem ao objeto? O que se pode e não se pode concluir da saliência?

Muitas vezes, mas nem sempre. Nos verdadeiros positivos, a ResNet realça sobretudo o objeto. O DINOv2 faz o mesmo no person e no car, mas no aeroplane e na motorbike espalha-se pelo céu e pela estrada. Os falsos positivos mostram dependência de contexto, e as mesmas imagens enganam a ResNet e o DINOv2: um barco num atrelado na rua como car, um dirigível como aeroplane.

Os nossos mapas são uma decomposição exata do score do SVM (média do mapa + bias = score), por isso mostram com rigor quanto cada posição contribuiu para esta decisão e revelam o uso de contexto. Não mostram que a evidência está no pixel realçado: cada posição do layer4 vê quase a imagem toda e a self-attention do ViT mistura os patches. Também não provam causalidade (seria preciso tapar o objeto ou o fundo e medir o score), mostram só a parte positiva, com resolução baixa (7×7, 16×16) e em poucos exemplos (Adebayo et al., 2018).

#### Que categorias e condições continuam difíceis, e que evidência o suporta?

O car e o person são as mais difíceis e, para os modelos mais fracos, também o horse e a motorbike. Os AP mais baixos são car (0.967) no DINOv2, horse e car (0.954) na ResNet e person (0.583) no CLIP zero-shot. Car e person têm a maior variação intra-classe e aparecem muitas vezes como elementos secundários de cenas de rua.

As condições difíceis são objetos pequenos ou distantes, no bordo ou fora do crop central de 224 px, silhuetas em contra-luz e cenas muito cheias. A evidência mais forte é que modelos treinados de forma muito diferente falham nas mesmas imagens: um cavaleiro minúsculo numa praia (horse 3611) e uma mota que não aparece no crop (8742) falham na ResNet, no DINOv2 e no CLIP. Quando modelos independentes concordam num erro, a causa está na imagem e não nas features.

A co-ocorrência também pesa: só 37 das 125 motos e 31 dos 148 cavalos do test têm uma única categoria, e os SVMs one-vs-background nunca viram as outras classes como negativos.

---

**Referências citadas**

- Adebayo et al., *Sanity checks for saliency maps*, NeurIPS 2018.
- Oquab et al., *DINOv2: learning robust visual features without supervision*, TMLR 2024.
- Radford et al., *Learning transferable visual models from natural language supervision*, ICML 2021.
- Razavian et al., *CNN features off-the-shelf: an astounding baseline for recognition*, CVPR Workshops 2014.
