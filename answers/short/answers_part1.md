## Stage A

#### Como escolher o número de palavras visuais?

Na validação, nunca no test: escolhe-se o K a partir do qual o val mAP satura, pesado pelo custo. Com tiles e Hellinger, K = 256 / 512 / 1024 dá 0.705 / 0.717 / 0.712, por isso escolhemos K = 512, que tem metade do custo de K = 1024.

#### Como é que o tiling espacial codifica o layout? Quando reduz a generalização?

Os 2x2 tiles guardam em que quadrante aparece cada palavra (céu em cima, rodas e asfalto em baixo). Ajuda as classes com layout estável (motorbike: +0.109 de test AP). Prejudica quando o objeto muda de posição ou escala (car −0.011, person −0.001), ou quando a dimensão 4K é grande para os dados: com K = 1024, no horse, o global ganha 0.10 na validação.

#### Como é que o tamanho do vocabulário afeta capacidade, erro de quantização, computação e armazenamento?

Um K maior reduz o erro de quantização (0.476 → 0.419 de K = 256 para 1024, quase igual em treino e test) e aumenta a capacidade. O custo cresce linearmente com K: fit do k-means 31 → 87 s, extração 186 → 282 ms por imagem, 8 → 32 KB por imagem, treino do SVM 0.07 → 0.28 s. O ganho de AP satura a partir de K = 512.

## Stage B

#### O que se espera dos resultados?

Um treino quase perfeito e um test bem pior: 2048 dimensões para 96 positivos são fáceis de separar. Esperava-se também que o modelo usasse o contexto, porque o BoVW não sabe onde está o objeto. Confirmou-se: AP 1.000 no treino e 0.632 no test.

#### As imagens do topo partilham características do objeto, do contexto ou do fundo?

Partilham as duas coisas: motos vistas de lado, mas quase todas em pista. As palavras com mais peso no top-1 do test são asfalto, barreiras de pneus e piso da pista, não a mota. Um ganso em água lisa chega ao rank 18 do test por causa de uma dessas palavras.

#### Porque é que um classificador pode ter treino alto e falhar em imagens novas?

Com muitas dimensões e poucos positivos, o SVM separa o treino memorizando particularidades e correlações espúrias (fundos, pistas específicas) que não se repetem em imagens novas. Por isso as escolhas são feitas no val AP e não no train AP.

## Stage C

#### Porque é que o bias é irrelevante para a ordenação?

O score é s(x) = w·x + b. Somar a mesma constante a todas as imagens não muda a ordem, e o AP e a curva PR só dependem da ordem. Verificado: AP = 0.632 com b = −3.47 (o aprendido), 0 ou +96.5. O bias só fixa o threshold de classificação: com s > 0, P = 0.70 e R = 0.49.

#### Como é que a ordem de positivos e negativos afeta a curva PR e o AP?

O AP é a média da precisão nas posições dos positivos, por isso pesa sobretudo o topo do ranking. Um negativo no início baixa a precisão de todos os positivos seguintes. Mover o positivo do rank 1 para o fim baixa o AP de 0.632 para 0.594; trazer o último positivo para o rank 18 só o sobe para 0.650. Um ranking aleatório dá um AP igual à prevalência (0.104).

## Stage D

#### Como explicam a variação intra-classe, o nº de amostras, a escala e o fundo as diferenças de AP?

O AP bruto não é comparável entre classes, porque o aleatório é a prevalência (0.104 em motorbike, 0.477 em person). Normalizado, (AP − aleatório) / (1 − aleatório), a ordem é aeroplane 0.654, car 0.651, horse 0.615, motorbike 0.589 e person 0.523.

- **Nº de amostras:** não explica tudo. O person tem 821 positivos e é o pior; o aeroplane tem 90 e é o melhor.
- **Variação intra-classe e escala:** o person aparece de corpo inteiro, cortado ou minúsculo, e um histograma da imagem inteira dilui objetos pequenos. 27 pessoas do test ficam nos últimos 10% do ranking.
- **Fundo:** os aviões têm um contexto consistente (céu, pista) que o background não tem.
- **Co-ocorrência:** 78% dos cavalos e 68% das motos do test têm pessoas. Como os negativos são só background, o SVM pode usar "pessoa + relva/pista" sem ser penalizado.

## Stage E

#### Como é que a normalização L2 afeta a self-similarity sob um kernel linear?

Com L2, K(h,h) = ‖h‖₂² = 1 para qualquer imagem: todas ficam na esfera unitária e o kernel passa a ser o cosseno entre histogramas. Compara a forma da distribuição de palavras e não a quantidade de descritores.

#### O que se conclui sobre K(h,h) de um histograma normalizado em L2?

É constante e igual a 1: nenhuma imagem é "mais semelhante a si própria" do que outra. O score deixa de depender do tamanho da imagem (a correlação entre o score e o nº de descritores passa de −0.20 para 0.00).

#### Como se compara K(h,h) com K(h,h′) para dois histogramas distintos normalizados em L2?

Pela desigualdade de Cauchy–Schwarz, K(h,h′) ≤ ‖h‖‖h′‖ = 1 = K(h,h), com igualdade só se tiverem a mesma direção. Como os histogramas são não-negativos, K(h,h′) ∈ [0, 1]. Medido em 2000 pares: máximo 0.98, mediana 0.30.

#### O mesmo vale para histogramas sem normalização ou normalizados em L1?

Não.
- **Sem normalização:** K(h,h) = ‖h‖² cresce com o nº de descritores (8 600 a 53 000 por imagem), e há pares em que K(h,h′) é 6.5× maior do que K(h,h).
- **Com L1:** a soma é 1, mas K(h,h) = Σhₖ² depende de quão concentrado é o histograma (0.001 a 0.198). Também há pares com K(h,h′) até 4× maior do que K(h,h).

#### Há correlação entre o desempenho e o uso de L2?

Com tiles sim: o L2 é o melhor em todos os K (val mAP 0.621 / 0.628 / 0.604, contra 0.557 / 0.607 / 0.592 sem normalização e 0.559 / 0.570 / 0.558 com L1). Com o histograma global as três ficam iguais (±0.01). O ganho é modesto; o grande salto vem do mapa √ (0.717, Stage F). O L1 também é prejudicado pela escala minúscula dos vetores com C fixo (train AP 0.80 no motorbike).

#### Porque é que remover a informação espacial ajuda umas classes e prejudica outras?

As classes com layout estável perdem com o histograma global: a motorbike perde 0.109 de test AP (rodas e asfalto em baixo, barreiras em cima). As classes que aparecem em qualquer posição e escala ganham ou ficam iguais (car +0.011, person +0.001), porque fixar "o quê está onde" não generaliza. O global também tem 4× menos dimensões, o que ajuda quando há poucos dados: no horse com K = 1024 (4096 dimensões para 111 positivos), o global ganha 0.10 na validação.

## Stage F

#### Como normalizar para o kernel de Hellinger? Antes ou depois da raiz?

O kernel de Hellinger usa histogramas normalizados em L1 antes da raiz: φ(h) = √(h / ‖h‖₁). Isto é idêntico a tirar a raiz e normalizar em L2 depois, porque ‖√(h/‖h‖₁)‖₂² = 1 (verificado numericamente). Assim respeita-se também a regra do L2. Normalizar em L1 depois da raiz está errado: a norma volta a variar entre imagens e o val mAP cai de 0.717 para 0.597.

#### Porque é que a raiz muda a contribuição das palavras frequentes e raras?

A raiz comprime muito mais as contagens altas do que as baixas (100 → 10, 1 → 1). Os histogramas BoVW são "bursty": texturas repetidas (céu, asfalto) enchem poucas palavras. Com o kernel linear, 70% da self-similarity vem das 10 palavras mais frequentes; com a raiz são 21%, e o nº efetivo de palavras passa de 21 para 200. As palavras raras, mais específicas do objeto, ganham peso: o test mAP sobe de 0.593 para 0.694.

## Stage G

#### Que desempenho se obtém com o kernel linear? E com o Hellinger?

Com todos os dados, o linear obtém val mAP 0.626 (test 0.593) e o Hellinger 0.716 (test 0.694). O Hellinger é melhor em todas as frações, por +0.07 a +0.10, e com metade dos dados (0.650) já supera o linear com todos.

#### O desempenho satura? Mais exemplos ou mais diversidade ajudariam mais?

Não satura: de 50% para 100% dos dados o mAP ainda sobe 0.07 (Hellinger) e 0.09 (linear). Mais exemplos devem ajudar, mas a diversidade parece mais útil:
- com poucos dados, a variância entre seeds chega a ±0.12, ou seja, *quais* imagens entram pesa tanto como *quantas*;
- o horse linear quase não melhora com o dobro dos dados (0.49 → 0.50);
- o train AP é ≈ 1 em todas as frações.

Imagens noutros contextos (motos na rua, cavalos sem cavaleiro) atacam diretamente os erros dos Stages B e D.
