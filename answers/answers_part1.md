## Stage A

#### Como escolher K?

Na validação, nunca no test. Procura-se o ponto onde o val mAP satura, pesado contra o custo. Aqui, 512 ≈ 1024 com metade do custo, logo 512. K pequeno junta descritores diferentes na mesma palavra (pouco discriminativo). K grande dá histogramas esparsos e ruidosos, e com poucos dados o SVM sobreajusta.

#### Tiling:

Guarda onde aparecem as palavras (céu em cima e avião ao centro; rodas em baixo na mota). Funciona como um spatial pyramid muito simples. Prejudica a generalização quando o objeto muda de posição, escala ou enquadramento. Também quadruplica a dimensão com os mesmos dados de treino, o que aumenta o sobreajuste. Vê-se no ganho que encolhe com K e no caso do horse.

#### Efeito do K:

Mais capacidade e menos erro de quantização (0.476 → 0.419). Em troca, custo linear em K para k-means, atribuição, armazenamento e SVM, e cada histograma fica mais esparso. Os números estão todos na tabela acima.

## Stage B

#### O que se espera?

Treino quase perfeito (alta dimensão, poucos positivos, dados separáveis) e um test bastante pior. Também é de esperar que o modelo use contexto (estrada, pista), porque o BoVW não sabe onde está o objeto. Confirmou-se: 1.000 vs 0.632.

#### O que partilham as imagens do topo?

Partilham tanto o objeto (rodas e carenagem, vistas de lado) como o contexto: asfalto, pistas, barreiras. As visual words mostram que o contexto pesa muito, ao ponto de um ganso em água lisa chegar a #18 no test.

#### Porque é que treino alto ≠ generalização?

Com 2048 dimensões e 96 positivos, o SVM consegue separar os dados de treino quase de qualquer maneira. Memoriza particularidades (fundos, pistas específicas) e correlações espúrias. Por isso é que a escolha de configurações é feita na validação e não no train AP.

## Stage C

#### Porque é que o bias não importa para a ordenação?

O score é s(x) = w·x + b. Somar uma constante a todos os scores não muda a ordem entre eles. A ordem fica idêntica e o AP é 0.6318 em todos os casos. O AP e a curva PR só dependem da ordem. O bias só decide onde está o threshold, ou seja, quantas imagens são classificadas como positivas. Mudar o b move-nos ao longo da mesma curva PR.

O b é bastante negativo (−3.47) porque há muito mais negativos do que positivos (815 vs 96).

#### Como é que a ordem afeta a curva e o AP?

Duas trocas no ranking real:

- Mandar o positivo do rank 1 para o fim baixa o AP de 0.632 para 0.594 (−0.038).
- Trazer o último positivo (rank 1185) para o rank 18 sobe o AP de 0.632 para 0.650 (+0.018).

Ou seja, o AP é sensível ao topo do ranking. Um negativo no topo baixa a precisão de todos os positivos que vêm a seguir, e um positivo perdido no fim só conta na última parcela. Com 10% de positivos, cada negativo acima de um positivo pesa muito, e por isso o AP aleatório é 0.104 e não 0.5.

## Stage D

#### Como é que a variação intra-classe, o nº de amostras, a escala e a diversidade do fundo explicam as diferenças de AP?

O AP bruto não é comparável entre classes: um ranking aleatório já dá um AP igual à fração de positivos (0.104 em motorbike, 0.477 em person). Normalizando, (AP − aleatório)/(1 − aleatório), a ordem muda: aeroplane 0.654 > car 0.651 > horse 0.615 > motorbike 0.589 > person 0.523. O person, que tem o AP mais alto (0.750), é afinal a classe mais difícil.

Nº de amostras. Ajuda (o Stage G mostra que o AP ainda sobe com mais dados), mas não explica tudo: o person tem 821 positivos de treino e é o pior, e o aeroplane, com 90, é o melhor.

Variação intra-classe e escala. É o fator principal para o person: pessoas de corpo inteiro, só mãos, minúsculas ao fundo ou cortadas. Um histograma de palavras de toda a imagem dilui um objeto pequeno no meio do fundo. 27 pessoas do test ficam nos últimos 10% do ranking; nas outras classes são 0 a 2.

Fundo e contexto. Os aviões aparecem num contexto muito consistente (céu liso, pista) que quase não existe no background, e isso ajuda. As motos dependem de asfalto e pista (Stage B), o que falha quando aparece outro fundo liso (o ganso no rank 18).

Contexto partilhado entre classes. 78% dos cavalos e 68% das motos do test também têm pessoas. Como os negativos são só background, o classificador nunca é penalizado por usar "pessoa + relva/pista" como pista. Isto é um limite do desenho one-vs-background: não mede se o SVM do horse rejeitaria uma imagem só com pessoas.

Nota. A validação das classes pequenas tem 22–28 positivos, por isso diferenças de ±0.05 no val AP são ruído (horse: val 0.596, test 0.662).

## Stage E

#### Como é que o L2 afeta a self-similarity?

K(h,h) = ‖h‖₂² = 1 para qualquer imagem (medido: min = max = 1). Todas as imagens ficam na esfera unitária, e o kernel passa a ser o cosseno entre histogramas. Compara a forma da distribuição de palavras e não a quantidade de descritores.

#### O que se conclui sobre K(h,h)?

É constante e igual a 1, por isso nenhuma imagem é "mais semelhante a si própria" do que outra. O score w·h deixa de depender do tamanho da imagem: a correlação passou de −0.20 para 0.00.

#### Como se compara K(h,h) com K(h,h′)?

Pela desigualdade de Cauchy–Schwarz, K(h,h′) ≤ ‖h‖‖h′‖ = 1 = K(h,h), com igualdade só se h′ = h (a mesma direção). Como os histogramas são ≥ 0, K(h,h′) ∈ [0, 1]. Medido: o máximo foi 0.98 e a mediana 0.30. Uma imagem é sempre a mais semelhante a si própria, que é o que se quer de uma medida de semelhança.

#### E sem normalização ou com L1?

Não.

Sem normalização: K(h,h) = ‖h‖² cresce com o nº de descritores, e encontrei pares onde K(h,h′) é 6.5× maior que K(h,h). Uma imagem grande e texturada fica "mais parecida" com outra do que esta consigo própria.
Com L1: Σhₖ = 1, mas K(h,h) = Σhₖ² depende de quão concentrado é o histograma. Um histograma de céu, com poucas palavras dominantes, dá 0.198, e um histograma espalhado dá 0.001. Também falha: 4×.
Há correlação entre desempenho e L2? Com tiles, sim: o L2 é o melhor em todos os K (+0.01 a +0.06 sobre none, +0.03 a +0.06 sobre L1). Com histograma global não há diferença (±0.01). O ganho grande vem de combinar o L2 com o √ (Hellinger, 0.717). A conclusão honesta é que o L2 ajuda de forma consistente mas modesta; o ganho grande vem do mapa √. Parte da vantagem sobre o L1 deve-se à escala dos vetores com C fixo.

#### Porque é que tirar a informação espacial ajuda umas classes e prejudica outras?

Os números estão na tabela "Effect of the tiles" do report (K = 512, sqrt_l2):

test: tiles − global val: tiles − global
motorbike +0.109 +0.137
aeroplane +0.008 +0.085
person −0.001 +0.021
horse +0.017 −0.010 (e −0.100 com K = 1024)
car −0.011 +0.009
A motorbike ganha porque o layout é estável: fotos de lado, rodas e asfalto em baixo, barreiras ou céu em cima.
Person e car não ganham ou perdem: aparecem em qualquer posição e escala, de perto ou ao fundo. Fixar "o quê está onde" não generaliza, e os tiles quadruplicam a dimensão com os mesmos dados (mais sobreajuste).
O horse varia com o K: com K = 1024 e tiles a dimensão é 4096 para 111 positivos, e o global ganha por 0.10 na validação.

## Stage F

#### Como normalizar para o kernel de Hellinger? Antes ou depois da raiz?

Pela definição, o histograma é normalizado em L1 antes da raiz: φ(h) = √(h/‖h‖₁). Isto é exatamente igual a aplicar a raiz e normalizar em L2 depois, porque ‖√(h/‖h‖₁)‖₂² = Σhₖ/‖h‖₁ = 1 (verificado numericamente). Assim o feature map do Hellinger cumpre também a regra do L2: todos os vetores têm norma 1 e K(h,h) = 1, por isso o kernel é o produto interno de vetores unitários. Normalizar em L1 depois da raiz (sqrt_l1) está errado: a norma volta a variar de imagem para imagem (0.026–0.137) e o val mAP cai para 0.597, contra 0.717 do sqrt_l2. Sem normalização (sqrt) obtém-se 0.680.

#### Porque é que a raiz muda a contribuição das palavras frequentes e raras?

A raiz é côncava: comprime as contagens altas muito mais do que as baixas (100 → 10, 1 → 1). Os histogramas BoVW são "bursty": texturas repetidas como céu ou asfalto enchem poucas palavras com milhares de descritores. Com o kernel linear, 70% da self-similarity vem das 10 palavras mais frequentes, e só cerca de 21 palavras contribuem de facto. Com a raiz são 21% e cerca de 200 palavras. As palavras raras, mais específicas do objeto, ganham peso face às texturas de fundo. O resultado é +0.09 de val mAP e +0.10 de test mAP (0.593 → 0.694).

## Stage G

#### Que desempenho se obtém com o kernel linear? E com o Hellinger?

Com todos os dados, o linear (L2) obtém val mAP = 0.626 (test 0.593) e o Hellinger 0.716 (test 0.694). O Hellinger é melhor em todas as classes e em todas as frações, por +0.07 a +0.10 de mAP. O ganho é maior nas classes com contexto texturado: no aeroplane, +0.33 a 10% e +0.21 a 100%. É tão grande que o Hellinger com metade dos dados (0.650) supera o linear com todos (0.626).

#### Satura? Mais exemplos ou mais diversidade?

Não satura: a última duplicação (50 → 100%) ainda sobe o mAP em +0.07 (Hellinger) e +0.09 (linear), e o motorbike sobe +0.14. Mais exemplos devem ajudar. Mas três sinais apontam para a diversidade como o fator mais útil:
(1) a variância entre seeds com poucos dados (±0.12 com 10 positivos) mostra que quais imagens entram pesa tanto como quantas;
(2) o horse linear quase não melhora com o dobro dos dados (0.49 → 0.50), o que indica que o limite está na representação e no contexto (78% dos cavalos têm pessoas, Stage D), não na quantidade;
(3) o train AP fica ≈ 1 em todas as frações, ou seja, o modelo memoriza o que vê. Mais imagens dos mesmos tipos (motos em pista) ensinam pouco; imagens noutros contextos (motos na rua, cavalos sem cavaleiro) atacam diretamente os erros dos Stages B e D.
