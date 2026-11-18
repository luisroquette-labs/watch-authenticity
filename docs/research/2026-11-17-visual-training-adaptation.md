# Treino visual para detectar réplicas — decisão de 17/11/2026

## O que a pesquisa muda

1. **Começar de uma base visual já treinada.** DINOv2 aprende representações
   transferíveis sem rótulos da tarefa; o projeto já carrega
   `facebook/dinov2-base` com pesos congelados. Treinaremos primeiro uma saída
   binária sobre essas representações. A técnica de transferência com pesos
   congelados também é documentada no tutorial oficial do PyTorch.
2. **Tratar origem como parte do rótulo.** As 3.661 imagens distintas do
   supercloneluxury são *réplicas declaradas pelo vendedor*, distribuídas por
   muitas marcas e referências. O catálogo contém um produto 126610LV com
   três URLs e **nenhum produto explicitamente 126610LN**. As seis fotos
   enviadas por Luis são *falsas declaradas pelo proprietário*. Nenhuma recebe
   grau 1–4 ou nota 0–1.000 automaticamente. O novo carregador experimental
   verifica os hashes e devolve 3.667 imagens; as 3.661 do catálogo ficam em
   1.019 grupos de produto ligados por bytes idênticos, mais um grupo
   provisório das seis fotos do proprietário. Grupo de produto ainda não prova
   identidade de relógio físico. No código, o campo chama-se
   `provisional_group` para impedir que seja tratado como divisão pronta.
3. **Separar por relógio e origem antes de medir acerto.** Fotos repetidas,
   ângulos e anúncios do mesmo relógio não podem cruzar treino e teste. Um
   lote de falsos de um único site contra originais de outro pode ensinar
   apenas o estilo do site. É o risco de *shortcut learning* descrito na
   literatura; um teste em outra fonte por classe é obrigatório antes de
   afirmar que o modelo reconhece falsificações. Hashes idênticos ligam
   anúncios repetidos, mas cortes ou recodificações do mesmo relógio não são
   detectados por hash. Antes de dividir dados, reconciliar manualmente ou por
   comparação visual os possíveis reposts e excluir grupos ambíguos. Nunca
   usar `provisional_group` diretamente como chave final de separação.
4. **Treinar o primeiro resultado observável como triagem binária.** Usar
   imagens de originais com proveniência explícita e réplicas declaradas,
   balanceadas por relógio e fonte. O teste precisa informar, por relógio,
   quantos falsos foram chamados de originais, quantos casos receberam resposta
   e quantos ficaram inconclusivos. O modelo não será ligado à página até
   superar uma regra simples e passar no teste de fonte independente.
5. **Adicionar detalhe fino depois do teste binário.** O corte de 224 pixels e
   uma foto de frente não comprovam gravação, movimento ou acabamento de
   superclone. Usar cortes de alta resolução por ângulo quando houver imagens
   adequadas. Conhecimento de manuais e fóruns alimenta verificações citadas
   separadamente; não vira rótulo de treino nem nota de qualidade por si só.

## Estado e próxima execução

O produtor IBM existente exige seis classes, seis ângulos, referência exata,
graus 1–4 e um conjunto confirmado. Ele continua sendo a rota de **liberação**.
O carregador `training/experimental_data.py` é uma entrada local de pesquisa
para aproveitar os rótulos já existentes; não altera o conjunto de liberação.
O primeiro lote CPO obteve **15 imagens de 51 URLs tentadas**, em **12 anúncios**:
11 imagens da Watches of Switzerland e quatro da Precision Watches. Goldsmiths
não rendeu arquivos nesta execução (falhas de DNS). O carregador confere os
hashes, preserva anúncio e origem, e registra `cpo_seller_claim`: trata-se de
uma afirmação do vendedor, não de autenticação independente. As 15 imagens
ainda não equilibram 3.667 imagens rotuladas falsas e não sustentam treino ou
teste de generalização. Falta a classe original em volume e diversidade
comparáveis. Há um experimento
anterior de 40 imagens de treino e 16 de teste com rótulos provisórios, mas
11 acertos e cinco inconclusivos naquele pequeno teste não medem desempenho
contra falsificações novas. Não há modelo binário liberado.

Para o piloto 126610LN, há ainda nove imagens distintas de **um único anúncio**
da VS Factory USA que declara réplica 126610LN. Elas representam no máximo um
grupo de produto, não nove relógios. As seis fotos de Luis têm rótulo falso,
mas a referência exata permanece não confirmada. Portanto, o acervo atual
não permite um teste independente de detecção da 126610LN: comparar as nove
fotos de um vendedor com os 12 anúncios CPO mediria principalmente a origem.

Próxima unidade: coletar originais de fontes independentes e ligar esse lado
ao mesmo inventário experimental; depois executar um treino binário local
limitado no tempo, com teste separado por relógio e fonte. Se a separação
independente não for possível, publicar o resultado como experimento sem
afirmar precisão de detecção.

## Fontes técnicas verificadas

- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021. Fundamenta o uso de um modelo visual pré-treinado; não substitui avaliação na tarefa.
- Geirhos et al., [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780), 2020.
- PyTorch, [Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), consultado em 17/11/2026.
