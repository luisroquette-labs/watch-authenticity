# Comparação das seis fotos do piloto — 17/11/2026

## Método

- Mesmo lote local de seis arquivos em `~/Downloads`: capturas de tela
  `21.34.27`, `.31`, `.36`, `.41`, `.45` e `517615334671816.webp`.
- Mesma ordem, todos com vista informada `dial`; mesmo Mac M5, mesmo checkpoint
  local `HuggingFaceTB/SmolVLM-500M-Instruct`, sem chamada paga.
- Código anterior: `pilot/server.py` do commit `28d82f7` (SHA do arquivo
  `e2e933de455f680ed1123a0eb417279f482ec4a2bb893cf23e06e8d5ee641ffb`).
  Código novo: commit `649351a`, antes do ajuste de versão para `0.2.0`.
- O script de comparação carregou cada serviço isoladamente e chamou
  `analyze_batch` com os seis arquivos. O código novo também recebeu o mesmo
  lote pela rota HTTP local `/predict/batch`: HTTP 200, seis fotos usadas e
  resposta igual à chamada direta. Isso ainda não cobre o navegador nem o
  ambiente publicado.
- SHA-256 dos arquivos, na ordem: `fe8a1fcaf86d487be3a258d4cc5e38d0ece3b247b7a390497b7406d86fc11aec`,
  `ee92d45114f23383b8ed0ca7aa68b7420c4312766d83952fc1a6d980559a6c44`,
  `08f83d34faeb0f08b3332a1635cd00c772fa16e152832ae7c995f37a540e4a22`,
  `4cbafdb8bce47cc8fcec915c5ee5c894cddf4c34d8fec08fe384753de3831f21`,
  `4fe1ec8371cd27a7e63ec479baa5b75cfe4b95fab40a1e6d44249330484074a6`,
  `1caaefe73fbb9a3d93ca8dc65810498f0d43b58b8b6ac2bd8bb2e7d95f29eabf`.

## Resultado observado

| Critério | Anterior | Novo | Avaliação |
| --- | --- | --- | --- |
| Fotos enviadas ao modelo | 6/6 | 6/6 | Igual |
| Mostrador por foto | Verde nas 6 | Verde nas 6 | Igual e visualmente plausível |
| Cor do aro por foto | Indefinida nas 6 | Indefinida nas 6 | Igual e insuficiente: aro verde é visível |
| Conclusão | Diverge da referência 126610LN; autenticidade inconclusiva | Mesma conclusão | Igual; não confunde referência divergente com falsificação |
| Referência candidata | Nenhuma | Nenhuma | Igual e insuficiente para a configuração visível |
| Transparência | Não indicava fotos efetivamente usadas, fatos citados ou métricas ausentes | Mostra 6 fotos usadas, 6 conteúdos distintos, 0 reaproveitamentos, 2 fatos oficiais citados, confiança e nota indisponíveis | Melhorou |
| Tempo de carga + análise | 3,86 + 42,53 = 46,39 s | 2,87 + 43,81 = 46,68 s | Cerca de 0,3 s pior nesta única rodada; diferença sem significância estatística |

As seis fotos parecem retratar **o mesmo relógio físico**, sobretudo de frente.
Seis conteúdos digitais distintos não são seis relógios nem seis provas
independentes. O piloto novo avisa que trata o lote como um relógio, mas ainda
conta seis conteúdos visuais distintos. Os arquivos incluem fotos com caixa,
cartão e selo; esses objetos não demonstram procedência ou autenticidade.

## Síntese

**Melhorou:** a resposta mostra o que foi analisado, quais fatos oficiais foram
citados e quais medidas não existem. Não inventa confiança calibrada nem nota
de qualidade da falsificação.

**Não mudou:** as seis mensagens individuais e o veredito conjunto são
idênticos. O sistema continua comparando o relógio à 126610LN de mostrador
preto, apesar do mostrador verde nas imagens.

**Piorou:** não há regressão visual demonstrada. A análise nesta execução foi
1,28 s mais lenta (42,53 → 43,81 s); a diferença total foi 0,29 s e uma
medição por versão não permite afirmar regressão de desempenho. Uma rodada
anterior no mesmo Mac mediu 67,09 s na versão antiga e 123,19 s na nova;
a variação entre rodadas reforça que falta um benchmark repetido e controlado.

**Melhorou, mas ainda não basta:** a comunicação da incerteza ficou mais
confiável, porém o detector não reconheceu o aro verde visível, não propôs a
configuração de referência apropriada e não inspecionou lupa, impressão,
gravações, fecho ou movimento. Não há classificador de autenticidade liberado,
probabilidade calibrada nem nota 0–1.000. **Não sabemos por essas fotos se o
relógio é original ou réplica.**

Próximo experimento técnico: corrigir e testar a localização do aro com fotos
verdes, pretas e controles difíceis; depois testar identificação da referência
em relógios físicos distintos e um conjunto confirmado de originais e réplicas.
Nenhuma métrica de acerto em falsificações pode ser inferida deste lote.

## Nova execução após corrigir o recorte do aro

O detector anterior exigia dois círculos completos. Nas seis fotos, o localizador
às vezes recorta parte da borda externa; por isso o aro visivelmente verde era
descartado. O ajuste aceita a borda interna do mostrador e exige cor consistente
em grande parte do anel, além de um exterior de outra cor. Um teste de regressão
mantém o fundo verde sem aro como `uncertain`.

Na nova execução local pela rota HTTP e pela página real em Chrome isolado:
seis fotos enviadas e usadas; mostrador verde em seis; aro verde em **duas**;
referência **116610LV (Hulk) sugerida apenas como configuração compatível**.
As outras quatro fotos continuam sem cor de aro verificada. O veredito é
`inconclusivo`; confiança calibrada, nota 0–1.000 e classificador de
autenticidade continuam ausentes. O navegador mostrou esses limites ao usuário.

O perfil de seis fotos atribuiu 28,71 s à localização do relógio, 25,90 s à
leitura visual do mostrador e 0,11 s à leitura do aro. Uma tentativa de reduzir
a resolução do localizador acelerou essa etapa, mas perdeu a detecção do aro;
não foi incorporada. A meta de velocidade ainda não foi comprovada.
