# Phase 7 qualitative error analysis

Original final-test saved predictions only; no inference, training, tuning or changed decisions.

Model: MLP seed 202; locked validation threshold: 0.3784696161746979.

Saved confusion matrix [[TN, FP], [FN, TP]]: [[1423, 77], [105, 1395]]. Available errors: {'FP': 77, 'FN': 105}; reviewed: {'FP': 10, 'FN': 10}.

FP: true 0, predicted 1, score descending. FN: true 1, predicted 0, score ascending. Both: example_id ascending for equal scores; first 10 or all available if fewer. Predicted Hallucinated iff score >= locked validation threshold.

## Findings and limits

Supported answers are sometimes rejected even when copied from the reference. Several long grounded answers and short erroneous answers are consistent with the existing style/length shortcut concern. Short supported false positives also occur. These cases do not identify which features caused a decision; length is not an explicit primary feature. The already reported strong answer-only control and shuffled-context results provide separate evidence of shortcut risk; no experiment was rerun here.

Clear date/entity mismatches coexist with apparent label anomalies and insufficient evidence. Franklin Square is explicitly supported despite label 1; EMC Corporation is supported for Dell EMC but the question leaves the company unspecified. Brahe is not established as a middle name. The Curry distance comparison requires geography absent from the reference; the Palmer attribution has unresolved pronouns. These judgments preserve all benchmark labels. Missing evidence is not proof of falsehood. No chunk-selection failure is asserted: selected chunks were not inspected.

One analyst (the implementing assistant) read each selected question, candidate answer and entire provided reference context. Only relevant verbatim excerpts are exported. These deliberately selected extreme-score errors are not representative population estimates. Scores are not calibrated probabilities or proven confidence. Categories describe evidence and possible failure modes, not established causal explanations. There was no external fact checking or label revision.

## Descriptive counts

Categories allow multiple assignments per case; zero categories are retained.

| Category | FP | FN | Total |
|---|---:|---:|---:|
| entity substitution | 0 | 3 | 3 |
| numerical hallucination | 0 | 1 | 1 |
| contradiction | 0 | 2 | 2 |
| high lexical overlap but incorrect meaning | 0 | 2 | 2 |
| plausible but unsupported | 0 | 2 | 2 |
| partial correctness | 1 | 1 | 2 |
| context ambiguity | 2 | 5 | 7 |
| evidence missed through chunk selection | 0 | 0 | 0 |
| dataset/label anomaly | 0 | 2 | 2 |
| other/unclear | 7 | 2 | 9 |

Analyst judgments: {'model error': 12, 'unclear judgment': 6, 'dataset/label anomaly': 2}

## False positives: Grounded predicted Hallucinated

| Rank | Example ID | Score | Score minus threshold | Absolute distance | Judgment |
|---:|---|---:|---:|---:|---|
| 1 | `6fb2c1c8065c1ee2cacbc0c01d6176dca6bfdc32d5aecec51d652b8f6c8325b7` | 0.999203742 | +0.620734125 | 0.620734125 | model error |
| 2 | `cce4e9e660c32087894e0e2e59c5b5d7c23e5827669ce111c5605bb443a13961` | 0.998336256 | +0.619866639 | 0.619866639 | model error |
| 3 | `d09c1c0ef42ad005a3cbf73f3c9441d0e617014e363f1453302aa6151606d44d` | 0.99558264 | +0.617113024 | 0.617113024 | model error |
| 4 | `cb73c13ec734c931071db5d7e59374e94f316ba3d3051ebc07d02327babe1042` | 0.990913272 | +0.612443656 | 0.612443656 | model error |
| 5 | `b4d0806e920850908040a0b0333a7959c97adb3b9ffba28ca14f18cb839f82fd` | 0.973666728 | +0.595197111 | 0.595197111 | model error |
| 6 | `e9814d95b5b6cd321a800fdca66936afdad9d108111ed335406daea5a423f10b` | 0.960350931 | +0.581881315 | 0.581881315 | unclear judgment |
| 7 | `c2dc0582437aba97a8ee86dddbdd89873ae82b78daf895d9423663c788e859b1` | 0.952946186 | +0.57447657 | 0.57447657 | unclear judgment |
| 8 | `3de63ecf11d3a0a71e8729527e09288061511308cf9e7242eee72bd69defdbbf` | 0.938436508 | +0.559966892 | 0.559966892 | model error |
| 9 | `10252113c9d395221f93853ca1a86924e1b6fcb833d90753601fc52862556aa2` | 0.931244135 | +0.552774519 | 0.552774519 | model error |
| 10 | `e9b561e7c6334606692fcd82ce769187e52fddb9f0bb7fe224d4aa6a84523cc1` | 0.924501121 | +0.546031505 | 0.546031505 | model error |

## False negatives: Hallucinated predicted Grounded

| Rank | Example ID | Score | Score minus threshold | Absolute distance | Judgment |
|---:|---|---:|---:|---:|---|
| 1 | `5b8b3102033b39d059e98f8cfc5fe80c3ff85277e7f88b62c1550aea3bda28f2` | 0.00060372724 | -0.377865889 | 0.377865889 | dataset/label anomaly |
| 2 | `3f8d862c7e69a9c5c1f79763d08932aca1113de72a214961d874057df155340f` | 0.000999241252 | -0.377470375 | 0.377470375 | dataset/label anomaly |
| 3 | `b1d5173beb71ece366b07bcdbbcdda371f46b785c27b15c3dd03c239786035ba` | 0.00152545306 | -0.376944163 | 0.376944163 | unclear judgment |
| 4 | `a7fa8802415a963225a222a1f4febd5f4bad26ff584cc85ff71f963e11faf4e1` | 0.00178059202 | -0.376689024 | 0.376689024 | model error |
| 5 | `ff388e1567f7211f44e47a55703eafbc7768839bdfbaaffa645be36ce4835d39` | 0.0023260389 | -0.376143577 | 0.376143577 | model error |
| 6 | `c846edde857af80683c4b09a75241470041dca1620aa50866f6207965514db56` | 0.00478725694 | -0.373682359 | 0.373682359 | unclear judgment |
| 7 | `c291aea3db7cedc18eadd310837dfe63f8bf881fd043a674d6bf7dd081863a3f` | 0.00614193315 | -0.372327683 | 0.372327683 | unclear judgment |
| 8 | `e8f2e2dd17e47da7ef2917c95e38af31c83bd29c8553c4aac2a02cf0e6349fdf` | 0.00635399297 | -0.372115623 | 0.372115623 | model error |
| 9 | `2df0e9170ef437201fbe6f353fa96f0c582df7f096a3a6f54dc2fd0d5e11f777` | 0.00673934352 | -0.371730273 | 0.371730273 | unclear judgment |
| 10 | `af7f6813d04d98455c360a891f8eae27c022283beeacfb27039c559d982f3d33` | 0.00828246772 | -0.370187148 | 0.370187148 | model error |

## Reviewed cases

### FP 1

- example_id: `6fb2c1c8065c1ee2cacbc0c01d6176dca6bfdc32d5aecec51d652b8f6c8325b7`
- source_id: `e5c9850fc858af84cf345d15c92943bacc19f48813c54c1991d749a2d8a38eb5`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9992037415504456; threshold: 0.3784696161746979.
- Score minus threshold: 0.6207341253757477; absolute distance: 0.6207341253757477.

**Question:** Who played the onscreen husband of a variety show banjo player?

**Candidate answer:** banjo player and former member of the television variety show "Hee Haw" gang having played the role of Ida Lee Nagger, the ironing, nagging wife of Laverne Nagger (Gordie Tapp).

**Reference evidence:**

> the ironing, nagging wife of Laverne Nagger (Gordie Tapp).

**Categories:** other/unclear.

**Judgment:** model error.

The reference explicitly identifies Gordie Tapp as the actor playing the husband. The candidate is an awkward, overlong copied fragment but includes that answer and makes no evident unsupported claim. Its rejection is an apparent grounding-classification failure; the cause is unclear, and verbosity is only a possible shortcut cue.

### FP 2

- example_id: `cce4e9e660c32087894e0e2e59c5b5d7c23e5827669ce111c5605bb443a13961`
- source_id: `e4bd77ac965a0264ecbd6cc235969fbdd19a93290ba06a06e0a7de71958fb31f`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9983362555503845; threshold: 0.3784696161746979.
- Score minus threshold: 0.6198666393756866; absolute distance: 0.6198666393756866.

**Question:** Why was the fort viewable from Willard Beach never used?

**Candidate answer:** Advancing military technology, including iron clad ships and long range guns, made the fort obsolete before it could be used.

**Reference evidence:**

> Advancing military technology, including iron clad ships and long range guns, made the fort obsolete before it could be used.

**Categories:** other/unclear.

**Judgment:** model error.

The answer repeats the reference explanation verbatim and directly answers why the fort was unused. The model rejects supported content. Its sentence length is consistent with a possible style shortcut, but the score alone cannot establish that mechanism.

### FP 3

- example_id: `d09c1c0ef42ad005a3cbf73f3c9441d0e617014e363f1453302aa6151606d44d`
- source_id: `834e3f07ddcfcfbf03dbd460d664ae9241f31277cbeb0b2edef276c9ed97a2be`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.995582640171051; threshold: 0.3784696161746979.
- Score minus threshold: 0.6171130239963531; absolute distance: 0.6171130239963531.

**Question:** How many novels has the author of the novel Child of God written?

**Candidate answer:** He has written ten novels

**Reference evidence:**

> based on the novel of the same name by Cormac McCarthy.

> He has written ten novels,

**Categories:** other/unclear.

**Judgment:** model error.

The context links the novel to McCarthy and explicitly gives ten novels. The candidate repeats that count. This is a supported numerical answer rejected by the model, not a numerical hallucination in the answer.

### FP 4

- example_id: `cb73c13ec734c931071db5d7e59374e94f316ba3d3051ebc07d02327babe1042`
- source_id: `fbd45ba3aedda66d9c0480255d6735bb10f6d33ffe477753c61d1c57dd4b1795`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9909132719039917; threshold: 0.3784696161746979.
- Score minus threshold: 0.6124436557292938; absolute distance: 0.6124436557292938.

**Question:** Who created Creature comforts and how many academy awards did he win?

**Candidate answer:** Park has been nominated for an Academy Award a total of six times, and won four with "Creature Comforts"

**Reference evidence:**

> It was created by Nick Park and Aardman Animations.

> Park has been nominated for an Academy Award a total of six times, and won four with "Creature Comforts"

**Categories:** partial correctness.

**Judgment:** model error.

The answer accurately quotes four wins and identifies Park by surname. It omits Aardman and the first name from the creation clause, making it incomplete for the two-part question, but no asserted fact contradicts the reference. Incompleteness should not itself be equated with hallucination; the high score rejects supported content.

### FP 5

- example_id: `b4d0806e920850908040a0b0333a7959c97adb3b9ffba28ca14f18cb839f82fd`
- source_id: `4ce16f419345e7d21b7fafd26f8ed001984413a942d7cedfe427f1b2bd90ad38`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9736667275428772; threshold: 0.3784696161746979.
- Score minus threshold: 0.5951971113681793; absolute distance: 0.5951971113681793.

**Question:** Was the president who signed the The Tax Reduction and Simplification Act of 1977 a Republican or Democrat?

**Candidate answer:** James Earl Carter Jr. (born October 1, 1924) is an American politician who served as the 39th President of the United States from 1977 to 1981. A member of the Democratic Party,

**Reference evidence:**

> signed into law by President Jimmy Carter on May 23, 1977.

> A member of the Democratic Party,

**Categories:** other/unclear.

**Judgment:** model error.

The reference links the signer Carter to the Democratic Party. The long biographical answer includes the correct affiliation and copies supported details. Rejecting it is a model error under the benchmark; a style-based cause remains speculative.

### FP 6

- example_id: `e9814d95b5b6cd321a800fdca66936afdad9d108111ed335406daea5a423f10b`
- source_id: `d20d7b47257bb34aa8438d8b459a34ea89e511a7de05a3e9f553dc5b286ad25c`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9603509306907654; threshold: 0.3784696161746979.
- Score minus threshold: 0.5818813145160675; absolute distance: 0.5818813145160675.

**Question:** Which was released first, The Plow That Broke the Plains or The Man Who Skied Down Everest?

**Candidate answer:** The Plow That Broke the Plains

**Reference evidence:**

> a Japanese alpinist who skied down Mt. Everest in 1970.

> The Plow That Broke the Plains is a 1936 short documentary film

**Categories:** context ambiguity.

**Judgment:** unclear judgment.

The reference explicitly dates Plow to 1936 and places the event depicted by the other documentary in 1970, supporting the apparent ordering. It does not give that documentary's release date. The benchmark answer is plausible, but this is indirect temporal evidence rather than a complete release-date comparison. The false-positive designation follows the unchanged label.

### FP 7

- example_id: `c2dc0582437aba97a8ee86dddbdd89873ae82b78daf895d9423663c788e859b1`
- source_id: `775869cdac23100e907b09866740ec98cf3b931632d8253721e4087baa465c4b`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9529461860656738; threshold: 0.3784696161746979.
- Score minus threshold: 0.574476569890976; absolute distance: 0.574476569890976.

**Question:** While both singer-songwriters have displayed incredible talent, what sets Robert Palmer apart from Wayne Coyne, causing him to gain populatiry in two continents?

**Candidate answer:** eclectic mix of musical styles

**Reference evidence:**

> He was known for his distinctive, soulful voice, eclectic mix of musical styles on his albums, combining soul, jazz, rock, pop, reggae, blues, and sartorial acumen.

**Categories:** context ambiguity.

**Judgment:** unclear judgment.

The answer is a verbatim trait in the reference and fits the question's requested distinction. However, the context uses unresolved male pronouns and does not name Palmer or establish two-continent popularity. The model may have rejected a supported phrase, but attribution and the causal framing cannot be fully checked here. This is not evidence that the candidate is false.

### FP 8

- example_id: `3de63ecf11d3a0a71e8729527e09288061511308cf9e7242eee72bd69defdbbf`
- source_id: `2827040d81508458c8db0a5fcf53322f5191ca005417cab91095edb0cba50f05`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9384365081787109; threshold: 0.3784696161746979.
- Score minus threshold: 0.5599668920040131; absolute distance: 0.5599668920040131.

**Question:** What mystery play can be dated back to medieval theater in the early 15th century?

**Candidate answer:** The Chester Mystery Plays

**Reference evidence:**

> The Chester Mystery Plays is a cycle of mystery plays dating back to at least the early part of the 15th century.

**Categories:** other/unclear.

**Judgment:** model error.

The named cycle and its early-15th-century dating are explicit. The short candidate directly answers the question and is supported. This false positive also shows that the observed errors cannot all be explained simply as rejection of long answers.

### FP 9

- example_id: `10252113c9d395221f93853ca1a86924e1b6fcb833d90753601fc52862556aa2`
- source_id: `abca19356aa9dd9ef688aa77515a4a6cf4030e93eb71705b9869c776aaf99044`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9312441349029541; threshold: 0.3784696161746979.
- Score minus threshold: 0.5527745187282562; absolute distance: 0.5527745187282562.

**Question:** Name the American actor, film producer and environmentalist activist who co-wrote the '11th Hour', a 2007 documentary film about the state of the natural environment, distributed by Warner Independent Pictures?

**Candidate answer:** Leonardo DiCaprio

**Reference evidence:**

> The 11th Hour is a 2007 documentary film, created, produced, co-written and narrated by Leonardo DiCaprio,

**Categories:** other/unclear.

**Judgment:** model error.

The reference directly names DiCaprio as co-writer of the requested film. The concise answer is supported, so the high score is a model classification error. The evidence does not establish which interaction or scalar feature drove it.

### FP 10

- example_id: `e9b561e7c6334606692fcd82ce769187e52fddb9f0bb7fe224d4aa6a84523cc1`
- source_id: `13ae2c5febfea5d84058c2fd599aaff8183c9fbc7ab3653e4c283e98119f36cd`
- True label: 0; predicted label: 1 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.9245011210441589; threshold: 0.3784696161746979.
- Score minus threshold: 0.5460315048694611; absolute distance: 0.5460315048694611.

**Question:** Mennonites are traditionally multilingual with which Low Prussian dialect of East Low German, with Dutch influence language was their first?

**Candidate answer:** Plautdietsch

**Reference evidence:**

> Plautdietsch or Mennonite Low German, was originally a Low Prussian dialect of East Low German, with Dutch influence,

**Categories:** other/unclear.

**Judgment:** model error.

The reference explicitly names Plautdietsch and describes the dialect requested. The question's first-language framing is not fully spelled out, but the candidate identifies the described language without adding that claim. The model rejects a supported identification.

### FN 1

- example_id: `5b8b3102033b39d059e98f8cfc5fe80c3ff85277e7f88b62c1550aea3bda28f2`
- source_id: `8694a0645879f39e6aeb39b2f893b1aeea9a100d7b0b75d07031a0e7754c2aed`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.0006037272396497428; threshold: 0.3784696161746979.
- Score minus threshold: -0.37786588893504813; absolute distance: 0.37786588893504813.

**Question:** Where is H. Frank Carey Junior-Senior High School?

**Candidate answer:** Franklin Square

**Reference evidence:**

> H. Frank Carey High School is a public high school located in Franklin Square, New York

**Categories:** dataset/label anomaly.

**Judgment:** dataset/label anomaly.

Franklin Square directly matches the stated school location. Although the benchmark marks this candidate Hallucinated, the provided evidence supports the model's Grounded decision. This is an apparent label anomaly, not a demonstrated factual failure by the model. The benchmark label remains 1.

### FN 2

- example_id: `3f8d862c7e69a9c5c1f79763d08932aca1113de72a214961d874057df155340f`
- source_id: `497de7a9d60b29769270f48e717f05984e9df3ee206c3caae981517334b2c118`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.0009992412524297833; threshold: 0.3784696161746979.
- Score minus threshold: -0.3774703749222681; absolute distance: 0.3774703749222681.

**Question:** What was the company orginally named first?

**Candidate answer:** EMC Corporation

**Reference evidence:**

> Dell Technologies Inc. is an American multinational information technology corporation

> Dell EMC (EMC Corporation until 2016)

**Categories:** dataset/label anomaly, context ambiguity.

**Judgment:** dataset/label anomaly.

EMC Corporation is explicitly the earlier name of Dell EMC. If that is the intended company, the candidate is supported despite label 1. The question only says the company and the context mentions two companies, so the referent is ambiguous. Flagged as an apparent label anomaly conditional on Dell EMC, without revising the benchmark.

### FN 3

- example_id: `b1d5173beb71ece366b07bcdbbcdda371f46b785c27b15c3dd03c239786035ba`
- source_id: `24ac1ab856e5dc106f4c870a0908687ea03dcfbb256f117cc5335d86ef5eab9d`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.001525453059002757; threshold: 0.3784696161746979.
- Score minus threshold: -0.3769441631156951; absolute distance: 0.3769441631156951.

**Question:** What is the middle name of Beate Clausdatter Bille's Son ?

**Candidate answer:** Brahe

**Reference evidence:**

> Beate Bille was the mother of astronomer Tycho Brahe.

> born Tyge Ottesen Brahe

**Categories:** high lexical overlap but incorrect meaning, context ambiguity.

**Judgment:** unclear judgment.

Brahe occurs in both displayed names, but occurrence alone does not establish it as the requested middle name. The birth-name sequence instead places Ottesen between Tyge and Brahe; the reference does not explicitly classify name components. Accepting Brahe may confuse name membership with name role, but the question's naming convention remains unclear.

### FN 4

- example_id: `a7fa8802415a963225a222a1f4febd5f4bad26ff584cc85ff71f963e11faf4e1`
- source_id: `284bbdd7a22991a6c3ae486e21562acb87571120528e5aa08c1843f9dcb2e56d`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.0017805920215323567; threshold: 0.3784696161746979.
- Score minus threshold: -0.3766890241531655; absolute distance: 0.3766890241531655.

**Question:** Surinder Sodhi composed the music for special 26, when was it relieced.

**Candidate answer:** January 5, 2013

**Reference evidence:**

> "Special 26" released on February 5, 2013

**Categories:** numerical hallucination, contradiction, high lexical overlap but incorrect meaning.

**Judgment:** model error.

The candidate changes February to January while retaining the day and year. It directly conflicts with the provided release date. The low score accepts a compact near-match containing a substantive date error; lexical similarity is a possible explanation, not an established cause.

### FN 5

- example_id: `ff388e1567f7211f44e47a55703eafbc7768839bdfbaaffa645be36ce4835d39`
- source_id: `bf04b284f95b0d3cb3b4a46f999b79a3faf488458fed085aee663880e73aad43`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.0023260388989001513; threshold: 0.3784696161746979.
- Score minus threshold: -0.3761435772757977; absolute distance: 0.3761435772757977.

**Question:** Which brand shop in Beverly Center is based in Maranello ?

**Candidate answer:** Lamborghini, based in Rome.

**Reference evidence:**

> Ferrari Store,

> Ferrari N.V.

> is an Italian sports car manufacturer based in Maranello.

**Categories:** entity substitution, plausible but unsupported.

**Judgment:** model error.

The reference links the listed Ferrari store to a manufacturer based in Maranello. The candidate substitutes Lamborghini and adds Rome, neither supported here, and the stated city fails the Maranello condition in the question. This supports a model error without asserting any external fact about Lamborghini's actual headquarters.

### FN 6

- example_id: `c846edde857af80683c4b09a75241470041dca1620aa50866f6207965514db56`
- source_id: `545f27fa69dcbf3917d3e6b903af692ddc6f2681868f53408796081ef01fdf97`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.004787256941199303; threshold: 0.3784696161746979.
- Score minus threshold: -0.3736823592334986; absolute distance: 0.3736823592334986.

**Question:** What action does Moves and The Artist's Magazine have in common?

**Candidate answer:** edited

**Reference evidence:**

> Moves was a wargaming magazine originally published by SPI

> The Artist's Magazine is a monthly magazine for artists published by F+W Media

**Categories:** plausible but unsupported, other/unclear.

**Judgment:** unclear judgment.

The shared action explicit in the reference is publication. The answer edited supplies a different action without reference support. Magazines might also be edited, but that cannot be confirmed or disproved from this context. The model accepts an unsupported response; the evidence does not show a direct factual contradiction.

### FN 7

- example_id: `c291aea3db7cedc18eadd310837dfe63f8bf881fd043a674d6bf7dd081863a3f`
- source_id: `4fcf0eb6f77d8bd971e6b2e91fc0193a0e4743e526ede008c3cac8df85978593`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.006141933146864176; threshold: 0.3784696161746979.
- Score minus threshold: -0.3723276830278337; absolute distance: 0.3723276830278337.

**Question:** Which is the largest non-Catholic Christian denomination in the United States headed by Russell D. Moore, an American evangelical theologian, ethicist, and preacher?

**Candidate answer:** The Baptist Church

**Reference evidence:**

> the Southern Baptist Convention, the largest non-Catholic Christian denomination in the United States,

**Categories:** partial correctness, context ambiguity.

**Judgment:** unclear judgment.

The reference names the Southern Baptist Convention; The Baptist Church is a broader and less specific designation. It shares Baptist terminology but does not identify the exact organization. The text also does not explicitly connect Moore to heading it. This is a plausible specificity failure, with insufficient evidence to resolve whether the generic wording is acceptable.

### FN 8

- example_id: `e8f2e2dd17e47da7ef2917c95e38af31c83bd29c8553c4aac2a02cf0e6349fdf`
- source_id: `bb21fd3d2a6c5e660c82eab333d347394e8a893a9960e51e6e0300d338baecd8`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.006353992968797684; threshold: 0.3784696161746979.
- Score minus threshold: -0.3721156232059002; absolute distance: 0.3721156232059002.

**Question:** The Champaign Aviation Museum is an aviation museum in Urbana, Ohio, in which country, the museum and its volunteers are restoring a B-17 Flying Fortress back to flying status, a our-engine heavy bomber developed in the 1930s for the United States Army Air Corps (USAAC)?

**Candidate answer:** Saudi Arabia

**Reference evidence:**

> The Champaign Aviation Museum is an aviation museum in Urbana, Ohio, United States.

**Categories:** entity substitution, contradiction.

**Judgment:** model error.

The requested country is explicitly United States, whereas the candidate says Saudi Arabia. This is a direct country substitution contradicted by the reference, accepted with a very low hallucination score. The long question and short answer do not by themselves explain the decision.

### FN 9

- example_id: `2df0e9170ef437201fbe6f353fa96f0c582df7f096a3a6f54dc2fd0d5e11f777`
- source_id: `df64954eeed8979b3f73935b6caea5c9245df38759099b926be5d01030e630bd`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.006739343516528606; threshold: 0.3784696161746979.
- Score minus threshold: -0.37173027265816927; absolute distance: 0.37173027265816927.

**Question:** Which players' team, Seth or Stephen Curry, is closer to New Orleans?

**Candidate answer:** Stephen Curry

**Reference evidence:**

> Seth Adham Curry

> for the Dallas Mavericks

> Wardell Stephen Curry II

> for the Golden State Warriors

**Categories:** context ambiguity, other/unclear.

**Judgment:** unclear judgment.

The context associates each player with a team but supplies neither team locations nor distances to New Orleans. It cannot establish the requested geographic comparison using only the provided evidence. The benchmark marks Stephen Curry wrong, but this review cannot independently resolve the answer or infer that missing distance information makes it false.

### FN 10

- example_id: `af7f6813d04d98455c360a891f8eae27c022283beeacfb27039c559d982f3d33`
- source_id: `4364394dc0dbef6eae67f0e8fc5de002187340cbc50aaa7fc7443d9e124b91de`
- True label: 1; predicted label: 0 (0=Grounded, 1=Hallucinated).
- Hallucination score: 0.008282467722892761; threshold: 0.3784696161746979.
- Score minus threshold: -0.3701871484518051; absolute distance: 0.3701871484518051.

**Question:** Which movie produced by Uri Singer stars the actress who won critical acclaim with her appearance as Lydia Deetz in Tim Burton's "Beetlejuice"?

**Candidate answer:** "Open Road"

**Reference evidence:**

> "Open Road" (Andy García, Juliette Lewis and Camilla Belle)

> "Experimenter" (written and directed by Michael Almereyda) with Peter Sarsgaard and Winona Ryder.

> As Lydia Deetz, a goth teenager in Tim Burton's "Beetlejuice" (1988), she won critical acclaim

**Categories:** entity substitution, context ambiguity.

**Judgment:** model error.

The supplied sequence links Winona Ryder with Experimenter and then describes the Lydia Deetz role. The candidate instead chooses another listed film, Open Road, whose listed performers differ. This is a likely wrong-film association accepted by the model. The final she is implicit and the cast lists need not be exhaustive, so the text supports the intended mismatch without proving Ryder could never appear in Open Road.
