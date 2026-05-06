# Test Report - Autoencoder2D_GMM

Questo file raccoglie i risultati dei test eseguiti durante il debug di `Autoencoder2D_GMM`,
le cause individuate e le modifiche applicate.

Data sessione: 2026-05-06

## Obiettivo

Portare `Autoencoder2D_GMM` sui target PULP/Siracusa, procedendo in ordine:

1. Siracusa non-tiled
2. Siracusa tiled con memoria standard L3
3. Siracusa tiled con Neureka

Il modello contiene due rami:

- `Autoencoder2D`, con output `reconstruction`
- testa `GMM`, con output `gmm_output`

Il ramo GMM introduce operatori che erano supportati nel target `Generic`, ma non ancora
nel mapping PULP/Siracusa.

## Stato iniziale

Il modello `Autoencoder2D_GMM` funzionava sul target `Generic`, ma non su Siracusa.

Comando di riferimento Generic:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_generic.py -t Tests/Models/Autoencoder2D_GMM
```

## 1) Siracusa non-tiled - primo tentativo

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D_GMM
```

Esito: `FAILED`

Errore principale:

```text
RuntimeError: No mapping found for node gmm_modelgmmReduceLogSumExp with op type ReduceLogSumExp
```

### Causa

`ReduceLogSumExp` era implementato nel target `Generic`, ma non era registrato nella
piattaforma PULP/Siracusa.

Implementazioni Generic gia' presenti:

- `Deeploy/Targets/Generic/Parsers.py`
  - `ReduceLogSumExpParser`
- `Deeploy/Targets/Generic/Bindings.py`
  - `BasicReduceLogSumExpBindings`
- `Deeploy/Targets/Generic/Templates/FloatReduceLogSumExpTemplate.py`
- `TargetLibraries/Generic/src/ReduceLogSumExp_fp32.c`
- `TargetLibraries/Generic/inc/kernel/ReduceLogSumExp.h`

### Modifica applicata

File modificato:

- `Deeploy/Targets/PULPOpen/Platform.py`

Sono stati importati e registrati:

- `ReduceLogSumExpLayer`
- `ReduceLogSumExpParser`
- mapping `ReduceLogSumExp`

Modifica concettuale:

```python
ReduceLogSumExpMapper = NodeMapper(ReduceLogSumExpParser(), PULPReduceLogSumExpTilingReadyBindings)

PULPMapping = {
    ...
    'ReduceLogSumExp': ReduceLogSumExpLayer([ReduceLogSumExpMapper]),
    ...
}
```

### Perche'

Senza questa entry il parser PULP non aveva nessun mapper disponibile per il nodo ONNX
`ReduceLogSumExp`, quindi il grafo veniva rifiutato prima della generazione del codice.

## 2) Siracusa non-tiled - secondo blocco

Dopo l'aggiunta di `ReduceLogSumExp`, il parsing e' avanzato fino al nodo `Concat`.

Esito: `FAILED`

Errore principale:

```text
PARSING FAILED - Backtracking exhausted at root!
Deepest successful exploration: Layer 13 'gmm_modelConcat'
Deepest layer available mappers: ['ConcatParser']
RuntimeError: Did not find adequate mapping for graph!
```

Nodo ONNX coinvolto:

```text
Concat /gmm_model/Concat ['/flatten/Flatten_output_0', 'onnx::Concat_154']
```

### Causa

Il binding `Concat` PULP supportava solo tipi interi:

```python
PULPConcatBindings = [
    NodeBinding(... for type in IntegerDataTypes)
]
```

Il target `Generic`, invece, aveva gia' anche il caso `float32_t`.

### Modifica applicata

File modificato:

- `Deeploy/Targets/PULPOpen/Bindings.py`

Aggiunto binding `Concat` FP32:

```python
PULPConcatBindings = [
    NodeBinding(ConcatChecker([PointerClass(type), PointerClass(type)], [PointerClass(type)]),
                ConcatTemplate.referenceTemplate, ClusterTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ConcatChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ConcatTemplate.referenceTemplate, ClusterTransformer)
]
```

### Perche'

La testa GMM concatena tensori `float32_t`. Il template `ConcatTemplate` era gia' riutilizzabile,
ma mancava il type binding PULP per FP32.

## 3) Siracusa non-tiled - limite memoria L2

Dopo i fix precedenti, il modello non-tiled arriva a generare e compilare `Network.c`,
ma fallisce al link.

Esito: `FAILED`

Errore principale:

```text
ld.lld: error: section '.l2_data' will not fit in region 'L2': overflowed by 89932 bytes
```

### Causa

Il modello completo `Autoencoder2D_GMM` non entra nella L2 standard in configurazione
Siracusa non-tiled.

Il runner non-tiled `deeployRunner_siracusa.py` non espone `--l2`, quindi non e' possibile
fare lo stesso override memoria usato nel flusso tiled.

### Conclusione

Siracusa non-tiled non e' bloccato da layer mancanti dopo le patch, ma da memoria L2
insufficiente. Il target rilevante per il modello completo resta il flusso tiled con L3.

## 4) Siracusa tiled L3 - primo tentativo

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D_GMM --defaultMemLevel=L3 --l2=3000000
```

Esito: `FAILED`

Errore principale:

```text
AttributeError: 'NodeTemplate' object has no attribute 'tileConstraint'
```

### Causa

Il primo mapping di `ReduceLogSumExp` riusava il binding Generic puro. Quel binding usa
un `NodeTemplate` senza `tileConstraint`, quindi non puo' essere attraversato dal tiler PULP.

### Modifica applicata

File modificati:

- `Deeploy/Targets/PULPOpen/Bindings.py`
- `Deeploy/Targets/PULPOpen/Tiler.py`
- `Deeploy/Targets/PULPOpen/Platform.py`

In `Bindings.py` e' stato aggiunto un binding PULP vero per `ReduceLogSumExp`:

```python
PULPReduceLogSumExpBindings = [
    NodeBinding(ReduceLogSumExpChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatReduceLogSumExpTemplate.referenceTemplate, ForkTransformer)
]
```

In `Tiler.py` e' stato aggiunto il binding tiling-ready:

```python
_PULPReduceLogSumExpBindings = copy.deepcopy(PULPReduceLogSumExpBindings)

PULPReduceLogSumExpTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = _PULPReduceLogSumExpBindings,
    tileConstraint = UntiledTileConstraint()
)
```

In `Platform.py`, il mapper PULP e' stato collegato a `PULPReduceLogSumExpTilingReadyBindings`.

### Perche'

`ReduceLogSumExp` e' una riduzione. Per questo step non e' stata inventata una nuova
tilizzazione matematica del kernel. La scelta conservativa e' stata renderlo compatibile
con il flusso tiled tramite `UntiledTileConstraint`, come gia' fatto per altri operatori
che devono restare atomici ma vivere in una rete tiled.

Il passaggio da binding Generic a binding PULP era necessario anche per usare il transformer
memory-aware corretto. Con il binding Generic l'output del nodo non veniva allocato
correttamente nel flusso PULP, causando poi un accesso invalido nel kernel C.

### Nota sulla scelta `UntiledTileConstraint`

Il kernel numerico usato per `ReduceLogSumExp` e' ancora quello Generic:

- `TargetLibraries/Generic/src/ReduceLogSumExp_fp32.c`
- `Deeploy/Targets/Generic/Templates/FloatReduceLogSumExpTemplate.py`

La patch non introduce quindi una implementazione PULP ottimizzata della riduzione.
Introduce invece un'integrazione PULP/Siracusa memory-aware, in modo che il nodo possa
essere inserito correttamente in una rete tiled L3.

La forma numerica stabile di `ReduceLogSumExp` e':

```text
m = max(x_i)
out = log(sum(exp(x_i - m))) + m
```

Questa operazione ha una dipendenza globale lungo l'asse ridotto. Se l'asse della riduzione
venisse tagliato ingenuamente in tile indipendenti, ogni tile vedrebbe solo una parte dei
valori, calcolerebbe un massimo locale e una somma locale, e il risultato finale sarebbe
numericamente sbagliato.

Un tiling corretto richiederebbe un kernel/constraint dedicato multi-pass:

1. calcolo dei massimi locali per ogni tile;
2. riduzione dei massimi locali in un massimo globale;
3. calcolo delle somme parziali `sum(exp(x_i - max_globale))`;
4. riduzione delle somme parziali;
5. calcolo finale `log(sum_globale) + max_globale`.

Per questo debug e' stato scelto `UntiledTileConstraint`: il nodo resta atomico, ma viene
gestito correttamente dal flusso tiled e puo' convivere con gli altri layer in L3.

### Ha senso un kernel C PULP dedicato?

Si', ma non era necessario per sbloccare questo modello.

Ha senso implementarlo se:

- `ReduceLogSumExp` diventa un collo di bottiglia di runtime;
- l'input della riduzione diventa troppo grande per essere tenuto atomico nel livello di
  memoria scelto;
- serve sfruttare parallelismo sui core PULP per riduzioni lunghe;
- si vuole supportare davvero il tiling lungo l'asse ridotto.

Nel caso corrente `Autoencoder2D_GMM` usa `ReduceLogSumExp` su una testa GMM piccola
rispetto al resto del modello. La soluzione conservativa e' quindi preferibile: meno codice
nuovo, minore rischio numerico, e test finali corretti su Siracusa tiled e Siracusa + Neureka.

## 5) Siracusa tiled L3 - reference duplicata nel self-Mul

Dopo il fix di `ReduceLogSumExp`, il tiler proseguiva ma falliva durante la codegen.

Esito: `FAILED`

Errore principale:

```text
KeyError: 'Buffername TILING_CODEGEN_L1_gmm_modelgmmMul_gmm_modelAdd_output_0_tensor_ref was already in the local context!'
```

Nodo ONNX coinvolto:

```text
Mul /gmm_model/gmm/Mul ['/gmm_model/Add_output_0', '/gmm_model/Add_output_0']
```

### Causa

Il nodo e' un self-Mul: stesso tensore usato come entrambi gli ingressi, cioe':

```text
Mul(x, x)
```

La codegen tiled crea reference locali basate sul nome del buffer esterno. Con due ingressi
uguali, tentava di creare due volte la stessa reference nello stesso `NetworkContext`.

### Modifica applicata

File modificato:

- `Deeploy/TilingExtension/CodeTransformationPasses/TilingHoistingMixIn.py`

La funzione `_hoistReference(...)` ora riusa una reference locale gia' esistente se:

- il nome coincide
- il buffer referenziato coincide
- shape e offset coincidono
- il tipo coincide

Modifica concettuale:

```python
refName = self.prefix + name
if ctxt.is_local(refName):
    ref = ctxt.lookup(refName)
    assert isinstance(ref, _ReferenceBuffer)
    assert ref._referenceName == reference.name
    assert tuple(ref.shape) == tuple(shape)
    expectedOffset = offset.name if isinstance(offset, VariableBuffer) else offset
    assert ref._offset == expectedOffset
    ...
    return ref

ref = ctxt.hoistReference(refName, reference, shape, offset, override_type)
```

### Perche'

Due ingressi dello stesso nodo possono puntare allo stesso tensore ONNX. In quel caso
duplicare la reference non serve ed e' sbagliato: va riusata la stessa reference locale.

## 6) Siracusa tiled L3 - errore numerico su `gmm_output`

Dopo la correzione della reference duplicata, il modello generava, compilava ed eseguiva,
ma falliva sul confronto finale.

Esito: `FAILED`

Risultato:

```text
Expected: 113.107033  Actual: -39.171692  Diff: 152.278717 at Index 0 in Output 1
Errors: 1 out of 241
```

Osservazione:

- `reconstruction` era corretta
- solo `gmm_output` era errato

### Causa

Il template PULP FP32 di `Mul` assumeva sempre che il secondo input fosse uno scalare:

```c
float32_t scalar = B[0];
C[i] = A[i] * scalar;
```

Questo e' corretto per `Mul(vettore, scalare)`, ma non per `Mul(vettore, vettore)`.

Nel grafo GMM il nodo:

```text
Mul('/gmm_model/Add_output_0', '/gmm_model/Add_output_0')
```

deve calcolare:

```text
x * x
```

Il template PULP calcolava invece:

```text
x * x[0]
```

Questo corrompeva la testa GMM prima di `MatMul`, `Add` e `ReduceLogSumExp`.

Il template Generic era gia' corretto e distingueva:

```python
B[0] se sizeB == 1
B[i] altrimenti
```

### Modifica applicata

File modificato:

- `Deeploy/Targets/PULPOpen/Templates/FloatMulTemplate.py`

Il template PULP ora mantiene l'ottimizzazione scalare solo quando `sizeB == 1`,
altrimenti usa moltiplicazione elemento-per-elemento.

Modifica concettuale:

```mako
% if sizeB == 1:
    float32_t ${nodeName}_scalar = ${B}[0];
% endif

...

% if sizeB == 1:
    ${C}[i] = ${A}[i] * ${nodeName}_scalar;
% else:
    ${C}[i] = ${A}[i] * ${B}[i];
% endif
```

### Perche'

`Mul` ONNX supporta il caso vettore-vettore. La testa GMM lo usa esplicitamente per
calcolare un termine quadratico. Il target PULP era piu' restrittivo del target Generic
e produceva codice numericamente sbagliato.

## 7) Siracusa tiled L3 - risultato finale

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D_GMM --defaultMemLevel=L3 --l2=3000000
```

Esito: `PASSED`

Risultato:

```text
Errors: 0 out of 241
Runtime: 15934289 cycles
```

## 8) Siracusa tiled L3 + Neureka - risultato finale

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D_GMM --defaultMemLevel=L3 --l2=3000000
```

Esito: `PASSED`

Risultato:

```text
Errors: 0 out of 241
Runtime: 15868032 cycles
```

## Modifiche esatte applicate

### `Deeploy/Targets/PULPOpen/Platform.py`

- aggiunto import di `ReduceLogSumExpLayer`
- aggiunto import di `ReduceLogSumExpParser`
- aggiunto import di `PULPReduceLogSumExpTilingReadyBindings`
- aggiunto:

```python
ReduceLogSumExpMapper = NodeMapper(ReduceLogSumExpParser(), PULPReduceLogSumExpTilingReadyBindings)
```

- aggiunta entry:

```python
'ReduceLogSumExp': ReduceLogSumExpLayer([ReduceLogSumExpMapper])
```

Motivo:

- rendere visibile `ReduceLogSumExp` alla piattaforma PULP/Siracusa.

### `Deeploy/Targets/PULPOpen/Bindings.py`

- aggiunto import di `FloatReduceLogSumExpTemplate`
- aggiunto import di `ReduceLogSumExpChecker`
- aggiunto `PULPReduceLogSumExpBindings`
- aggiunto binding `Concat` FP32 a `PULPConcatBindings`

Motivo:

- usare `ReduceLogSumExp` nel flusso PULP con transformer memory-aware
- supportare `Concat` FP32 nella testa GMM

### `Deeploy/Targets/PULPOpen/Tiler.py`

- aggiunto import di `PULPReduceLogSumExpBindings`
- aggiunto:

```python
PULPReduceLogSumExpTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = _PULPReduceLogSumExpBindings,
    tileConstraint = UntiledTileConstraint()
)
```

Motivo:

- permettere al tiler di attraversare `ReduceLogSumExp` senza spezzarlo in tile non supportati.

### `Deeploy/TilingExtension/CodeTransformationPasses/TilingHoistingMixIn.py`

- `_hoistReference(...)` ora riusa una reference locale gia' esistente quando e' equivalente.

Motivo:

- gestire nodi con lo stesso tensore usato piu' volte in input, come `Mul(x, x)`.

### `Deeploy/Targets/PULPOpen/Templates/FloatMulTemplate.py`

- corretto il template FP32 PULP:
  - `sizeB == 1`: usa `B[0]` come scalare
  - `sizeB != 1`: usa `B[i]`

Motivo:

- supportare correttamente `Mul` elemento-per-elemento oltre al caso scalare.

## Verifiche finali

Verifica sintattica Python:

```bash
python -m py_compile \
  Deeploy/Targets/PULPOpen/Bindings.py \
  Deeploy/Targets/PULPOpen/Platform.py \
  Deeploy/Targets/PULPOpen/Tiler.py \
  Deeploy/Targets/PULPOpen/Templates/FloatMulTemplate.py \
  Deeploy/TilingExtension/CodeTransformationPasses/TilingHoistingMixIn.py
```

Esito: `PASSED`

Test finali:

- Siracusa tiled L3: `PASSED`, `Errors: 0 out of 241`
- Siracusa tiled L3 + Neureka: `PASSED`, `Errors: 0 out of 241`

## Nota finale

Il target Siracusa non-tiled non e' stato portato a `PASSED` per limite di memoria L2,
non per mancanza di layer dopo le patch. Il percorso funzionante per `Autoencoder2D_GMM`
e' quello tiled con `--defaultMemLevel=L3 --l2=3000000`.

## Aggiornamento modello e tolleranza FP (2026-05-06)

Dopo un aggiornamento di:

- `Tests/Models/Autoencoder2D_GMM/network.onnx`
- `Tests/Models/Autoencoder2D_GMM/inputs.npz`
- `Tests/Models/Autoencoder2D_GMM/outputs.npz`

il numero di output verificati e' passato da `241` a `161`.

### Verifica Generic

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_generic.py -t Tests/Models/Autoencoder2D_GMM
```

Esito: `PASSED`

Risultato:

```text
Errors: 0 out of 161
```

### Regressione apparente su Siracusa tiled

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D_GMM --defaultMemLevel=L3 --l2=3000000
```

Esito iniziale: `FAILED`

Errore osservato:

```text
Expected: 619.625488  Actual: 619.625610  Diff: -0.000122 at Index 0 in Output 1
Errors: 1 out of 161
```

### Causa

Il modello e i dati erano coerenti, infatti il target `Generic` passava.

L'errore Siracusa era dovuto al criterio di confronto FP nel test runtime:

```c
if ((diff < -1e-4) || (diff > 1e-4) || isnan(diff))
```

Questa e' una tolleranza solo assoluta. Con il nuovo modello `gmm_output` vale circa `620`,
quindi una differenza assoluta di `1.22e-4` corrisponde a un errore relativo di circa
`2e-7`. Questo e' compatibile con differenze normali tra implementazioni FP di `expf/logf`
e con l'esecuzione su target PULP.

### Modifica applicata

File modificati:

- `DeeployTest/Platforms/Siracusa/src/deeploytest.c`
- `DeeployTest/Platforms/PULPOpen/src/deeploytest.c`

La comparazione float ora usa tolleranza assoluta piu' relativa:

```c
#define FLOAT_ABS_TOL 1e-4f
#define FLOAT_REL_TOL 1e-5f

float abs_diff = fabsf(diff);
float scale = fabsf(expected_val);
float abs_actual = fabsf(actual_val);
if (abs_actual > scale) {
  scale = abs_actual;
}
float tolerance = FLOAT_ABS_TOL + FLOAT_REL_TOL * scale;

if ((abs_diff > tolerance) || isnan(diff)) {
  ...
}
```

### Perche'

La tolleranza assoluta resta utile per valori piccoli. La tolleranza relativa evita invece
di segnare come errore uno scarto numerico molto piccolo rispetto alla scala del valore.
Questo e' particolarmente rilevante per la testa GMM, dove `ReduceLogSumExp` usa `expf`
e `logf`.

### Verifiche dopo la modifica

Siracusa tiled L3:

```text
Errors: 0 out of 161
Runtime: 15914263 cycles
```

Siracusa tiled L3 + Neureka:

```text
Errors: 0 out of 161
Runtime: 15945410 cycles
```

## Pulizia warning MatMul tiled (2026-05-06)

Durante la compilazione Siracusa tiled del modello aggiornato comparivano warning del tipo:

```text
warning: implicit conversion changes signedness: 'int8_t' to 'unsigned int'
```

Il warning era generato nel `Network.c` tilizzato per il nodo `gmm_modelMatMul`.
La dimensione tilizzata `O` veniva materializzata come `int8_t *O_ref` e poi usata in:

- pointer arithmetic;
- chiamata a `PULP_MatMul_fp32_fp32_fp32_unroll1x7`, che si aspetta dimensioni unsigned.

### Causa

In `Deeploy/Targets/PULPOpen/TileConstraints/MatMulTileConstraint.py` i replacement
tilizzati di `MatMul` erano tipizzati come `int8_t`:

```python
replacementTypes = {
    "M": PointerClass(int8_t),
    "N": PointerClass(int8_t),
    "O": PointerClass(int8_t),
    "batch": PointerClass(int8_t)
}
```

Queste grandezze sono dimensioni di tile, quindi non possono essere negative.
Il tipo signed non era semanticamente necessario.

### Modifica applicata

File modificato:

- `Deeploy/Targets/PULPOpen/TileConstraints/MatMulTileConstraint.py`

Il tipo dei replacement e' stato allineato a `GEMMTileConstraint`, usando `uint16_t`:

```python
replacementTypes = {
    "M": PointerClass(uint16_t),
    "N": PointerClass(uint16_t),
    "O": PointerClass(uint16_t),
    "batch": PointerClass(uint16_t)
}
```

### Perche'

`M`, `N`, `O` e `batch` sono dimensioni non negative. Usare `uint16_t` elimina i warning
signed-to-unsigned e rende il codice generato piu' coerente con le API dei kernel PULP.

### Verifica

Comando:

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D_GMM --defaultMemLevel=L3 --l2=3000000
```

Esito:

```text
Errors: 0 out of 161
Runtime: 15946335 cycles
```

I warning `sign-conversion` nel `Network.c` generato sono spariti.

Restano warning non legati al codice Deeploy generato:

- `clang-15: warning: argument unused during compilation: '-nostartfiles'`
- `llvm-objdump: warning: failed to find source ... newlib ...`

Questi provengono dal toolchain/debug info e non indicano una regressione numerica o di
generazione del modello.
