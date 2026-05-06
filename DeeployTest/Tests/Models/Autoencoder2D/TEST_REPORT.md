# Test Report - Autoencoder2D

Questo file raccoglie i risultati dei test eseguiti durante il debug e le osservazioni tecniche.
Aggiornare questo report ad ogni nuovo test.

Nota organizzativa:
- dal 2026-05-04 questo report e' mantenuto in `Tests/Models/Autoencoder2D/TEST_REPORT.md` (prima era stato salvato per errore sotto `Autoencoder2D_GMM`).

## Sessione corrente (2026-05-04)

### 1) Generic - Autoencoder2D (tentativo iniziale)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_generic.py -t Tests/Models/Autoencoder2D`
- Esito: `FAILED`
- Errore principale:
  - parsing fallito su nodo `encoderlayer1paddingPad`
  - `Did not find adequate mapping for graph ... Candidates: ['Pad1DParser', 'Pad2DParser']`

### 2) Generic - Autoencoder2D (retry)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_generic.py -t Tests/Models/Autoencoder2D`
- Esito: `PASSED`
- Risultato:
  - `Errors: 0 out of 160`
- Note:
  - warning di compilazione non bloccanti (unused vars / conversioni implicite).

### 3) Siracusa no-tiling con simulazione
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D`
- Esito: `FAILED`
- Stato:
  - build completata
  - errore in fase gvsoc/runtime (`Invalid fetch request`), con fallimento target `gvsoc_Autoencoder2D`.

### 4) Siracusa no-tiling senza simulazione (build-only)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D --skipsim`
- Esito: `PASSED`
- Note:
  - confermato che `--skipsim` evita l'esecuzione della simulazione e valida solo generate/build.

### 5) Siracusa tiled senza simulazione (default L2)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --skipsim`
- Esito: `FAILED`
- Errore principale:
  - tiling/memory allocation fallita
  - `Memory allocator failed ... L2 with capacity of -409076 bytes`
  - `minimalloc` invocato con capacity negativa.

### 6) Siracusa tiled senza simulazione (default L3, L2 aumentata)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --skipsim --defaultMemLevel=L3 --l2=3000000`
- Esito: `PASSED`
- Note:
  - configurazione stabile per generate/build tiled su Autoencoder2D.

### 7) Siracusa no-tiling con simulazione (retry)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D`
- Esito: `FAILED`
- Stato:
  - build completata
  - errore in fase post-build gvsoc: `Error copying file ... build_master/*.bin ... gvsoc_workdir/`
  - durante l'esecuzione compare ancora `Invalid fetch request` su PE cluster.

### 8) Siracusa tiled con simulazione (default L3, L2 aumentata)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- Esito: `FAILED`
- Stato:
  - build completata
  - in simulazione il network gira fino al confronto finale
  - mismatch numerico completo: `Errors: 160 out of 160`
  - runtime riportato: `12910631 cycles`

### 9) testMVP Siracusa_w_neureka con `--doublebuffer`
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python testMVP.py -p Siracusa_w_neureka -t /workspaces/Deeploy/DeeployTest/Tests/Models/Autoencoder2D --doublebuffer`
- Esito: `FAILED`
- Errore principale:
  - binding/tiling fallisce prima della simulazione
  - `RuntimeError: ERROR: Some geometrical constraints are infeasible`
  - stack principale in:
    - `Deeploy/TilingExtension/TilerModel.py:269` (`debugConstraints`)
    - `Deeploy/TilingExtension/TilerModel.py:358` (`trySolveModel`)
    - `Deeploy/TilingExtension/TilerExtension.py:316` (`computeTilingSchedule`)
- Note tecniche:
  - senza `--doublebuffer` lo stesso comando non fallisce in geometria, ma dopo in allocazione memoria (`minimalloc`, capacity L2 negativa).
  - questo indica che il blocco introdotto da `--doublebuffer` rende il sistema di vincoli geometrici infeasible.

### 10) Verifica path di import Python (run reale vs codice locale)
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `python -c "import Deeploy; print(Deeploy.__file__)"`
- Esito:
  - import di default da `/app/Deeploy/Deeploy/__init__.py` (non da `/workspaces/Deeploy/...`).
- Verifica aggiuntiva:
  - forzando `PYTHONPATH=/workspaces/Deeploy` il traceback punta al sorgente locale ma l'errore geometrico resta identico.
- Conclusione:
  - il problema non dipende da una mismatch di package, ma da vincoli realmente infeasible nel flusso `--doublebuffer`.

## Sessione corrente (2026-05-05)

Obiettivo:
- riprendere il debug dopo i microblocchi `Autoencoder2D_MicroBlocks`;
- confermare i target richiesti:
  - Generic
  - Siracusa tiled con memoria standard L3
  - Siracusa tiled con Neureka
- capire perche' `Encoder_mini` era stato risolto ma `Autoencoder2D` completo falliva ancora.

### 11) Generic - Autoencoder2D dopo fix ConvTranspose
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_generic.py -t Tests/Models/Autoencoder2D`
- Esito: `PASSED`
- Risultato:
  - `Errors: 0 out of 160`
- Nota:
  - il backend Generic non era piu' il problema per il modello completo.

### 12) Siracusa tiled L3 - Autoencoder2D prima del fix Conv FP
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- Esito: `FAILED`
- Risultato:
  - mismatch numerico completo: `Errors: 160 out of 160`
  - runtime osservato: circa `12924041 cycles`
- Nota:
  - il modello compilava ed eseguiva, ma l'output finale era completamente errato.

### 13) Siracusa tiled L3 + Neureka - Autoencoder2D prima del fix Conv FP
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- Esito: `FAILED`
- Risultato:
  - mismatch numerico completo: `Errors: 160 out of 160`
  - runtime osservato: circa `12961863 cycles`
- Nota:
  - il fallimento era coerente con Siracusa tiled senza Neureka.

### 14) Diagnostica per output intermedi
- Sono stati creati test ONNX intermedi in `/tmp/deeploy_diag_auto2d` e `/tmp/deeploy_diag_auto2d_enc`.
- Punti principali generati:
  - encoder Conv/Pool/ReLU/Flatten/Gemm
  - decoder Gemm/ConvTranspose/Conv/BatchNorm/last Conv/Slice
- Osservazione iniziale:
  - `auto_enc_linear` falliva in Siracusa tiled, ma questo non indicava un bug GEMM.
  - Risalendo la catena, `auto_relu_enc2` era gia' errato.
  - Risalendo ancora, `auto_conv_enc1` falliva con `Errors: 6400 out of 6400`.
- Conclusione:
  - la divergenza nasceva gia' dalla primissima Conv FP dell'encoder.
  - i layer lineari dei microblocchi erano corretti; nel modello completo ricevevano input gia' corrotto.

### 15) Causa trovata: overlap L1 tra bias e buffer im2col
- Nel C generato per `auto_conv_enc1`:
  - il kernel chiamato era `PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC`;
  - la bias era allocata in L1 a un offset che ricadeva dentro l'area usata dal buffer transient `im2col`;
  - il kernel parallelizza sul numero di core e usa una porzione di `im2col` per ogni core.
- Causa tecnica:
  - `PULP2DFloatConvIm2ColTemplate.computeTransientBuffersSize(...)` dimensionava il transient buffer usando `operatorRepresentation["n_cores"]`;
  - il valore `operatorRepresentation["n_cores"]` arrivava da `generateNetwork.py --cores`;
  - `deeployRunner.py` passava `args.cores` a CMake come `-DNUM_CORES=...`, ma non lo propagava anche agli argomenti di generazione;
  - quindi `generateNetwork.py` usava il suo default `--cores=1`, mentre il C compilato/eseguito usava `NUM_CORES=8`;
  - quindi veniva riservato spazio solo per 1 core, ma il kernel ne usava 8, sovrascrivendo la bias.
- Effetto:
  - la prima Conv produceva output sbagliato;
  - tutto il resto della rete divergeva, inclusi i GEMM successivi.

### 16) Verifica diagnostica dopo fix Conv FP
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t /tmp/deeploy_diag_auto2d_enc/auto_conv_enc1 --defaultMemLevel=L3 --l2=3000000`
- Esito: `PASSED`
- Risultato:
  - `Errors: 0 out of 6400`
  - runtime: `583115 cycles`
- Verifica aggiuntiva:
  - con `--cores=1`, il runner passa sia `-DNUM_CORES=1` a CMake sia `--cores=1` a `generateNetwork.py`;
  - `auto_conv_enc1` passa anche a 1 core con `Errors: 0 out of 6400`;
  - runtime osservato a 1 core: `1648612 cycles`.

### 17) Generic - Autoencoder2D finale
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_generic.py -t Tests/Models/Autoencoder2D`
- Esito: `PASSED`
- Risultato:
  - `Errors: 0 out of 160`

### 18) Siracusa tiled L3 - Autoencoder2D finale
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- Esito: `PASSED`
- Risultato:
  - `Errors: 0 out of 160`
  - runtime: `14048962 cycles`

### 19) Siracusa tiled L3 + Neureka - Autoencoder2D finale
- Comando:
  - `cd /workspaces/Deeploy/DeeployTest`
  - `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- Esito: `PASSED`
- Risultato:
  - `Errors: 0 out of 160`
  - runtime: `14061183 cycles`

## Cambiamenti al codice (solo fix efficaci)

### Fix ConvTranspose stride parsing
- File:
  - `Deeploy/Targets/Generic/Parsers.py`
- Modifica:
  - nel parser comune `ConvTransposeParser`, `stride_x` e `stride_y` erano assegnati invertiti.
  - Corretto in modo che:
    - `stride_x = node.attrs["strides"][0]`
    - `stride_y = node.attrs["strides"][1]` se presente, altrimenti uguale a `stride_x`
- Impatto:
  - risolve il caso progressivo dei microblocchi dove una Conv dopo `ConvTranspose` produceva layout/risultati errati.
  - `Encoder_mini` passa su Generic, Siracusa tiled L3 e Siracusa tiled L3 + Neureka.

### Fix dimensionamento im2col per Conv FP PULP tiled
- File:
  - `DeeployTest/testUtils/deeployRunner.py`
  - `Deeploy/Targets/PULPOpen/Templates/FloatConvTemplate.py`
- Modifica:
  - il runner propaga ora `--cores=...` anche ai generation args, non solo a CMake:
    - `--cores=<args.cores>` oppure `--cores=<args.num_cores>`
  - `PULP2DFloatConvIm2ColTemplate.computeTransientBuffersSize(...)`
  - `PULP2DFloatDWConvIm2ColTemplate.computeTransientBuffersSize(...)`
  - il numero di core usato per dimensionare il transient buffer viene letto direttamente da `operatorRepresentation["n_cores"]`, ora coerente con `NUM_CORES`.
  - la patch conservativa usata durante il debug resta commentata nel template:
    - `# n_cores = max(int(operatorRepresentation.get("n_cores", 8)), 8)`
- Impatto:
  - evita overlap in L1 tra `im2col` e bias/altre tile;
  - mantiene corretto anche il caso reale a 1 core senza allocare inutilmente per 8 core;
  - risolve `Autoencoder2D` completo su Siracusa tiled L3 e Siracusa tiled L3 + Neureka.

## Discussione memoria (verificata su codice generato)

### Domanda
I dati usati spesso (es. filtri convolutivi) vengono ricaricati da L3 ogni volta o rimangono in livelli intermedi?

### Evidenza osservata
Nel codice generato (`DeeployTest/TEST_SIRACUSA/Tests/Models/Autoencoder2D/Network.c`):
- i pesi vengono inizialmente caricati in L3 (`cl_ram_malloc` + `load_file_to_ram(...)`)
- nelle closure tiled vengono copiati da L3 a L2 (`pi_cl_ram_copy_2d(... weight_ExternalToLocal ...)`)
- nel loop interno vengono trasferiti da L2 a L1 per il compute (`mchan_transfer_1d(... weight_ref ...)`)

### Conclusione pratica
- L3 funziona da backing store capiente.
- L2 è staging/intermedio.
- L1 è il livello vicino al compute dove i tile vengono processati.
- Quindi non è un pattern "usa e rimetti sempre in L3 ad ogni micro-step"; c'è buffering e riuso nei livelli intermedi secondo la strategia di tiling.

## Comandi di riferimento usati
- `python deeployRunner_generic.py -t Tests/Models/Autoencoder2D`
- `python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D`
- `python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D --skipsim`
- `python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --skipsim`
- `python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --skipsim --defaultMemLevel=L3 --l2=3000000`
- `python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- `PYTHONPATH=/workspaces/Deeploy python deeployRunner_generic.py -t Tests/Models/Autoencoder2D`
- `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D --defaultMemLevel=L3 --l2=3000000`
- `PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t /tmp/deeploy_diag_auto2d_enc/auto_conv_enc1 --defaultMemLevel=L3 --l2=3000000`
