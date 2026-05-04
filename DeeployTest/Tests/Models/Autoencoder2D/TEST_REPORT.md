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

## Cambiamenti al codice (solo fix efficaci)

Al momento non ci sono nuovi cambiamenti al codice da registrare in questa sessione che abbiano risolto un problema.
I risultati ottenuti sopra derivano da variazioni di configurazione test/runtime.

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
