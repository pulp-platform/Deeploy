# Autoencoder2D_GMM

Questa cartella contiene il modello `Autoencoder2D_GMM` e i file necessari per usarlo con Deeploy e convertirlo in codice C per il target `Generic`.

## Contenuto della cartella

- `network.onnx`
  Modello ONNX finale. E' il modello che viene dato in ingresso a Deeploy per la generazione del codice C.

- `inputs.npz`
  Input di test del modello. Vengono usati per eseguire il network durante la validazione.

- `outputs.npz`
  Output attesi del modello. Vengono usati come riferimento per verificare che il codice C prodotto da Deeploy dia il risultato corretto.

- `README.md`
  Questo file.

## Cosa rappresenta il modello

`Autoencoder2D_GMM` e' un modello composto da due sottosistemi principali:

- un ramo `Autoencoder2D`, che produce l'output di ricostruzione
- un ramo `GMM`, integrato nello stesso grafo, che produce un secondo output

Il modello ha quindi due output finali:

- `reconstruction`
- `gmm_output`

## Dove finisce il codice C generato

Quando esegui Deeploy sul target `Generic`, i file C generati per questo modello vengono scritti in:

- [Network.c](/workspaces/Deeploy/DeeployTest/TEST_GENERIC/Tests/Models/Autoencoder2D_GMM/Network.c)
- [Network.h](/workspaces/Deeploy/DeeployTest/TEST_GENERIC/Tests/Models/Autoencoder2D_GMM/Network.h)
- [testinputs.h](/workspaces/Deeploy/DeeployTest/TEST_GENERIC/Tests/Models/Autoencoder2D_GMM/testinputs.h)
- [testoutputs.h](/workspaces/Deeploy/DeeployTest/TEST_GENERIC/Tests/Models/Autoencoder2D_GMM/testoutputs.h)

Questi file hanno il seguente ruolo:

- `Network.c`
  Contiene il codice C generato da Deeploy per eseguire l'inferenza del modello.

- `Network.h`
  Espone le funzioni principali del network e i puntatori ai buffer di input e output.

- `testinputs.h`
  Contiene gli input del file `inputs.npz` convertiti in array C.

- `testoutputs.h`
  Contiene gli output del file `outputs.npz` convertiti in array C.

## Main di esecuzione

Il programma che esegue il network generato e confronta i risultati si trova qui:

- [main.c](/workspaces/Deeploy/DeeployTest/Platforms/Generic/main.c)

Questo file:

- inizializza il network
- copia gli input nei buffer del modello
- esegue `RunNetwork`
- confronta gli output reali con quelli di riferimento

## Librerie C del target Generic

Le implementazioni dei kernel C usati dal network generato si trovano in:

- `/workspaces/Deeploy/TargetLibraries/Generic/src`

Alcuni file utili da conoscere sono:

- [MatMul_fp32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/MatMul_fp32.c)
- [Gemm_fp32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/Gemm_fp32.c)
- [Convolution_fp32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/Convolution_fp32.c)
- [ConvTranspose2d_fp32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/ConvTranspose2d_fp32.c)
- [ReduceLogSumExp_fp32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/ReduceLogSumExp_fp32.c)
- [Div_fp32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/Div_fp32.c)
- [Div_s32.c](/workspaces/Deeploy/TargetLibraries/Generic/src/Div_s32.c)

In pratica:

- `Network.c` descrive il tuo modello convertito
- le librerie in `TargetLibraries/Generic/src` implementano le operazioni numeriche chiamate da `Network.c`

## Flusso di lavoro

Il flusso standard e':

1. prepari `network.onnx`, `inputs.npz` e `outputs.npz`
2. Deeploy genera il codice C del modello
3. il target `Generic` compila il codice generato insieme alle librerie C di supporto
4. `main.c` esegue il modello e verifica i risultati

## Comando per fare tutto

Per generare, compilare ed eseguire il modello sul target `Generic`:

```bash
cd /workspaces/Deeploy/DeeployTest
python deeployRunner_generic.py -t Tests/Models/Autoencoder2D_GMM -v
```

## Nota importante

Se modifichi uno di questi file:

- `network.onnx`
- `inputs.npz`
- `outputs.npz`

devi rigenerare il codice Deeploy prima di ricompilare, altrimenti i file C generati potrebbero non essere piu' allineati con il contenuto del modello o dei dati di test.
