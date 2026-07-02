from Deeploy.DeeployTypes import NodeTemplate

# snrt_l1alloc currently does not support free-ing of memory (spatz/sw/snRuntime/src/alloc.c)
spatzLocalTemplate = NodeTemplate("")
spatzGlobalTemplate = NodeTemplate("")