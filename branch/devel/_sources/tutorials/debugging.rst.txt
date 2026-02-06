.. SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
..
.. SPDX-License-Identifier: Apache-2.0

Debugging
=========

Printing Tensor Values
----------------------

Deeploy provides two primary approaches for printing the input and output tensors of computational kernels:

1. **Topology Pass**: This approach modifies the graph's topology by inserting print nodes. These nodes can be added before or after a specified operator, which can be selected using a regular expression.

2. **Code Transformation**: This approach inserts code-level print statements to output tensor values at various stages of execution. Deeploy also offers memory-aware versions that allow printing values at specific memory levels (e.g., per tile).

Topology Pass
~~~~~~~~~~~~~

.. currentmodule:: Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.DebugPasses

The :py:class:`DebugPrint` topology pass modifies the graph by inserting print nodes either before or after specified operators. The target operator(s) can be selected using a regular expression pattern.

To enable this, extend the optimization passes by adding :py:class:`DebugPrintPass`. For example, to modify the ``GenericOptimizer`` in ``Deeploy/Targets/Generic/Platform.py``, you can add:

.. code-block:: python

    GenericOptimizer = TopologyOptimizer([
        # ... existing passes ...
        DebugPrintPass(r'.*[Mm]at[Mm]ul.*', position='after'),
    ])

**Ensure that your platform provides a valid implementation and mapping for the ``DebugPrint`` node.**

Code Transformation
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: Deeploy.CommonExtensions.CodeTransformationPasses.PrintInputs

The :py:class:`PrintInputGeneration` and :py:class:`PrintOutputGeneration` code transformations offer a flexible way to insert print statements directly into the generated code. These transformations allow you to log tensor values at any point during execution, making them useful for in-depth debugging.
For cases where memory layout is important—such as debugging tiled execution—Deeploy also provides memory-aware variants: :py:class:`MemoryAwarePrintInputGeneration` and :py:class:`MemoryAwarePrintOutputGeneration`.

To use these transformations, add them to the code transformation pipeline in your target bindings. For example, you can extend the ``BasicTransformer`` in ``Deeploy/Targets/Generic/Bindings.py``:

.. code-block:: python

    BasicTransformer = CodeTransformation([
        # ... existing passes ...
        PrintInputGeneration(),
        PrintOutputGeneration()
    ])

For memory-aware platforms, use the memory-aware transformations instead. For example, extend ``ForkTransformer`` in ``Deeploy/Targets/PULPOpen/Platform.py``:

.. code-block:: python

    ForkTransformer = CodeTransformation([
        # ... existing passes ...
        MemoryAwarePrintInputGeneration("L1"),
        MemoryAwarePrintOutputGeneration("L1")
    ])

To apply these code transformations across all bindings, refer to the test implementation in ``DeeployTest/testPrintInputOutputTransformation.py``.

The output of the print statements will be directed to the standard output stream.
.. code-block::

    Add_0 DeeployNetwork_input_0: int8_t, [1, 5, 5, 5], 0x2062b0
    [[[[ -63, -76,-116, -22, -35,],
    [-105, -69, -51, -95, -69,],
    [-104,  -6, -37, -12, -63,],
    [ -32, -10,  -8, -29, -15,],
    [-111, -18,-120,-106, -50,],
    ],
    [[ -62, -22, -60,-109, -13,],
    [ -78, -52, -42,-104,-100,],
    [-115,-105,-119,-104, -62,],
    [ -57, -81,-104, -39, -13,],
    [ -51, -47, -18, -14,-123,],
    ],
    [[-111, -92, -91, -84,-121,],
    [ -41,-118,-128,-109,  -7,],
    [-120, -66,  -9, -66, -55,],
    [ -96, -34, -47,-105, -91,],
    [ -94,-106,  -3,-121, -55,],
    ],
    [[ -15, -65,-126, -11,-101,],
    [ -85,  -6, -98, -46, -84,],
    [ -55,  -8, -53, -99, -79,],
    [ -72, -12,  -2,-103, -55,],
    [ -95, -48, -16, -78, -95,],
    ],
    [[ -18, -40, -43,-104, -78,],
    [ -33,-102, -46,-110, -40,],
    [-128, -24, -68, -70,-113,],
    [ -73,-123,-114, -51,  -3,],
    [ -69, -68, -66, -26,-124,],
    ],
    ],
    ],