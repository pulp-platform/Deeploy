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
