# Model Inspection Guide

The purpose of inspection is to recover the **active computation graph**, not to produce a raw framework graph dump.

## General evidence order

Prefer evidence in this order:

1. executed/declared forward call path;
2. active configuration controlling branch/block counts and dimensions;
3. framework graph/tracing/export when reliable;
4. model summaries / repr output as supporting evidence;
5. comments, names, and documentation as semantic hints only.

If sources disagree, the active execution path wins.

## PyTorch

Inspect:

- subclasses of `torch.nn.Module`;
- `__init__` for construction and registered modules;
- `forward()` for execution order, reuse, branching, residuals, and merges;
- `ModuleList` / `Sequential` / loops for repeat counts;
- configuration/checkpoint metadata when it changes active topology.

Useful corroboration when executable:

- `torch.fx` symbolic tracing for traceable graphs;
- hooks or shape summaries for tensor transitions;
- exported ONNX graphs for deployment paths.

Caveats:

- tracing may fail or simplify dynamic control flow;
- module `repr` can include unused modules;
- repeated invocation of one module does not mean multiple independent parameter sets.

## TensorFlow / Keras

For Functional models, inspect graph connectivity directly.

For subclassed models, inspect `call()` and configuration that affects branches or loops.

`model.summary()` is useful for dimensions and parameter counts but may not explain semantic branches well enough for a paper figure.

## JAX / Flax / Haiku

Inspect the module call path and explicit transforms. Pay special attention to:

- scans / repeated blocks;
- residual additions;
- shared parameters;
- explicit attention/operator functions;
- shape changes hidden in pure functions.

`jax.make_jaxpr` or framework tabulation can corroborate the graph when the model is executable.

## ONNX

ONNX provides an explicit operator graph. Use it to recover topology and tensor flow, then collapse low-level implementation operators into meaningful scientific modules.

Do not draw every `Reshape`, `Cast`, or bookkeeping node in a manuscript figure unless it is methodologically important.

## Dynamic routing and conditional computation

For MoE, early-exit, conditional branches, recurrent loops, or input-dependent execution:

- represent the routing decision explicitly;
- distinguish potential topology from a single realized trace;
- do not imply that all experts/branches execute if only a subset does.

## Shared weights

When the same learned module is reused:

- use a `shared` annotation or shared group when the distinction matters;
- do not imply independent parameter sets by duplicating unlabeled modules.

## Training-only paths

Losses, augmentations, teacher models, target encoders, stop-gradient branches, and regularizers should be marked `training` unless the requested figure is exclusively a training diagram.

## Shapes and dimensions

Derive shapes from explicit code/config or reliable runtime evidence. Never infer a tensor dimension only because a conventional architecture would normally have it.
