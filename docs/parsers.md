# Model parser design

The parsers convert source code into Architecture IR without treating a source file as a flat list of layers.

## Safety and fidelity

The default parser path is static AST inspection. It does not import the target module and does not execute `forward()` / `call()`.

This has two deliberate consequences:

1. parsing works without installing the user's ML framework;
2. runtime-only facts may remain unresolved.

The second outcome is preferable to silently inventing architecture facts.

## PyTorch parser

Command:

```bash
ml-arch parse-pytorch model.py --class MyModel --detail system
```

Detail levels:

- `system`: suppress low-value activations/reshapes where possible;
- `block`: keep more functional operators and custom calls;
- `operation`: preserve more implementation detail for debugging or block diagrams.

Common recognized patterns:

```python
x = self.encoder(x)
x = F.relu(x)
x = torch.cat([x, context], dim=-1)
x = x + residual
for block in self.blocks:
    x = block(x)
return self.head(x)
```

`ModuleList([... for _ in range(N)])` can be compressed to one node with `repeat: N` when the loop follows the common reusable-block pattern.

### PyTorch limitations

Manual/agent review is recommended for:

- dynamic module construction inside `forward()`;
- branch conditions driven by tensor values;
- dispatch through dictionaries/callbacks;
- shape-dependent topology;
- custom operator overloading;
- modules created by external factories or config not visible in the parsed file;
- models whose true call path is controlled by wrappers outside the selected class.

Future versions may add an opt-in `torch.fx`/runtime backend, but it should remain separate from the safe static default.

## Keras parser

Command:

```bash
ml-arch parse-keras model.py --detail system
```

Common recognized patterns:

```python
image = keras.Input(...)
x = layers.Conv2D(...)(image)
x = layers.GlobalAveragePooling2D()(x)
meta = layers.Dense(...)(metadata)
fused = layers.Concatenate()([x, meta])
out = layers.Dense(...)(fused)
model = keras.Model(inputs=[image, metadata], outputs=out)
```

Also supported in the initial parser:

- simple `keras.Sequential([...])`;
- subclassed `keras.Model` / `keras.layers.Layer` with `call()`;
- common merge layers and pooling layers.

### Keras limitations

Manual/agent review is recommended for complex Lambda layers, nested Model composition, custom `tf.function` control flow, dynamically reused layers, and graph construction spread across multiple factories/files.

## Why not pretend static parsing is universal?

No static parser can reliably reconstruct arbitrary Python programs. The repository therefore keeps the Architecture IR independent of the parser. An agent, a future runtime tracer, or another framework adapter can all emit the same IR and use the same renderer and validation pipeline.
