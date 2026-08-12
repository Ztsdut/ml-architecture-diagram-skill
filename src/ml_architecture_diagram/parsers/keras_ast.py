from __future__ import annotations

"""Conservative static TensorFlow/Keras architecture parser.

Supports common subclassed ``keras.Model``/``Layer`` code, functional API chains,
and simple ``Sequential([...])`` definitions without executing user code.
"""

from pathlib import Path
import ast
from typing import Any

from .pytorch_ast import ParseState, ModuleInfo, _dotted, _short_type, _safe_text, _human_module_label


_ROLE = {
    "Dense": "head", "Embedding": "representation",
    "Conv1D": "backbone", "Conv2D": "backbone", "Conv3D": "backbone",
    "LSTM": "backbone", "GRU": "backbone", "SimpleRNN": "backbone", "Bidirectional": "backbone",
    "MultiHeadAttention": "novel", "Attention": "novel",
    "LayerNormalization": "preprocess", "BatchNormalization": "preprocess", "Dropout": "preprocess",
    "MaxPooling1D": "operator", "MaxPooling2D": "operator", "AveragePooling1D": "operator", "AveragePooling2D": "operator",
    "GlobalAveragePooling1D": "operator", "GlobalAveragePooling2D": "operator", "Flatten": "operator", "Reshape": "operator",
    "Concatenate": "fusion", "Add": "fusion", "Multiply": "fusion",
}

_MERGE = {"Concatenate": "Concat", "Add": "+", "Multiply": "×"}


def _label(name: str, typ: str) -> str:
    aliases = {
        "Conv1D": "1-D Convolution", "Conv2D": "2-D Convolution", "Conv3D": "3-D Convolution",
        "Dense": "Dense", "LayerNormalization": "Layer Norm", "BatchNormalization": "Batch Norm",
        "MultiHeadAttention": "Multi-Head Attention", "GlobalAveragePooling1D": "Global Avg Pool",
        "GlobalAveragePooling2D": "Global Avg Pool", "MaxPooling1D": "Max Pool", "MaxPooling2D": "Max Pool",
    }
    if typ in aliases:
        return aliases[typ]
    return _human_module_label(name, typ)


class KerasAstParser:
    def __init__(self, source: str, source_name: str = "model.py", class_name: str | None = None, detail: str = "system"):
        self.source = source
        self.source_name = source_name
        self.tree = ast.parse(source, filename=source_name)
        self.requested_class = class_name
        self.detail = detail
        self.state = ParseState()
        self.class_node = self._select_class_optional()
        self.class_name = self.class_node.name if self.class_node else None
        if self.class_node:
            self._collect_modules()

    def _select_class_optional(self) -> ast.ClassDef | None:
        classes = [n for n in self.tree.body if isinstance(n, ast.ClassDef)]
        if self.requested_class:
            for c in classes:
                if c.name == self.requested_class:
                    return c
            raise ValueError(f"Class {self.requested_class!r} not found in {self.source_name}.")
        candidates = []
        for c in classes:
            bases = {_dotted(b).split(".")[-1] for b in c.bases}
            has_call = any(isinstance(x, ast.FunctionDef) and x.name in {"call", "__call__"} for x in c.body)
            if has_call and ({"Model", "Layer"} & bases or any(b.endswith(("Model", "Layer")) for b in bases)):
                candidates.append(c)
        return candidates[-1] if candidates else None

    def _collect_modules(self) -> None:
        init = next((x for x in self.class_node.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)
        if not init:
            self.state.unresolved.append("No __init__() found; Keras layer construction metadata is unavailable.")
            return
        for stmt in ast.walk(init):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    name = target.attr
                    typ = _short_type(value)
                    self.state.modules[name] = ModuleInfo(name, typ, _ROLE.get(typ, "backbone"), 1, _safe_text(value), typ in {"Sequential"}, "")

    def _input(self, name: str, label: str | None = None) -> str:
        if name in self.state.vars:
            return self.state.vars[name]
        nid = self.state.add_node(f"input_{name}", label or name.replace("_", " ").title(), role="input", kind="data")
        self.state.vars[name] = nid
        return nid

    def _resolve(self, expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name):
            return self.state.vars.get(expr.id)
        if isinstance(expr, (ast.Attribute, ast.Subscript)):
            base = expr.value if isinstance(expr, ast.Attribute) else expr.value
            return self._resolve(base)
        return None

    def _producers(self, expr: ast.AST) -> list[str]:
        if isinstance(expr, ast.Name):
            n = self.state.vars.get(expr.id)
            return [n] if n else []
        if isinstance(expr, (ast.List, ast.Tuple)):
            out = []
            for e in expr.elts:
                out.extend(self._producers(e))
            return list(dict.fromkeys(out))
        if isinstance(expr, ast.Call):
            n = self._expr(expr)
            return [n] if n else []
        if isinstance(expr, ast.BinOp):
            n = self._expr(expr)
            return [n] if n else []
        n = self._resolve(expr)
        return [n] if n else []

    def _new_layer_node(self, name: str, typ: str, sources: list[str]) -> str:
        if typ in _MERGE:
            nid = self.state.add_node(name, _MERGE[typ], role="fusion", kind="merge")
            for i, src in enumerate(sources):
                self.state.add_edge(src, nid, "main" if i == 0 or typ != "Add" else "residual")
            return nid
        role = _ROLE.get(typ, "backbone")
        kind = "operator" if role == "operator" else "module"
        nid = self.state.add_node(name, _label(name, typ), role=role, kind=kind)
        for src in sources:
            self.state.add_edge(src, nid, "main")
        return nid

    def _expr(self, expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name):
            return self.state.vars.get(expr.id)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Mult)):
            left, right = self._expr(expr.left) or self._resolve(expr.left), self._expr(expr.right) or self._resolve(expr.right)
            sym = "+" if isinstance(expr.op, ast.Add) else "×"
            nid = self.state.add_node("add" if sym == "+" else "multiply", sym, role="fusion", kind="merge")
            self.state.add_edge(left, nid, "main")
            self.state.add_edge(right, nid, "residual" if sym == "+" else "main")
            return nid
        if not isinstance(expr, ast.Call):
            return self._resolve(expr)

        # layers.Dense(...)(x): outer call whose func is the layer-constructor call.
        if isinstance(expr.func, ast.Call):
            typ = _short_type(expr.func)
            sources = []
            for a in expr.args:
                sources.extend(self._producers(a))
            return self._new_layer_node(typ.lower(), typ, list(dict.fromkeys(sources)))

        name = _dotted(expr.func)
        short = name.split(".")[-1]
        if short == "Input":
            nid = self.state.add_node("keras_input", "Input", role="input", kind="data")
            return nid
        if name.startswith("self."):
            attr = name.split(".")[1]
            info = self.state.modules.get(attr, ModuleInfo(attr, "", "backbone"))
            sources = []
            for a in expr.args:
                sources.extend(self._producers(a))
            return self._new_layer_node(attr, info.type_name or attr, list(dict.fromkeys(sources)))
        if short in {"concatenate", "concat", "stack"}:
            label = "Concat" if short != "stack" else "Stack"
            sources = self._producers(expr.args[0]) if expr.args else []
            nid = self.state.add_node(label.lower(), label, role="fusion", kind="merge")
            for src in sources:
                self.state.add_edge(src, nid, "main")
            return nid
        if isinstance(expr.func, ast.Attribute):
            base = self._resolve(expr.func.value)
            if base and short in {"mean", "sum"}:
                nid = self.state.add_node(short, "Mean Pool" if short == "mean" else "Sum", role="operator", kind="operator")
                self.state.add_edge(base, nid, "main")
                return nid
        sources = []
        for a in expr.args:
            sources.extend(self._producers(a))
        if self.detail != "system" and short not in {"Model", "Sequential"}:
            return self._new_layer_node(short.lower(), short, list(dict.fromkeys(sources)))
        return sources[0] if sources else None

    def _assign(self, target: ast.AST, value: ast.AST) -> None:
        producer = self._expr(value)
        if isinstance(target, ast.Name) and producer:
            self.state.vars[target.id] = producer
        elif isinstance(target, (ast.Tuple, ast.List)) and producer:
            for e in target.elts:
                if isinstance(e, ast.Name):
                    self.state.vars[e.id] = producer

    def _stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            # Functional API model = keras.Model(inputs=..., outputs=...)
            if isinstance(stmt.value, ast.Call) and _short_type(stmt.value) == "Model":
                outputs = None
                for kw in stmt.value.keywords:
                    if kw.arg in {"outputs", "output"}:
                        outputs = kw.value
                if outputs is None and len(stmt.value.args) >= 2:
                    outputs = stmt.value.args[1]
                for src in self._producers(outputs) if outputs is not None else []:
                    out = self.state.add_node("output", "Output", role="output", kind="output")
                    self.state.add_edge(src, out, "main")
                return
            for t in stmt.targets:
                self._assign(t, stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            self._assign(stmt.target, stmt.value)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            vals = stmt.value.elts if isinstance(stmt.value, (ast.Tuple, ast.List)) else [stmt.value]
            for i, v in enumerate(vals, 1):
                src = self._expr(v) or self._resolve(v)
                out = self.state.add_node(f"output_{i}", "Output" if len(vals) == 1 else f"Output {i}", role="output", kind="output")
                self.state.add_edge(src, out, "main")
        elif isinstance(stmt, ast.If):
            self.state.unresolved.append(f"Conditional control flow at line {stmt.lineno} was not executed; review if it changes Keras topology.")
            for x in stmt.body:
                self._stmt(x)
        elif isinstance(stmt, ast.For):
            self.state.unresolved.append(f"Loop at line {stmt.lineno} requires manual review unless represented by a reusable Keras layer/container.")
            for x in stmt.body:
                self._stmt(x)

    def _parse_sequential_top_level(self) -> bool:
        for stmt in self.tree.body:
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call) or _short_type(stmt.value) != "Sequential":
                continue
            if not stmt.value.args or not isinstance(stmt.value.args[0], (ast.List, ast.Tuple)):
                continue
            prev = None
            for i, layer in enumerate(stmt.value.args[0].elts):
                if not isinstance(layer, ast.Call):
                    continue
                typ = _short_type(layer)
                if typ == "Input" or typ == "InputLayer":
                    prev = self.state.add_node("input", "Input", role="input", kind="data")
                else:
                    nid = self._new_layer_node(f"{typ.lower()}_{i}", typ, [prev] if prev else [])
                    prev = nid
            if prev:
                out = self.state.add_node("output", "Output", role="output", kind="output")
                self.state.add_edge(prev, out, "main")
            return True
        return False

    def parse(self) -> dict[str, Any]:
        title = "Keras model architecture"
        if self.class_node:
            title = f"{self.class_name} architecture"
            fn = next((x for x in self.class_node.body if isinstance(x, ast.FunctionDef) and x.name in {"call", "__call__"}), None)
            if fn is None:
                raise ValueError(f"Class {self.class_name!r} has no call() method.")
            for arg in [a.arg for a in fn.args.args if a.arg not in {"self", "training", "mask"}]:
                self._input(arg)
            for stmt in fn.body:
                self._stmt(stmt)
        elif not self._parse_sequential_top_level():
            for stmt in self.tree.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and _short_type(stmt.value) == "Input":
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            self.state.vars[target.id] = self.state.add_node(f"input_{target.id}", target.id.replace("_", " ").title(), role="input", kind="data")
                else:
                    self._stmt(stmt)
            if not any(n.get("role") == "output" for n in self.state.nodes):
                self.state.unresolved.append("Functional Keras output could not be identified. Prefer an explicit keras.Model(inputs=..., outputs=...) assignment.")

        return {
            "version": "1.0",
            "figure": {"title": title, "direction": "LR", "target": "paper", "theme": "paper-light", "font": "Arial", "width_mm": 178},
            "metadata": {
                "source_framework": "keras",
                "source_file": self.source_name,
                "source_class": self.class_name,
                "parser": "keras-ast",
                "parser_detail": self.detail,
                "unresolved": list(dict.fromkeys(self.state.unresolved)),
            },
            "nodes": self.state.nodes,
            "edges": self.state.edges,
            "style": {"compact": self.detail == "system", "show_shapes": True, "show_repeat_badges": True, "visual_level": "publication-rich"},
        }


def parse_keras_source(source: str, *, source_name: str = "model.py", class_name: str | None = None, detail: str = "system") -> dict[str, Any]:
    if detail not in {"system", "block", "operation"}:
        raise ValueError("detail must be one of: system, block, operation")
    return KerasAstParser(source, source_name=source_name, class_name=class_name, detail=detail).parse()


def parse_keras_file(path: str | Path, *, class_name: str | None = None, detail: str = "system") -> dict[str, Any]:
    path = Path(path)
    return parse_keras_source(path.read_text(encoding="utf-8"), source_name=str(path), class_name=class_name, detail=detail)
