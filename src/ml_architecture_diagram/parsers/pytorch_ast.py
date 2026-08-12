from __future__ import annotations

"""Conservative static PyTorch architecture parser.

The parser intentionally does *not* execute model code.  It recognizes common
``nn.Module`` construction patterns and follows the selected ``forward`` AST to
produce a framework-neutral Architecture IR.  Ambiguities are recorded in
``metadata.unresolved`` instead of being silently guessed.
"""

from dataclasses import dataclass, field
from pathlib import Path
import ast
import re
from typing import Any

import yaml


_SIMPLE_ROLE = {
    "Conv1d": "backbone", "Conv2d": "backbone", "Conv3d": "backbone",
    "Linear": "head", "Embedding": "representation",
    "LSTM": "backbone", "GRU": "backbone", "RNN": "backbone",
    "Transformer": "novel", "TransformerEncoder": "novel", "TransformerDecoder": "novel",
    "MultiheadAttention": "novel",
    "BatchNorm1d": "preprocess", "BatchNorm2d": "preprocess", "BatchNorm3d": "preprocess",
    "LayerNorm": "preprocess", "GroupNorm": "preprocess",
    "Dropout": "preprocess", "Dropout1d": "preprocess", "Dropout2d": "preprocess",
    "ReLU": "operator", "GELU": "operator", "SiLU": "operator", "Sigmoid": "operator", "Tanh": "operator",
    "MaxPool1d": "operator", "MaxPool2d": "operator", "AvgPool1d": "operator", "AvgPool2d": "operator",
    "AdaptiveAvgPool1d": "operator", "AdaptiveAvgPool2d": "operator",
    "Flatten": "operator", "Identity": "operator",
}

_LOW_VALUE_SYSTEM = {
    "ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Dropout", "Dropout1d", "Dropout2d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "Identity",
}

_FUNCTION_LABELS = {
    "relu": "ReLU", "gelu": "GELU", "silu": "SiLU", "sigmoid": "Sigmoid", "tanh": "Tanh",
    "softmax": "Softmax", "dropout": "Dropout", "layer_norm": "Layer Norm",
    "max_pool1d": "Max Pool", "max_pool2d": "Max Pool", "avg_pool1d": "Avg Pool", "avg_pool2d": "Avg Pool",
    "flatten": "Flatten", "reshape": "Reshape", "view": "Reshape", "permute": "Permute",
}


def _dotted(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return _dotted(node.value)
    return ""


def _call_name(call: ast.Call) -> str:
    return _dotted(call.func)


def _short_type(call: ast.Call) -> str:
    return _call_name(call).split(".")[-1]


def _const_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
        return -int(node.operand.value)
    return None


def _range_repeat(node: ast.AST) -> int | None:
    # range(N), [x for _ in range(N)], tuple/list multiplication is handled elsewhere.
    if isinstance(node, ast.Call) and _call_name(node).split(".")[-1] == "range" and node.args:
        return _const_int(node.args[-1])
    return None


def _infer_repeat(expr: ast.AST) -> int | None:
    if isinstance(expr, ast.ListComp):
        for gen in expr.generators:
            n = _range_repeat(gen.iter)
            if n is not None:
                return max(1, n)
    if isinstance(expr, (ast.List, ast.Tuple)):
        return len(expr.elts) or None
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
        left = _const_int(expr.left)
        right = _const_int(expr.right)
        if left is not None:
            return max(1, left)
        if right is not None:
            return max(1, right)
    return None


def _safe_text(node: ast.AST, limit: int = 70) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _human_module_label(name: str, type_name: str) -> str:
    friendly = name.replace("_", " ").strip().title()
    aliases = {
        "Conv1d": "1-D Convolution", "Conv2d": "2-D Convolution", "Conv3d": "3-D Convolution",
        "Linear": "Linear", "Embedding": "Embedding",
        "LSTM": "LSTM", "GRU": "GRU", "RNN": "RNN",
        "MultiheadAttention": "Multi-Head Attention",
        "TransformerEncoder": "Transformer Encoder", "TransformerDecoder": "Transformer Decoder",
        "LayerNorm": "Layer Norm", "BatchNorm1d": "Batch Norm", "BatchNorm2d": "Batch Norm",
        "MaxPool1d": "Max Pool", "MaxPool2d": "Max Pool", "AdaptiveAvgPool1d": "Adaptive Avg Pool",
        "AdaptiveAvgPool2d": "Adaptive Avg Pool", "Flatten": "Flatten",
    }
    type_label = aliases.get(type_name, type_name)
    semantic_attr = name.lower()
    # A descriptive attribute name can carry more paper-level meaning than the primitive
    # constructor type. For example ``self.map_encoder = nn.Linear(...)`` is still the
    # model's Map Encoder at system level, not merely an anonymous Linear layer.
    if friendly and any(term in semantic_attr for term in ("encoder", "decoder", "projector", "adapter", "stem")):
        return friendly.replace("Mha", "MHA")
    # Preserve descriptive user names for custom blocks, while avoiding "Conv1 Conv2D Convolution" noise.
    if type_name in _SIMPLE_ROLE and name.lower() in {"conv", "conv1", "conv2", "conv3", "fc", "head", "norm", "pool", "dropout", "embedding"}:
        return type_label
    if type_name == "MultiheadAttention" and name.lower() not in {"attention", "mha", "multihead_attention", "multi_head_attention"}:
        return friendly.replace("Mha", "MHA")
    if type_name not in _SIMPLE_ROLE and friendly and friendly.lower() != type_name.lower():
        return friendly
    return type_label or friendly or name


@dataclass
class ModuleInfo:
    name: str
    type_name: str
    role: str = "backbone"
    repeat: int = 1
    detail: str = ""
    is_container: bool = False
    item_type: str = ""


@dataclass
class ParseState:
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    node_ids: set[str] = field(default_factory=set)
    vars: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    resolved_control_flow: list[str] = field(default_factory=list)
    call_counter: dict[str, int] = field(default_factory=dict)

    def unique_id(self, base: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", base).strip("_").lower() or "node"
        if base not in self.node_ids:
            self.node_ids.add(base)
            return base
        i = self.call_counter.get(base, 2)
        while f"{base}_{i}" in self.node_ids:
            i += 1
        self.call_counter[base] = i + 1
        nid = f"{base}_{i}"
        self.node_ids.add(nid)
        return nid

    def add_node(self, base: str, label: str, role: str = "backbone", kind: str = "module", **extra: Any) -> str:
        nid = self.unique_id(base)
        node: dict[str, Any] = {"id": nid, "label": label, "role": role, "kind": kind}
        for k, v in extra.items():
            if v not in (None, "", [], 1, False):
                node[k] = v
            elif k == "repeat" and isinstance(v, int) and v > 1:
                node[k] = v
        self.nodes.append(node)
        return nid

    def add_edge(self, src: str | None, dst: str | None, edge_type: str = "main", label: str = "") -> None:
        if not src or not dst or src == dst:
            return
        edge: dict[str, Any] = {"from": src, "to": dst, "type": edge_type}
        if label:
            edge["label"] = label
        key = (src, dst, edge_type, label)
        if not any((e.get("from"), e.get("to"), e.get("type", "main"), e.get("label", "")) == key for e in self.edges):
            self.edges.append(edge)


class PytorchAstParser:
    def __init__(self, source: str, source_name: str = "model.py", class_name: str | None = None, detail: str = "system"):
        self.source = source
        self.source_name = source_name
        self.tree = ast.parse(source, filename=source_name)
        self.requested_class = class_name
        self.detail = detail
        self.state = ParseState()
        self.class_node = self._select_class()
        self.class_name = self.class_node.name
        self.init_defaults = self._collect_init_defaults()
        self.attr_defaults = self._collect_attr_defaults()
        self._collect_modules()

    def _select_class(self) -> ast.ClassDef:
        classes = [n for n in self.tree.body if isinstance(n, ast.ClassDef)]
        if self.requested_class:
            for c in classes:
                if c.name == self.requested_class:
                    return c
            raise ValueError(f"Class {self.requested_class!r} not found in {self.source_name}.")
        candidates = []
        for c in classes:
            bases = {_dotted(b).split(".")[-1] for b in c.bases}
            has_forward = any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == "forward" for x in c.body)
            if has_forward and ("Module" in bases or any("Module" in b for b in bases)):
                candidates.append(c)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            # Prefer the last top-level model class: helper blocks are commonly defined first.
            return candidates[-1]
        for c in classes:
            if any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == "forward" for x in c.body):
                return c
        raise ValueError("No class with a forward() method was found.")

    def _collect_init_defaults(self) -> dict[str, Any]:
        """Collect literal constructor defaults for conservative branch resolution."""
        fn = next((x for x in self.class_node.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)
        if fn is None:
            return {}
        out: dict[str, Any] = {}
        pos_args = [a.arg for a in fn.args.args]
        if fn.args.defaults:
            for name, default in zip(pos_args[-len(fn.args.defaults):], fn.args.defaults):
                if isinstance(default, ast.Constant):
                    out[name] = default.value
        for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            if isinstance(default, ast.Constant):
                out[arg.arg] = default.value
        return out

    def _collect_attr_defaults(self) -> dict[str, Any]:
        """Resolve assignments such as ``self.use_map = use_map`` when the argument has a literal default."""
        fn = next((x for x in self.class_node.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)
        out: dict[str, Any] = {}
        if fn is None:
            return out
        for stmt in ast.walk(fn):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            value = stmt.value
            for target in targets:
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
                    continue
                if isinstance(value, ast.Name) and value.id in self.init_defaults:
                    out[target.attr] = self.init_defaults[value.id]
                elif isinstance(value, ast.Constant):
                    out[target.attr] = value.value
        return out

    def _eval_bool_condition(self, expr: ast.AST) -> bool | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
            return bool(expr.value)
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
            value = self.attr_defaults.get(expr.attr)
            return value if isinstance(value, bool) else None
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            v = self._eval_bool_condition(expr.operand)
            return None if v is None else not v
        if isinstance(expr, ast.BoolOp):
            vals = [self._eval_bool_condition(v) for v in expr.values]
            if isinstance(expr.op, ast.And):
                if any(v is False for v in vals): return False
                if all(v is True for v in vals): return True
            if isinstance(expr.op, ast.Or):
                if any(v is True for v in vals): return True
                if all(v is False for v in vals): return False
        return None

    def _init_fn(self) -> ast.FunctionDef | None:
        return next((x for x in self.class_node.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)

    def _forward_fn(self) -> ast.FunctionDef:
        fn = next((x for x in self.class_node.body if isinstance(x, ast.FunctionDef) and x.name == "forward"), None)
        if fn is None:
            raise ValueError(f"Class {self.class_name!r} has no forward() method.")
        return fn

    def _collect_modules(self) -> None:
        init = self._init_fn()
        if not init:
            self.state.unresolved.append("No __init__() found; module construction metadata is unavailable.")
            return
        for stmt in ast.walk(init):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            value = stmt.value
            if value is None:
                continue
            for target in targets:
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
                    continue
                name = target.attr
                if isinstance(value, ast.Call):
                    type_name = _short_type(value)
                    role = _SIMPLE_ROLE.get(type_name, "backbone")
                    repeat = 1
                    is_container = type_name in {"Sequential", "ModuleList", "ModuleDict"}
                    item_type = ""
                    if is_container:
                        # A Sequential with several heterogeneous arguments is one logical block,
                        # not a repeated block. Only explicit list/list-comprehension construction
                        # carries repetition semantics.
                        if value.args:
                            first = value.args[0]
                            source = first.value if isinstance(first, ast.Starred) else first
                            if len(value.args) == 1 or isinstance(first, ast.Starred):
                                repeat = _infer_repeat(source) or 1
                                if isinstance(source, ast.ListComp) and isinstance(source.elt, ast.Call):
                                    item_type = _short_type(source.elt)
                                elif isinstance(source, (ast.List, ast.Tuple)) and source.elts and isinstance(source.elts[0], ast.Call):
                                    item_type = _short_type(source.elts[0])
                            else:
                                repeat = 1
                        elif value.keywords:
                            repeat = 1
                    detail = _safe_text(value)
                    self.state.modules[name] = ModuleInfo(name, type_name, role, repeat, detail, is_container, item_type)

    def _input_nodes(self, fn: ast.FunctionDef) -> None:
        args = [a.arg for a in fn.args.args if a.arg not in {"self"}]
        for arg in args:
            nid = self.state.add_node(f"input_{arg}", arg.replace("_", " ").title(), role="input", kind="data")
            self.state.vars[arg] = nid

    def _module_node(self, attr: str) -> str:
        info = self.state.modules.get(attr, ModuleInfo(attr, "", "backbone"))
        label = _human_module_label(attr, info.item_type or info.type_name)
        role = info.role
        if info.type_name in {"Sequential", "ModuleList"} and info.item_type:
            role = _SIMPLE_ROLE.get(info.item_type, "backbone")
        lname = attr.lower()
        if "encoder" in lname:
            role = "backbone"
        if lname.endswith("_head") or lname == "head":
            role = "head"
        if "gate" in lname:
            role = "fusion"
        extra: dict[str, Any] = {}
        if info.repeat > 1:
            extra["repeat"] = info.repeat
        if self.detail == "operation" and info.detail:
            extra["subtitle"] = info.detail
        elif info.type_name in {"Sequential", "ModuleList"} and info.item_type:
            extra["subtitle"] = info.item_type
        elif info.type_name == "MultiheadAttention" and label != "Multi-Head Attention":
            extra["subtitle"] = "Multi-head attention"
        return self.state.add_node(attr, label, role=role, kind="module", **extra)

    def _resolve_name(self, expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name):
            return self.state.vars.get(expr.id)
        if isinstance(expr, ast.Attribute):
            # tuple outputs such as out.hidden are approximated to the base producer.
            base = self._resolve_name(expr.value)
            return base
        if isinstance(expr, ast.Subscript):
            return self._resolve_name(expr.value)
        return None

    def _source_nodes(self, expr: ast.AST) -> list[str]:
        if isinstance(expr, ast.Name):
            n = self.state.vars.get(expr.id)
            return [n] if n else []
        if isinstance(expr, (ast.List, ast.Tuple)):
            out: list[str] = []
            for e in expr.elts:
                out.extend(self._source_nodes(e))
            return list(dict.fromkeys(out))
        if isinstance(expr, ast.Attribute):
            return self._source_nodes(expr.value)
        if isinstance(expr, ast.Subscript):
            return self._source_nodes(expr.value)
        if isinstance(expr, ast.Call):
            return self._source_nodes(expr.args[0]) if expr.args else []
        if isinstance(expr, ast.BinOp):
            return list(dict.fromkeys(self._source_nodes(expr.left) + self._source_nodes(expr.right)))
        return []

    def _producers(self, expr: ast.AST) -> list[str]:
        """Resolve producer nodes while evaluating nested architecture calls once."""
        if isinstance(expr, ast.Name):
            n = self.state.vars.get(expr.id)
            return [n] if n else []
        if isinstance(expr, (ast.List, ast.Tuple)):
            out: list[str] = []
            for item in expr.elts:
                out.extend(self._producers(item))
            return list(dict.fromkeys(out))
        if isinstance(expr, ast.Call):
            n = self._expr(expr)
            return [n] if n else []
        if isinstance(expr, ast.BinOp):
            n = self._expr(expr)
            return [n] if n else []
        if isinstance(expr, ast.Attribute):
            n = self._resolve_name(expr)
            return [n] if n else []
        if isinstance(expr, ast.Subscript):
            n = self._resolve_name(expr)
            return [n] if n else []
        return []

    def _call_expr(self, call: ast.Call) -> str | None:
        name = _call_name(call)
        short = name.split(".")[-1]
        # self.module(...)
        if name.startswith("self."):
            attr = name.split(".")[1] if len(name.split(".")) >= 2 else short
            nid = self._module_node(attr)
            info = self.state.modules.get(attr)
            # Preserve query/key/value semantics for attention. These edge labels are
            # much more informative in a paper figure than three anonymous arrows.
            if info and info.type_name == "MultiheadAttention" and len(call.args) >= 3:
                q = self._producers(call.args[0])
                k = self._producers(call.args[1])
                v = self._producers(call.args[2])
                for src in list(dict.fromkeys(q)):
                    self.state.add_edge(src, nid, "main", "Q")
                kset = list(dict.fromkeys(k)); vset = list(dict.fromkeys(v))
                if kset == vset:
                    for src in kset:
                        self.state.add_edge(src, nid, "main", "K,V")
                else:
                    for src in kset:
                        self.state.add_edge(src, nid, "main", "K")
                    for src in vset:
                        self.state.add_edge(src, nid, "main", "V")
                return nid
            sources: list[str] = []
            for a in call.args:
                sources.extend(self._producers(a))
            for src in list(dict.fromkeys(sources)):
                self.state.add_edge(src, nid, "main")
            return nid

        # torch.cat / concatenate
        if short in {"cat", "concat", "concatenate", "stack"}:
            label = "Concat" if short != "stack" else "Stack"
            nid = self.state.add_node(label.lower(), label, role="fusion", kind="merge")
            sources = self._producers(call.args[0]) if call.args else []
            for src in sources:
                self.state.add_edge(src, nid, "main")
            return nid

        # Functional operations: draw only at requested detail, otherwise preserve source producer.
        if short in _FUNCTION_LABELS:
            sources: list[str] = []
            for a in call.args:
                sources.extend(self._producers(a))
            if self.detail == "system" and short in {"relu", "gelu", "silu", "sigmoid", "tanh", "dropout", "flatten", "reshape", "view", "permute"}:
                return sources[0] if sources else None
            nid = self.state.add_node(short, _FUNCTION_LABELS[short], role="operator", kind="operator")
            for src in list(dict.fromkeys(sources)):
                self.state.add_edge(src, nid, "main")
            return nid

        # Method calls such as x.flatten(), head(x).squeeze(), tensor.expand().
        if isinstance(call.func, ast.Attribute):
            base_src = self._expr(call.func.value) or self._resolve_name(call.func.value)
            method = call.func.attr
            passthrough = {
                "squeeze", "unsqueeze", "expand", "expand_as", "transpose", "permute",
                "reshape", "view", "flatten", "contiguous", "detach", "clone", "to",
                "float", "double", "half", "type", "type_as", "repeat", "repeat_interleave",
            }
            if base_src and method in passthrough and self.detail == "system":
                return base_src
            if base_src and method in _FUNCTION_LABELS:
                if self.detail == "system" and method in {"flatten", "reshape", "view", "permute"}:
                    return base_src
                nid = self.state.add_node(method, _FUNCTION_LABELS[method], role="operator", kind="operator")
                self.state.add_edge(base_src, nid, "main")
                return nid
            if base_src and method in {"mean", "sum", "max"}:
                label = {"mean": "Mean Pool", "sum": "Sum", "max": "Max"}[method]
                nid = self.state.add_node(method, label, role="operator", kind="operator")
                self.state.add_edge(base_src, nid, "main")
                return nid
            if base_src and method in passthrough:
                nid = self.state.add_node(method, method.replace("_", " ").title(), role="operator", kind="operator")
                self.state.add_edge(base_src, nid, "main")
                return nid

        # Unknown pure function: at block/operation detail keep it, otherwise retain first input source.
        sources: list[str] = []
        for a in call.args:
            sources.extend(self._producers(a))
        sources = list(dict.fromkeys(sources))
        if self.detail in {"block", "operation"} and short and short not in {"range", "len"}:
            nid = self.state.add_node(short, short.replace("_", " ").title(), role="operator", kind="operator")
            for src in sources:
                self.state.add_edge(src, nid, "main")
            return nid
        return sources[0] if sources else None

    def _expr(self, expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name):
            return self.state.vars.get(expr.id)
        if isinstance(expr, ast.Call):
            return self._call_expr(expr)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Mult)):
            left = self._expr(expr.left) or self._resolve_name(expr.left)
            right = self._expr(expr.right) or self._resolve_name(expr.right)
            symbol = "+" if isinstance(expr.op, ast.Add) else "×"
            nid = self.state.add_node("add" if symbol == "+" else "multiply", symbol, role="fusion", kind="merge")
            # The primary stream is main; secondary branch is visually treated as residual/conditioning for add.
            if left:
                self.state.add_edge(left, nid, "main")
            if right:
                self.state.add_edge(right, nid, "residual" if symbol == "+" else "main")
            return nid
        if isinstance(expr, ast.Tuple):
            # Keep first producer for a tuple assignment unless separately unpacked.
            producers = [self._expr(e) for e in expr.elts]
            return next((p for p in producers if p), None)
        if isinstance(expr, ast.IfExp):
            a = self._expr(expr.body)
            b = self._expr(expr.orelse)
            if a and b and a != b:
                nid = self.state.add_node("conditional", "Conditional", role="fusion", kind="merge")
                self.state.add_edge(a, nid, "main")
                self.state.add_edge(b, nid, "auxiliary")
                return nid
            return a or b
        return self._resolve_name(expr)

    def _assign(self, target: ast.AST, value: ast.AST) -> None:
        producer = self._expr(value)
        if isinstance(target, ast.Name) and producer:
            self.state.vars[target.id] = producer
        elif isinstance(target, (ast.Tuple, ast.List)):
            names = [e.id for e in target.elts if isinstance(e, ast.Name)]
            if producer:
                for n in names:
                    self.state.vars[n] = producer
                if len(names) > 1:
                    self.state.unresolved.append(f"Tuple assignment {', '.join(names)} shares one inferred producer; inspect multi-output semantics if important.")

    def _loop(self, stmt: ast.For) -> None:
        # Common pattern: for layer in self.layers: x = layer(x)
        container = ""
        if isinstance(stmt.iter, ast.Attribute) and isinstance(stmt.iter.value, ast.Name) and stmt.iter.value.id == "self":
            container = stmt.iter.attr
        elif isinstance(stmt.iter, ast.Call) and _call_name(stmt.iter).split(".")[-1] == "enumerate" and stmt.iter.args:
            arg = stmt.iter.args[0]
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "self":
                container = arg.attr
        if container:
            info = self.state.modules.get(container)
            # Find assignments inside the loop that call the loop variable.
            loop_names: set[str] = set()
            if isinstance(stmt.target, ast.Name):
                loop_names.add(stmt.target.id)
            elif isinstance(stmt.target, (ast.Tuple, ast.List)):
                loop_names.update(e.id for e in stmt.target.elts if isinstance(e, ast.Name))
            handled = False
            for inner in stmt.body:
                if isinstance(inner, (ast.Assign, ast.AnnAssign)):
                    val = inner.value
                    if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in loop_names:
                        label_type = (info.item_type if info else "") or container.rstrip("s").title()
                        label = _human_module_label(container, label_type)
                        repeat = info.repeat if info else 1
                        nid = self.state.add_node(container, label, role=_SIMPLE_ROLE.get(label_type, "backbone"), kind="module", repeat=repeat, subtitle=label_type if info and info.item_type else "")
                        sources = []
                        for a in val.args:
                            sources.extend(self._producers(a))
                        for src in list(dict.fromkeys(sources)):
                            self.state.add_edge(src, nid, "main")
                        targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                        for t in targets:
                            if isinstance(t, ast.Name):
                                self.state.vars[t.id] = nid
                        handled = True
            if handled:
                return
        self.state.unresolved.append(f"Loop at line {getattr(stmt, 'lineno', '?')} was not fully normalized; inspect dynamic control flow if architecturally important.")
        for inner in stmt.body:
            self._stmt(inner)

    def _stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                self._assign(t, stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._assign(stmt.target, stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            # x += branch(x)
            left = self._resolve_name(stmt.target)
            right = self._expr(stmt.value)
            symbol = "+" if isinstance(stmt.op, ast.Add) else "×"
            nid = self.state.add_node("add" if symbol == "+" else "multiply", symbol, role="fusion", kind="merge")
            self.state.add_edge(left, nid, "main")
            self.state.add_edge(right, nid, "residual" if symbol == "+" else "main")
            if isinstance(stmt.target, ast.Name):
                self.state.vars[stmt.target.id] = nid
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._expr(stmt.value)
        elif isinstance(stmt, ast.For):
            self._loop(stmt)
        elif isinstance(stmt, ast.If):
            resolved = self._eval_bool_condition(stmt.test)
            if resolved is not None:
                chosen = stmt.body if resolved else stmt.orelse
                for x in chosen:
                    self._stmt(x)
                self.state.resolved_control_flow.append(
                    f"Conditional at line {stmt.lineno} resolved from constructor default to {'body' if resolved else 'else'} branch."
                )
            else:
                before = dict(self.state.vars)
                for x in stmt.body:
                    self._stmt(x)
                body = dict(self.state.vars)
                self.state.vars = dict(before)
                for x in stmt.orelse:
                    self._stmt(x)
                other = dict(self.state.vars)
                merged = dict(before)
                for var in set(body) | set(other):
                    a, b = body.get(var), other.get(var)
                    if a and b and a != b:
                        nid = self.state.add_node(f"conditional_{var}", "Conditional", role="fusion", kind="merge")
                        self.state.add_edge(a, nid, "main")
                        self.state.add_edge(b, nid, "auxiliary")
                        merged[var] = nid
                    else:
                        merged[var] = a or b or before.get(var)
                self.state.vars = merged
                self.state.unresolved.append(f"Conditional control flow at line {stmt.lineno} was summarized; runtime branch conditions were not evaluated.")
        elif isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Dict):
                for i, (key, val) in enumerate(zip(stmt.value.keys, stmt.value.values), start=1):
                    src = self._expr(val) or self._resolve_name(val)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        raw = key.value
                        label = raw.replace("_", " ").title()
                        base = f"output_{raw}"
                    else:
                        label = f"Output {i}"
                        base = f"output_{i}"
                    out_id = self.state.add_node(base, label, role="output", kind="output")
                    self.state.add_edge(src, out_id, "main")
                return
            vals: list[ast.AST]
            if isinstance(stmt.value, (ast.Tuple, ast.List)):
                vals = list(stmt.value.elts)
            elif stmt.value is not None:
                vals = [stmt.value]
            else:
                vals = []
            for i, val in enumerate(vals, start=1):
                src = self._expr(val) or self._resolve_name(val)
                label = "Output" if len(vals) == 1 else f"Output {i}"
                out_id = self.state.add_node(f"output_{i}", label, role="output", kind="output")
                self.state.add_edge(src, out_id, "main")
        elif isinstance(stmt, (ast.While, ast.Try, ast.With, ast.Match)):
            self.state.unresolved.append(f"Dynamic/compound statement {type(stmt).__name__} at line {getattr(stmt, 'lineno', '?')} requires manual review if it changes model topology.")
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.stmt):
                    self._stmt(child)

    def parse(self) -> dict[str, Any]:
        fn = self._forward_fn()
        self._input_nodes(fn)
        for stmt in fn.body:
            self._stmt(stmt)

        # Remove unused input nodes only when no edge leaves them; keep them because user-visible model inputs matter.
        # Avoid empty output if return producer was not recoverable.
        if not any(n.get("role") == "output" for n in self.state.nodes):
            self.state.unresolved.append("No explicit return value could be normalized into an output node.")

        return {
            "version": "1.0",
            "figure": {
                "title": f"{self.class_name} architecture",
                "direction": "LR",
                "target": "paper",
                "theme": "paper-light",
                "font": "Arial",
                "width_mm": 178,
            },
            "metadata": {
                "source_framework": "pytorch",
                "source_file": self.source_name,
                "source_class": self.class_name,
                "parser": "pytorch-ast",
                "parser_detail": self.detail,
                "unresolved": list(dict.fromkeys(self.state.unresolved)),
                "resolved_control_flow": list(dict.fromkeys(self.state.resolved_control_flow)),
                "constructor_defaults": dict(self.init_defaults),
            },
            "nodes": self.state.nodes,
            "edges": self.state.edges,
            "style": {
                "compact": self.detail == "system",
                "show_shapes": True,
                "show_repeat_badges": True,
                "visual_level": "publication-rich",
            },
        }


def parse_pytorch_source(source: str, *, source_name: str = "model.py", class_name: str | None = None, detail: str = "system") -> dict[str, Any]:
    if detail not in {"system", "block", "operation"}:
        raise ValueError("detail must be one of: system, block, operation")
    return PytorchAstParser(source, source_name=source_name, class_name=class_name, detail=detail).parse()


def parse_pytorch_file(path: str | Path, *, class_name: str | None = None, detail: str = "system") -> dict[str, Any]:
    path = Path(path)
    return parse_pytorch_source(path.read_text(encoding="utf-8"), source_name=str(path), class_name=class_name, detail=detail)


def dump_spec(spec: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output
