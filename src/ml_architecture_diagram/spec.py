from __future__ import annotations

from pathlib import Path
import json
import yaml


def load_spec(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Architecture spec must be a mapping/object.")
    return data


def schema_path() -> Path:
    # Development checkout: prefer the human-visible schema in templates/.
    root = Path(__file__).resolve().parents[2]
    candidate = root / "templates" / "architecture.schema.json"
    if candidate.exists():
        return candidate
    # Installed package: use bundled resource schema.
    bundled = Path(__file__).resolve().parent / "resources" / "architecture.schema.json"
    return bundled


def validate_semantics(spec: dict) -> list[str]:
    errors: list[str] = []
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    ids = [n.get("id") for n in nodes]
    if len(ids) != len(set(ids)):
        errors.append("Node IDs must be unique.")
    known = set(ids)
    for i, e in enumerate(edges):
        if e.get("from") not in known:
            errors.append(f"Edge {i}: unknown source node {e.get('from')!r}.")
        if e.get("to") not in known:
            errors.append(f"Edge {i}: unknown target node {e.get('to')!r}.")
    panels = spec.get("panels", []) or []
    panel_ids = {p.get("id") for p in panels}
    if panels:
        for n in nodes:
            p = n.get("panel")
            if p and p not in panel_ids:
                errors.append(f"Node {n.get('id')!r}: unknown panel {p!r}.")
    return errors


def validate_spec(spec: dict, use_jsonschema: bool = True) -> list[str]:
    errors = validate_semantics(spec)
    if use_jsonschema:
        try:
            from jsonschema import Draft202012Validator
            sp = schema_path()
            if sp.exists():
                schema = json.loads(sp.read_text(encoding="utf-8"))
                validator = Draft202012Validator(schema)
                for err in sorted(validator.iter_errors(spec), key=lambda e: list(e.path)):
                    loc = ".".join(str(x) for x in err.path)
                    errors.append(f"Schema {loc or '<root>'}: {err.message}")
        except ImportError:
            pass
    return errors
