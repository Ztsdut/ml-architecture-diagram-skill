from __future__ import annotations

THEMES = {
    "publication": {
        "background": "#FFFFFF",
        "input_text": "#2D69A4",
        "backbone_text": "#28796F",
        "novel_text": "#C66A12",
        "output_text": "#4F7E43",
        "text": "#1F2933",
        "stroke": "#2F3B45",
        "edge": "#36454F",
        "input": "#DCEAF7",
        "preprocess": "#F1F3F5",
        "representation": "#DCEFEA",
        "backbone": "#DCEFEA",
        "novel": "#F3D7A7",
        "auxiliary": "#E7DDF2",
        "operator": "#EEF1F3",
        "fusion": "#DDEBD8",
        "head": "#DDEBD8",
        "output": "#E8F0E4",
        "training": "#F2E0E5",
        "neutral": "#F6F7F8",
        "panel": "#A9B2BA",
    },
    "paper-light": {
        "background": "#FFFFFF",
        "input_text": "#2B67A0",
        "backbone_text": "#2B7C73",
        "novel_text": "#C86A12",
        "output_text": "#4E7D45",
        "text": "#263238",
        "stroke": "#455A64",
        "edge": "#546E7A",
        "input": "#E3F2FD",
        "preprocess": "#ECEFF1",
        "representation": "#D9F3F0",
        "backbone": "#D9F3F0",
        "novel": "#FDE8C8",
        "auxiliary": "#EDE2F7",
        "operator": "#ECEFF1",
        "fusion": "#E1F1DB",
        "head": "#E1F1DB",
        "output": "#E8F5E9",
        "training": "#FCE4EC",
        "neutral": "#F5F5F5",
        "panel": "#B0BEC5",
    },
    "grayscale": {
        "background": "#FFFFFF",
        "input_text": "#424242",
        "backbone_text": "#424242",
        "novel_text": "#424242",
        "output_text": "#424242",
        "text": "#212121",
        "stroke": "#424242",
        "edge": "#616161",
        "input": "#F5F5F5",
        "preprocess": "#EEEEEE",
        "representation": "#E0E0E0",
        "backbone": "#E0E0E0",
        "novel": "#D6D6D6",
        "auxiliary": "#EEEEEE",
        "operator": "#F5F5F5",
        "fusion": "#E8E8E8",
        "head": "#E8E8E8",
        "output": "#F5F5F5",
        "training": "#E0E0E0",
        "neutral": "#F5F5F5",
        "panel": "#9E9E9E",
    },
}


def get_theme(name: str) -> dict[str, str]:
    return THEMES.get(name, THEMES["paper-light"]).copy()


def theme_for_spec(spec: dict) -> dict[str, str]:
    fig = spec.get("figure", {})
    theme = get_theme(fig.get("theme", "paper-light"))
    overrides = (spec.get("style", {}) or {}).get("colors", {}) or {}
    for key, value in overrides.items():
        if isinstance(value, str):
            theme[key] = value
    return theme
