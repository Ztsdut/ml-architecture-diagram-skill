# Adaptive Layout and Obstacle-Aware Routing

## Goal

A paper architecture figure is not valid merely because every node has coordinates. The composition must remain readable after the real model changes. Node geometry is therefore computed before stage geometry, and edge routes are computed only after both are final.

## Layout order

1. Measure each node from label, subtitle, repeat badge, visual primitive, and scientific illustration.
2. Infer local stage topology (stack, layered branch, fan-in, fan-out, or serial chain).
3. Compute minimum stage width/height.
4. Place stages and nodes without overlap.
5. Compute stage/group containers.
6. Route connectors around final obstacles.
7. Expand the canvas if a legal route needs additional space.
8. Run publication geometry linting before export.

Never begin with fixed stage coordinates and then squeeze content into them.

## Routing rules

- Main inter-stage flow normally enters from left and exits right.
- Same-stage edges use local geometry. A source clearly above its target should normally use bottom-to-top ports.
- Conditioning, training, and auxiliary connections treat unrelated stages as soft obstacles and prefer gutters or outer buses.
- Connectors may cross stage borders only when entering/leaving a source or target stage.
- Connectors must not cross unrelated node interiors.
- Exact collinear reuse by unrelated edges should be avoided because it makes two semantic paths look like one.
- A small number of line crossings can be preferable to an extreme detour. Crossings are a soft metric; node/stage intersections are hard errors.

## High fan-in

When three or more paper-facing edges enter the same fusion/merge node, render them as a shared fan-in bus when this improves readability:

- all original edges remain in Architecture IR;
- source branches meet a shared bus;
- the bus has one final arrow into the target;
- no synthetic computation node is added.

The bus orientation is selected from source geometry (top, bottom, left, or right).

## Lint requirements

`lint-publication` must fail on:

- `node-overlap`
- `stage-overlap`
- `edge-through-node`
- `edge-through-stage`

`edge-crossings` is a softer warning/note. For dense graphs, first improve node ordering and fan-in abstraction before enlarging the canvas excessively.

## Renderer parity

SVG and PowerPoint should use the same route geometry. In PPTX, a routed polyline is represented as multiple native straight connectors so the path remains editable. draw.io stores route waypoints where possible.
