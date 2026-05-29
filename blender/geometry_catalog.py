"""
Geometry catalog: procedural fallback library for CGI dispatch.

PRIMARY path: CLIP search → Objaverse GLB → Blender import (asset_search.py)
FALLBACK path: CLIP text similarity → geometry_function here → procedural mesh

Each entry has a rich semantic description used for CLIP cosine similarity.
The story planner also reads this catalog to know what procedural shapes exist.

To add a new procedural fallback:
  1. Write make_<name>(obj_def, suffix) in insert_cgi.py
  2. Add an entry here with a rich natural-language description
  3. No keywords, no hardcoding — CLIP embedding handles all future queries.
"""

CATALOG: dict[str, dict] = {
    "make_kitsune_fox": {
        "description": (
            "A four-legged animal figure with a pointed snout, two upright pointed ears, "
            "an oval body, and a large bushy tail curving upward. Rendered as a glowing "
            "wireframe skeleton for a holographic or spirit-like appearance. "
            "Best for: fox, kitsune, wolf, dog, cat, bear, tanuki, raccoon, deer, rabbit, "
            "badger, any quadruped mammal, spirit animal, mythical creature, yokai."
        ),
        "placement": "ground",
        "motion_style": "walks, dances, leaps, prowls along the ground",
    },
    "make_lantern_drone": {
        "description": (
            "A floating spherical glowing orb with an equatorial ring and a hanging chain. "
            "Hovers and drifts smoothly through the air with gentle sine oscillation. "
            "Best for: lantern, orb, drone, UFO, will-o-wisp, spirit ball, floating light, "
            "magic sphere, firefly, aerial object, ghost light, fairy lantern, hovering device."
        ),
        "placement": "air",
        "motion_style": "hovers, drifts, floats, orbits above ground level",
    },
    "make_robot": {
        "description": (
            "A bipedal humanoid figure with a spherical body, two eyes, two arm cylinders, "
            "an antenna with a glowing tip, and an emissive metallic material. "
            "Best for: robot, android, mech, cyborg, automaton, machine, droid, "
            "service robot, guard, humanoid, AI figure."
        ),
        "placement": "ground",
        "motion_style": "walks, rotates, scans, waves",
    },
    "make_hologram_panel": {
        "description": (
            "A flat rectangular glowing panel with horizontal scan lines and a bright border frame. "
            "Looks like a floating holographic screen or billboard. "
            "Best for: holographic display, billboard, floating screen, digital sign, "
            "information panel, projection screen, advertisement board, data display."
        ),
        "placement": "any",
        "motion_style": "drifts slowly, rotates to face camera, pulses",
    },
}


def catalog_for_prompt() -> str:
    """Return a compact human-readable catalog for injection into LLM prompts."""
    lines = ["Available CGI geometry functions (pick the best match for each character):"]
    for fn, info in CATALOG.items():
        lines.append(f'\n  "{fn}":')
        lines.append(f'    {info["description"]}')
        lines.append(f'    Placement: {info["placement"]} | Motion: {info["motion_style"]}')
    return "\n".join(lines)
