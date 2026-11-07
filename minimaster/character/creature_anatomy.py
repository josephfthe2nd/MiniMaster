"""What an appendage does to the body that carries it.

A wing bolted onto an unchanged human back reads as a costume, because in real
animals a limb and its host are one system: the limb is powered by muscles
anchored on the trunk, and those muscles reshape the trunk. This module encodes
those consequences as adjustments to the *existing* morph sliders, so adding
wings broadens the chest and back the way it would in an animal that flies.

The mappings below follow comparative anatomy:

* **Wings are modified forelimbs.** In both birds and bats the downstroke is
  driven by the pectoralis anchored on an enlarged sternum/keel, with the
  latissimus and scapular muscles anchoring the recovery stroke. Pectoralis
  major is roughly 8-25 % of body mass in birds (mean ~15.5 %); the ventral
  thoracic flight muscles of bats average ~9.1 %. That mass has to live
  somewhere: chest circumference, pectoral and dorsi bulk, shoulder breadth.
* **A tail continues the spine.** Caudal vertebrae begin immediately after the
  sacrum, so a heavy tail needs a broad sacrum and pelvis and strong caudal /
  gluteal musculature — the same package bipedal hoppers and theropods use to
  make a tail work as a counterbalance.
* **Horns are bone, not decoration.** A horn is a cornual process of the
  frontal bone under a keratin sheath, permanently attached. Carrying one
  loads the neck, and the load shows up as neck girth and a heavier brow.
* **An extra arm pair needs a second girdle**, the same reasoning as the
  primary one: broader shoulders, thicker traps and lats to hang it from.

Nothing here invents new geometry — it only turns sliders the character
already has, so the result stays printable and still responds to every other
control.

THE DERIVED LAYER. The body a character is printed with is NOT the body it was saved
with. A document stores the user's own slider values, the list of appendages and two
numbers (the switch and the cap of this module); :func:`effective_values` recomputes the
adapted body from scratch every time -- ``user values + the consequences of the ENABLED
appendages``, clamped on the sum -- and never writes it back. So saving and loading
changes nothing, applying twice is the same as applying once, and removing an appendage
gives the original body back exactly. :func:`apply_to` (which adds into a Character in
place and cannot be undone) is kept for the older tools and demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Reference figures used by the flight report, from comparative-flight
# literature. They are cited so the numbers can be checked rather than trusted.
BIRD_PECTORALIS_FRACTION = 0.155   # mean of the 8-25% range in birds
BAT_FLIGHT_MUSCLE_FRACTION = 0.091
MAX_SOARING_MASS_KG = 41.0         # scaling limit from albatross/petrel data
MAX_SOARING_SPAN_M = 5.1
MAX_POWERED_BAT_MASS_KG = 2.2      # upper end of the 1.1-2.2 kg estimate
BODY_DENSITY_G_CM3 = 1.01          # human body is a touch denser than water

# A MakeHuman unit is one decimetre: the neutral base measures 16.66 units tall
# (1.67 m) and encloses 54.9 units^3, which at 1.01 kg/L is 55 kg. So a mesh
# volume in units^3 is already litres.
MH_UNIT_M = 0.1


@dataclass
class AnatomyEffect:
    """Body changes one appendage imposes, per unit of 'load'."""

    sliders: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    reason: str = ""

    def scaled(self, load: float) -> tuple[dict, dict]:
        return ({k: v * load for k, v in self.sliders.items()},
                {k: v * load for k, v in self.macro.items()})


# load 1.0 == an appendage of "typical" size for its kind
CONSEQUENCES: dict[str, AnatomyEffect] = {
    "wing": AnatomyEffect(
        sliders={
            # the flight-muscle mass has to be carried on the trunk
            "measure.measure_bust_circ": 0.75,
            "torso.torso_muscle_pectoral": 0.85,
            "torso.torso_muscle_dorsi": 0.70,
            "measure.measure_shoulder_dist": 0.55,
            "armslegs.upperarm_shoulder_muscle": 0.40,
            "neck.neck_scale_horiz": 0.30,
            # flyers are light: the abdomen and legs pay for the chest
            "stomach.stomach_tone": 0.45,
            "measure.measure_waist_circ": -0.35,
            "measure.measure_thigh_circ": -0.30,
        },
        macro={"weight": -0.10},
        reason="pectoralis drives the downstroke off an enlarged sternum; "
               "latissimus and scapular muscles anchor the recovery stroke",
    ),
    "tail": AnatomyEffect(
        sliders={
            "hip.hip_scale_horiz": 0.45,
            "hip.hip_scale_depth": 0.35,
            "buttocks.buttocks_volume": 0.55,
            "torso.torso_muscle_dorsi": 0.30,
            "measure.measure_hips_circ": 0.35,
        },
        reason="caudal vertebrae continue the sacrum, so the pelvis broadens "
               "and the caudal/gluteal muscles that swing the tail thicken",
    ),
    "horn": AnatomyEffect(
        sliders={
            "neck.neck_scale_horiz": 0.55,
            "neck.neck_scale_depth": 0.40,
            "forehead.forehead_nubian": 0.35,
            "forehead.forehead_temple": 0.30,
        },
        reason="a horn is a cornual process of the frontal bone; carrying it "
               "loads the neck and thickens the brow it grows from",
    ),
    "harvest": AnatomyEffect(   # an extra limb pair
        sliders={
            "measure.measure_shoulder_dist": 0.60,
            "torso.torso_muscle_dorsi": 0.55,
            "torso.torso_muscle_pectoral": 0.40,
            "measure.measure_bust_circ": 0.45,
            "neck.neck_scale_horiz": 0.25,
        },
        reason="a second limb pair needs a second girdle to hang from: "
               "broader shoulders, thicker traps and lats",
    ),
}


# Kinds whose *typical* installation is a symmetric pair. The coefficients
# above are written for the typical case, so a pair of wings scores load 1.0
# and a lone wing scores less -- not the other way round. Getting this backwards
# saturates every chest slider on a default pair and leaves a 20 m wingspan
# looking exactly like a 9 m one.
PAIRED_KINDS = frozenset({"wing", "horn", "harvest"})
PAIR_FACTOR = 1.45      # a pair costs more than one, but not double


def _registered_kind(app):
    """The registry :class:`~minimaster.character.kinds.Kind` of a document record
    (an ``AppendageSpec``), or ``None`` for a legacy ``Appendage`` or an unknown kind.
    The registry owns ``load()`` and ``paired`` for the records it knows; the tables in
    this module keep serving everything else."""
    from .appendage_spec import AppendageSpec
    if not isinstance(app, AppendageSpec):
        return None
    from . import kinds
    return kinds.KINDS.get(app.kind)


def appendage_load(app) -> float:
    """How much body adaptation this appendage demands.

    Scaled from the appendage's own size so a small decorative horn costs
    almost nothing and a nine-metre wing costs a lot.
    """
    spec_kind = _registered_kind(app)
    if spec_kind is not None:
        # a document record: its kind says how big it is and whether a pair is typical,
        # and only the placement keys the kind honours scale it (a harvested limb ignores
        # ``scale``, so it must not cost anatomy either)
        try:
            values, _ = spec_kind.clean(app.params or {})
        except ValueError:
            return 0.0                  # cannot be built (unknown style): no anatomy either
        load = spec_kind.load(values)
        if "scale" in spec_kind.placement_keys:
            load *= float(app.scale)
        if app.mirror and not spec_kind.paired:
            load *= PAIR_FACTOR
        elif spec_kind.paired and not app.mirror:
            load /= PAIR_FACTOR
        return max(0.0, load)
    from .wings import DEFAULT_SPAN
    p = app.params or {}
    kind = app.kind
    if kind == "wing":
        # read the builder's own default so the two can never drift apart
        base = p.get("span", DEFAULT_SPAN) / DEFAULT_SPAN
    elif kind == "tail":
        base = p.get("length", 6.0) / 6.0
    elif kind == "horn":
        base = p.get("length", 2.2) / 2.2
    else:
        base = 1.0
    load = base * float(app.scale)
    paired = kind in PAIRED_KINDS
    if app.mirror and not paired:
        load *= PAIR_FACTOR         # two tails demand more than one
    elif paired and not app.mirror:
        load /= PAIR_FACTOR         # a single wing is a lighter commitment
    return max(0.0, load)


def consequences(appendages, cap: float = 1.0):
    """Total slider/macro adjustments for a set of appendages.

    Returns ``(sliders, macro, notes)``. Values are clamped so that stacking
    several appendages cannot drive a slider past its usable range.
    """
    sliders: dict[str, float] = {}
    macro: dict[str, float] = {}
    notes: list[str] = []
    for app in appendages:
        if getattr(app, "enabled", True) is False:       # a switched-off record costs nothing
            continue
        eff = CONSEQUENCES.get(app.kind)
        if eff is None:
            continue
        load = appendage_load(app)
        if load <= 0:
            continue
        s, m = eff.scaled(load)
        for k, v in s.items():
            sliders[k] = sliders.get(k, 0.0) + v
        for k, v in m.items():
            macro[k] = macro.get(k, 0.0) + v
        notes.append(f"{app.kind} (load {load:.2f}): {eff.reason}")
    sliders = {k: max(-cap, min(cap, v)) for k, v in sliders.items()}
    return sliders, macro, notes


def anatomy_delta(appendages, catalog=None, cap: float = 1.0):
    """What the (enabled) appendages ask of the body, WITHOUT touching anything:
    ``(delta, notes)`` with ``delta = {"sliders": {...}, "macro": {...}}`` (amounts to
    add; slider amounts are capped to ``+-cap`` on the sum of all appendages, and unknown
    slider keys are left out when a ``catalog`` is given)."""
    sliders, macro, notes = consequences(appendages, cap=cap)
    if catalog is not None:
        sliders = {k: v for k, v in sliders.items() if k in catalog}
    return {"sliders": sliders, "macro": macro}, notes


def effective_values(user_sliders, user_macro, appendages, catalog=None, cap: float = 1.0):
    """``(sliders, macro, notes)`` of the body the appendages call for: the user's values
    plus :func:`anatomy_delta`, clamped on the sum (sliders to -1..1, macros to 0..1).

    A pure function of its arguments: the inputs are not modified, so the result can be
    recomputed whenever a record is added, removed, switched off or edited, and it is
    never saved. It gives the same numbers :func:`apply_to` would, minus the damage."""
    delta, notes = anatomy_delta(appendages, catalog=catalog, cap=cap)
    sliders = dict(user_sliders)
    for key, amount in delta["sliders"].items():
        sliders[key] = max(-1.0, min(1.0, float(user_sliders.get(key, 0.0)) + amount))
    macro = dict(user_macro)
    for key, amount in delta["macro"].items():
        if key in macro:
            macro[key] = max(0.0, min(1.0, float(user_macro[key]) + amount))
    return sliders, macro, notes


def apply_to(character, appendages, catalog=None, cap: float = 1.0):
    """Fold appendage consequences into a Character, in place.

    Existing user slider values are ADDED to, not replaced, so a deliberate
    setting is never silently overwritten. Unknown slider keys are dropped
    (the asset set may differ), which is why a catalog can be passed.

    This BAKES the adaptation into the character: a second call adds it again, and nothing
    undoes it. It is kept for the older tools; a character document uses the derived
    layer instead (:func:`effective_values`, ``Character.effective``).
    """
    sliders, macro, notes = consequences(appendages, cap=cap)
    for key, val in sliders.items():
        if catalog is not None and key not in catalog:
            continue
        character.sliders[key] = max(
            -1.0, min(1.0, character.sliders.get(key, 0.0) + val))
    for key, val in macro.items():
        if key in character.macro:
            character.macro[key] = max(
                0.0, min(1.0, character.macro[key] + val))
    return notes


# --------------------------------------------------------------------------
# the honest physics


@dataclass
class FlightReport:
    body_mass_kg: float
    wingspan_m: float
    required_muscle_kg: float
    can_soar: bool
    can_power_fly: bool
    verdict: str
    notes: list

    @property
    def wing_loading_kg_m2(self) -> float:
        """Mass carried per square metre of wing, the standard flight metric.

        Wing area is estimated as a rectangle of span x (span/6), the rough
        aspect ratio of a soaring bird; birds that fly sit near 1-20 kg/m2.
        """
        area = self.wingspan_m * (self.wingspan_m / 6.0)
        return self.body_mass_kg / area if area > 0 else float("inf")

    def as_text(self) -> str:
        lines = [f"body mass      ~{self.body_mass_kg:.1f} kg (from mesh volume)",
                 f"wingspan       {self.wingspan_m:.2f} m",
                 f"flight muscle  ~{self.required_muscle_kg:.1f} kg needed "
                 f"({BIRD_PECTORALIS_FRACTION * 100:.0f}% of mass, avian mean)",
                 f"wing loading   ~{self.wing_loading_kg_m2:.0f} kg/m2 "
                 f"(flying birds sit at 1-20)",
                 f"verdict        {self.verdict}"]
        lines += [f"note           {n}" for n in self.notes]
        return "\n".join(lines)


def wingspan_of(meshes) -> float:
    """Widest extent of a set of wing shells, whichever axis they span.

    Wings are swept and drooped, so the span is not simply the ``span``
    parameter: measure the built geometry instead of trusting the input.
    """
    import numpy as np
    pts = [m.vertices for m in meshes]
    if not pts:
        return 0.0
    v = np.vstack(pts)
    return float(np.max(v.max(axis=0) - v.min(axis=0)))


def flight_report(body_volume: float, wingspan: float,
                  unit_m: float = MH_UNIT_M) -> FlightReport:
    """Could this creature actually fly? Almost certainly not — say so, with
    numbers, rather than pretending.

    ``body_volume`` and ``wingspan`` are both in *mesh units*, so they can be
    handed straight from ``mesh.volume()`` and a wing's bounding box. The mass
    estimate is therefore the character's own, and tracks every morph.
    """
    volume_l = body_volume * (unit_m ** 3) * 1000.0     # m^3 -> litres
    mass = volume_l * BODY_DENSITY_G_CM3                # 1 g/cm3 == 1 kg/L
    span = wingspan * unit_m
    needed = mass * BIRD_PECTORALIS_FRACTION
    can_soar = mass <= MAX_SOARING_MASS_KG and span <= MAX_SOARING_SPAN_M
    can_power = mass <= MAX_POWERED_BAT_MASS_KG
    notes = []
    if not can_power:
        notes.append(
            f"powered flight tops out around {MAX_POWERED_BAT_MASS_KG:.1f} kg "
            "in bats (wingbeat frequency required overtakes what is attainable)")
    if not can_soar:
        notes.append(
            f"even soaring scales out at ~{MAX_SOARING_MASS_KG:.0f} kg and "
            f"~{MAX_SOARING_SPAN_M:.1f} m span (albatross/petrel scaling)")
    if mass > MAX_SOARING_MASS_KG:
        ratio = mass / MAX_SOARING_MASS_KG
        notes.append(f"this body is {ratio:.1f}x the soaring mass limit")
    if can_power:
        verdict = "powered flight is plausible at this mass"
    elif can_soar:
        verdict = "soaring only, and marginally"
    else:
        verdict = "fantasy flight: no real anatomy supports this"
    return FlightReport(mass, span, needed, can_soar, can_power, verdict, notes)
