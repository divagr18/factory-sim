"""The seed builder program: `expert.Builder`, rewritten against `World` alone.

It is the ancestor every evolved program descends from, so it has to be both
correct and legible to the model that edits it. It follows the scripted
expert step for step -- the canonical layout first, the same walk, the same
two placements and the same fuel -- but reads only what `World` publishes, and
stays inside the sandbox's language: no imports (`math` is a global), no
lambda, no method calls other than `world.*` and `math.*`.

One difference is forced by the observation. The expert reads every ore tile
on the map; the grid reports ore within 12 tiles of the character, and the
start can be 13 from the patch centre. So the program re-plans the layout
from what it sees before every step of the walk. It settles on the expert's
layout as soon as the patch is in view, which is before the walk ends.
"""

SOURCE = '''
def build(world):
    """Walk beside the ore patch, place a drill over ore and a furnace under its
    drop, and fuel both with coal."""
    radius = 5
    tolerance = 0.3
    stride = 38 / 256
    facings = ("N", "E", "S", "W")
    index = {"N": 0, "E": 1, "S": 2, "W": 3}

    def turn(offset, quarters):
        x = offset[0]
        y = offset[1]
        for _ in range(quarters % 4):
            x, y = -y, x
        return (x, y)

    def turn_facing(facing, quarters):
        return facings[(index[facing] + quarters) % 4]

    def square(anchor):
        ax = anchor[0]
        ay = anchor[1]
        return [(ax, ay), (ax + 1, ay), (ax, ay + 1), (ax + 1, ay + 1)]

    def plan():
        ore = set(world.ore_tiles("iron-ore"))
        taken = set(world.blocked_tiles())
        for e in world.entities():
            if e.kind != "item-entity":
                taken = taken | {(math.floor(e.x), math.floor(e.y))}
        patch = world.patch()
        if patch is None:
            here = world.me()
            patch = (here[0], here[1])
        canonical = (math.floor(patch[0]), math.floor(patch[1]))
        anchors = []
        if set(square(canonical)) <= ore:
            anchors = [canonical]
        for a in sorted(ore):
            if a != canonical and set(square(a)) <= ore:
                anchors = anchors + [a]
        for anchor in anchors:
            centre = (anchor[0] + 1, anchor[1] + 1)
            for quarters in range(4):
                s = turn((3.5, 0.5), quarters)
                stand = (math.floor(centre[0] + s[0]), math.floor(centre[1] + s[1]))
                f = turn((0.0, 2.0), quarters)
                furnace = (round(centre[0] + f[0]) - 1, round(centre[1] + f[1]) - 1)
                if any(t in taken for t in square(furnace) + [stand]):
                    continue
                far = False
                for t in (anchor, furnace):
                    if abs(t[0] - stand[0]) > radius or abs(t[1] - stand[1]) > radius:
                        far = True
                if not far:
                    return (anchor, quarters)
        return (canonical, 0)

    def layout(anchor, quarters):
        centre = (anchor[0] + 1, anchor[1] + 1)
        f = turn((0.0, 2.0), quarters)
        furnace_centre = (centre[0] + f[0], centre[1] + f[1])
        furnace = (round(furnace_centre[0]) - 1, round(furnace_centre[1]) - 1)
        s = turn((3.5, 0.5), quarters)
        stand = (centre[0] + s[0], centre[1] + s[1])
        return centre, furnace_centre, furnace, stand

    def step_towards(goal):
        """One walking decision towards `goal`; False once within tolerance."""
        here = world.me()
        dx = goal[0] - here[0]
        dy = goal[1] - here[1]
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return False
        if abs(dx) >= abs(dy):
            distance = abs(dx)
            direction = "E" if dx > 0 else "W"
        else:
            distance = abs(dy)
            direction = "S" if dy > 0 else "N"
        if distance >= 30 * stride:
            world.move(direction, "long")
        elif distance >= 7 * stride:
            world.move(direction, "step")
        else:
            world.move(direction, "nudge")
        return True

    def nearest(kind, point):
        best = None
        best_d = 0.0
        for e in world.entities():
            if e.kind == kind:
                d = math.hypot(e.x - point[0], e.y - point[1])
                if best is None or d < best_d:
                    best = e
                    best_d = d
        return best

    anchor, quarters = plan()
    while step_towards(layout(anchor, quarters)[3]):
        anchor, quarters = plan()

    centre, furnace_centre, furnace, stand = layout(anchor, quarters)
    world.place("burner-mining-drill", anchor[0], anchor[1], turn_facing("S", quarters))
    world.place("stone-furnace", furnace[0], furnace[1], turn_facing("N", quarters))
    drill = nearest("mining-drill", centre)
    if drill is not None:
        world.give(drill, "coal", 20)
    oven = nearest("furnace", furnace_centre)
    if oven is not None:
        world.give(oven, "coal", 20)
'''
