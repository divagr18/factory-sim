"""The seed that knows nothing: a program that waits until the build phase ends.

It is the program-search counterpart of PPO from scratch. The builder seed
hands the search the scripted expert's whole layout rule, and PPO's recipe
hands the policy that same expert's demonstrations; starting here instead
takes that away, so whatever the search reaches is the model's own doing plus
whatever the game notes in the prompt say. It scores zero everywhere, which
is the point: every family is a failure the model is shown.
"""

SOURCE = '''def build(world):
    """Build nothing: wait until the build phase ends."""
    while world.decisions_left() > 0:
        world.wait()
'''
