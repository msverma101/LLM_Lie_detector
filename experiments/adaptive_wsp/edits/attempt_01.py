"""Attenuate the earliest probe's overconfident disagreement; no fitting."""
def edit(theta, context):
    theta['W'][0] *= 0.5
    theta['b'][0] *= 0.5
    return theta
