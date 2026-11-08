"""Attenuate a shared negative contribution in a newly inspected hard contradiction."""
def edit(theta, context):
    theta['W'][2,1652] *= 0.5
    return theta
