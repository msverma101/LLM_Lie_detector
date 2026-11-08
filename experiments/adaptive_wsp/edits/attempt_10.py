"""Moderate a second saturated probe while retaining its learned direction."""
def edit(theta, context):
    theta['W'][1] *= 0.75
    theta['b'][1] *= 0.75
    return theta
