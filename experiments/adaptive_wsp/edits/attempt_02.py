"""Sparse strengthening of an observed contrastive late-probe contribution."""
def edit(theta, context):
    theta['W'][3:5,169] *= 2.0
    return theta
