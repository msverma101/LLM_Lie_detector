"""One incremental refinement of the sparse contribution supported in attempts 2/3."""
def edit(theta, context):
    theta['W'][3:5,169] *= 1.5
    return theta
