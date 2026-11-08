"""Test whether the same sparse coordinate also helps in the preceding block."""
def edit(theta, context):
    theta['W'][2,169] *= 2.0
    return theta
