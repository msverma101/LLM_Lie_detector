"""Reduce an overcorrection after finding counterexamples to the coordinate hypothesis."""
def edit(theta, context):
    theta['W'][3:5,169] *= (2.0/3.0)
    return theta
