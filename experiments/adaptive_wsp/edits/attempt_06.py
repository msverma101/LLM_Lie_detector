"""Remove a late coordinate that opposes the named contradiction/control contrast."""
def edit(theta, context):
    theta['W'][4:6,1581] = 0.0
    return theta
