"""Remove two observed nuisance contributions; retain the surrounding learned probe."""
def edit(theta, context):
    theta['W'][4:6,1856] = 0.0
    return theta
