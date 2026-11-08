"""Further attenuate an early confidence contribution after inspecting hard failures."""
def edit(theta, context):
    theta['W'][0] *= 0.5
    theta['b'][0] *= 0.5
    return theta
