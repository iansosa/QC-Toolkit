import numpy as np
import sys
import tensorflow.compat.v1 as tf

# the Hessian is only recomputed under some criterion during the iteration.
def nonlinear_solve(Jx, Rx, x, maxIters=20,
                    relativeTolerance=1e-9, absoluteTolerance=1e-13, rateTolerance=5e-1, normTolerance=20,
                    linear_solver=None, args=None):
    tf.reset_default_graph()
    J = Jx(x)
    # Newton iteration loop:
    converged = False
    for i in range(maxIters):
        R = Rx(x, args)
        # J = Jx(x)
        currentNorm = np.linalg.norm(R)
        if i == 0:
            referenceError = currentNorm
            tempNorm = currentNorm
        relativeNorm = currentNorm / referenceError
        print("Solver iteration: " + str(i) + " , Relative norm: " + str(relativeNorm))
        sys.stdout.flush()
        if relativeNorm < relativeTolerance or currentNorm < absoluteTolerance:
            converged = True
            print('converged!')
            break
        if relativeNorm > normTolerance:
            print("ERROR: Nonlinear solver failed to converge.")
            exit()
        if i > 0:
            if currentNorm > tempNorm * rateTolerance:
                print('recomputing J matrix...')
                J = Jx(x)
                tf.reset_default_graph()
            tempNorm = currentNorm

        if linear_solver is None:
            dx = np.linalg.solve(J, R)
        else:
            dx = linear_solver(J * 1e13, R * 1e13)
            dx = dx[0]
        x = x - dx
    if not converged:
        print("ERROR: Nonlinear solver failed to converge.")
        exit()
    return x

