""" This file defines utility classes and functions for dynamics. """
import numpy as np


def guess_dynamics(gains, acc, dX, dU, dt):
    """
    Initial guess at the model using position-velocity assumption.
    Note: This code assumes joint positions occupy the first dU state
          indices and joint velocities occupy the next dU.
    Args:
        gains: dU dimensional joint gains.
        acc: dU dimensional joint acceleration.
        dX: Dimensionality of the state.
        dU: Dimensionality of the action.
        dt: Length of a time step.
    Returns:
        Fd: A dX by dX+dU transition matrix.
        fc: A dX bias vector.
    """
    Fd = np.vstack([
        np.hstack([
            np.eye(dU), dt * np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt ** 2 * np.diag(gains)
        ]),
        np.hstack([
            np.zeros((dU, dU)), np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt * np.diag(gains)
        ]),
        np.zeros((dX - dU*2, dX+dU))
    ])
    fc = np.hstack([acc * dt ** 2, acc * dt, np.zeros((dX - dU*2))])
    return Fd, fc

def guess_dynamics_rr(dX, dU):
    """
    Initial guess at the model using position-velocity assumption.
    Note: This code assumes joint positions occupy the first dU state
          indices and joint velocities occupy the next dU.
    Args:
        gains: dU dimensional joint gains.
        acc: dU dimensional joint acceleration.
        dX: Dimensionality of the state.
        dU: Dimensionality of the action.
        dt: Length of a time step.
    Returns:
        Fd: A dX by dX+dU transition matrix.
        fc: A dX bias vector.
    """
    # factor = 0.9
    # Fd = np.hstack(((1-factor)*np.eye(dX), factor*np.eye(dU)))

    factor = 1
    Fd = np.hstack((np.eye(dX), factor * np.eye(dU)))
    # Fd = np.vstack([
    #     np.hstack([np.eye(dU), np.zeros((dU, dX - dU)), factor * np.eye(dU)
    #     ]),
    #     np.zeros((dX - dU, dX + dU))
    # ])

    fc = np.zeros(dX)
    return Fd, fc