import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import argparse

def ssh_hamiltonian(N : int, t1 : float, t2 : float, periodic=False) -> np.ndarray:
    """
    Construct the SSH model Hamiltonian for a chain of N unit cells.
    
    Parameters:
        N : int
            Number of unit cells (2N sites total).
        t1 : float
            Intracell hopping amplitude.
        t2 : float
            Intercell hopping amplitude.
        periodic : bool
            Whether to use periodic boundary conditions.
    
    Returns:
        H : ndarray
            2N x 2N SSH Hamiltonian matrix.
    """
    size = 2 * N
    H = np.zeros((size, size))

    for i in range(N):
        a = 2 * i     # site A in unit cell i
        b = a + 1     # site B in unit cell i

        # Intra-cell hopping
        H[a, b] = t1
        H[b, a] = t1

        # Inter-cell hopping
        if i < N - 1:
            H[b, a + 2] = t2
            H[a + 2, b] = t2

    if periodic:
        # Connect last B to first A
        H[size - 1, 0] = t2
        H[0, size - 1] = t2

    return H


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSH Model Spectrum")
    parser.add_argument("--N", type=int, default=50, help="Number of unit cells")
    parser.add_argument("--t1", type=float, default=0.5, help="Intra-cell hopping amplitude")
    parser.add_argument("--t2", type=float, default=1.0, help="Inter-cell hopping amplitude")
    parser.add_argument("--periodic", action='store_true', help="Use periodic boundary conditions")
    args = parser.parse_args()

    # Parameters
    N = 50      # Number of unit cells
    t1 = 0.5    # Intra-cell hopping
    t2 = 1.0
    # Inter-cell hopping

    # Construct and diagonalize
    H = ssh_hamiltonian(N, t1, t2, periodic=False)
    energies, _ = eigh(H)

    # Show a diagram of the lattice using matplotlib
    # plt.figure(figsize=(8,2))
    # for i in range(N):
    #     plt.plot([2*i, 2*i+1], [0, 0], 'k-', lw=2)  # intra-cell
    #     if i < N - 1:
    #         plt.plot([2*i+1, 2*(i+1)], [0, 0], 'r--', lw=2)  # inter-cell
    # plt.title('SSH Model Lattice')
    
    
    
    # Plot the spectrum
    plt.figure(figsize=(6,4))
    plt.plot(energies, '.', markersize=4)
    plt.xlabel('State index')
    plt.ylabel('Energy')
    plt.title(f'SSH spectrum: t1 = {t1}, t2 = {t2}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
