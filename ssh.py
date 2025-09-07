import numpy as np
from scipy.linalg import eigh


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import argparse


def bloch_hamiltonian(k, t1, t2):
    # Bloch Hamiltonian H(k) in the sublattice basis (A,B)
    off_diag = t1 + t2*np.exp(-1j*k)
    return np.array([[0, off_diag],[np.conj(off_diag), 0]], dtype=complex)

def bandstructure(t1, t2, Nk=401):
    ks = np.linspace(-np.pi, np.pi, Nk, endpoint=True)
    E = np.zeros((Nk, 2))
    vecs = np.zeros((Nk, 2, 2), dtype=complex)
    for i,k in enumerate(ks):
        e, v = eigh(bloch_hamiltonian(k, t1, t2))
        E[i] = e  # smallest to largest
        vecs[i] = v
    return ks, E, vecs

def zak_phase(t1, t2, Nk=2001):
    # Berry (or Zak) phase for the lower band with periodic gauge fixing
    ks, _, vecs = bandstructure(t1, t2, Nk)
    u = vecs[:, :, 0]  # take lower band eigenvectors (last idx 0, 1st idx is k-vec and 2nd idx is sublattice idx)
    # Ensure phase continuity between k points (the eigenvector solver may flip signs arbitrarily due to the gauge freedom)
    u = u / np.linalg.norm(u, axis=1, keepdims=True) # normalize and keep shape
    overlaps = np.einsum('ij,ij->i', np.conj(u[:-1]), u[1:])
    print(f"overlaps={overlaps}")
    phase = np.angle(np.prod(overlaps / np.abs(overlaps)))  # total Berry phase
    
    # close the loop
    phase += np.angle(np.vdot(u[-1], u[0]) / np.abs(np.vdot(u[-1], u[0])))
    # return in [0, 2π) mapped to {0, π} numerically
    phase = (phase + 2*np.pi) % (2*np.pi)
    return phase

def winding_number(t1, t2, Nk=4001):
    ks = np.linspace(-np.pi, np.pi, Nk)
    q = t1 + t2*np.exp(1j*ks)
    # total change of arg q around the loop divided by 2π
    dtheta = np.unwrap(np.angle(q))
    W = int(np.rint((dtheta[-1] - dtheta[0])/(2*np.pi)))
    return W



def ssh_hamiltonian(N, t1, t2, periodic=False):
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
    parser.add_argument("--periodic", action='store_true', help="Use periodic boundary conditions(set false if you want to see edge states.")
    args = parser.parse_args()


    # Parameters
    N = args.N      # Number of unit cells
    t1 = args.t1    # Intra-cell hopping
    t2 = args.t2
    ks, E, _ = bandstructure(t1, t2)
    plt.figure(figsize=(5,3))
    plt.plot(ks, E[:,0], '.', ms=2)
    plt.plot(ks, E[:,1], '.', ms=2)
    plt.xlabel('k'); plt.ylabel('E(k)'); plt.title('SSH Bloch bands'); plt.tight_layout()

    phi = zak_phase(t1, t2)
    W = winding_number(t1, t2)
    print(f"Zak phase (lower band) ≈ {phi:.3f} rad  (~π means topological/ 0 means trivial)")
    print(f"Winding number W = {W}  (1 means topological/ 0 means trivial)")
    

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
    
    
    
    #Plot the spectrum
    plt.figure(figsize=(6,4))
    plt.plot(energies, '.', markersize=4)
    plt.xlabel('State index')
    plt.ylabel('Energy')
    plt.title(f'SSH spectrum: t1 = {t1}, t2 = {t2}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

