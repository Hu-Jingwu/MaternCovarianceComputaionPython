import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import multiprocessing
from functools import partial
import os
import multiprocessing
import concurrent.futures
from scipy import stats
from scipy.fft import fft, fft2, ifft, ifft2
from scipy.special import kv, gamma
from numpy.fft import fftn, ifftshift
from multiprocessing import Pool
import pyfftw
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

#define sincx = sinx / x
def sinc(x):
    #asarray confirm the input of x could be worked as the array
    x = np.array(x)
    eps = np.finfo(float).eps
    
    # Calculate the threshold x0
    x0 = (240 * eps) ** (1 / 4)
    f = np.sin(x)
    ok = np.abs(x) > x0
    f[ok] = f[ok] / x[ok]
    f[~ok] = 1 - x[~ok] ** 2 / 6
    
    return f


def S_fun(omega, ht, gamma_v, alpha_v, DIM):
        return nonsep_space_cross_spectrum(np.sqrt(np.sum(omega**2, axis=1)), ht, gamma_v, alpha_v, DIM)

# Compute folded spectrum for spatial discretisation
def fold_spectrum(omega, S_fun, ht, h,gamma_v, alpha_v, DIM, pointwise=True, n_folds=10, fold=True):
    S = 0
    done = False

    # Generate folding indices
    if fold:
        kk = np.arange(-n_folds, n_folds + 1)
    else:
        kk = np.array([0])

    # Initialize the k_ vector to iterate through folds
    k_ = np.ones(len(h), dtype=int)

    while not done:
        # Compute folded frequencies omega_ = omega + 2 * pi * kk / h
        omega_ = omega + np.kron(np.ones((omega.shape[0], 1)), (2 * np.pi * kk[k_ - 1] / h).reshape(1, -1))

        # Scaling factor for sinc correction, if pointwise is False
        scaling = 1
        if not pointwise:
            for d in range(len(h)):
                scaling *= sinc(omega_[:, d] * h[d] / 2) ** 2

        # Sum the spectrum evaluated at folded frequencies
        S += S_fun(omega_, ht, gamma_v, alpha_v, DIM) * scaling

        # Update the folding indices k_
        idx = np.where(k_ < len(kk))[0]
        if len(idx) == 0:
            done = True
        else:
            idx = np.max(idx)
            k_[idx] += 1
            if idx < len(h) - 1:
                k_[idx + 1:] = 1

    # Global scaling to restrict frequencies to base band
    global_scaling = np.ones(omega.shape[0])
    for d in range(len(h)):
        global_scaling *= (omega[:, d] >= -np.pi / h[d]) * (omega[:, d] < np.pi / h[d])

    # Apply global scaling to the result
    S *= global_scaling

    return S

def fftshift(x, dim=None):
    if dim is None:
        dim = x.shape
    elif np.prod(dim) != x.size:
        raise ValueError(f"Product of dimensions {dim} does not match the size of the array.")
    
    x = np.reshape(x, dim)

    if len(dim) == 1:
        mid = dim[0] // 2
        x = np.concatenate([x[mid:], x[:mid]])
    elif len(dim) == 2:
        mid0, mid1 = dim[0] // 2, dim[1] // 2
        x = np.concatenate([x[mid0:], x[:mid0]], axis=0)
        x = np.concatenate([x[:, mid1:], x[:, :mid1]], axis=1)
    else:
        raise NotImplementedError(f"Dimension {len(dim)} not implemented.")
    
    return x

def S2C(S, dim, h):
    if len(dim) == 1:
        fft = pyfftw.interfaces.numpy_fft.fft
    elif len(dim) == 2:
        fft = pyfftw.interfaces.numpy_fft.fft2
    else:
        raise NotImplementedError(f"Dimension {len(dim)} is not implemented.")
    
    # Apply fftshift, FFT, and reverse fftshift (ifftshift equivalent)
    shifted_S = fftshift(S, dim)
    fft_result = fft(shifted_S)
    C = fftshift(np.real(fft_result), dim)
    
    # Compute the final scaling factor
    scaling_factor = np.prod(2 * np.pi / np.array(h) / np.array(dim))
    
    return C * scaling_factor

def S2sample(S, dim, h, seed=None, conjugate=False):
    if seed is not None:
        np.random.seed(seed)

    if len(dim) == 1:
        fft_func = fft
    elif len(dim) == 2:
        fft_func = fft2
    else:
        raise ValueError(f"Dimension {len(dim)} is not implemented.")
    
    # Standard deviation calculation
    SD = np.sqrt(S * np.prod(2 * np.pi / (np.array(h) * np.array(dim))))
    
    # Generate complex random samples
    z = (np.random.normal(size=np.prod(dim)) + 1j * np.random.normal(size=np.prod(dim))) * SD
    
    if conjugate:
        k = np.array(np.meshgrid(*(np.arange(d) - (d / 2) for d in dim), indexing='ij')).reshape(len(dim), -1).T
        
        if len(dim) == 1:
            pair = np.empty(len(k), dtype=int)
            for idx in range(len(k)):
                pair_ = k[idx, 0] == -k[:, 0]
                if np.any(pair_):
                    pair[idx] = np.where(pair_)[0][0]
                else:
                    pair[idx] = -1  # Assign an invalid index if no pair found
                    
            idx_ = (k[:, 0] < 0) & (pair >= 0)
            z[idx_] = np.real(z[pair[idx_]]) - 1j * np.imag(z[pair[idx_]])
            z[k[:, 0] == 0] = np.real(z[k[:, 0] == 0])  # Real part for k = 0
            
        elif len(dim) == 2:
            pair = np.empty(len(k), dtype=int)
            for idx in range(len(k)):
                pair_ = (k[idx, 0] == -k[:, 0]) & (k[idx, 1] == -k[:, 1])
                if np.any(pair_):
                    pair[idx] = np.where(pair_)[0][0]
                else:
                    pair[idx] = -1  # Assign an invalid index if no pair found
            
            idx_ = (k[:, 0] < 0) & (pair >= 0)
            z[idx_] = np.real(z[pair[idx_]]) - 1j * np.imag(z[pair[idx_]])
            z[(k[:, 0] == 0) & (k[:, 1] < 0)] = np.real(z[(k[:, 0] == 0) & (k[:, 1] < 0)]) - 1j * np.imag(z[pair[(k[:, 0] == 0) & (k[:, 1] < 0)]])
            z[(k[:, 0] == 0) & (k[:, 1] == 0)] = np.real(z[(k[:, 0] == 0) & (k[:, 1] == 0)])  # Real part for (0,0)
            
    else:  # !conjugate
        k = np.array(np.meshgrid(*(np.arange(d) - (d / 2) for d in dim), indexing='ij')).reshape(len(dim), -1).T
        
        if len(dim) == 1:
            edge = k[:, 0] == -dim[0] / 2
            z[edge] = 0
        else:  # length(dim) == 2
            edge = (k[:, 0] == -dim[0] / 2) | (k[:, 1] == -dim[1] / 2)
            z[edge] = 0

    # Apply FFT shift
    z = fftshift(z)
    
    # Perform FFT
    if len(dim) == 1:
        sample = fft_func(z)
    else:
        sample = fft_func(z.reshape(dim))

    return sample

#this can give us the list of x1.... x2.....
def make_x_list(dim, L):
    x = {}
    for d in range(len(dim)):
        x[f'x{d+1}'] = (np.arange(1, dim[d] + 1) - 1 - dim[d] / 2) / dim[d] * L[d]
    return x

def make_x_array(dim, L):
    # Create a list to store arrays for each dimension
    x_list = []
    
    # Loop through dimensions and generate arrays
    for d in range(len(dim)):
        # Generate the x values for each dimension
        x_values = (np.arange(1, dim[d] + 1) - 1 - dim[d] / 2) / dim[d] * L[d]
        # Append each array to the list
        x_list.append(x_values)
    
    # Use meshgrid to generate the grid for all combinations of x values
    mesh = np.meshgrid(*x_list, indexing='ij')
    
    # Stack the mesh arrays to create a single array of coordinates
    x_array = np.stack(mesh, axis=-1)
    
    return x_array

#the output is also list like {'x1': array([-5., -3., -1.,  1.,  3.]), 'x2': array([-4., -2.,  0.,  2.])}
def make_x_sampling_list(dim, h):
    L = [d * h_i for d, h_i in zip(dim, h)]  # Calculate lengths for each dimension
    x = {}
    
    for d in range(len(dim)):
        x[f'x{d+1}'] = (np.arange(dim[d]) / dim[d]) * L[d]
    
    return x

def make_x_sampling_array(dim, h):
    # Calculate lengths for each dimension
    L = [d * h_i for d, h_i in zip(dim, h)]
    
    # Create a list to store arrays for each dimension
    x_list = []
    
    # Loop through dimensions and generate arrays
    for d in range(len(dim)):
        # Generate the x values for each dimension
        x_values = (np.arange(dim[d]) / dim[d]) * L[d]
        # Append each array to the list
        x_list.append(x_values)
    
    # Use meshgrid to generate the grid for all combinations of x values
    mesh = np.meshgrid(*x_list, indexing='ij')
    
    # Stack the mesh arrays to create a single array of coordinates
    x_array = np.stack(mesh, axis=-1)
    
    return x_array

#the output is the list again 
def make_omega(dim, L):
    dim = np.array(dim)  # Convert dim to a NumPy array
    L = np.array(L)      # Convert L to a NumPy array
    h = L / dim          # Calculate spatial steps for each dimension
    w = {}
    
    for d in range(len(dim)):
        w[f'w{d+1}'] = (np.arange(-dim[d] / 2, dim[d] / 2) / (dim[d] / 2)) * (np.pi / h[d])
    
    return w

def make_omega_array(dim, L):
    dim = np.array(dim)  # Convert dim to a NumPy array
    L = np.array(L)      # Convert L to a NumPy array
    h = L / dim          # Calculate spatial steps for each dimension
    
    # Create a list to store omega arrays for each dimension
    w_list = []
    
    for d in range(len(dim)):
        # Generate omega values for each dimension
        w_values = (np.arange(-dim[d] / 2, dim[d] / 2) / (dim[d] / 2)) * (np.pi / h[d])
        w_list.append(w_values)
    
    # Use meshgrid to generate the grid for all combinations of omega values
    omega_mesh = np.meshgrid(*w_list, indexing='ij')
    
    # Stack the mesh arrays to create a single array of omega coordinates
    omega_array = np.stack(omega_mesh, axis=-1)
    
    return omega_array



#output is list
def make_omega_sampling(dim, h):
    w = {}
    
    for d in range(len(dim)):
        w[f'w{d+1}'] = (np.arange(-dim[d] / 2, dim[d] / 2) / (dim[d] / 2)) * (np.pi / h[d])
    
    return 

def make_omega_sampling_array(dim, h):
    # Create a list to store omega arrays for each dimension
    w_list = []
    
    for d in range(len(dim)):
        # Generate omega values for each dimension
        w_values = (np.arange(-dim[d] / 2, dim[d] / 2) / (dim[d] / 2)) * (np.pi / h[d])
        w_list.append(w_values)
    
    # Use meshgrid to generate the grid for all combinations of omega values
    omega_mesh = np.meshgrid(*w_list, indexing='ij')
    
    # Stack the mesh arrays to create a single array of omega coordinates
    omega_array = np.stack(omega_mesh, axis=-1)
    
    return omega_array



def nonsep_spectrum1d(w_s, w_t, gamma_v, alpha_v, DIM):
    """
    Compute the non-separable spectrum for given parameters.

    Parameters:
    - w_s: Norm of the spatial (angular) wave number vector
    - w_t: Temporal norm of the temporal (angular) frequency
    - gamma_v: List containing [gamma_t, gamma_s, gamma_0]
    - alpha_v: List containing [alpha_t, alpha_s, alpha_e]
    - DIM: The spatial domain dimension

    Returns:
    - S: The non-separable spectrum
    """
    # Calculate spatial spectrum
    S_spat = gamma_v[1] ** 2 + w_s ** 2

    # Compute the non-separable spectrum
    S = ((gamma_v[0] ** 2 * w_t ** 2 + S_spat ** alpha_v[1]) ** alpha_v[0]) * (S_spat ** alpha_v[2])
    S = 1 / ((2 * np.pi) ** (1 + DIM) * gamma_v[2] ** 2 * S)

    return S

"""
computation for matern covariance by nu kappa this part is hard to calculate since we nont know how to compute"
"""
def INLA_matern_cov(nu, kappa, ht):
    # Placeholder for the Matérn covariance function. In practice, you would need to implement this.
    kh_nu =(kappa * ht) ** nu
    bessel_term = kv(nu, kappa * ht)
    C_h = (1 / (2**(nu - 1) * gamma(nu))) * kh_nu * bessel_term
    return C_h


def nonsep_space_cross_spectrum(w_s, ht, gamma_v, alpha_v, DIM):
    # Unpack gamma and alpha values
    gamma_t, gamma_s, gamma_0 = gamma_v
    alpha_t, alpha_s, alpha_e = alpha_v

    # Spatial spectrum calculation
    S_spat = gamma_s**2 + w_s**2
    
    # Time-integral parameter
    kappa = (S_spat**(alpha_s / 2)) / gamma_t
    
    # Correlation part of time-integral using Matern covariance
    S_corr = INLA_matern_cov(nu=alpha_t - 1/2, kappa=kappa, ht=ht)  # Define or import this function
    
    # Variance part of time-integral
    S_variance = S_corr * (gamma(alpha_t - 1/2) / gamma(alpha_t) / 
                  (kappa**(2 * alpha_t - 1)) / np.sqrt(4 * np.pi))
    
    # Final spectrum calculation
    S =  S_variance / (((2 * np.pi)**DIM) * gamma_0**2 * (S_spat**alpha_e) * (gamma_t**(2 * alpha_t)))
    
    return S

def S_fun(omega, ht, gamma_v, alpha_v, DIM):
        return nonsep_space_cross_spectrum(np.sqrt(np.sum(omega**2, axis=1)), ht, gamma_v, alpha_v, DIM)

def nonsep_covar(hx, ht, gamma_v, alpha_v, DIM, expand_factor=2):
    """
    Compute the covariance of a non-separable spatial-temporal process.

    Parameters:
    - hx: An increasing regularly spaced vector starting at 0.
    - ht: Time lag (can be a scalar).
    - gamma_v: List containing [gamma_t, gamma_s, gamma_0].
    - alpha_v: List containing [alpha_t, alpha_s, alpha_e].
    - DIM: The dimension of the spatial domain (1 or 2).
    - expand_factor: Factor to expand the FFT grid size.

    Returns:
    - C: Covariance matrix.
    """
    
    # Check that hx starts at 0 and is non-negative
    assert hx[0] == 0, "hx must start at 0."
    assert np.all(hx >= 0), "All elements of hx must be non-negative."
    
    ht = np.abs(ht)  # Ensure ht is non-negative

    
    # Calculate dimensions for FFT
    calc_dim = 2 ** np.ceil(np.log2(len(hx))).astype(int) * 2 * expand_factor
    dim = np.array([1] * DIM) * calc_dim
    h = np.full(DIM, hx[1] - hx[0])
    L = dim * h
    
    # Create the frequency grid
    omega_ = make_omega(dim, L)
    omega = np.array(np.meshgrid(*[omega_[f'w{i+1}'] for i in range(DIM)], indexing='ij')).reshape(DIM, -1).T

    # Compute the folded spectrum
    S = fold_spectrum(omega=omega, S_fun=S_fun, h=h, ht=ht, gamma_v=gamma_v, alpha_v=alpha_v, DIM=DIM)
    
    # Compute covariance from the folded spectrum
    C = fftshift(S2C(S, dim, h), dim)
    
    if DIM == 1:
        C = C[:len(hx)]  # Covariance for 1D
    elif DIM == 2:
        C = C[:len(hx), :len(hx)]  # Covariance for 2D
    else:
        raise NotImplementedError("Dimension greater than 2 is not implemented.")

    return C

def compute_nonsep_covar(j, hx, ht, gamma_v, alpha_v, DIM):
    try:
        res = nonsep_covar(hx, ht[j], gamma_v, alpha_v, DIM)
        return res[:, 0]
    except Exception as e:
        print(f"Error in compute_nonsep_covar: {e}")
        return np.nan
# Define parameters for Model A: Separable order 1
alpha_t = 1
alpha_s = 2
alpha_e = 1
DIM = 2
rt_factor = 1
alpha_v = [alpha_t, alpha_s, alpha_e]
N=64
alpha = alpha_e + alpha_s * (alpha_t - 1 / 2)
nu_spatial = alpha - DIM / 2
range_s = 1
gamma_s = np.sqrt(8 * nu_spatial) / range_s

# Temporal correlation range
nu_time = min(alpha_t - 1 / 2, nu_spatial / alpha_s) if alpha_s != 0 else alpha_t - 1 / 2
range_t = 1 * rt_factor
gamma_t = range_t * gamma_s**alpha_s / np.sqrt(8 * (alpha_t - 1 / 2)) if alpha_t != 1 / 2 else 1.0

# Variance setting for gamma_0
sigma2 = (
    gamma(alpha_t - 1 / 2) * gamma(alpha - DIM / 2) /
    (gamma(alpha_t) * gamma(alpha) * ((4 * np.pi)**((DIM+1)/2)) * gamma_t * gamma_s**(2 * (alpha - DIM / 2)))
)
gamma_0 = np.sqrt(sigma2)
gamma_v = [gamma_t, gamma_s, gamma_0]


# Parameters for the spatial domain
dim = [64, 64]     # Spatial dimensions
L = [1.25*rt_factor, 1.25*rt_factor]       # Spatial range in each dimension
h = [L[i] / dim[i] for i in range(len(dim))]  # Spatial steps based on dim and L
hx = np.linspace(0, L[0], dim[0])  # Regularly spaced vector for covariance
ht = np.linspace(0, L[1], dim[1])

# Example values for covariance function parameters (from previous steps)
#gamma_v = [1.0, 1.0, 1.0]  # gamma_t, gamma_s, gamma_0
#alpha_v = [1.5, 1.0, 1.0]  # alpha_t, alpha_s, alpha_e
DIM = 2                    # 2D spatial domain                 # Temporal correlation lag
# Generate the covariance matrix
C = np.zeros((len(hx), len(ht)))  # Initialize C matrix

C = np.zeros((len(hx), len(ht)))  # Initialize C matrix


C = np.zeros((len(hx), len(ht)))
for j in range(N):
    print(".", end="")
    res = nonsep_covar(hx, ht[j], gamma_v, alpha_v, DIM)
    C[:, j] = res[:, 0]
C = C.T

# Reshape or slice the covariance matrix for 2D plotting if needed
# Here assuming `C` is already a 2D array
x = np.linspace(0, L[0], dim[0])
y = np.linspace(0, L[1], dim[1])
X, Y = np.meshgrid(x, y)


    # Contour lines
# Temporal and spatial decays (replace with actual calculations if available)
temporal_decay = np.where(C[0, :] != 0, C / C[0, :], 0)  # Placeholder for actual temporal decay
spatial_decay = np.where(C[:, 0] != 0, C / C[:, 0][:, np.newaxis], 0)   # Placeholder for actual spatial decay

# Set up a single plot
fig, ax = plt.subplots(figsize=(6, 4))

# Contour fill for covariance levels
contour = ax.contourf(X, Y, C, levels=np.linspace(0, 1, 11), cmap="Blues")

# Contour lines for covariance levels
cset = ax.contour(X, Y, C, levels=np.linspace(0, 1, 11), colors="black", linewidths=0.5)
ax.clabel(cset, fmt="%.1f", colors="black", fontsize=8)

# Temporal and spatial decay overlays (grey lines)
ax.contour(X, Y, temporal_decay, levels=[0.2, 0.4, 0.6, 0.8], colors="grey", linestyles="--")
ax.contour(X, Y, spatial_decay, levels=[0.2, 0.4, 0.6, 0.8], colors="grey", linestyles="--")

# Labels and title
ax.set_title("B: Critical Diffusion")
ax.set_xlabel(r"$h_s$")
ax.set_ylabel(r"$h_t$")

# Colorbar
fig.colorbar(contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()