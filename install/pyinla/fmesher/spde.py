"""SPDE model functions for spatial modeling.

This module provides pure Python implementations of SPDE-related functions
for creating Matern-based spatial models using the SPDE approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.special import gamma

from .mesh import FmMesh, FmFEM
from .core import fm_fem

from scipy.spatial import Delaunay


def _write_fmesher_file(A: Any, filename: str) -> str:
    """Write matrix in fmesher binary format.

    Internal function to avoid circular imports with pyinla.fmesher_io.
    """
    import struct

    version = 0

    with open(filename, "wb") as fp:
        if isinstance(A, np.ndarray) and A.ndim == 2:
            # Dense matrix - column major
            nrow, ncol = A.shape
            elems = nrow * ncol
            datatype = 0  # dense
            valuetype = 0 if np.issubdtype(A.dtype, np.integer) else 1
            matrixtype = 0  # general
            storagetype = 1  # columnmajor
            h = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)

            fp.write(struct.pack("=i", h.size))
            h.astype(np.int32).tofile(fp)

            if valuetype == 0:
                np.asarray(A, dtype=np.int32).ravel(order="F").tofile(fp)
            else:
                np.asarray(A, dtype=np.float64).ravel(order="F").tofile(fp)

        elif sparse.issparse(A):
            # Sparse matrix
            M = A.tocoo()
            nrow, ncol = M.shape
            i = M.row.astype(np.int32)
            j = M.col.astype(np.int32)
            values = M.data

            # Sort in column-major order
            sort_idx = np.lexsort((i, j))
            i = i[sort_idx]
            j = j[sort_idx]
            values = values[sort_idx]

            elems = i.size
            datatype = 1  # sparse
            valuetype = 0 if np.issubdtype(values.dtype, np.integer) else 1
            matrixtype = 0  # general
            storagetype = 1  # columnmajor
            h = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)

            fp.write(struct.pack("=i", h.size))
            h.tofile(fp)
            i.tofile(fp)
            j.tofile(fp)
            if valuetype == 0:
                values.astype(np.int32).tofile(fp)
            else:
                values.astype(np.float64).tofile(fp)

        elif isinstance(A, np.ndarray) and A.ndim == 1:
            # Diagonal matrix from 1D array
            n = A.size
            i = np.arange(n, dtype=np.int32)
            j = np.arange(n, dtype=np.int32)
            values = np.asarray(A)
            elems = n
            datatype = 1  # sparse
            valuetype = 0 if np.issubdtype(values.dtype, np.integer) else 1
            matrixtype = 2  # diagonal
            storagetype = 1
            h = np.array([version, elems, n, n, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)

            fp.write(struct.pack("=i", h.size))
            h.tofile(fp)
            i.tofile(fp)
            j.tofile(fp)
            if valuetype == 0:
                values.astype(np.int32).tofile(fp)
            else:
                values.astype(np.float64).tofile(fp)
        else:
            raise TypeError(f"Unsupported type: {type(A)}")

    return filename


@dataclass
class SPDE2PcMatern:
    """SPDE model with Penalized Complexity (PC) priors for Matern fields.

    This class represents an SPDE approximation to a Matern Gaussian field
    using the finite element method. It uses PC priors for the practical
    range and marginal standard deviation parameters.

    Attributes
    ----------
    mesh : FmMesh
        The triangular mesh for the SPDE discretization.
    fem : FmFEM
        The FEM matrices (mass and stiffness matrices).
    alpha : float
        Smoothness parameter. The Matern smoothness nu = alpha - d/2.
        For alpha=2 in 2D, nu=1 (once differentiable fields).
    prior_range : tuple of (float, float)
        PC prior for practical range: (r0, prob) such that P(range < r0) = prob.
    prior_sigma : tuple of (float, float)
        PC prior for marginal std: (s0, prob) such that P(sigma > s0) = prob.
    n_spde : int
        Number of mesh vertices (dimension of the latent field).
    lambda_range : float
        Rate parameter for the exponential prior on range.
    lambda_sigma : float
        Rate parameter for the exponential prior on sigma.

    Notes
    -----
    The SPDE approach (Lindgren et al., 2011) approximates a Matern field
    by solving the SPDE:

        (kappa^2 - Delta)^(alpha/2) u(s) = W(s)

    where Delta is the Laplacian, W is spatial white noise, and:
    - kappa = sqrt(8 * nu) / range
    - tau controls the marginal variance

    The precision matrix for alpha=2 is:
        Q = tau^2 * (kappa^4 * C + 2 * kappa^2 * G + G * C^{-1} * G)

    PC priors (Simpson et al., 2017) penalize departure from a base model
    (infinite range, zero variance) using exponential distributions.

    Examples
    --------
    >>> from fmesher import fm_mesh_2d
    >>> from fmesher.spde import spde2_pcmatern
    >>>
    >>> # Create mesh
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>>
    >>> # Create SPDE model with PC priors
    >>> spde = spde2_pcmatern(mesh, prior_range=(1.0, 0.5), prior_sigma=(1.0, 0.5))
    >>>
    >>> # Compute precision matrix for given parameters
    >>> theta = [np.log(0.5), np.log(1.0)]  # log(range), log(sigma)
    >>> Q = spde.precision(theta)
    """

    mesh: FmMesh
    fem: FmFEM
    alpha: float = 2.0
    prior_range: Tuple[float, float] = (1.0, 0.5)
    prior_sigma: Tuple[float, float] = (1.0, 0.5)
    n_spde: int = field(init=False)
    lambda_range: float = field(init=False)
    lambda_sigma: float = field(init=False)
    _d: int = 2  # spatial dimension

    def __post_init__(self) -> None:
        """Initialize derived quantities."""
        self.n_spde = self.mesh.n

        # Compute PC prior rate parameters
        # P(range < r0) = p  =>  lambda_r = -log(p) / r0
        r0, p_r = self.prior_range
        self.lambda_range = -np.log(p_r) / r0

        # P(sigma > s0) = p  =>  lambda_s = -log(p) / s0
        s0, p_s = self.prior_sigma
        self.lambda_sigma = -np.log(p_s) / s0

        # Compute B matrix constants for PC-Matern parameterization
        # Based on R-INLA's inla.spde2.pcmatern for alpha=2, nu=1, d=2
        self._compute_b_matrices()

    @property
    def nu(self) -> float:
        """Matern smoothness parameter nu = alpha - d/2."""
        return self.alpha - self._d / 2.0

    def _compute_kappa_tau(
        self, range_val: float, sigma: float
    ) -> Tuple[float, float]:
        """Compute kappa and tau from range and sigma.

        Parameters
        ----------
        range_val : float
            Practical correlation range.
        sigma : float
            Marginal standard deviation.

        Returns
        -------
        kappa : float
            Scale parameter in SPDE.
        tau : float
            Precision scaling parameter.
        """
        nu = self.nu
        d = self._d

        # kappa = sqrt(8 * nu) / range
        kappa = np.sqrt(8.0 * nu) / range_val

        # tau from the variance formula
        # Var(u) = sigma^2 = Gamma(nu) / (Gamma(alpha) * (4*pi)^(d/2) * kappa^(2*nu) * tau^2)
        # Solving for tau:
        # tau^2 = Gamma(nu) / (Gamma(alpha) * (4*pi)^(d/2) * kappa^(2*nu) * sigma^2)
        # tau = sqrt(...) / sigma
        scaling = gamma(nu) / (gamma(self.alpha) * (4.0 * np.pi) ** (d / 2.0))
        tau = np.sqrt(scaling) / (sigma * kappa**nu)

        return kappa, tau

    def precision(
        self, theta: Union[np.ndarray, Tuple[float, float], list]
    ) -> sparse.csr_matrix:
        """Compute the SPDE precision matrix Q.

        Parameters
        ----------
        theta : array-like of shape (2,)
            Hyperparameters: [log(range), log(sigma)].

        Returns
        -------
        Q : sparse.csr_matrix
            Precision matrix of shape (n_spde, n_spde).

        Notes
        -----
        For alpha=2, the precision matrix is:
            Q = tau^2 * (kappa^4 * C + 2 * kappa^2 * G + G * C^{-1} * G)

        where C is the lumped mass matrix (diagonal) and G is the stiffness matrix.
        """
        log_range, log_sigma = theta[0], theta[1]
        range_val = np.exp(log_range)
        sigma = np.exp(log_sigma)

        kappa, tau = self._compute_kappa_tau(range_val, sigma)

        # Get FEM matrices
        C_diag = self.fem.c0  # lumped mass (diagonal)
        G = self.fem.g1  # stiffness matrix

        if self.alpha == 2:
            # Q = tau^2 * (kappa^4 * C + 2 * kappa^2 * G + G * C^{-1} * G)
            C_inv_diag = 1.0 / C_diag
            GCinvG = G @ sparse.diags(C_inv_diag) @ G

            Q = tau**2 * (
                kappa**4 * sparse.diags(C_diag)
                + 2.0 * kappa**2 * G
                + GCinvG
            )
        elif self.alpha == 1:
            # Q = tau^2 * (kappa^2 * C + G)
            Q = tau**2 * (kappa**2 * sparse.diags(C_diag) + G)
        else:
            raise NotImplementedError(
                f"alpha={self.alpha} not implemented. Use alpha=1 or alpha=2."
            )

        return Q.tocsr()

    def log_prior(
        self, theta: Union[np.ndarray, Tuple[float, float], list]
    ) -> float:
        """Compute log-prior density for the hyperparameters.

        Parameters
        ----------
        theta : array-like of shape (2,)
            Hyperparameters: [log(range), log(sigma)].

        Returns
        -------
        log_pi : float
            Log-prior density (up to normalizing constant).

        Notes
        -----
        PC priors are exponential on the natural scale:
        - range ~ Exp(lambda_range) with lambda_range = -log(p) / r0
        - sigma ~ Exp(lambda_sigma) with lambda_sigma = -log(p) / s0

        On log scale, the Jacobian adjustment gives:
        - log p(log(range)) = log(lambda_r) - lambda_r * range + log(range)
        - log p(log(sigma)) = log(lambda_s) - lambda_s * sigma + log(sigma)
        """
        log_range, log_sigma = theta[0], theta[1]
        range_val = np.exp(log_range)
        sigma = np.exp(log_sigma)

        # Log prior on log(range): exponential prior on range with Jacobian
        log_pi_range = (
            np.log(self.lambda_range) - self.lambda_range * range_val + log_range
        )

        # Log prior on log(sigma): exponential prior on sigma with Jacobian
        log_pi_sigma = (
            np.log(self.lambda_sigma) - self.lambda_sigma * sigma + log_sigma
        )

        return log_pi_range + log_pi_sigma

    def summary(self) -> str:
        """Return a summary string of the SPDE model."""
        lines = [
            "SPDE2PcMatern:",
            f"  Mesh vertices (n_spde): {self.n_spde}",
            f"  alpha: {self.alpha}",
            f"  nu (smoothness): {self.nu}",
            f"  Prior range: P(range < {self.prior_range[0]}) = {self.prior_range[1]}",
            f"  Prior sigma: P(sigma > {self.prior_sigma[0]}) = {self.prior_sigma[1]}",
            f"  lambda_range: {self.lambda_range:.4f}",
            f"  lambda_sigma: {self.lambda_sigma:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SPDE2PcMatern(n_spde={self.n_spde}, alpha={self.alpha}, "
            f"prior_range={self.prior_range}, prior_sigma={self.prior_sigma})"
        )

    def _compute_b_matrices(self) -> None:
        """Compute B matrices for the SPDE2 parameterization.

        The B matrices define the transformation from user hyperparameters
        theta = [log(range), log(sigma)] to the internal SPDE coefficients
        phi = [phi0, phi1, phi2] such that:

        Q = exp(phi0) * M0 + exp(phi1) * M1 + exp(phi2) * M2

        where M0=C (mass), M1=G (stiffness), M2=G*C^{-1}*G.

        For each mesh vertex i:
        phi[i] = B0[i, 0] + B0[i, 1] * theta[0] + B0[i, 2] * theta[1]

        For standard PC-Matern (all vertices share same hyperparameters),
        all rows of B0, B1, B2 are identical.
        """
        n = self.n_spde
        nu = self.nu

        if self.alpha == 2 and self._d == 2:
            # PC-Matern with alpha=2, d=2, nu=1
            # Based on R-INLA's empirical values
            # These constants match R-INLA's inla.spde2.pcmatern output

            # B0: coefficient matrix for phi0 = log(coef of M0)
            # phi0 = -0.5*log(32*pi) + theta0 - theta1
            b0_const = -0.5 * np.log(32.0 * np.pi)  # ≈ -2.30523289
            b0_theta0 = 1.0
            b0_theta1 = -1.0

            # B1: coefficient matrix for phi1 = log(coef of M1)
            # phi1 = log(8) - 2*theta0
            b1_const = np.log(8.0)  # ≈ 2.07944154
            b1_theta0 = -2.0
            b1_theta1 = 0.0

            # B2: coefficient matrix for phi2 = log(coef of M2)
            # phi2 = 1 (constant)
            b2_const = 1.0
            b2_theta0 = 0.0
            b2_theta1 = 0.0

            # Create (n_spde, 3) matrices with repeated rows
            self._B0 = np.column_stack([
                np.full(n, b0_const),
                np.full(n, b0_theta0),
                np.full(n, b0_theta1)
            ])
            self._B1 = np.column_stack([
                np.full(n, b1_const),
                np.full(n, b1_theta0),
                np.full(n, b1_theta1)
            ])
            self._B2 = np.column_stack([
                np.full(n, b2_const),
                np.full(n, b2_theta0),
                np.full(n, b2_theta1)
            ])

            # BLC: Linear combinations for posterior output
            # Format: 6 rows for different outputs
            # Row 0: log(range) = theta0
            # Row 1: log(sigma) = theta1
            # Row 2: log(tau) = -0.5*log(32*pi) + theta0 - theta1
            # Row 3: log(kappa) = 0.5*log(8) - theta0
            # Row 4: log(variance) = 2*theta1
            # Row 5: log(range) = theta0 (duplicate for compatibility)
            self._BLC = np.array([
                [0.0, 1.0, 0.0],           # log(range) = theta0
                [0.0, 0.0, 1.0],           # log(sigma) = theta1
                [b0_const, 1.0, -1.0],     # log(tau)
                [0.5 * np.log(8.0), -1.0, 0.0],  # log(kappa)
                [0.0, 0.0, 2.0],           # log(variance) = 2*log(sigma)
                [0.0, 1.0, 0.0],           # log(range) duplicate
            ])
        else:
            raise NotImplementedError(
                f"B matrices not implemented for alpha={self.alpha}, d={self._d}. "
                "Currently only alpha=2, d=2 (nu=1) is supported."
            )

    @property
    def M0(self) -> sparse.csr_matrix:
        """Mass matrix C (lumped/diagonal)."""
        return sparse.diags(self.fem.c0)

    @property
    def M1(self) -> sparse.csr_matrix:
        """Stiffness matrix G."""
        return self.fem.g1

    @property
    def M2(self) -> sparse.csr_matrix:
        """G * C^{-1} * G matrix for alpha=2."""
        C_inv_diag = 1.0 / self.fem.c0
        return self.fem.g1 @ sparse.diags(C_inv_diag) @ self.fem.g1

    @property
    def B0(self) -> np.ndarray:
        """B0 matrix for phi0 parameterization, shape (n_spde, 3)."""
        return self._B0

    @property
    def B1(self) -> np.ndarray:
        """B1 matrix for phi1 parameterization, shape (n_spde, 3)."""
        return self._B1

    @property
    def B2(self) -> np.ndarray:
        """B2 matrix for phi2 parameterization, shape (n_spde, 3)."""
        return self._B2

    @property
    def BLC(self) -> np.ndarray:
        """BLC matrix for linear combinations in posterior output, shape (6, 3)."""
        return self._BLC

    @property
    def n_theta(self) -> int:
        """Number of hyperparameters (2 for PC-Matern: range and sigma)."""
        return 2

    def write_spde_files(self, prefix: str) -> str:
        """Write SPDE model files in fmesher binary format.

        This writes the FEM matrices (M0, M1, M2) and parameterization
        matrices (B0, B1, B2, BLC) required by R-INLA to the specified
        prefix location.

        Parameters
        ----------
        prefix : str
            File path prefix for the output files. Files will be created
            as prefix.M0, prefix.M1, prefix.M2, prefix.B0, prefix.B1,
            prefix.B2, prefix.BLC.

        Returns
        -------
        str
            The prefix path (for use in Model.ini as spde2.prefix).

        Examples
        --------
        >>> spde = spde2_pcmatern(mesh, prior_range=(30, 0.05), prior_sigma=(1, 0.05))
        >>> prefix = spde.write_spde_files("/tmp/myspde/spde")
        >>> # Files created: /tmp/myspde/spde.M0, .M1, .M2, .B0, .B1, .B2, .BLC
        """
        # Ensure directory exists
        dirname = os.path.dirname(prefix)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        # Write FEM matrices
        _write_fmesher_file(self.M0, f"{prefix}.M0")
        _write_fmesher_file(self.M1, f"{prefix}.M1")
        _write_fmesher_file(self.M2, f"{prefix}.M2")

        # Write B matrices (dense)
        _write_fmesher_file(self.B0, f"{prefix}.B0")
        _write_fmesher_file(self.B1, f"{prefix}.B1")
        _write_fmesher_file(self.B2, f"{prefix}.B2")

        # Write BLC matrix
        _write_fmesher_file(self.BLC, f"{prefix}.BLC")

        return prefix


def spde2_pcmatern(
    mesh: FmMesh,
    alpha: float = 2.0,
    prior_range: Tuple[float, float] = (1.0, 0.5),
    prior_sigma: Tuple[float, float] = (1.0, 0.5),
    fem_order: int = 2,
) -> SPDE2PcMatern:
    """Create an SPDE model with PC priors for Matern spatial fields.

    This function creates an SPDE approximation to a Matern Gaussian field
    using finite element methods on a triangular mesh, with Penalized
    Complexity (PC) priors on the range and marginal standard deviation.

    Parameters
    ----------
    mesh : FmMesh
        A triangular mesh from fm_mesh_2d.
    alpha : float, optional
        SPDE order parameter. Default is 2, giving Matern fields with
        smoothness nu = alpha - d/2 = 1 in 2D.
    prior_range : tuple of (float, float), optional
        PC prior specification for practical range: (r0, p) such that
        P(range < r0) = p. Default is (1.0, 0.5).
    prior_sigma : tuple of (float, float), optional
        PC prior specification for marginal std: (s0, p) such that
        P(sigma > s0) = p. Default is (1.0, 0.5).
    fem_order : int, optional
        Order of FEM computation. Default is 2.

    Returns
    -------
    SPDE2PcMatern
        An SPDE model object with methods to compute precision matrices
        and priors.

    Notes
    -----
    The SPDE approach approximates a Matern field by solving:

        (kappa^2 - Delta)^(alpha/2) u(s) = W(s)

    The PC priors penalize departure from a "base model" with infinite
    range (no spatial correlation) and zero variance.

    References
    ----------
    Lindgren, F., Rue, H., & Lindstrom, J. (2011). An explicit link between
    Gaussian fields and Gaussian Markov random fields: the stochastic
    partial differential equation approach. JRSS-B, 73(4), 423-498.

    Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H. (2017).
    Penalising Model Component Complexity: A Principled, Practical Approach
    to Constructing Priors. Statistical Science, 32(1), 1-28.

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_mesh_2d
    >>> from fmesher.spde import spde2_pcmatern
    >>>
    >>> # Create mesh
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>>
    >>> # Create SPDE model
    >>> spde = spde2_pcmatern(
    ...     mesh,
    ...     prior_range=(1.0, 0.5),  # P(range < 1) = 0.5
    ...     prior_sigma=(1.0, 0.5),  # P(sigma > 1) = 0.5
    ... )
    >>> print(spde.summary())
    >>>
    >>> # Get precision matrix for specific hyperparameters
    >>> theta = [np.log(0.5), np.log(1.0)]  # range=0.5, sigma=1.0
    >>> Q = spde.precision(theta)
    >>> print(f"Precision matrix: {Q.shape}, nnz={Q.nnz}")
    """
    # Compute FEM matrices
    fem = fm_fem(mesh, order=fem_order)

    return SPDE2PcMatern(
        mesh=mesh,
        fem=fem,
        alpha=alpha,
        prior_range=prior_range,
        prior_sigma=prior_sigma,
    )


@dataclass
class SpdeGridProjector:
    """Pure Python grid evaluator for projecting mesh fields to regular grids.

    This class provides functionality similar to R's fm_evaluator/fm_pixels
    for creating regular grids and projecting spatial fields onto them,
    without requiring R/rpy2.

    Attributes
    ----------
    mesh : FmMesh
        The mesh to project from.
    xlim : tuple of (float, float)
        X axis limits (min, max).
    ylim : tuple of (float, float)
        Y axis limits (min, max).
    dims : tuple of (int, int)
        Number of grid points in each dimension (nx, ny).
    x : np.ndarray
        X coordinates of grid points.
    y : np.ndarray
        Y coordinates of grid points.
    lattice : np.ndarray
        All grid point coordinates, shape (nx*ny, 2).
    proj : sparse.csr_matrix
        Projection matrix A from mesh vertices to grid points.
    inside : np.ndarray
        Boolean mask indicating which grid points are inside the mesh.

    Examples
    --------
    >>> from fmesher.spde import spde_grid_projector, spde2_pcmatern
    >>> # Create projector with automatic bounds from mesh
    >>> projector = spde_grid_projector(mesh, dims=(100, 100))
    >>> # Project spatial field to grid
    >>> spatial_grid = projector.project(spatial_mean)
    >>> # Plot with matplotlib
    >>> plt.imshow(spatial_grid, origin='lower',
    ...            extent=[*projector.xlim, *projector.ylim])
    """

    mesh: FmMesh
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    dims: Tuple[int, int]
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    lattice: np.ndarray = field(init=False)
    proj: sparse.csr_matrix = field(init=False)
    inside: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize grid and projection matrix."""
        # Create grid coordinates
        self.x = np.linspace(self.xlim[0], self.xlim[1], self.dims[0])
        self.y = np.linspace(self.ylim[0], self.ylim[1], self.dims[1])
        xx, yy = np.meshgrid(self.x, self.y)
        self.lattice = np.column_stack([xx.ravel(), yy.ravel()])

        # Create projection matrix
        self.proj = spde_make_A(self.mesh, self.lattice)

        # Determine which points are inside the mesh
        row_sums = np.asarray(self.proj.sum(axis=1)).flatten()
        self.inside = row_sums > 0.99

    @property
    def n_grid(self) -> int:
        """Total number of grid points."""
        return len(self.lattice)

    @property
    def n_inside(self) -> int:
        """Number of grid points inside the mesh."""
        return int(np.sum(self.inside))

    @property
    def inside_2d(self) -> np.ndarray:
        """2D mask of points inside the mesh, shape (ny, nx)."""
        return self.inside.reshape(self.dims[1], self.dims[0])

    def project(
        self,
        field: np.ndarray,
        mask_outside: bool = True
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Project a field from mesh vertices to the grid.

        Parameters
        ----------
        field : np.ndarray
            Field values at mesh vertices, shape (n_vertices,).
        mask_outside : bool
            If True, return a masked array with points outside mesh masked.

        Returns
        -------
        field_2d : np.ndarray or np.ma.MaskedArray
            Field projected to grid, shape (ny, nx).
        """
        field_grid = self.proj @ field
        field_2d = field_grid.reshape(self.dims[1], self.dims[0])

        if mask_outside:
            return np.ma.array(field_2d, mask=~self.inside_2d)
        return field_2d

    def summary(self) -> str:
        """Return a summary string of the projector."""
        lines = [
            "SpdeGridProjector:",
            f"  Grid dimensions: {self.dims[0]} x {self.dims[1]} = {self.n_grid} points",
            f"  X range: [{self.xlim[0]:.2f}, {self.xlim[1]:.2f}]",
            f"  Y range: [{self.ylim[0]:.2f}, {self.ylim[1]:.2f}]",
            f"  Points inside mesh: {self.n_inside} ({100*self.n_inside/self.n_grid:.1f}%)",
            f"  Projection matrix: {self.proj.shape}, nnz={self.proj.nnz}",
        ]
        return "\n".join(lines)


def spde_grid_projector(
    mesh: FmMesh,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dims: Tuple[int, int] = (100, 100),
    padding: float = 0.0,
) -> SpdeGridProjector:
    """Create a pure Python grid projector for visualizing mesh fields.

    This function creates a regular grid and projection matrix for
    visualizing spatial fields defined on a mesh. Unlike fm_evaluator
    (which requires R/rpy2), this uses pure Python via spde_make_A.

    Parameters
    ----------
    mesh : FmMesh
        The mesh to project from.
    xlim : tuple of (float, float), optional
        X axis limits. If None, uses mesh bounds.
    ylim : tuple of (float, float), optional
        Y axis limits. If None, uses mesh bounds.
    dims : tuple of (int, int)
        Number of grid points in each dimension (nx, ny). Default (100, 100).
    padding : float
        Fraction of range to pad the bounds. Default 0.0 (no padding).
        Negative values shrink the bounds.

    Returns
    -------
    SpdeGridProjector
        Grid projector object with projection matrix, grid coordinates,
        and a project() method for easy field projection.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from fmesher.mesh import FmMesh
    >>> from fmesher.spde import spde_grid_projector, spde2_pcmatern
    >>>
    >>> # Create mesh and SPDE model
    >>> mesh = FmMesh(loc=mesh_locs, tv=mesh_triangles)
    >>> spde = spde2_pcmatern(mesh, prior_range=(50, 0.05), prior_sigma=(2, 0.05))
    >>>
    >>> # After running INLA, get spatial random effect
    >>> spatial_mean = result.summary_random['field']['mean'].values
    >>>
    >>> # Create projector and project to grid
    >>> projector = spde_grid_projector(mesh, dims=(100, 100))
    >>> spatial_grid = projector.project(spatial_mean)  # Returns masked array
    >>>
    >>> # Plot
    >>> plt.imshow(spatial_grid, origin='lower',
    ...            extent=[*projector.xlim, *projector.ylim])
    >>> plt.colorbar(label='Spatial effect')
    """
    mesh_x = mesh.loc[:, 0]
    mesh_y = mesh.loc[:, 1]

    # Determine bounds
    if xlim is None:
        x_min, x_max = mesh_x.min(), mesh_x.max()
        x_range = x_max - x_min
        xlim = (x_min - padding * x_range, x_max + padding * x_range)

    if ylim is None:
        y_min, y_max = mesh_y.min(), mesh_y.max()
        y_range = y_max - y_min
        ylim = (y_min - padding * y_range, y_max + padding * y_range)

    return SpdeGridProjector(mesh=mesh, xlim=xlim, ylim=ylim, dims=dims)


def _compute_barycentric(
    point: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute barycentric coordinates of a point in a triangle.

    Parameters
    ----------
    point : np.ndarray
        Point coordinates (x, y).
    v0, v1, v2 : np.ndarray
        Triangle vertex coordinates (x, y).

    Returns
    -------
    np.ndarray
        Barycentric coordinates (w0, w1, w2) that sum to 1.
    """
    # Using the formula from https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0

    # Compute dot products
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        # Degenerate triangle, return equal weights
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return np.array([w0, w1, w2])


def _find_nearest_vertex(mesh: FmMesh, point: np.ndarray) -> int:
    """Find the index of the nearest mesh vertex to a point.

    Parameters
    ----------
    mesh : FmMesh
        The mesh.
    point : np.ndarray
        Point coordinates (x, y).

    Returns
    -------
    int
        Index of the nearest vertex.
    """
    dists = np.sum((mesh.loc[:, :2] - point[:2]) ** 2, axis=1)
    return int(np.argmin(dists))


def spde_make_A(
    mesh: FmMesh,
    loc: np.ndarray,
) -> sparse.csr_matrix:
    """Create projector matrix from mesh vertices to observation locations.

    This function creates a sparse matrix A that maps the latent field values
    at mesh vertices to values at observation locations using barycentric
    interpolation within the triangles.

    Parameters
    ----------
    mesh : FmMesh
        A triangular mesh from fm_mesh_2d.
    loc : np.ndarray
        Observation locations, shape (n_obs, 2) or (n_obs, 3).

    Returns
    -------
    A : sparse.csr_matrix
        Projector matrix of shape (n_obs, n_vertices). Each row has at most
        3 non-zeros (the barycentric weights for the containing triangle).

    Notes
    -----
    For each observation location:
    1. Find which triangle contains the point
    2. Compute barycentric coordinates within that triangle
    3. The row of A has non-zeros at the triangle's vertex indices

    Points outside the mesh domain have zero weights (row sum = 0),
    matching R's inla.spde.make.A behavior.

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_mesh_2d
    >>> from fmesher.spde import spde_make_A
    >>>
    >>> # Create mesh
    >>> mesh_locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=mesh_locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>>
    >>> # Create observation locations
    >>> obs_locs = np.random.randn(50, 2)
    >>>
    >>> # Create projector matrix
    >>> A = spde_make_A(mesh, obs_locs)
    >>> print(f"A shape: {A.shape}, nnz: {A.nnz}")
    >>>
    >>> # Row sums are 1 for points inside mesh, 0 for points outside
    >>> print(f"Row sums: {A.sum(axis=1).flatten()}")
    """
    loc = np.asarray(loc, dtype=float)
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)

    n_obs = loc.shape[0]
    n_vertices = mesh.n

    # Use scipy Delaunay to find containing triangles
    # Note: We use the mesh's own triangulation, not Delaunay's
    # but Delaunay helps with point location queries

    # Build Delaunay triangulation for point location
    mesh_points = mesh.loc[:, :2]
    try:
        tri = Delaunay(mesh_points)
    except Exception:
        # Fallback: use nearest vertex for all points
        rows, cols, data = [], [], []
        for i, point in enumerate(loc):
            nearest = _find_nearest_vertex(mesh, point)
            rows.append(i)
            cols.append(nearest)
            data.append(1.0)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_vertices))

    # Find which simplex each point belongs to
    simplex_indices = tri.find_simplex(loc[:, :2])

    rows, cols, data = [], [], []

    for i, (point, simplex_idx) in enumerate(zip(loc, simplex_indices)):
        if simplex_idx >= 0:
            # Point is inside the convex hull of mesh vertices
            # Get the triangle vertices from Delaunay
            delaunay_verts = tri.simplices[simplex_idx]
            v0, v1, v2 = mesh_points[delaunay_verts]

            # Compute barycentric coordinates
            bary = _compute_barycentric(point[:2], v0, v1, v2)

            # Add to sparse matrix
            for j, (v_idx, w) in enumerate(zip(delaunay_verts, bary)):
                if abs(w) > 1e-12:  # Skip near-zero weights
                    rows.append(i)
                    cols.append(v_idx)
                    data.append(w)
        else:
            # Point is outside the mesh - leave row as zeros (R behavior)
            pass

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_vertices))
    return A
