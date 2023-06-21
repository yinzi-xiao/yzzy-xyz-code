import operator
import functools
import itertools
import logging
import json
import time
import statistics
import multiprocessing

import numpy as np
from mpmath import mp

from qecsim import paulitools as pt, tensortools as tt
from qecsim.model import Decoder, cli_description
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarPauli, RotatedPlanarRMPSDecoder

NUM_CORES = 32


#--------------------------------- Class RotatedPlanarYZPauli used in Rotated YZZY code --------------------------------
class RotatedPlanarYZPauli(RotatedPlanarPauli):
    """
    Defines a Pauli operator on a rotated planar YZ/ZY lattice.

    Notes:

    * This is a utility class used by rotated planar implementations of the core models.
    * It is typically instantiated using :meth:`rotatedplanaryz.RotatedPlanarYZCode.new_pauli`

    Use cases:

    * Construct a rotated planar YZ Pauli operator by applying site, plaquette and logical operators:
    :meth:`site`, :meth:`plaquette`, :meth:`logical_y`, :meth:`logical_z`, :meth:`logical_x`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a rotated planar YZ Pauli operator: :meth:`copy`.
    """

    def plaquette(self, index):
        """
        Apply a plaquette operator at the given index.

        Notes:

        * Index is in the format (x, y).
        * Z operators are applied to SW and NE qubits. Y operators are applied to NW and SE qubits.
        * Applying plaquette operators on plaquettes that lie outside the lattice have no effect on the lattice.

        :param index: Index identifying the plaquette in the format (x, y).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanarXZPauli
        """
        x, y = index
        # apply if index within lattice
        if self.code.is_in_plaquette_bounds(index):
            # flip plaquette sites
            self.site('Z', (x, y))  # SW
            self.site('Y', (x, y + 1))  # NW
            self.site('Z', (x + 1, y + 1))  # NE
            self.site('Y', (x + 1, y))  # SE
        return self

    def logical_y(self):
        """
        Apply a logical Y operator, i.e. alternate Y and Z between lower-left and lower-right corners.

        Notes:

        * Operators are applied to the bottom row to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: RotatedPlanarYZPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('Y', *((x, 0) for x in range(0, max_site_x + 1, 2)))
        self.site('Z', *((x, 0) for x in range(1, max_site_x + 1, 2)))
        return self

    def logical_z(self):
        """
        Apply a logical Z operator, i.e. alternate Z and Y between lower-right and upper-right corners.

        Notes:

        * Operators are applied to the rightmost column to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: RotatedPlanarYZPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('Z', *((max_site_x, y) for y in range(0, max_site_y + 1, 2)))
        self.site('Y', *((max_site_x, y) for y in range(1, max_site_y + 1, 2)))
        return self

    def logical_x(self):
        """
        Apply a logical X operator simply by multiplying logical Y & Z together.

        Notes:

        :return: self (to allow chaining)
        :rtype: RotatedPlanarYZPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('Y', *((x, 0) for x in range(0, max_site_x + 1, 2)))
        self.site('Z', *((x, 0) for x in range(1, max_site_x + 1, 2)))
        self.site('Z', *((max_site_x, y) for y in range(0, max_site_y + 1, 2)))
        self.site('Y', *((max_site_x, y) for y in range(1, max_site_y + 1, 2)))
        return self

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.code, self.to_bsf())

#---------------------------------- Rotated Planar Code with Z & Y stabilizers -----------------------------------------

#---------------------------------------------- Rotated YZZY code ------------------------------------------------------
class RotatedPlanarYZCode(RotatedPlanarCode):
    r"""
    Implements a rotated planar mixed boundary code with YZ/ZY plaquettes defined by its lattice size.

    In addition to the members defined in :class:`qecsim.model.StabilizerCode`, it provides several lattice methods as
    described below.

    Lattice methods:

    * Get size: :meth:`size`.
    * Get plaquette type: :meth:`is_virtual_plaquette`.
    * Get and test bounds: :meth:`site_bounds`, :meth:`is_in_site_bounds`, :meth:`is_in_plaquette_bounds`.
    * Resolve a syndrome to plaquettes: :meth:`syndrome_to_plaquette_indices`.
    * Construct a Pauli operator on the lattice: :meth:`new_pauli`.

    Indices:

    * Indices are in the format (x, y).
    * Qubit sites (vertices) are indexed by (x, y) coordinates with the origin at the lower left qubit.
    * Stabilizer plaquettes are indexed by (x, y) coordinates such that the lower left corner of the plaquette is on the
      qubit site at (x, y).

    For example, qubit site indices on a 3 x 3 lattice:
    ::

             (0,2)-----(1,2)-----(2,2)
               |         |         |
               |         |         |
               |         |         |
             (0,1)-----(1,1)-----(2,1)
               |         |         |
               |         |         |
               |         |         |
             (0,0)-----(1,0)-----(2,0)

    For example, stabilizer plaquette indices on a 3 x 3 lattice:
    ::

                 -------
                /       \
               |Z (0,2) Y|
               +---------+---------+-----
               |Y       Z|Y       Z|Y    \
               |  (0,1)  |  (1,1)  |(2,1) |
               |Z       Y|Z       Y|Z    /
          -----+---------+---------+-----
         /    Z|Y       Z|Y       Z|
        |(-1,0)|  (0,0)  |  (1,0)  |
         \    Y|Z       Y|Z       Y|
          -----+---------+---------+
                         |Y       Z|
                          \ (1,-1)/
                           -------
    """

    def __init__(self, distance):
        """
        Initialise new rotated planar YZ code.

        :param distance: Number of rows/columns in lattice.
        :type distance: int
        :raises ValueError: if size smaller than 3.
        :raises ValueError: if size is even.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if operator.index(distance) < self.MIN_SIZE[0]:
                raise ValueError('{} minimum distance is {}.'.format(type(self).__name__, self.MIN_SIZE[0]))
            if distance % 2 == 0:
                raise ValueError('{} size must be odd.'.format(type(self).__name__))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        super().__init__(distance, distance)

    # < StabilizerCode interface methods >

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Rotated planar YZ {}'.format(self.n_k_d[2])

    # </ StabilizerCode interface methods >

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.n_k_d[2])

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Rotated planar YZ Pauli
        :rtype: RotatedPlanarYZPauli
        """
        return RotatedPlanarYZPauli(self, bsf)
    
#------------------- MPS decoder for Rotated Y/Z surface code and for non-IID error model ------------------------------
logger = logging.getLogger(__name__)
class RotatedPlanarYZRNIIDMPSDecoder(RotatedPlanarRMPSDecoder):
    r"""
    Implements a rotated planar Rotated Matrix Product State (RMPS) decoder which has Z/Y stabilizer and also support 
    for non-IID error model.
    
    * Note: because the input parameter "prob_dist" has been changed its form, so it should be used manually, 
    rather than use the app method for implementing it automatically. 
    
    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by applying a path of Y operators between each plaquette,
      identified by the syndrome, along a diagonal to an appropriate boundary.
    * The probability of the left coset :math:`fG` of the stabilizer group :math:`G` of the planar code with respect
      to :math:`f` is found by contracting an appropriately defined MPS-based tensor network (see
      https://arxiv.org/abs/1405.4883).
    * Since this is a rotated MPS decoder, the links of the network are rotated 45 degrees by splitting each stabilizer
      node into 4 delta nodes that are absorbed into the neighbouring qubit nodes.
    * The complexity of the algorithm can managed by defining a bond dimension :math:`\chi` to which the MPS bond
      dimension is truncated after each row/column of the tensor network is contracted into the MPS.
    * The probability of cosets :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G` are calculated similarly.
    * The default contraction is column-by-column but can be set using the mode parameter to row-by-row or the average
      of both contractions.
    * A sample recovery operation from the most probable coset is returned.

    Notes:

    * Specifying chi=None gives an exact contract (up to rounding errors) but is exponentially slow in the size of
      the lattice.
    * Modes:

        * mode='c': contract by columns
        * mode='r': contract by rows
        * mode='a': contract by columns and by rows and, for each coset, take the average of the probabilities.

    * Contracting by columns (i.e. truncating vertical links) may give different coset probabilities to contracting by
      rows (i.e. truncating horizontal links). However, the effect is symmetric in that transposing the sample_pauli on
      the lattice and exchanging Y and Z single Paulis reverses the difference between Y and Z cosets probabilities.

    Tensor network example:

    3x3 rotated planar code with H or V indicating qubits and hashed/blank plaquettes indicating Y/Z stabilizers:
    ::

           /---\
           |   |
           H---V---H--\
           |###|   |##|
           |###|   |##|
           |###|   |##|
        /--V---H---V--/
        |##|   |###|
        |##|   |###|
        |##|   |###|
        \--H---V---H
               |   |
               \---/


    MPS tensor network as per https://arxiv.org/abs/1405.4883 (s=stabilizer):
    ::

             s
            / \
           H   V   H
            \ / \ / \
             s   s   s
            / \ / \ /
           V   H   V
          / \ / \ /
         s   s   s
          \ / \ / \
           H   V   H
                \ /
                 s

    Links are rotated by splitting stabilizers and absorbing them into neighbouring qubits.
    For even columns of stabilizers (according to indexing defined in :class:`qecsim.models.planar.RotatedPlanarCode`),
    a 'lucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H V
          s    =>    s s    =>  | |
         / \         | |        V-H
        V   H        s-s
                    /   \
                   V     H

    For odd columns, an 'unlucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H-V
          s    =>    s-s    =>  | |
         / \         | |        V H
        V   H        s s
                    /   \
                   V     H

    Resultant MPS tensor network, where horizontal (vertical) bonds have dimension 2 (4) respectively.
    ::

          0 1 2
        0 H-V-H
          | | |
        1 V-H-V
          | | |
        2 H-V-H
    """

    @classmethod
    def sample_recovery(cls, code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path of Y operators between each
        plaquette, identified by the syndrome, along a diagonal to an appropriate boundary.

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as rotated planar pauli.
        :rtype: RotatedPlanarPauli
        """
        # prepare sample
        sample_recovery = code.new_pauli()
        # ask code for syndrome plaquette_indices
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)
        # for each plaquette
        max_site_x, max_site_y = code.site_bounds
        for plaq_index in plaquette_indices:
            # NOTE: plaquette index coincides with site on lower left corner
            plaq_x, plaq_y = plaq_index
            # if upper-left even diagonals or lower-right odd diagonals
            if (plaq_x < plaq_y and (plaq_x - plaq_y) % 2 == 0) or (plaq_x > plaq_y and (plaq_x - plaq_y) % 2 == 1):
                # join with X to lower-left boundary
                site_x, site_y = plaq_x, plaq_y
                while site_x >= 0 and site_y >= 0:
                    sample_recovery.site('Y', (site_x, site_y))
                    site_x -= 1
                    site_y -= 1
            else:
                # join with X to upper-right boundary
                site_x, site_y = plaq_x + 1, plaq_y + 1
                while site_x <= max_site_x and site_y <= max_site_y:
                    sample_recovery.site('Y', (site_x, site_y))
                    site_x += 1
                    site_y += 1
        # return sample
        return sample_recovery
#TODO: calculating cos_ps with non-IID error prob_dist
    def _coset_probabilities(self, prob_dist_list, sample_pauli):
        r"""
        Return the (approximate) probability and sample Pauli for the left coset :math:`fG` of the stabilizer group
        :math:`G` of the planar code with respect to the given sample Pauli :math:`f`, as well as for the cosets
        :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G`.

        :param prob_dist_list: A list of tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)); 
        the list size equals the code size and the ith tuple in the list represents the prob_dist of ith qubit in the 
        code. The 1D index definition is the same as in class RotatedPlanarPauli.
        :type prob_dist_list: A list with 4-tuple of float as each list element
        :param sample_pauli: Sample planar Pauli.
        :type sample_pauli: PlanarPauli
        :return: Coset probabilities, Sample Paulis (both in order I, X, Y, Z)
            E.g. (0.20, 0.10, 0.05, 0.10), (PlanarPauli(...), PlanarPauli(...), PlanarPauli(...), PlanarPauli(...))
        :rtype: 4-tuple of mp.mpf, 4-tuple of PlanarPauli
        """
        # NOTE: all list/tuples in this method are ordered (i, x, y, z)
        # empty log warnings
        log_warnings = []
        # the list length should be the same as code size
        if (len(prob_dist_list)!= sample_pauli.code.n_k_d[0]):
            print("The length of probability distribution list should be the same as code qubit number!")
        # sample paulis
        sample_paulis = (
            sample_pauli,
            sample_pauli.copy().logical_x(),
            sample_pauli.copy().logical_x().logical_z(),
            sample_pauli.copy().logical_z()
        )
        # TODO: This needs to be changed according to the definition of tn.
        # tensor networks: tns are common to both contraction by column and by row (after transposition)
        tns = [self._tnc.create_tn(prob_dist_list, sp) for sp in sample_paulis]
        # probabilities
        coset_ps = (0.0, 0.0, 0.0, 0.0)  # default coset probabilities
        coset_ps_col = coset_ps_row = None  # undefined coset probabilities by column and row
        # N.B. After multiplication by mult, coset_ps will be of type mp.mpf so don't process with numpy!
        if self._mode in ('c', 'a'):
            # evaluate coset probabilities by column
            coset_ps_col = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # TODO: One change is made here for X to Y
            # note: I,Z and Y,X cosets differ only in the last column (logical Z)
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1)  # tns.i
                coset_ps_col[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_col.i
                coset_ps_col[3] = tt.mps.inner_product(bra_i, tns[3][:, -1]) * mult  # coset_ps_col.z
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR I/Z COSET FAILED: {!r}'.format(ex))
            try:
                bra_y, mult = tt.mps2d.contract(tns[2], chi=self._chi, tol=self._tol, stop=-1)  # tns.y
                coset_ps_col[2] = tt.mps.inner_product(bra_y, tns[2][:, -1]) * mult  # coset_ps_col.y
                coset_ps_col[1] = tt.mps.inner_product(bra_y, tns[1][:, -1]) * mult  # coset_ps_col.x
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR X/Y COSET FAILED: {!r}'.format(ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_col = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_col]
        if self._mode in ('r', 'a'):
            # evaluate coset probabilities by row
            coset_ps_row = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # transpose tensor networks
            tns = [tt.mps2d.transpose(tn) for tn in tns]
            # note: I,Y and Z,X cosets differ only in the last row (logical Y)
            # TODO: One change is made here for X to Y
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1)  # tns.i
                coset_ps_row[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_row.i
                coset_ps_row[2] = tt.mps.inner_product(bra_i, tns[2][:, -1]) * mult  # coset_ps_row.y
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR I/X COSET FAILED: {!r}'.format(ex))
            try:
                bra_z, mult = tt.mps2d.contract(tns[3], chi=self._chi, tol=self._tol, stop=-1)  # tns.z
                coset_ps_row[3] = tt.mps.inner_product(bra_z, tns[3][:, -1]) * mult  # coset_ps_row.z
                coset_ps_row[1] = tt.mps.inner_product(bra_z, tns[1][:, -1]) * mult  # coset_ps_row.x
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR Z/Y COSET FAILED: {!r}'.format(ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_row = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_row]
        if self._mode == 'c':
            coset_ps = coset_ps_col
        elif self._mode == 'r':
            coset_ps = coset_ps_row
        elif self._mode == 'a':
            # average coset probabilities
            coset_ps = [sum(coset_p) / len(coset_p) for coset_p in zip(coset_ps_col, coset_ps_row)]
        # logging
        if log_warnings:
            log_data = {
                # instance
                'decoder': repr(self),
                # method parameters
                'prob_dist_list': prob_dist_list,
                'sample_pauli': pt.pack(sample_pauli.to_bsf()),
                # variables (convert to string because mp.mpf)
                'coset_ps_col': [repr(p) for p in coset_ps_col] if coset_ps_col else None,
                'coset_ps_row': [repr(p) for p in coset_ps_row] if coset_ps_row else None,
                'coset_ps': [repr(p) for p in coset_ps],
            }
            logger.warning('{}: {}'.format(' | '.join(log_warnings), json.dumps(log_data, sort_keys=True)))
        # results
        return tuple(coset_ps), sample_paulis

    # TODO: need to change here to support non-IID error model
    def decode(self, code, syndrome, prob_dist_list, **kwargs):
        """
        See :meth:`qecsim.model.Decoder.decode`

        Note: The optional keyword parameters ``error_model`` and ``error_probability`` are used to determine the prior
        probability distribution for use in the decoding algorithm. Any provided error model must implement
        :meth:`~qecsim.model.ErrorModel.probability_distribution`.

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :param error_model: Error model. (default=DepolarizingErrorModel())
        :type prob_dist_list: A list of tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)); 
        the list size equals the code size and the ith tuple in the list represents the prob_dist of ith qubit in the 
        code. The 1D index definition is the same as in class RotatedPlanarPauli.
        :type prob_dist_list: A list with 4-tuple of float as each list element
        :return: Recovery operation as binary symplectic vector.
        :rtype: numpy.array (1d)
        """
        # any recovery
        any_recovery = self.sample_recovery(code, syndrome)
        # TODO: This also needs to be changed according to definition of method: _coset_probabilities
        # coset probabilities, recovery operations
        coset_ps, recoveries = self._coset_probabilities(prob_dist_list, any_recovery)
        # most likely recovery operation
        max_coset_p, max_recovery = max(zip(coset_ps, recoveries), key=lambda coset_p_recovery: coset_p_recovery[0])
        # logging
        if not (mp.isfinite(max_coset_p) and max_coset_p > 0):
            log_data = {
                # instance
                'decoder': repr(self),
                # method parameters
                'code': repr(code),
                'syndrome': pt.pack(syndrome),
                'prob_dist_list': prob_dist_list,
                # variables
                'coset_ps': [repr(p) for p in coset_ps],  # convert to string because mp.mpf
                # context
                'error': pt.pack(kwargs['error']) if 'error' in kwargs else None,
            }
            logger.warning('NON-POSITIVE-FINITE MAX COSET PROBABILITY: {}'.format(json.dumps(log_data, sort_keys=True)))
        # return most likely recovery operation as bsf
        return max_recovery.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        params = [('chi', self._chi), ('mode', self._mode), ('tol', self._tol), ]
        return 'Rotated planar YZZY RMPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(type(self).__name__, self._chi, self._mode, self._tol)

    class TNC:
        """Tensor network creator"""
        # TODO: One change is add here for X to Y
        @functools.lru_cache()
        def h_node_value(self, prob_dist, f, n, e, s, w):
            """Return horizontal edge tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            I, X, Y, Z = pt.pauli_to_bsf(paulis)
            # n, e, s, w are in {0, 1} so multiply op to turn on or off
            op = (f + (n * Z) + (e * Y) + (s * Z) + (w * Y)) % 2
            return op_to_pr[pt.bsf_to_pauli(op)]

        @functools.lru_cache()
        def v_node_value(self, prob_dist, f, n, e, s, w):
            """Return V-node qubit tensor element value."""
            # N.B. with YZ/ZY plaquettes, H-node and V-node values are both as per H-node values of the CSS code
            return self.h_node_value(prob_dist, f, n, e, s, w)

        @functools.lru_cache(maxsize=256)
        def create_q_node(self, prob_dist, f, h_node, even_column, compass_direction=None):
            """Create q-node for tensor network.

            Notes:

            * H-nodes have Z-plaquettes above and below (i.e. in NE and SW directions).
            * V-nodes have Z-plaquettes on either side (i.e. in NW and SE directions).
            * Columns are considered even/odd according to indexing defined in :class:`RotatedPlanarCode`.

            :param h_node: If H-node, else V-node.
            :type h_node: bool
            :param prob_dist: Probability distribution in the format (Pr(I), Pr(X), Pr(Y), Pr(Z)).
            :type prob_dist: (float, float, float, float)
            :param f: Pauli operator on qubit as 'I', 'X', 'Y', or 'Z'.
            :type f: str
            :param even_column: If even column, else odd column.
            :type even_column: bool
            :param compass_direction: Compass direction as 'n', 'ne', 'e', ..., 'nw', or falsy for bulk.
            :type compass_direction: str
            :return: Q-node for tensor network.
            :rtype: numpy.array (4d)
            """

            # H indicates h-node with shape (n,e,s,w).
            # * indicates delta nodes with shapes (n,I,j), (e,J,k), (s,K,l), (w,L,i) for n-, e-, s-, and w-deltas
            #   respectively.
            # n,e,s,w,i,j,k,I,J,K are bond labels
            #
            #   i     I
            #   |     |
            # L-*     *-j
            #    \   /
            #    w\ /n
            #      H
            #    s/ \e
            #    /   \
            # l-*     *-J
            #   |     |
            #   K     k
            #
            # Deltas are absorbed into h-node over n,e,s,w legs and reshaped as follows:
            # nesw -> (iI)(jJ)(Kk)(Ll)

            # define shapes # q_node:(n, e, s, w); delta_nodes: n:(n,I,j), e:(e,J,k), s:(s,K,l), w:(w,L,i)
            if h_node:
                # bulk h-node
                q_shape = (2, 2, 2, 2)
                if even_column:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 2), (2, 1, 2), (2, 2, 2), (2, 1, 2)
                else:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 2)
                # modifications for directions
                if compass_direction == 'n':
                    q_shape = (2, 2, 2, 1)
                    n_shape, w_shape = (2, 1, 2), (1, 1, 1)
                elif compass_direction == 'ne':
                    q_shape = (1, 2, 2, 1)
                    n_shape, e_shape, w_shape = (1, 1, 1), (2, 1, 2), (1, 1, 1)
                elif compass_direction == 'e':
                    q_shape = (1, 2, 2, 2)
                    n_shape, e_shape = (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'se':  # always even
                    q_shape = (1, 1, 2, 2)
                    n_shape, e_shape, s_shape = (1, 1, 1), (1, 1, 1), (2, 1, 2)
                elif compass_direction == 's':  # always even
                    q_shape = (2, 1, 2, 2)
                    e_shape, s_shape = (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'sw':  # always even
                    q_shape = (2, 1, 1, 2)
                    e_shape, s_shape, w_shape = (1, 1, 1), (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'w':  # always even
                    q_shape = (2, 2, 1, 2)
                    s_shape, w_shape = (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'nw':  # always even
                    q_shape = (2, 2, 1, 1)
                    n_shape, s_shape, w_shape = (2, 1, 2), (1, 1, 1), (1, 1, 1)
            else:
                # bulk v-node
                q_shape = (2, 2, 2, 2)
                if even_column:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 2), (2, 1, 2), (2, 2, 2), (2, 1, 2)
                else:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 2)
                # modifications for directions
                if compass_direction == 'n':
                    q_shape = (1, 2, 2, 2)
                    n_shape, w_shape = (1, 1, 1), (2, 2, 1)
                elif compass_direction == 'ne':
                    q_shape = (1, 1, 2, 2)
                    n_shape, e_shape, w_shape = (1, 1, 1), (1, 1, 1), (2, 2, 1)
                elif compass_direction == 'e':
                    q_shape = (2, 1, 2, 2)
                    n_shape, e_shape = (2, 2, 1), (1, 1, 1)
                elif compass_direction == 'se':  # always odd
                    q_shape = (2, 1, 1, 2)
                    n_shape, e_shape, s_shape = (2, 2, 1), (1, 1, 1), (1, 1, 1)
                elif compass_direction == 's':  # always odd
                    q_shape = (2, 2, 1, 2)
                    e_shape, s_shape = (2, 2, 1), (1, 1, 1)
                elif compass_direction == 'sw':  # not possible
                    raise ValueError('Cannot have v-node in SW corner of lattice.')
                elif compass_direction == 'w':  # always even
                    q_shape = (2, 2, 2, 1)
                    s_shape, w_shape = (2, 2, 1), (1, 1, 1)
                elif compass_direction == 'nw':  # always even
                    q_shape = (1, 2, 2, 1)
                    n_shape, s_shape, w_shape = (1, 1, 1), (2, 2, 1), (1, 1, 1)

            # create deltas
            n_delta = tt.tsr.delta(n_shape)
            e_delta = tt.tsr.delta(e_shape)
            s_delta = tt.tsr.delta(s_shape)
            w_delta = tt.tsr.delta(w_shape)
            # create q_node and fill values
            q_node = np.empty(q_shape, dtype=np.float64)
            for n, e, s, w in np.ndindex(q_node.shape):
                if h_node:
                    q_node[(n, e, s, w)] = self.h_node_value(prob_dist, f, n, e, s, w)
                else:
                    q_node[(n, e, s, w)] = self.v_node_value(prob_dist, f, n, e, s, w)
            # derive combined node shape
            shape = (w_shape[2] * n_shape[1], n_shape[2] * e_shape[1], e_shape[2] * s_shape[1], s_shape[2] * w_shape[1])
            # create combined node by absorbing deltas into q_node: nesw -> (iI)(jJ)(Kk)(Ll)
            node = np.einsum('nesw,nIj,eJk,sKl,wLi->iIjJKkLl', q_node, n_delta, e_delta, s_delta, w_delta).reshape(
                shape)
            # return combined node
            return node
        # TODO: Need change to adapt non-IID error prob_dist
        def create_tn(self, prob_dist_list, sample_pauli):
            """Return a network (numpy.array 2d) of tensors (numpy.array 4d).
            Note: The network contracts to the coset probability of the given sample_pauli.
            """

            def _xy_to_rc_index(index, code):
                """Convert code site index in format (x, y) to tensor network q-node index in format (r, c)"""
                x, y = index
                return code.site_bounds[1] - y, x

            def _compass_direction(index, code):
                """if the code site index in format (x, y) lies on border then give that direction, else empty string"""
                direction = {code.site_bounds[1]: 'n', 0: 's'}.get(index[1], '')
                direction += {0: 'w', code.site_bounds[0]: 'e'}.get(index[0], '')
                return direction

            # extract code
            code = sample_pauli.code
            # initialise empty tn
            tn = np.empty(code.size, dtype=object)
            # iterate over site indices
            max_site_x, max_site_y = code.site_bounds
            for code_index in itertools.product(range(max_site_x + 1), range(max_site_y + 1)):
                # prepare parameters
                is_h_node = code.is_z_plaquette(code_index)
                q_node_index = _xy_to_rc_index(code_index, code)
                is_even_column = not (q_node_index[1] % 2)
                q_pauli = sample_pauli.operator(code_index)
                q_direction = _compass_direction(code_index, code)
                # TODO: The prob_dist here should be specificlly of the qubit on the code_index
                # get the 1D representation of the code index
                code_index_1d = sample_pauli._flatten_site_index(code_index)
                # create q-node
                q_node = self.create_q_node(prob_dist_list[code_index_1d], q_pauli, is_h_node, is_even_column, q_direction)
                # add q-node to tensor network
                tn[q_node_index] = q_node
            return tn



#-------------------------------------- MPS decoder for Rotated YZZY code ----------------------------------------------
class RotatedPlanarYZRMPSDecoder(RotatedPlanarRMPSDecoder):
    @classmethod
    def sample_recovery(cls, code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path of Y operators between each
        plaquette, identified by the syndrome, along a diagonal to an appropriate boundary.

        :param code: Rotated planar YZ code.
        :type code: RotatedPlanarYZCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as rotated planar pauli.
        :rtype: RotatedPlanarYZPauli
        """
        # prepare sample
        sample_recovery = code.new_pauli()
        # ask code for syndrome plaquette_indices
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)
        # for each plaquette
        max_site_x, max_site_y = code.site_bounds
        for plaq_index in plaquette_indices:
            # NOTE: plaquette index coincides with site on lower left corner
            plaq_x, plaq_y = plaq_index
            # if upper-left even diagonals or lower-right odd diagonals
            if (plaq_x < plaq_y and (plaq_x - plaq_y) % 2 == 0) or (plaq_x > plaq_y and (plaq_x - plaq_y) % 2 == 1):
                # join with Y to lower-left boundary
                site_x, site_y = plaq_x, plaq_y
                while site_x >= 0 and site_y >= 0:
                    sample_recovery.site('Y', (site_x, site_y))
                    site_x -= 1
                    site_y -= 1
            else:
                # join with Y to upper-right boundary
                site_x, site_y = plaq_x + 1, plaq_y + 1
                while site_x <= max_site_x and site_y <= max_site_y:
                    sample_recovery.site('Y', (site_x, site_y))
                    site_x += 1
                    site_y += 1
        # return sample
        return sample_recovery

    def _coset_probabilities(self, prob_dist, sample_pauli):
        r"""
        Return the (approximate) probability and sample Pauli for the left coset :math:`fG` of the stabilizer group
        :math:`G` of the planar code with respect to the given sample Pauli :math:`f`, as well as for the cosets
        :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G`.

        :param prob_dist_list: A list of tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)); 
        the list size equals the code size and the ith tuple in the list represents the prob_dist of ith qubit in the 
        code. The 1D index definition is the same as in class RotatedPlanarPauli.
        :type prob_dist_list: A list with 4-tuple of float as each list element
        :param sample_pauli: Sample planar Pauli.
        :type sample_pauli: PlanarPauli
        :return: Coset probabilities, Sample Paulis (both in order I, X, Y, Z)
            E.g. (0.20, 0.10, 0.05, 0.10), (PlanarPauli(...), PlanarPauli(...), PlanarPauli(...), PlanarPauli(...))
        :rtype: 4-tuple of mp.mpf, 4-tuple of PlanarPauli
        """
        # NOTE: all list/tuples in this method are ordered (i, x, y, z)
        # empty log warnings
        log_warnings = []
        # sample paulis
        sample_paulis = (
            sample_pauli,
            sample_pauli.copy().logical_x(),
            sample_pauli.copy().logical_x().logical_z(),
            sample_pauli.copy().logical_z()
        )
        # TODO: This needs to be changed according to the definition of tn.
        # tensor networks: tns are common to both contraction by column and by row (after transposition)
        tns = [self._tnc.create_tn(prob_dist, sp) for sp in sample_paulis]
        # probabilities
        coset_ps = (0.0, 0.0, 0.0, 0.0)  # default coset probabilities
        coset_ps_col = coset_ps_row = None  # undefined coset probabilities by column and row
        # N.B. After multiplication by mult, coset_ps will be of type mp.mpf so don't process with numpy!
        if self._mode in ('c', 'a'):
            # evaluate coset probabilities by column
            coset_ps_col = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # TODO: One change is made here for X to Y
            # note: I,Z and Y,X cosets differ only in the last column (logical Z)
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1)  # tns.i
                coset_ps_col[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_col.i
                coset_ps_col[3] = tt.mps.inner_product(bra_i, tns[3][:, -1]) * mult  # coset_ps_col.z
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR I/Z COSET FAILED: {!r}'.format(ex))
            try:
                bra_z, mult = tt.mps2d.contract(tns[2], chi=self._chi, tol=self._tol, stop=-1)  # tns.y
                coset_ps_col[2] = tt.mps.inner_product(bra_z, tns[2][:, -1]) * mult  # coset_ps_col.x
                coset_ps_col[1] = tt.mps.inner_product(bra_z, tns[1][:, -1]) * mult  # coset_ps_col.y
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR Y/X COSET FAILED: {!r}'.format(ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_col = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_col]
        if self._mode in ('r', 'a'):
            # evaluate coset probabilities by row
            coset_ps_row = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # transpose tensor networks
            tns = [tt.mps2d.transpose(tn) for tn in tns]
            # note: I,Y and Z,X cosets differ only in the last row (logical Y)
            # TODO: One change is made here for X to Y
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1)  # tns.i
                coset_ps_row[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_row.i
                coset_ps_row[2] = tt.mps.inner_product(bra_i, tns[2][:, -1]) * mult  # coset_ps_row.y
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR I/X COSET FAILED: {!r}'.format(ex))
            try:
                bra_y, mult = tt.mps2d.contract(tns[3], chi=self._chi, tol=self._tol, stop=-1)  # tns.z
                coset_ps_row[3] = tt.mps.inner_product(bra_y, tns[3][:, -1]) * mult  # coset_ps_row.z
                coset_ps_row[1] = tt.mps.inner_product(bra_y, tns[1][:, -1]) * mult  # coset_ps_row.x
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR Z/Y COSET FAILED: {!r}'.format(ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_row = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_row]
        if self._mode == 'c':
            coset_ps = coset_ps_col
        elif self._mode == 'r':
            coset_ps = coset_ps_row
        elif self._mode == 'a':
            # average coset probabilities
            coset_ps = [sum(coset_p) / len(coset_p) for coset_p in zip(coset_ps_col, coset_ps_row)]
        # logging
        if log_warnings:
            log_data = {
                # instance
                'decoder': repr(self),
                # method parameters
                'prob_dist': prob_dist,
                'sample_pauli': pt.pack(sample_pauli.to_bsf()),
                # variables (convert to string because mp.mpf)
                'coset_ps_col': [repr(p) for p in coset_ps_col] if coset_ps_col else None,
                'coset_ps_row': [repr(p) for p in coset_ps_row] if coset_ps_row else None,
                'coset_ps': [repr(p) for p in coset_ps],
            }
            logger.warning('{}: {}'.format(' | '.join(log_warnings), json.dumps(log_data, sort_keys=True)))
        # results
        return tuple(coset_ps), sample_paulis

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        params = [('chi', self._chi), ('mode', self._mode), ('tol', self._tol), ]
        return 'Rotated planar YZ RMPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    class TNC(RotatedPlanarRMPSDecoder.TNC):
        """Tensor network creator"""
        @functools.lru_cache()
        def h_node_value(self, prob_dist, f, n, e, s, w):
            """Return horizontal edge tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            I, X, Y, Z = pt.pauli_to_bsf(paulis)
            # n, e, s, w are in {0, 1} so multiply op to turn on or off
            op = (f + (n * Z) + (e * Y) + (s * Z) + (w * Y)) % 2
            return op_to_pr[pt.bsf_to_pauli(op)]

        @functools.lru_cache()
        def v_node_value(self, prob_dist, f, n, e, s, w):
            """Return V-node qubit tensor element value."""
            # N.B. with YZ/ZY plaquettes, H-node and V-node values are both as per H-node values of the CSS code
            return self.h_node_value(prob_dist, f, n, e, s, w)

# ------------------------------------ code run for non-IID error model ------------------------------------------------
def run_niid(code,decoder,prob_dist_list,error_probability,max_runs):
    """
    This is a packaged code sequence for running simulation max_runs times a for non-IID error model, and returns to a 
    dictionary of running data.
    code: YZZY code
    decoder: YZZY non-IID MPS decoder
    prob_dist_list: a list of prob_dist for each qubits in the code
    error_probability: error probability parameter used in simulation
    max_runs: number of simulation running times
    return: a dictionary with running data
    """
    wall_time_start = time.perf_counter()
    # initialize runs_data
    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': 1, # 1 for ideal simulation
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': 0.0, # 0 for ideal simulation
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'n_logical_commutations': None,
        'custom_totals': None,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }
    error_weights = []  # list of error_weight from current run

    # initialize rng
    rng = np.random.default_rng()

    # each error probability is simulated max_run times
    for run in range(max_runs):
        # generate a random error
        error_pauli = ''
        for i_qubit in range(code.n_k_d[0]):
            error_pauli += ''.join(rng.choice(('I', 'X', 'Y', 'Z'),
            size=1, p=prob_dist_list[i_qubit]))
        error = pt.pauli_to_bsf(error_pauli)
        # transform error to syndrome
        syndrome = pt.bsp(error, code.stabilizers.T)
        # decode to find recovery
        recovery = decoder.decode(code, syndrome, prob_dist_list)
        # check if recovery is success or not
        # check if recovery communicate with stabilizers
        commutes_with_stabilizers = np.all(pt.bsp(recovery^error, code.stabilizers.T) == 0)
        if not commutes_with_stabilizers:
            log_data = {  # enough data to recreate issue
                # models
                'code': repr(code), 'decoder': repr(decoder),
                # variables
                'error': pt.pack(error), 'recovery': pt.pack(recovery),
            }
            logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
        # check if recovery communicate with logical operations
        commutes_with_logicals = np.all(pt.bsp(recovery^error, code.logicals.T) == 0)
        # success if recovery communicate with both stabilizers and logical operations
        success = commutes_with_stabilizers and commutes_with_logicals
        # increment run counts
        runs_data['n_run'] += 1
        if success:
            runs_data['n_success'] += 1
        else:
            runs_data['n_fail'] += 1
        # append error weight
        error_weights.append(pt.bsf_wt(np.array(error)))

    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    # add rate statistics
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0]

    runs_data['logical_failure_rate'] = n_fail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run
    
    return runs_data

# write a sub function for Pool to call
def run_once(index,code,decoder,prob_dist_list,error_paulis):
    #index parameter is used for Pool
    id = index
    error_pauli = error_paulis[id]
    error = pt.pauli_to_bsf(error_pauli)
    # transform error to syndrome
    syndrome = pt.bsp(error, code.stabilizers.T)
    # decode to find recovery
    recovery = decoder.decode(code, syndrome, prob_dist_list)
    # check if recovery is success or not
    # check if recovery communicate with stabilizers
    commutes_with_stabilizers = np.all(pt.bsp(recovery^error, code.stabilizers.T) == 0)
    if not commutes_with_stabilizers:
        log_data = {  # enough data to recreate issue
            # models
            'code': repr(code), 'decoder': repr(decoder),
            # variables
            'error': pt.pack(error), 'recovery': pt.pack(recovery),
        }
        logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
    # check if recovery communicate with logical operations
    commutes_with_logicals = np.all(pt.bsp(recovery^error, code.logicals.T) == 0)
    # respectively check if recovery communicate with logical X and Z
    commutes_with_logicalx = np.all(pt.bsp(recovery^error, code.logical_xs.T) == 0)
    commutes_with_logicalz = np.all(pt.bsp(recovery^error, code.logical_zs.T) == 0)
    # success if recovery communicate with both stabilizers and logical operations
    success = commutes_with_stabilizers and commutes_with_logicals
    # record the logical x and z failures seperately
    failure_x = commutes_with_stabilizers and not commutes_with_logicalx
    failure_z = commutes_with_stabilizers and not commutes_with_logicalz
    error_weight = pt.bsf_wt(np.array(error))
    # return to a list containing success and error_weight
    return [success,error_weight,failure_x,failure_z]

#----------------------------------- write a code to run simulation in multi-cores ------------------------------------
def run_niid_multicore(code,decoder,prob_dist_list,error_probability,max_runs):
    """
    This is a packaged code sequence for running simulation max_runs times a for non-IID error model in multi-cores 
    parallely, and returns to a dictionary of running data.
    code: YZZY code
    decoder: YZZY non-IID MPS decoder
    prob_dist_list: a list of prob_dist for each qubits in the code
    error_probability: error probability parameter used in simulation
    max_runs: number of simulation running times
    return: a dictionary with running data
    """
    wall_time_start = time.perf_counter()
    # initialize runs_data
    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': 1, # 1 for ideal simulation
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': 0.0, # 0 for ideal simulation
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'n_xfail' : 0,
        'n_zfail' : 0,
        'n_logical_commutations': None,
        'custom_totals': None,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'logicalx_failure_rate': 0.0,
        'logicalz_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }
    #count cpu cores
    num_cores = NUM_CORES
    # initialize rng
    rng = np.random.default_rng()
    # generate errors in advance to make sure all the errors are different (otherwise if directly generate error in
    # sub function run_once, the errors are all identical)
    error_paulis = []
    for _ in range(max_runs):
        # generate a random error
        error_pauli = ''
        for i_qubit in range(code.n_k_d[0]):
            error_pauli += ''.join(rng.choice(('I', 'X', 'Y', 'Z'),
            size=1, p=prob_dist_list[i_qubit]))
        error_paulis.append(error_pauli)
    
    # create a Pool and run the sub function in parallel.
    pool = multiprocessing.Pool(processes=num_cores)
    results = [pool.apply_async(run_once,args=(index,code,decoder,prob_dist_list,error_paulis)) for index in range(max_runs)]
    pool.close()
    pool.join()
    
    # get the results from ApplyResult object
    true_results = []
    for result in results:
        true_results.append(result.get()) 
    
    # update run counts
    runs_data['n_run'] = int(len(true_results))
    success, failure_x, failure_z = 0, 0, 0
    error_weights = []  # list of error_weight from current run
    for result in true_results:
        success += int(result[0])
        error_weights.append(int(result[1]))
        failure_x += int(result[2])
        failure_z += int(result[3])
    runs_data['n_success'] = success
    runs_data['n_fail'] = runs_data['n_run'] - runs_data['n_success']
    runs_data['n_xfail'] = failure_x
    runs_data['n_zfail'] = failure_z
    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    # add rate statistics
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    n_xfail = runs_data['n_xfail']
    n_zfail = runs_data['n_zfail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0]

    runs_data['logical_failure_rate'] = n_fail / n_run
    runs_data['logicalx_failure_rate'] = n_xfail / n_run
    runs_data['logicalz_failure_rate'] = n_zfail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run
    
    return runs_data

#--------------------------- sub-functions used in run_xyz_multicore --------------
def generate_xyz_error(error_model,code,probability,rng=None):
    """
    Generate a error for XYZ code.
    error_model: error model used for XYZ code
    code: the YZZY code mapped from XYZ code
    probability: error probability in error model
    rng: random number generator
    return: a error with length 2*d^2 (a list with two elements with length d^2) for XYZ code
    """
    rng = np.random.default_rng() if rng is None else rng
    n_qubits = code.n_k_d[0]
    error_pauli1 = ''.join(rng.choice(
        ('I', 'X', 'Y', 'Z'),
        size=n_qubits,
        p=error_model.probability_distribution(probability)
    ))
    error_pauli2 = ''.join(rng.choice(
        ('I', 'X', 'Y', 'Z'),
        size=n_qubits,
        p=error_model.probability_distribution(probability)
    ))
    error_pauli_xyz = [error_pauli1,error_pauli2]
    return error_pauli_xyz

def map_from_xyz_to_yzzy(error_model,error_probability,error_paulis_xyz):
    """
    Based by the error model and error probability for XYZ, map the given error paulis in XYZ code to corresponding error paulis in YZZY code with their error probability distribution.
    error_model: error model used in XYZ code
    error_probability: error probability in the error model
    error_paulis_xyz: a list of error paulis in XYZ code
    return: a tuple of mapping error paulis in YZZY code and their corresponding error probability distributions.
    """
    pi,px,py,pz = error_model.probability_distribution(error_probability)
    error_paulis = []
    prob_dist_lists = []
    # calculate the error probability distribution when there's link syndrome or not.
    prob_nolink = np.array([pi*pi + px*px, py*py + pz*pz, pz*py + py*pz, pi*px + px*pi])
    sum_nolink = np.sum(prob_nolink)
    prob_nolink = tuple(prob_nolink/sum_nolink)
    
    prob_link = np.array([pz*pi + py*px, px*py + pi*pz, pi*py + px*pz, pz*px + py*pi])
    sum_link = np.sum(prob_link)
    # if the total prob is 0, then is no problem for normalization, assign it another value to avoid 'divide 0' error
    if sum_link == 0:
        sum_link = 1
    prob_link = tuple(prob_link/sum_link)
    # a list to define link error types which light up a syndrome
    z_link_errors = ['IY','YI','XZ','ZX','IZ','ZI','XY','YX']
    # a dictionary to the map from link errors in XYZ code to single qubit errors in YZZY code 
    link_error_map = {
        'YY' : 'X',
        'ZZ' : 'X',
        'YZ' : 'Y',
        'ZY' : 'Y',
        'IX' : 'Z',
        'XI' : 'Z',
        'II' : 'I',
        'XX' : 'I'
    }
    for error_pauli_xyz in error_paulis_xyz:
        error_pauli = ''
        prob_dist_list = []
        length = len(error_pauli_xyz[0])
        for i in range(length):
            error1 = error_pauli_xyz[0][i]
            error2 = error_pauli_xyz[1][i]
            link_error = error1 + error2
            if link_error in z_link_errors:
                # if link error lights up a syndrome, add a Z error to the first qubit
                if error1 == 'I':
                    error1 = 'Z'
                elif error1 == 'X':
                    error1 = 'Y'
                elif error1 == 'Y':
                    error1 = 'X'
                elif error1 == 'Z':
                    error1 = 'I'
                link_error_aft = error1 + error2
                prob_dist_list.append(prob_link)
                error_pauli += link_error_map.get(link_error_aft)
            else:
                prob_dist_list.append(prob_nolink)
                error_pauli += link_error_map.get(link_error)
        prob_dist_lists.append(prob_dist_list)
        error_paulis.append(error_pauli)
    return (error_paulis,prob_dist_lists)

def run_xyz_once(index,code,decoder,prob_dist_lists,error_paulis):
    #index parameter is used for Pool
    id = index
    error_pauli = error_paulis[id]
    prob_dist_list = prob_dist_lists[id]
    # print(error_pauli)
    error = pt.pauli_to_bsf(error_pauli)
    # transform error to syndrome
    syndrome = pt.bsp(error, code.stabilizers.T)
    # decode to find recovery
    recovery = decoder.decode(code, syndrome, prob_dist_list)
    # check if recovery is success or not
    # check if recovery communicate with stabilizers
    commutes_with_stabilizers = np.all(pt.bsp(recovery^error, code.stabilizers.T) == 0)
    if not commutes_with_stabilizers:
        log_data = {  # enough data to recreate issue
            # models
            'code': repr(code), 'decoder': repr(decoder),
            # variables
            'error': pt.pack(error), 'recovery': pt.pack(recovery),
        }
        logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
    # check if recovery communicate with logical operations
    commutes_with_logicals = np.all(pt.bsp(recovery^error, code.logicals.T) == 0)
    # respectively check if recovery communicate with logical X and Z
    commutes_with_logicalx = np.all(pt.bsp(recovery^error, code.logical_xs.T) == 0)
    commutes_with_logicalz = np.all(pt.bsp(recovery^error, code.logical_zs.T) == 0)
    # success if recovery communicate with both stabilizers and logical operations
    success = commutes_with_stabilizers and commutes_with_logicals
    # record the logical x and z failures seperately
    failure_x = commutes_with_stabilizers and not commutes_with_logicalx
    failure_z = commutes_with_stabilizers and not commutes_with_logicalz
    # return success
    return [success, failure_x, failure_z]

#--------------------------- write a code to run simulation for xyz code ---------------------------------
def run_xyz_multicore(code,decoder,error_model,error_probability,max_runs):
    """
    This is a packaged code sequence for running simulation max_runs times a for XYZ code in multi-cores 
    parallely, and returns to a dictionary of running data.
    code: YZZY code which maps from XYZ code
    decoder: YZZY non-IID MPS decoder
    error_model: the error model used for XYZ code
    error_probability: the error probability used in error model
    max_runs: number of simulation running times
    return: a dictionary with running data
    """
    wall_time_start = time.perf_counter()
    # initialize runs_data
    runs_data = {
        'code': 'd = {} XYZ code'.format(code.n_k_d[2]),
        'n_k_d': [2*code.n_k_d[0],code.n_k_d[1],code.n_k_d[2]],
        'error_model' : error_model.label,
        'time_steps': 1, # 1 for ideal simulation
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': 0.0, # 0 for ideal simulation
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'n_xfail' : 0,
        'n_zfail' : 0,
        'n_logical_commutations': None,
        'custom_totals': None,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'logicalx_failure_rate': 0.0,
        'logicalz_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }
    #count cpu cores
    num_cores = multiprocessing.cpu_count()
    # initialize rng
    rng = np.random.default_rng()
    # generate errors in advance to make sure all the errors are different (otherwise if directly generate error in
    # sub function run_once, the errors are all identical)
    error_paulis_xyz = []
    error_weights = []
    for _ in range(max_runs):
        error_pauli_xyz = generate_xyz_error(error_model,code,error_probability,rng)
        error_paulis_xyz.append(error_pauli_xyz)
        # get error weight straight from XYZ error chain
        error1 = pt.pauli_to_bsf(error_pauli_xyz[0])
        error2 = pt.pauli_to_bsf(error_pauli_xyz[1])
        error_weight1 = pt.bsf_wt(np.array(error1))
        error_weight2 = pt.bsf_wt(np.array(error2))
        error_weight = error_weight1 + error_weight2
        error_weights.append(error_weight)
    
    # map XYZ error chain to YZZY error chain & error probability distribution
    error_paulis, prob_dist_lists = map_from_xyz_to_yzzy(error_model,error_probability,error_paulis_xyz)
    
    # create a Pool and run the sub function in parallel.
    pool = multiprocessing.Pool(processes=num_cores)
    results = [pool.apply_async(run_xyz_once,args=(index,code,decoder,prob_dist_lists,error_paulis)) for index in range(max_runs)]
    pool.close()
    pool.join()

    # get the results from ApplyResult object
    true_results = []
    for result in results:
        true_results.append(result.get()) 

    # update run counts
    runs_data['n_run'] = int(len(true_results))
    success, failure_x, failure_z = 0, 0, 0
    for result in true_results:
        success += int(result[0])
        failure_x += int(result[1])
        failure_z += int(result[2])
    runs_data['n_success'] = success
    runs_data['n_fail'] = runs_data['n_run'] - runs_data['n_success']
    runs_data['n_xfail'] = failure_x
    runs_data['n_zfail'] = failure_z
    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    # add rate statistics
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    n_xfail = runs_data['n_xfail']
    n_zfail = runs_data['n_zfail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0] # number of qubits in XYZ code

    runs_data['logical_failure_rate'] = n_fail / n_run
    runs_data['logicalx_failure_rate'] = n_xfail / n_run
    runs_data['logicalz_failure_rate'] = n_zfail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run

    return runs_data