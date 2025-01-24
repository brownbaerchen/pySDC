from firedrake import assemble
from firedrake.__future__ import interpolate

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh, IMEX_firedrake_mesh


class MeshToMeshFiredrake(SpaceTransfer):
    """
    This implementation can restrict and prolong between Firedrake meshes
    """

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        if isinstance(F, firedrake_mesh):
            u_coarse = self.coarse_prob.dtype_u(assemble(interpolate(F.functionspace, self.coarse_prob.init)))
        elif isinstance(F, IMEX_firedrake_mesh):
            u_coarse = IMEX_firedrake_mesh(self.coarse_prob.init)
            u_coarse.impl.functionspace.assign(assemble(interpolate(F.impl.functionspace, self.coarse_prob.init)))
            u_coarse.expl.functionspace.assign(assemble(interpolate(F.expl.functionspace, self.coarse_prob.init)))
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, firedrake_mesh):
            u_fine = self.fine_prob.dtype_u(assemble(interpolate(G.functionspace, self.fine_prob.init)))
        elif isinstance(G, IMEX_firedrake_mesh):
            u_fine = IMEX_firedrake_mesh(self.fine_prob.init)
            u_fine.impl.functionspace.assign(assemble(interpolate(G.impl.functionspace, self.fine_prob.init)))
            u_fine.expl.functionspace.assign(assemble(interpolate(G.expl.functionspace, self.fine_prob.init)))
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
