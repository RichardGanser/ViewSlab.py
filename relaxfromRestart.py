import os, sys
import shutil
import h5py
import numpy as np
from ase import Atoms
from ase.io import read
from ase.calculators.aims import Aims
from ase.build import stack, rotate
from ase.constraints import FixAtoms, FixConstraint, ExpCellFilter, Filter
from ase.visualize import view
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import PropertyNotImplementedError
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

XC = 'pbesol'
# SLAB_ORIENTATION = 0
# TETR_ORIENTATION = 0
L_SLAB_SIZE = 4
R_SLAB_SIZE = 2


class Region137(Filter):
    def __init__(self, atoms, indices, quasi_cell, tetra_axis, slab_axis, cell_factor=None):
        # def __init__(self, atoms, indices, tetra_axis, slab_axis, qmask_O_gr1, qmask_O_gr2, cell_factor=None):
        Filter.__init__(self, atoms, indices=range(len(atoms)))

        self.atoms = atoms
        self.atom_positions = atoms.get_positions()
        self.orig_cell = atoms.get_cell()

        self.deform_grad = np.asarray([1, 1, 1])
        self.cell_factor = cell_factor
        if self.cell_factor is None:
            self.cell_factor = float(len(atoms))

        self.spg_num = 137

        self.tetra_axis = tetra_axis
        self.slab_axis = slab_axis

        self.qcell = quasi_cell
        self.orig_qcell = quasi_cell.copy()

        self.qmask = np.full(len(atoms), False)
        self.qmask[indices] = True

        # self.qmask_O_gr1 = qmask_O_gr1
        # self.qmask_O_gr2 = qmask_O_gr2

        qatoms = Atoms(
            numbers=atoms.get_atomic_numbers()[self.qmask],
            cell=self.qcell,
            positions=atoms.get_positions()[self.qmask],
            pbc=True,
        )
        qmask_O_gr1 = np.full(len(qatoms), False)
        qmask_O_gr2 = np.full(len(qatoms), False)
        scaled_tetra_axis_positions = qatoms.get_scaled_positions()[:, self.tetra_axis]
        ref = scaled_tetra_axis_positions[qatoms.get_atomic_numbers() == 8][0]
        number_sc12 = len(qatoms) / 12
        frac_shifts = np.arange(0, 1, 0.5 / number_sc12)
        for i, (a, s) in enumerate(zip(qatoms.get_atomic_numbers(), scaled_tetra_axis_positions)):
            if a != 8:
                continue
            if any([np.isclose((s + x) % 1, ref) for x in frac_shifts]):
                qmask_O_gr1[i] = True
            else:
                qmask_O_gr2[i] = True

        self.qmask_O_gr1 = np.full(len(atoms), False)
        self.qmask_O_gr2 = np.full(len(atoms), False)
        self.qmask_O_gr1[indices] = qmask_O_gr1
        self.qmask_O_gr2[indices] = qmask_O_gr2

    # def _check_spg(self, atoms):
    #     print('_check_spg')
    #     print(atoms.get_atomic_numbers()[self.qmask])
    #     print(self.qcell)
    #     print(atoms.get_positions()[self.qmask])
    #     qatoms = Atoms(
    #         numbers=atoms.get_atomic_numbers()[self.qmask],
    #         cell=self.qcell,
    #         positions=atoms.get_positions()[self.qmask],
    #         pbc=True,
    #     )
    #     qstruct = AseAtomsAdaptor.get_structure(qatoms)
    #     print(qatoms)
    #     print(qstruct)
    #     analyzer = SpacegroupAnalyzer(qstruct, symprec=0.01)
    #     if analyzer.get_space_group_number() != self.spg_num:
    #         light_analyzer = SpacegroupAnalyzer(qstruct, symprec=0.1)
    #         if light_analyzer.get_space_group_number() != self.spg_num:
    #             raise RuntimeError(f'Spacegroup symmetry {self.spg_num} is broken.')
    #         print('WARNING: symprec was temporally set to 0.1 in order to find the wished symmetry')
    #     print()
    #     print()
    #     print()

    def get_positions(self):
        natoms = len(self.atoms)
        pos = np.zeros((natoms + 1, 3))
        pos[:natoms] = self.atom_positions
        pos[-1] = self.cell_factor * self.deform_grad
        return pos

    def set_positions(self, new, **kwargs):
        natoms = len(self.atoms)
        self.atom_positions[:] = new[:natoms]
        self.deform_grad = new[-1] / self.cell_factor
        self.atoms.set_positions(self.atom_positions, **kwargs)
        self.atoms.set_cell(self.orig_cell, scale_atoms=False)
        self.atoms.set_cell(np.dot(self.orig_cell, self.deform_grad.T), scale_atoms=True)

    def get_forces(self, apply_constraint=False):
        atoms_forces = self.atoms.get_forces()

        atoms_forces = np.dot(atoms_forces, np.diag(self.deform_grad))

        new_f = atoms_forces.copy()
        new_f[self.qmask] = np.mean(atoms_forces[self.qmask], axis=0)
        moving_O_f_symmetrized = (np.mean(atoms_forces[self.qmask_O_gr1, self.tetra_axis]) - np.mean(
            atoms_forces[self.qmask_O_gr2, self.tetra_axis])) / 2
        new_f[self.qmask_O_gr1, self.tetra_axis] += moving_O_f_symmetrized
        new_f[self.qmask_O_gr2, self.tetra_axis] -= moving_O_f_symmetrized
        atoms_forces = new_f

        stress = self.atoms.get_stress(voigt=False)
        volume = self.atoms.get_volume()
        virial = - volume * stress
        dg_inv = np.linalg.inv(np.diag(self.deform_grad))
        virial = np.dot(virial, dg_inv.T)
        virial = np.diag(virial).copy()

        natoms = len(self.atoms)
        forces = np.zeros((natoms + 1, 3))
        forces[:natoms] = atoms_forces
        forces[-1] = virial / self.cell_factor

        return forces

    def get_stress(self):
        raise PropertyNotImplementedError

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return (len(self.atoms) + 1)


class OutputBackupper:
    def __init__(self, calc, optimizer):
        self.calc = calc
        self.optimizer = optimizer
        self.absdir = os.path.abspath(self.calc.directory)

    def __call__(self):
        backup_dir = '{:03}'.format(len([name for name in os.listdir('.') if os.path.isdir()])+self.optimizer.get_number_of_steps())
        dst = os.path.join(self.absdir, backup_dir)
        os.makedirs(backup_dir, exist_ok=True)
        for f in ['geometry.in', 'control.in', 'parameters.ase', self.calc.out]:
            try:
                shutil.copy2(os.path.join(self.absdir, f), dst)
            except shutil.Error as e:
                print('WARNING: failed to copy file. {}'.format(e))


class HDF5Logger:
    def __init__(self, atoms, constrained_atoms, optimizer):
        self.atoms = atoms
        self.constrained_atoms = constrained_atoms
        self.optimizer = optimizer
        self.filename = os.path.join(os.curdir, 'log.h5')
        with h5py.File(self.filename, 'w') as f:
            f.create_dataset('atomic_numbers', data=atoms.get_atomic_numbers())
            gr = f.create_group('constraint_info')
            gr['tetra_axis'] = self.constrained_atoms.tetra_axis
            gr.create_dataset('mask', data=self.constrained_atoms.qmask)
            gr.create_dataset('mask_O_gr1', data=self.constrained_atoms.qmask_O_gr1)
            gr.create_dataset('mask_O_gr2', data=self.constrained_atoms.qmask_O_gr2)

    def __call__(self):
        with h5py.File(self.filename, 'r+') as f:
            gr = f.create_group(str(self.optimizer.get_number_of_steps()))
            gr.create_dataset('cell', data=self.atoms.get_cell()[:])
            gr.create_dataset('positions', data=self.atoms.get_positions())
            gr.create_dataset('positions_constrained', data=self.constrained_atoms.get_positions())
            gr.create_dataset('forces', data=self.atoms.get_forces())
            gr.create_dataset('forces_constrained', data=self.constrained_atoms.get_forces())
            gr.create_dataset('stress', data=self.atoms.get_stress(voigt=False))

if __name__ == "__main__":
    s137 = read("s137", format='cif')
    si = read("geometry.in")
    view(si)
    print(np.argmax(np.diag(si.get_cell())))
    k_grid = [4, 4, 4]
    k_grid[np.argmax(np.diag(si.get_cell()))] = 2
    parameters_dict = {
    'species_dir': '/dss/dssfs02/lwp-dss-0001/p6001/p6001-dss-0000/di98nom2/aimsSpeciesDefault/23_11_2018/tight',
    'xc': XC,
    'relativistic': 'atomic_zora scalar',
    'k_grid': k_grid,
    'compute_forces': '.true.',
    'compute_analytical_stress': '.true.',
    'final_forces_cleaned': '.true.',
    'override_illconditioning': '.true.',
    'mixer': 'pulay',
    'charge_mix_param': 0.2,
    }

    calc = Aims(
    aims_command='mpiexec -n 56 /dss/dssfs02/lwp-dss-0001/p6001/p6001-dss-0000/di98nom2/aimsBuild/current/aims',
    outfilename='out.x',
    tier=2,
    **parameters_dict,
    )

    # with h5py.File('previous.h5', 'r') as f:
    #     numbers = f['atomic_numbers'][...]
    #     cell = f['8/cell'][...]
    #     positions = f['8/positions'][...]
    #     slab = Atoms(numbers=numbers, cell=cell, positions=positions, pbc=True)
    #     qmask_O_gr1 = f['constraint_info/mask_O_gr1'][...]
    #     qmask_O_gr2 = f['constraint_info/mask_O_gr2'][...]

    # s137 = read(os.path.join(sys.path[0],"s137averaged"),format="cif")
    # si  = read(os.path.join(sys.path[0],"slab"),format="cif")

    si.set_calculator(calc)
    # constrained_slab = Region137(slab, range(48, 72), TETR_ORIENTATION, SLAB_ORIENTATION, qmask_O_gr1, qmask_O_gr2, 1)
    # constrained_slab = Region137(slab, range(48, 72), s137.get_cell()[:], TETR_ORIENTATION, SLAB_ORIENTATION, 1)
    constrained_slab = Region137(si, range(len(s137.get_atomic_numbers())), s137.get_cell()[:],
                             np.argmax(s137.get_cell()), np.argmax(si.get_cell()),
                             1)  # das *4 steht für 4 137er zellen unten auch
    """
    if all(s137.get_positions().round().flatten() == si.get_positions()[:len(s137.get_atomic_numbers())].round().flatten()):
    #print("treffer a, anfang")
    constrained_slab = Region137(si, range(len(s137.get_atomic_numbers())*4), s137.get_cell()[:],np.argmax(np.diag(s137.get_cell())), np.argmax(np.diag(si.get_cell())), 1) #das *4 steht für 4 137er zellen unten auch
    #if all(slarray[i].get_positions().round().flatten()== si[i].get_positions()[-len(slarray[i].get_atomic_numbers()):].round().flatten()):
    else:
    constrained_slab = Region137(si, range(len(si.get_atomic_numbers())-(len(s137.get_atomic_numbers()*4)),len(si.get_atomic_numbers())), s137.get_cell()[:],np.argmax(np.diag(s137.get_cell())), np.argmax(np.diag(si.get_cell())), 1)
    """
    # constrained_slab = Region137(slab, range(48, 72), TETR_ORIENTATION, SLAB_ORIENTATION, qmask_O_gr1, qmask_O_gr2, 1)
    # constrained_slab = Region137(slab, range(48, 72), s137.get_cell()[:], TETR_ORIENTATION, SLAB_ORIENTATION, 1)

    dynamics = BFGS(constrained_slab)
    dynamics.attach(Trajectory('slab.traj', 'w', si))
    dynamics.attach(OutputBackupper(calc, dynamics))
    dynamics.attach(HDF5Logger(si, constrained_slab, dynamics))
    dynamics.run(fmax=1e-3)
