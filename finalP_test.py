import numpy as np
import matplotlib.pyplot as plt

import sys

# OpenMM Imports
import simtk.openmm as mm
import simtk.unit as su
import simtk.openmm.app as app

import simtk.openmm.app.charmmpsffile as om_psf
import simtk.openmm.app.charmmparameterset as om_paramset

# ParmEd Imports
from parmed.charmm import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet
from parmed.openmm.reporters import StateDataReporter
from parmed import unit as u


# Load CHARMM parameter, psf and pdb files from the Solution Builder
drive_path = ''
om_params = om_paramset.CharmmParameterSet(drive_path + 'top_all36_prot.rtf', drive_path + 'par_all36m_prot.prm',
                               drive_path + 'top_all36_cgenff.rtf', drive_path + 'par_all36_cgenff.prm',
                               drive_path + 'ben.rtf', drive_path + 'ben.prm',
                               drive_path + 'toppar_water_ions.str')
om_solv = om_psf.CharmmPsfFile(drive_path + 'step3_pbcsetup.psf')
om_crds = app.PDBFile(drive_path + 'step3_pbcsetup.pdb')


# Fetching the corners of our solvant box for the PBCs.
coords = om_crds.positions
min_crds = [coords[0][0], coords[0][1], coords[0][2]]
max_crds = [coords[0][0], coords[0][1], coords[0][2]]

for coord in coords:
    min_crds[0] = min(min_crds[0], coord[0])
    min_crds[1] = min(min_crds[1], coord[1])
    min_crds[2] = min(min_crds[2], coord[2])
    max_crds[0] = max(max_crds[0], coord[0])
    max_crds[1] = max(max_crds[1], coord[1])
    max_crds[2] = max(max_crds[2], coord[2])

om_solv.setBox(max_crds[0]-min_crds[0], max_crds[1]-min_crds[1], max_crds[2]-min_crds[2])
print("Sidelengths of solvant box: ", om_solv.boxLengths)


# Simulation parameters
step_size = 2.0* su.femtosecond
sim_len = 1.0 * su.nanosecond
steps = round(sim_len / step_size)

report_time = 10.0*su.picosecond
report_steps = round(report_time / step_size)
print(f"Initialized| step_size = {step_size}, sim_len = {sim_len}, Num steps = {steps}, retport_time = {report_time}")


# Create our OpenMM system with the CHARMM parameters
system = om_solv.createSystem(om_params, nonbondedMethod=app.PME,
                              nonbondedCutoff=1 * su.nanometer,
                              constraints=app.HBonds)


# Create the integrator for the NVT ensemble
heat_bath = 303.15*su.kelvin
integrator = mm.LangevinIntegrator(heat_bath,              # Temperature of heat bath
                                   1.0/su.picosecond,      # Friction coefficient
                                   step_size)              # Time step


# Change to CUDA platform
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaDeviceIndex':1} # you can add other things like the precision here


# Create our Simulation object with our system and chosen integrator
sim = app.Simulation(om_solv.topology, system, integrator, platform, properties)


# Set the positions from our PDB file
sim.context.setPositions(om_crds.positions)


# Minimize Energy
print("Minimizing Energy")
sim.minimizeEnergy()


# Reporters
print("Appending Reporters")
sim.reporters.append(StateDataReporter(sys.stdout, round(steps/100), step=True,
                                       potentialEnergy=False, kineticEnergy=False,
                                       temperature=True, volume=True, density=False))

# Write out to DCD file
sim.reporters.append(app.DCDReporter('tryps_ben_solv.dcd', report_steps))


# Run (100ns)
sim.step(steps)
