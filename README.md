# Mujoco_Simulation and Dataset Generation


## Simulation

- launch_simulation.py (or simulation file) → Handles the MuJoCo-based physics simulation of table tennis ball trajectories, allowing parameterized launches and trajectory recording.
- Required packages to run this file.
  
    ```
    pip install mujoco numpy matplotlib h5py
    ```
 ### Usage

 - mujoco version 3.3.5

     
    ```
     python launch_simulation.py
    ```

  ### Features
  - Analyse Parameters with Plots alone
  - Analyse only Parameters
  - Analyse Parameters with 3D Viewer
  - Test System Parameter consistency with 3D Trajectory and Deviation Plots
    -  Noise based variations
    -  Realistic variations
---   
## Dataset Generation (Hdf5 Format)

  - launch_data2.py (or HDF5 generator file) → Processes the simulated trajectories and stores them in structured HDF5 datasets for training and analysis.
  - Trajectories saved in .hdf5 format under trajectories/ group.
  - Each entry contains launch parameters, positions, velocities, and metadata.

  ### Usage

   ```
    python launch_data2.py
   ```

---

## Future Plan

- Simulation (Implement realistic ball bounce mechanics after table contact).
- Dataset Generation (Done for single Trajectory , Automate batch trajectory generation with parameter sweeps).

---
