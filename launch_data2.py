import h5py
import numpy as np
from launch8 import SimpleTableTennisLauncher
import os
from tqdm import tqdm
import time

"""
 TABLE TENNIS TRAJECTORY DATASET CREATION
 --------------------------------------------
Automated trajectory generation :
* Runs table tennis simulations with 5 input parameters (phi, theta, rpm_tl, rpm_tr, rpm_bc)
* Records complete ball trajectories until first contact (table or ground)
* Stores ONLY trajectories that hit the table first (filters out ground-first hits)
* Saves data in HDF5 format 

- Cuts trajectories at exact contact point for clean datasets
-  HDF5 storage
- Error handling and trajectory filtering

"""
class SingleShotDatasetCreator:
    def __init__(self, xml_path):
        self.launcher = SimpleTableTennisLauncher(xml_path)
    
    def cut_trajectory_on_first_contact(self, positions, velocities, times, 
                                  hit_table, table_hit_position, 
                                  hit_ground, ground_hit_position):
     """ 
        1. Determine which contact happened first (table vs ground)
        2. Find closest trajectory point to actual contact position
        3. Cut all arrays (positions, velocities, times) at that index
        4. Replace final point with exact contact coordinates

    """
    
     if not hit_table and not hit_ground:
        return positions, velocities, times
    
     contact_pos = None
     contact_type = ""
    
     if hit_table:
        contact_pos = np.array(table_hit_position)
        contact_type = "table"
     elif hit_ground:
        contact_pos = np.array(ground_hit_position)
        contact_type = "ground"
     cutoff_idx = self._find_closest_point_to_contact(positions, contact_pos)
    
     if cutoff_idx is not None:
        cut_positions = positions[:cutoff_idx + 1]
        cut_velocities = velocities[:cutoff_idx + 1]
        cut_times = times[:cutoff_idx + 1]
        
        cut_positions[-1] = contact_pos
        
        print(f"  Cut trajectory at {contact_type} contact (index {cutoff_idx})")
        print(f"  Exact contact position: ({contact_pos[0]:.3f}, {contact_pos[1]:.3f}, {contact_pos[2]:.3f})")
        print(f"  Final trajectory length: {len(cut_positions)} points")
        
        return cut_positions, cut_velocities, cut_times
    
     return positions, velocities, times

    def _find_closest_point_to_contact(self, positions, contact_position):
      #Finds trajectory index closest to contact point
      min_distance = float('inf')
      best_index = 0
    
      for i, pos in enumerate(positions):
        distance = np.linalg.norm(pos - contact_position)
        if distance < min_distance:
            min_distance = distance
            best_index = i
    
      return best_index

    def save_single_shot_to_hdf5(self, phi, theta, rpm_tl, rpm_tr, rpm_bc, 
                                filename="single_shot_trajectory.hdf5",
                                use_system_effects=False, ramp_time=3.0,
                                stroke_gain=0.10, pinch_diameter=37.4,
                              cut_at_contact=True):
        """
        1. Run MuJoCo simulation with given parameters
        2. Extract trajectory data (positions, velocities, times)
        3. Apply trajectory cutting if requested
        4. Filter out ground-first trajectories (only save table-first)
        5. Structure data in HDF5 format with metadata
        6. Append to existing dataset or create new file
        
        HDF5 Structure:
        /trajectories/
        ├── 000000/  (trajectory index)
        │   ├── launch_parameters  [phi, theta, rpm_tl, rpm_tr, rpm_bc]
        │   ├── positions         [N×3 array: x,y,z coordinates]
        │   ├── velocities        [N×3 array: vx,vy,vz components]  
        │   ├── times             [N×1 array: simulation timestamps]
        │   ├── table_hit_position [3×1 array: contact coordinates]
        │   └── attributes       (metadata: hit_table, hit_ground, etc.)
        """
        
        print(f"Generating trajectory for parameters:")
        print(f"  φ = {phi:.4f} rad ({phi*180/np.pi:.1f}°)")
        print(f"  θ = {theta:.4f} rad ({theta*180/np.pi:.1f}°)")
        print(f"  RPMs: TL={rpm_tl}, TR={rpm_tr}, BC={rpm_bc}")
        
        result = self.launcher.get_hitting_position(
            phi, theta, rpm_tl, rpm_tr, rpm_bc,
            use_system_effects=use_system_effects,
            ramp_time=ramp_time,
            stroke_gain=stroke_gain,
            pinch_diameter=pinch_diameter
        )
        
        if len(result['trajectory']) == 0:
            print("ERROR: No trajectory data generated!")
            return None
    
        if result['hit_found_ground'] and not result['hit_found_table']:
           print("SKIPPING: Ball hit ground before table - not storing trajectory")
           return None
        
        print(f"Trajectory generated successfully!")
        print(f"  Original trajectory length: {len(result['trajectory'])} points")
        print(f"  Original simulation time: {result['times'][-1]:.3f}s")
        print(f"  Hit table: {result['hit_found_table']}")
        print(f"  Hit ground: {result['hit_found_ground']}")
        
        positions = result['trajectory']
        velocities = result['velocities']  
        times = result['times']
        
        if cut_at_contact:
            print("\nCutting trajectory at first contact...")
            positions, velocities, times = self.cut_trajectory_on_first_contact(
                positions, velocities, times,
                result['hit_found_table'], result['hit_position_table'],
                result['hit_found_ground'], result['hit_position_ground']
            )
        
        dataset_exists = self.check_dataset_exists(filename)

        if dataset_exists:
          traj_index = self.get_next_trajectory_index(filename)
          file_mode = 'a' 
          print(f"Appending as trajectory_{traj_index}")
        else:
          traj_index = 0
          file_mode = 'w'  
          print(f"Creating new dataset")

        with h5py.File(filename, file_mode) as hf:

         
         if file_mode == 'w':
          trajectories_group = hf.create_group('trajectories')
         else:
          trajectories_group = hf['trajectories']
         launch_params = [phi, theta, rpm_tl, rpm_tr, rpm_bc]
        
        
         traj_group = trajectories_group.create_group(f'{traj_index:06d}')
         
         traj_group.attrs['phi'] = phi
         traj_group.attrs['theta'] = theta
         traj_group.attrs['rpm_tl'] = rpm_tl
         traj_group.attrs['rpm_tr'] = rpm_tr
         traj_group.attrs['rpm_bc'] = rpm_bc
        
         traj_group.create_dataset('launch_parameters', data=launch_params)
         traj_group.create_dataset('positions', data=positions, compression='gzip', compression_opts=6)
         traj_group.create_dataset('velocities', data=velocities, compression='gzip', compression_opts=6)
         traj_group.create_dataset('times', data=times, compression='gzip', compression_opts=6)
         

         traj_group.attrs['hit_table'] = result['hit_found_table']
         traj_group.attrs['hit_ground'] = result['hit_found_ground']
         traj_group.attrs['original_trajectory_length'] = len(result['trajectory'])
         traj_group.attrs['final_trajectory_length'] = len(positions)
         traj_group.attrs['original_simulation_time'] = result['hit_found_ground']
         traj_group.attrs['final_simulation_time'] = times[-1] if len(times) > 0 else 0.0 
         
         if result['hit_found_table']:
          traj_group.create_dataset('table_hit_position', data=result['hit_position_table'])
          traj_group.attrs['hit_table'] = True
         if result['hit_found_ground']:
          traj_group.create_dataset('ground_hit_position', data=result['hit_position_ground'])
          traj_group.attrs['hit_ground'] = True

         traj_group.attrs['hit_table'] = result['hit_found_table']
         traj_group.attrs['hit_ground'] = result['hit_found_ground']

         return filename
    def load_single_shot_from_hdf5(self, filename="trajectory_dataset.hdf5"):
        
        print(f"Loading trajectory from: {filename}")
        print("=" * 50)
        
        try:
            with h5py.File(filename, 'r') as hf:
                meta = hf['metadata']
                print("METADATA:")
                print(f"  Creation time: {meta.attrs['creation_time']}")
                
                traj_data = hf['trajectory_data']
                launch_params = traj_data['launch_parameters'][:]
                phi, theta, rpm_tl, rpm_tr, rpm_bc = launch_params[:5]
                
                print(f"  Launch parameters:")
                print(f"    φ = {phi:.4f} rad ({phi*180/np.pi:.1f}°)")
                print(f"    θ = {theta:.4f} rad ({theta*180/np.pi:.1f}°)")
                print(f"    RPMs: TL={int(rpm_tl)}, TR={int(rpm_tr)}, BC={int(rpm_bc)}")
                
                if 'cut_at_contact' in meta.attrs and meta.attrs['cut_at_contact']:
                    print(f"  Trajectory cut at first contact: YES")
                    if 'original_trajectory_length' in meta.attrs:
                        orig_len = meta.attrs['original_trajectory_length']
                        final_len = meta.attrs['final_trajectory_length']
                        print(f"  Original length: {orig_len} -> Final length: {final_len}")
                else:
                    print(f"  Trajectory cut at first contact: NO")
                
                positions = traj_data['positions'][:]
                velocities = traj_data['velocities'][:]
                times = traj_data['times'][:]
                
                print(f"\nRESULTS:")
                print(f"  Hit table: {meta.attrs['hit_table']}")
                print(f"  Hit ground: {meta.attrs['hit_ground']}")
                print(f"  Final simulation time: {meta.attrs['final_simulation_time']:.3f}s")
                
                print(f"\nTRAJECTORY DATA:")
                print(f"  Launch Parameters shape: {launch_params.shape}")
                print(f"  Positions shape: {positions.shape}")
                print(f"  Velocities shape: {velocities.shape}")
                print(f"  Times shape: {times.shape}")
                print(f"  Time range: {times[0]:.3f} - {times[-1]:.3f}s")
                
                print(f"\n  First 3 positions:")
                for i in range(min(3, len(positions))):
                    print(f"    t={times[i]:.3f}s: pos=({positions[i,0]:.3f}, {positions[i,1]:.3f}, {positions[i,2]:.3f})")
                print(f"\n  Last 3 positions:")
                for i in range(max(0, len(positions)-3), len(positions)):
                    print(f"    t={times[i]:.3f}s: pos=({positions[i,0]:.3f}, {positions[i,1]:.3f}, {positions[i,2]:.3f})")
                
                return {
                    'positions': positions,
                    'velocities': velocities,
                    'times': times,
                    'launch_parameters': launch_params,
                    'metadata': dict(meta.attrs)
                }
                
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def check_dataset_exists(self, filename):
     #file existence validation
     return os.path.exists(filename)

    def get_next_trajectory_index(self, filename):
     #Scan existing trajectory keys, find maximum index, increment by 1
     try:
        with h5py.File(filename, 'r') as hf:
            if 'trajectories' in hf:
                existing = list(hf['trajectories'].keys())

                if existing:
                    nums = [int(traj) for traj in existing if traj.isdigit()]
                    return max(nums) + 1 if nums else 0
            return 0
     except:
        return 0
def save_single_trajectory():
    #for single trajectory generation
    
    xml_path = 'balllauncher/balllaunch.xml'
    creator = SingleShotDatasetCreator(xml_path)

    phi = float(input("Enter phi: "))
    theta = float(input("Enter theta: "))
    rpm_tl = int(input("Enter rpm_tl: "))
    rpm_tr = int(input("Enter rpm_tr: "))
    rpm_bc = int(input("Enter rpm_bc: "))
    
    cut_choice = input("Cut trajectory at first contact? (y/n, default y): ").strip().lower()
    cut_at_contact = cut_choice != 'n'
    
    filename = creator.save_single_shot_to_hdf5(
        phi=phi, theta=theta, 
        rpm_tl=rpm_tl, rpm_tr=rpm_tr, rpm_bc=rpm_bc,
        filename="trajectory_dataset.hdf5",
        use_system_effects=False,
        cut_at_contact=cut_at_contact
    )
    
    return filename
    

def load_and_analyze_trajectory(filename="trajectory_dataset.hdf5"):

    xml_path = 'balllauncher/balllaunch.xml'
    creator = SingleShotDatasetCreator(xml_path)
    
    data = creator.load_single_shot_from_hdf5(filename)
    
    return data


def generate_small_dataset():
    #Batch wise trajectory
    """
     BATCH GENERATION MODE: Automated dataset creation with parameter sweeps
     Generate large training datasets for machine learning
    - Nested parameter loops (phi × theta × rpm combinations)
    - Success/skip/error counting
    
    Dataset Coverage:
    - phi: -0.2 to 0.2 rad (horizontal angles, 5 values)
    - theta: 0.1 to 0.4 rad (vertical angles, 4 values)  
    - rpm: 500 to 1000 (all 3 wheels, 11 values each)
    - Total: 5 × 4 × 11³ = 26,620 combinations
    """

    xml_path = 'balllauncher/balllaunch.xml'
    creator = SingleShotDatasetCreator(xml_path)
    out_file = "small_trajectory_dataset.hdf5"

    # Small test ranges
    phi_values = np.arange(-0.2, 0.21, 0.1)    
    theta_values = np.arange(0.1, 0.5, 0.1)      
    rpm_values = np.arange(500, 1001, 50)       

    total_combinations = len(phi_values) * len(theta_values) * len(rpm_values)**3
    print(f"Total combinations: {total_combinations}")
    
    pbar = tqdm(total=total_combinations, desc="Generating trajectories")
    success, skipped, errors = 0, 0, 0
    start_time = time.time()

    for phi in phi_values:
        for theta in theta_values:
            for rpm_tl in rpm_values:
                for rpm_tr in rpm_values:
                    for rpm_bc in rpm_values:
                        try:
                            result = creator.save_single_shot_to_hdf5(
                                phi=phi, theta=theta,
                                rpm_tl=rpm_tl, rpm_tr=rpm_tr, rpm_bc=rpm_bc,
                                filename=out_file,
                                use_system_effects=False,
                                cut_at_contact=True
                            )
                            if result is not None:
                                success += 1
                            else:
                                skipped += 1
                        except Exception as e:
                            print(f"Error [{phi}, {theta}, {rpm_tl}, {rpm_tr}, {rpm_bc}]: {e}")
                            errors += 1
                        pbar.update(1)
    pbar.close()
    total = success + skipped + errors
    elapsed = time.time() - start_time

    print(f"\nCompleted {total}/{total_combinations} simulations in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Success: {success} | Skipped: {skipped} | Errors: {errors}")
    print(f"Final dataset: {out_file}")

    return out_file

if __name__ == "__main__":
    generate_small_dataset()
