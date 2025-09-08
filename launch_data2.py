import h5py
import numpy as np
import pathlib
from datetime import datetime
from launch_simulation import SimpleTableTennisLauncher
import os
#cutting trajectory till first contact

class SingleShotDatasetCreator:

    def __init__(self, xml_path):
        self.launcher = SimpleTableTennisLauncher(xml_path)
    
    def cut_trajectory_on_first_contact(self, positions, velocities, times, 
                                  hit_table, table_hit_position, 
                                  hit_ground, ground_hit_position):
    
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
     return os.path.exists(filename)

    def get_next_trajectory_index(self, filename):
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


if __name__ == "__main__":
    print("Saving single trajectory...")
    filename = save_single_trajectory()
    
    print(f"\n{'='*60}")
    print("Loading and analyzing saved trajectory...")
    data = load_and_analyze_trajectory(filename)
    
    if data is not None:
        print(f"\nTrajectory successfully loaded!")
        print(f"Data contains {len(data['positions'])} trajectory points")
