import h5py
import numpy as np
import pathlib
from datetime import datetime
from launch8 import SimpleTableTennisLauncher
#cutting trajectory till first contact

class SingleShotDatasetCreator:

    def __init__(self, xml_path):
        self.launcher = SimpleTableTennisLauncher(xml_path)
    """
    def cut_trajectory_on_first_contact(self, positions, velocities, times, 
                                      hit_table, table_hit_position, 
                                      hit_ground, ground_hit_position):
        
        if not hit_table and not hit_ground:
            print("  No contact detected, keeping full trajectory")
            return positions, velocities, times
        
        contact_pos = None
        contact_type = ""

        if hit_table and hit_ground:
            
            table_pos = np.array(table_hit_position)
            ground_pos = np.array(ground_hit_position)
            
            table_idx = self._find_contact_index(positions, table_pos)
            ground_idx = self._find_contact_index(positions, ground_pos)
            
            if table_idx < ground_idx:
                contact_pos = table_pos
                contact_type = "table"
                cutoff_idx = table_idx
            else:
                contact_pos = ground_pos
                contact_type = "ground"
                cutoff_idx = ground_idx
                
        elif hit_table:
            contact_pos = np.array(table_hit_position)
            contact_type = "table"
            cutoff_idx = self._find_contact_index(positions, contact_pos)
            
        elif hit_ground:
            contact_pos = np.array(ground_hit_position)
            contact_type = "ground"
            cutoff_idx = self._find_contact_index(positions, contact_pos)
        
        if cutoff_idx is not None and cutoff_idx > 0:
            print(f"  Cutting trajectory at {contact_type} contact (index {cutoff_idx})")
            print(f"  Original length: {len(positions)} -> Cut length: {cutoff_idx + 1}")
            
            return positions[:cutoff_idx + 1], velocities[:cutoff_idx + 1], times[:cutoff_idx + 1]
        else:
            print("  Contact index not found, keeping full trajectory")
            return positions, velocities, times
    
    def _find_contact_index(self, positions, contact_position, threshold=0.05):
        min_distance = float('inf')
        best_index = None
        
        for i, pos in enumerate(positions):
            distance = np.linalg.norm(pos - contact_position)
            if distance < min_distance:
                min_distance = distance
                best_index = i
                
            if distance < threshold:
                return i
        
        return best_index  """
    
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
    
     cutoff_idx = self._find_last_point_before_contact(positions, contact_pos)
    
     if cutoff_idx is not None:
        cut_positions = positions[:cutoff_idx + 1]
        cut_velocities = velocities[:cutoff_idx + 1]
        cut_times = times[:cutoff_idx + 1]
        
        contact_time = self._estimate_contact_time(
            positions[cutoff_idx], velocities[cutoff_idx], 
            contact_pos, times[cutoff_idx]
        )
        cut_positions = np.vstack([cut_positions, contact_pos])
        contact_velocity = velocities[cutoff_idx] * 0.5 
        cut_velocities = np.vstack([cut_velocities, contact_velocity])
        cut_times = np.append(cut_times, contact_time)
        
        print(f"  Added exact {contact_type} contact point at t={contact_time:.3f}s")
        print(f"  Contact position: ({contact_pos[0]:.3f}, {contact_pos[1]:.3f}, {contact_pos[2]:.3f})")
        
        return cut_positions, cut_velocities, cut_times
    
     return positions, velocities, times

    def _find_last_point_before_contact(self, positions, contact_position):
      min_distance = float('inf')
      best_index = 0
    
      for i, pos in enumerate(positions):
        distance = np.linalg.norm(pos - contact_position)
        if distance < min_distance:
            min_distance = distance
            best_index = i
        else:
            return best_index - 1 if best_index > 0 else 0
    
      return best_index

    def _estimate_contact_time(self, last_pos, last_vel, contact_pos, last_time):
     distance_to_contact = np.linalg.norm(contact_pos - last_pos)
     velocity_magnitude = np.linalg.norm(last_vel)
    
     if velocity_magnitude > 0:
        time_to_contact = distance_to_contact / velocity_magnitude
        return last_time + time_to_contact
     else:
        return last_time + 0.001  
     
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
        
        with h5py.File(filename, 'w') as hf:
            meta_group = hf.create_group('metadata')
            meta_group.attrs['creation_time'] = datetime.now().isoformat()
            
            launch_params = [phi, theta, rpm_tl, rpm_tr, rpm_bc]
            if use_system_effects:
                launch_params.extend([ramp_time, stroke_gain, pinch_diameter])
            
            meta_group.attrs['phi'] = phi
            meta_group.attrs['theta'] = theta
            meta_group.attrs['rpm_tl'] = rpm_tl
            meta_group.attrs['rpm_tr'] = rpm_tr
            meta_group.attrs['rpm_bc'] = rpm_bc
            meta_group.attrs['system_effects_enabled'] = use_system_effects
            meta_group.attrs['cut_at_contact'] = cut_at_contact
            
            if use_system_effects:
                meta_group.attrs['ramp_time'] = ramp_time
                meta_group.attrs['stroke_gain'] = stroke_gain
                meta_group.attrs['pinch_diameter'] = pinch_diameter
            
            meta_group.attrs['hit_table'] = result['hit_found_table']
            meta_group.attrs['hit_ground'] = result['hit_found_ground']
            meta_group.attrs['original_trajectory_length'] = len(result['trajectory'])
            meta_group.attrs['final_trajectory_length'] = len(positions)
            meta_group.attrs['original_simulation_time'] = result['times'][-1] if len(result['times']) > 0 else 0.0
            meta_group.attrs['final_simulation_time'] = times[-1] if len(times) > 0 else 0.0
            
            if result['hit_found_table']:
                meta_group.create_dataset('table_hit_position', data=result['hit_position_table'])
            if result['hit_found_ground']:
                meta_group.create_dataset('ground_hit_position', data=result['hit_position_ground'])
            
            traj_group = hf.create_group('trajectory_data')
            
            traj_group.create_dataset('positions', data=positions, 
                                    compression='gzip', compression_opts=6)
            traj_group.create_dataset('velocities', data=velocities,
                                    compression='gzip', compression_opts=6) 
            traj_group.create_dataset('times', data=times,
                                    compression='gzip', compression_opts=6)
            
            traj_group.create_dataset('launch_parameters', data=launch_params)
        
        print(f"\nTrajectory saved to: {filename}")
        print(f"Final trajectory contains {len(positions)} points over {times[-1]:.3f}s")
        return filename

    def load_single_shot_from_hdf5(self, filename="single_shot_trajectory.hdf5"):
        
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


def load_and_analyze_trajectory(filename="my_trajectory.hdf5"):
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
