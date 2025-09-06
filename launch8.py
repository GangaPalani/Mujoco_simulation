# launch.py
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from mpl_toolkits.mplot3d import Axes3D

class SimpleTableTennisLauncher:
    
    def __init__(self, xml_file):
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_free'
        )
        self.table_size = (2.74, 1.525, 0.76) 
        self.table_height = 0.76
        self.hit_points = []  
        self.print_environment_info()
        
    def print_environment_info(self):
        print("\n=== Environment Info ===")
        default_launch = [-0.5, 0.0, 1.25]
        print(f"Default launch position: {default_launch}")
        qposadr = self.model.jnt_qposadr[self.ball_joint_id]
        ball_init = self.data.qpos[qposadr:qposadr+3]
        print(f"Ball initial position from sim: {ball_init}")
        geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
              for i in range(self.model.ngeom)]
        print(geom_names)
        for i, name in enumerate(geom_names):
          if "table" in name:
            gsize = self.model.geom_size[i]
            gpos = self.model.geom_pos[i]
            print(f"Table geom '{name}': size={gsize}, pos={gpos}")
            print(f"   Full size: {2*gsize[0]:.3f} x {2*gsize[1]:.3f} m")
            print(f"   Height (top surface): {gpos[2] + gsize[2]:.3f} m") 

    def parameters_to_velocity_spin(self, phi, theta, rpm_tl, rpm_tr, rpm_bc):
        v_base = self.implement_aimy_rpm_conversion(rpm_tl, rpm_tr, rpm_bc)
        vx = v_base * np.cos(theta) * np.cos(phi)
        vy = v_base * np.cos(theta) * np.sin(phi)
        vz = v_base * np.sin(theta)
        print("velocities:", vx, vy, vz)
        spin_x = (rpm_bc - (rpm_tl + rpm_tr)/2) * 0.1047   #(2*pi / 60)
        spin_y = (rpm_tr - rpm_tl) * 0.1047
        spin_z = 0.0    
        return np.array([vx, vy, vz], dtype=float), np.array([spin_x, spin_y, spin_z], dtype=float)
    
    def implement_aimy_rpm_conversion(self, rpm_tl, rpm_tr, rpm_bc):
        """rpm_curves = {
            'rpm_tl': [0, 489, 1278, 1865, 2354, 2694, 2959, 3170, 3321, 3434, 3524, 3615, 3688, 3931, 3968, 3970],
            'rpm_tr': [0, 470, 1250, 1828, 2325, 2667, 2939, 3149, 3299, 3418, 3509, 3600, 3678, 3937, 3961, 3963], 
            'rpm_bc': [0, 524, 1291, 1900, 2369, 2722, 2977, 3184, 3331, 3435, 3529, 3613, 3681, 3933, 3957, 3960],
            'actuation': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        }
        
        def interpolate_rpm(motor_type, actuation_level):
            return np.interp(actuation_level, rpm_curves['actuation'], rpm_curves[motor_type]) """
        
        
        avg_rpm = (rpm_tl + rpm_tr + rpm_bc) / 3
        factor = 0.00523 #(2*pi*50mm(0.05)/60)
        return avg_rpm * factor

    def launch_ball_in_mujoco(self, initial_pos, initial_vel, spin=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        
        qposadr = self.model.jnt_qposadr[self.ball_joint_id]
        qveladr = self.model.jnt_dofadr[self.ball_joint_id]
        self.data.qpos[qposadr:qposadr+3] = np.array(initial_pos, dtype=float)
        self.data.qpos[qposadr+3:qposadr+7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[qveladr:qveladr+3] = np.array(initial_vel, dtype=float)
        if spin is not None:
            self.data.qvel[qveladr+3:qveladr+6] = np.array(spin, dtype=float)
        else:
            self.data.qvel[qveladr+3:qveladr+6] = np.zeros(3, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step1(self.model, self.data)
    
    def get_hitting_position(self, phi, theta, rpm_tl, rpm_tr, rpm_bc, launch_pos=None, max_time=4.0,
                        use_system_effects=False, ramp_time=3.0, stroke_gain=0.10, pinch_diameter=37.4):
     if launch_pos is None:
        launch_pos = [-0.5, 0.0, 1.25]
     v0, spin = self.parameters_to_velocity_spin(phi, theta, rpm_tl, rpm_tr, rpm_bc)
     if use_system_effects:
        v0 = self.apply_ramp_time_effects(v0, ramp_time)
        spin = self.apply_stroke_gain_effects(spin, stroke_gain)
        v0, spin = self.apply_pinch_diameter_effects(v0, spin, pinch_diameter)
     self.launch_ball_in_mujoco(launch_pos, v0, spin)
     positions, velocities, times = [], [], []
     hit_position_table = hit_position_ground = None
     hit_found_table = hit_found_ground = False

     qposadr = self.model.jnt_qposadr[self.ball_joint_id]
     qveladr = self.model.jnt_dofadr[self.ball_joint_id]

     while self.data.time < max_time:
        mujoco.mj_step(self.model, self.data)
        current_pos = self.data.qpos[qposadr:qposadr+3].copy()
        positions.append(current_pos)
        velocities.append(self.data.qvel[qveladr:qveladr+3].copy())
        times.append(self.data.time)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            if (('ball' in geom1 or 'ball' in geom2) and ('table_geom' in geom1 or 'table_geom' in geom2)) and not hit_found_table:
                hit_position_table = contact.pos.copy()
                hit_found_table = True
                print(f"Table hit at: {hit_position_table}")
            if (('ball' in geom1 or 'ball' in geom2) and ('ground' in geom1 or 'ground' in geom2)) and not hit_found_ground:
                hit_position_ground = contact.pos.copy()
                hit_found_ground = True
                print(f"Ground hit at: {hit_position_ground}")
        if hit_found_ground or current_pos[2] < 0.01 or abs(current_pos).max() > 15.0:
            if current_pos[2] < 0.01 or abs(current_pos).max() > 15.0:
                print("Ball went out of bounds")
            break
     positions = np.array(positions) if positions else np.array([])
     times = np.array(times) if times else np.array([])
     velocities = np.array(velocities) if velocities else np.array([])
     return {
        'hit_position_table': hit_position_table.tolist() if hit_position_table is not None else None,
        'hit_position_ground': hit_position_ground.tolist() if hit_position_ground is not None else None,
        'hit_found_table': hit_found_table,
        'hit_found_ground': hit_found_ground,
        'trajectory': positions,
        'times': times,
        'velocities': velocities,
        'parameters_used': {
            'phi': phi, 'theta': theta, 'rpm_tl': rpm_tl, 'rpm_tr': rpm_tr, 'rpm_bc': rpm_bc, 'launch_position': launch_pos,
            'system_params': {'ramp_time': ramp_time, 'stroke_gain': stroke_gain, 'pinch_diameter': pinch_diameter} if use_system_effects else None
        }
    }
 
    def plot_trajectory(self, positions, times, hit_points=None, title="Ball Trajectory"):
        if len(positions) == 0:
            print("No trajectory data to plot")
            return
        table_length = 2.74  
        table_width  = 1.525 
        table_top    = self.table_height  # 0.76 m
        net_height   = 0.1525  # m
        net_x        = table_length / 2   # 1.37 m
        plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(positions[:,0], positions[:,2], '-b', linewidth=2, label='Ball trajectory')
        ax1.fill_between([0, table_length], 0, table_top,
                       color='lightgray', alpha=0.5, label='Table')
        ax1.plot([net_x, net_x], [table_top, table_top+net_height],
               'k-', linewidth=3, label='Net')
        if hit_points is not None and len(hit_points) > 0:
            ax1.scatter(hit_points[:,0], hit_points[:,2],
                        color='red', s=80, marker='o', zorder=5, label='Hits')
            for i, hit in enumerate(hit_points):
                ax1.annotate(f'Hit {i+1}', (hit[0], hit[2]), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, color='red')
        ax1.set_xlim(-0.5, table_length + 0.5) 
        ax1.set_ylim(0, table_top + net_height + 0.5)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Z Position (m)')
        ax1.set_title(f'{title} - Side View (X-Z)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(positions[:,0], positions[:,1], '-r', linewidth=2, label='Ball trajectory')
        table_corners_x = [0, table_length, table_length, 0, 0]
        table_corners_y = [-table_width/2, -table_width/2, table_width/2, table_width/2, -table_width/2]
        ax2.plot(table_corners_x, table_corners_y, 'k-', linewidth=3, label='Table edge')
        ax2.axvline(net_x, color='gray', linestyle='--', linewidth=2, label='Net')
        if hit_points is not None and len(hit_points) > 0:
            ax2.scatter(hit_points[:,0], hit_points[:,1],
                        color='red', s=80, marker='o', zorder=5, label='Hits')
            for i, hit in enumerate(hit_points):
                ax2.annotate(f'Hit {i+1}', (hit[0], hit[1]), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, color='red')
        ax2.set_xlim(-0.5, table_length + 0.5)
        ax2.set_ylim(-table_width/2 - 0.2, table_width/2 + 0.2)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title(f'{title} - Top View (X-Y)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
    
    def run_3d_viewer(self, phi, theta, rpm_tl, rpm_tr, rpm_bc, launch_pos=None):
        if launch_pos is None:
            launch_pos = [-0.5,0.0,1.25]
        v0, spin = self.parameters_to_velocity_spin(phi, theta, rpm_tl, rpm_tr, rpm_bc)
        self.launch_ball_in_mujoco(launch_pos, v0, spin)
        print(f"\nLaunching 3D viewer with parameters:")
        print(f"   φ = {phi:.4f} rad ({phi*180/np.pi:.1f}°)") 
        print(f"   θ = {theta:.4f} rad ({theta*180/np.pi:.1f}°)")  
        print(f"   RPM: TL={rpm_tl}, TR={rpm_tr}, BC={rpm_bc}")
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                paused = False
                
                def key_callback(keycode):
                    nonlocal paused
                    if chr(keycode) == ' ':
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif chr(keycode) == 'r':
                        self.launch_ball_in_mujoco(launch_pos, v0, spin)
                        print("Ball relaunched!")
                
                viewer.user_callback = key_callback
                while viewer.is_running():
                    step_start = time.time()
                    if not paused:
                        with viewer.lock():
                            mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except Exception as e:
            print(f"Passive viewer error: {e}")
            print("Falling back to blocking viewer...")
            try:
                mujoco.viewer.launch(self.model, self.data)
            except Exception as e2:
                print(f"Blocking viewer error: {e2}")

    def analyze_parameters(self, phi, theta, rpm_tl, rpm_tr, rpm_bc,show_plot=True, show_3d=False,use_system_effects=False,
                      ramp_time=3.0, stroke_gain=0.10, pinch_diameter=37.4):
    
        print(f"\nANALYZING PARAMETERS:")
        print(f"   φ (horizontal): {phi:.4f} rad ({phi*180/np.pi:.1f}°)")
        print(f"   θ (vertical): {theta:.4f} rad ({theta*180/np.pi:.1f}°)")
        print(f"   RPM - TL: {rpm_tl}, TR: {rpm_tr}, BC: {rpm_bc}")
        if use_system_effects:
          print(f"   System Effects: ENABLED")
          print(f"   System Params: Ramp={ramp_time}s, Stroke={stroke_gain}, Pinch={pinch_diameter}mm")
          result = self.get_hitting_position(
            phi, theta, rpm_tl, rpm_tr, rpm_bc, 
            ramp_time=ramp_time, stroke_gain=stroke_gain, pinch_diameter=pinch_diameter,use_system_effects=True
        )
        else:
          print(f"System Effects: DISABLED ")
          result = self.get_hitting_position(phi, theta, rpm_tl, rpm_tr, rpm_bc,use_system_effects=False)
        print("-" * 50)
        print("RESULTS:")
        
        if result['hit_found_table']:
            ideal_landing = np.array(result['hit_position_table'])
            realistic_landing, scatter_info = self.system_parameter_effects(
                ideal_landing, ramp_time, stroke_gain, pinch_diameter
            )
            result['hit_position_table_realistic'] = realistic_landing.tolist()
            result['scatter_info'] = scatter_info
            
            print(f"TABLE HIT (IDEAL): ({ideal_landing[0]:.3f}, {ideal_landing[1]:.3f}, {ideal_landing[2]:.3f})")
            print(f"TABLE HIT (REALISTIC): ({realistic_landing[0]:.3f}, {realistic_landing[1]:.3f}, {realistic_landing[2]:.3f})")
            print(f"Expected scatter: σₓ={scatter_info['sigma_x']*1000:.1f}mm, σᵧ={scatter_info['sigma_y']*1000:.1f}mm")
            side = "Own" if ideal_landing[1] < 0 else "Opponent"
            direction = "Left" if ideal_landing[0] < 0 else "Right"
            print(f"  Side: {side}")
            print(f"  Direction: {direction}")
            self.hit_points.append((ideal_landing[0], ideal_landing[1]))
        else:
            print("NO TABLE HIT - Ball missed table")
        if result['hit_found_ground']:
            hit_pos_ground = result['hit_position_ground']
            print(f"GROUND HIT FOUND:")
            print(f"  Position: ({hit_pos_ground[0]:.3f}, {hit_pos_ground[1]:.3f}, {hit_pos_ground[2]:.3f})")
            distance_from_launch = hit_pos_ground[0] - (-0.8)
            print(f"  Distance from launch: {distance_from_launch:.3f} meters")
        else:
            print("NO GROUND HIT - Ball still in air")

        print(f"Trajectory length: {len(result['trajectory'])} points")
        print(f"Simulation time: {result['times'][-1]:.2f}s" if len(result['times']) > 0 else "No time data")

        if show_plot and len(result['trajectory']) > 0:
            title = f"φ={phi:.3f}rad θ={theta:.3f}rad"
            hit_points = []
            if result['hit_found_table']:
                hit_points.append(result['hit_position_table'])
            if result['hit_found_ground']:
                hit_points.append(result['hit_position_ground'])
            
            hit_points = np.array(hit_points, dtype=float) if hit_points else np.array([])
            self.plot_trajectory(result['trajectory'], result['times'], hit_points, title)

        if show_3d:
            self.run_3d_viewer(phi, theta, rpm_tl, rpm_tr, rpm_bc)
        
        return result

    def apply_ramp_time_effects(self, velocity, ramp_time):
     if ramp_time < 0.5:
        velocity_error = np.random.normal(0, 0.05) 
     elif ramp_time > 5.0:
        velocity_error = np.random.normal(0, 0.03) 
     else:
        velocity_error = np.random.normal(0, 0.01) 
     return velocity * (1 + velocity_error)

    def apply_stroke_gain_effects(self, spin, stroke_gain):
     if stroke_gain < 0.1:
        spin_error = np.random.normal(0, 0.1)
     elif stroke_gain > 1.0:
        spin_error = np.random.normal(0, 0.5)  
     else:
        spin_error = np.random.normal(0, 0.2)  
     return spin + spin_error

    def apply_pinch_diameter_effects(self, velocity, spin, pinch_diameter):
     optimal_pinch = 37.4
     deviation = abs(pinch_diameter - optimal_pinch)
    
     velocity_loss = deviation * 0.01  
     velocity_modified = velocity * (1 - velocity_loss)
    
     spin_efficiency = 1.0 - deviation * 0.02 
     spin_modified = spin * spin_efficiency
    
     return velocity_modified, spin_modified

     
    def test_system_parameter_consistency(self, phi, theta, rpm_tl, rpm_tr, rpm_bc, 
                                     num_shots=10, use_physics_variations=False,
                                     ramp_time=3.0, stroke_gain=0.10, pinch_diameter=37.4, 
                                     plot_results=True):
    
     print(f"\n TESTING {num_shots} SHOTS ") 
     print(f"Parameters: φ={phi:.3f}, θ={theta:.3f}, RPMs=[{rpm_tl},{rpm_tr},{rpm_bc}]")
     print(f"System: Ramp={ramp_time}s, Stroke={stroke_gain}, Pinch={pinch_diameter}mm")
     print("-" * 60)
    
     landing_positions = []
     ideal_landing = None
     scatter_info = None
    
     if use_physics_variations:
        print("Using PHYSICS VARIATIONS (each shot has different MuJoCo trajectory)")
        
        for shot in range(num_shots):
            result = self.get_hitting_position(
                phi, theta, rpm_tl, rpm_tr, rpm_bc, 
                ramp_time=ramp_time, stroke_gain=stroke_gain, pinch_diameter=pinch_diameter , use_system_effects=True
            )
            
            if result['hit_found_table']:
                landing_positions.append(result['hit_position_table'])
                print(f"Shot {shot+1:2d}: ({result['hit_position_table'][0]:.3f}, {result['hit_position_table'][1]:.3f}, {result['hit_position_table'][2]:.3f})")
                
                if ideal_landing is None:
                    ideal_landing = np.array(result['hit_position_table'])
            else:
                print(f"Shot {shot+1:2d}: MISSED TABLE")
        if landing_positions:
            positions_array = np.array(landing_positions)
            scatter_info = {
                'sigma_x': np.std(positions_array[:, 0]),
                'sigma_y': np.std(positions_array[:, 1])
            }
            
     else:
        print("Using NOISE-BASED VARIATIONS (faster statistical approach)")
        base_result = self.get_hitting_position(phi, theta, rpm_tl, rpm_tr, rpm_bc,use_system_effects=False)
        
        if not base_result['hit_found_table']:
            print("ERROR: No table hit found in base simulation")
            return []
        
        ideal_landing = np.array(base_result['hit_position_table'])
        print(f"Ideal landing position: ({ideal_landing[0]:.3f}, {ideal_landing[1]:.3f}, {ideal_landing[2]:.3f})")
        print("-" * 60)

        for shot in range(num_shots):
            realistic_landing, scatter_info = self.system_parameter_effects(
                ideal_landing, ramp_time, stroke_gain, pinch_diameter
            )
            
            landing_positions.append(realistic_landing)
            print(f"Shot {shot+1:2d}: ({realistic_landing[0]:.3f}, {realistic_landing[1]:.3f}, {realistic_landing[2]:.3f})")
    
     if landing_positions:
        positions = np.array(landing_positions)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        print("-" * 60)
        print(f"STATISTICS:")
        
        if ideal_landing is not None:
            print(f"Reference(First shot): ({ideal_landing[0]:.3f}, {ideal_landing[1]:.3f}, {ideal_landing[2]:.3f})")
        
        print(f"Mean:      ({mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f})")
        print(f"Std:       ({std_pos[0]*1000:.1f}mm, {std_pos[1]*1000:.1f}mm, {std_pos[2]*1000:.1f}mm)")
        
        if scatter_info:
            print(f"Scatter:   σₓ={scatter_info['sigma_x']*1000:.1f}mm, σᵧ={scatter_info['sigma_y']*1000:.1f}mm")
        hit_rate = len(landing_positions) / num_shots * 100
        print(f"Hit Rate:  {hit_rate:.1f}% ({len(landing_positions)}/{num_shots} shots)")
        if plot_results:
            try:
                if ideal_landing is not None and scatter_info is not None:
                    self.plot_consistency_test_results(positions, ideal_landing, scatter_info)
                else:
                    print("Cannot plot: Missing reference data")
            except Exception as e:
                print(f"Plotting error: {e}")
                print("Continuing without plots...")
     else:
        print("No successful shots to analyze")
    
     return landing_positions

    def plot_consistency_test_results(self, positions, ideal_pos, scatter_info):
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    
     table_length = 2.74
     table_width = 1.525
    
     table_corners_x = [0, table_length, table_length, 0, 0]
     table_corners_y = [-table_width/2, -table_width/2, table_width/2, table_width/2, -table_width/2]
     ax1.plot(table_corners_x, table_corners_y, 'k-', linewidth=3, label='Table edge')
     ax1.axvline(table_length/2, color='gray', linestyle='--', linewidth=2, label='Net')

     ax1.scatter(ideal_pos[0], ideal_pos[1], c='blue', marker='x', s=100, label='Ideal', zorder=5)
     ax1.scatter(positions[:,0], positions[:,1], c='red', marker='o', s=60, alpha=0.7, label='Actual shots')
     circle = plt.Circle((ideal_pos[0], ideal_pos[1]), 
                       radius=2*np.mean([scatter_info['sigma_x'], scatter_info['sigma_y']]),
                       fill=False, color='orange', linestyle='--', linewidth=2, label='2σ scatter')
     ax1.add_patch(circle)
    
     ax1.set_xlabel('X Position (m)')
     ax1.set_ylabel('Y Position (m)')
     ax1.set_title('Shot Consistency - Top View')
     ax1.legend()
     ax1.grid(True, alpha=0.3)
     ax1.axis('equal')
     
     deviations_x = positions[:,0] - ideal_pos[0]
     deviations_y = positions[:,1] - ideal_pos[1]
    
     ax2.hist(deviations_x * 1000, bins=max(3, len(positions)//2), alpha=0.5, label='X deviation', color='blue')
     ax2.hist(deviations_y * 1000, bins=max(3, len(positions)//2), alpha=0.5, label='Y deviation', color='red')
     ax2.axvline(0, color='black', linestyle='--', alpha=0.7, label='Perfect accuracy')
     ax2.set_xlabel('Deviation (mm)')
     ax2.set_ylabel('Frequency')
     ax2.set_title('Deviation Distribution')
     ax2.legend()
     ax2.grid(True, alpha=0.3)
    
     plt.tight_layout()
     plt.show()
    
    def stroke_gain_effect(self, stroke_gain):
        stroke_data = {
            0.05: {'sigma_x': 0.0197, 'sigma_y': 0.0143, 'launch_time': 8.83},
            0.10: {'sigma_x': 0.0156, 'sigma_y': 0.0192, 'launch_time': 4.21},
            0.50: {'sigma_x': 0.0241, 'sigma_y': 0.0239, 'launch_time': 1.60},
            1.00: {'sigma_x': 0.0213, 'sigma_y': 0.0223, 'launch_time': 1.39},
            3.00: {'sigma_x': 0.0237, 'sigma_y': 0.0196, 'launch_time': 0.81},
            5.00: {'sigma_x': 0.0227, 'sigma_y': 0.0223, 'launch_time': 0.61},
            10.00: {'sigma_x': 0.0199, 'sigma_y': 0.0215, 'launch_time': 0.67},
            30.00: {'sigma_x': 0.0279, 'sigma_y': 0.0150, 'launch_time': 0.62}
        }
        stroke_gains = list(stroke_data.keys())
        closest_gain = min(stroke_gains, key=lambda x: abs(x - stroke_gain))
        
        return stroke_data[closest_gain]['sigma_x'], stroke_data[closest_gain]['sigma_y']
    
    def pinch_diameter_effect(self, pinch_diameter):
        pinch_data = {
            35.3: {'sigma_x': 0.03194, 'sigma_y': 0.03062},
            35.8: {'sigma_x': 0.01932, 'sigma_y': 0.02658},
            36.4: {'sigma_x': 0.01906, 'sigma_y': 0.02171},
            37.0: {'sigma_x': 0.01965, 'sigma_y': 0.02480},
            37.4: {'sigma_x': 0.01866, 'sigma_y': 0.02208}, 
            38.6: {'sigma_x': 0.02361, 'sigma_y': 0.02428}
        }
        pinch_diameters = list(pinch_data.keys())
        closest_pinch = min(pinch_diameters, key=lambda x: abs(x - pinch_diameter))
        
        return pinch_data[closest_pinch]['sigma_x'], pinch_data[closest_pinch]['sigma_y']
    
    def ramp_time_effect(self, ramp_time):
        ramp_data = {
            0.01: {'sigma_x': 0.0402, 'sigma_y': 0.0166},
            0.05: {'sigma_x': 0.0400, 'sigma_y': 0.0143},
            0.10: {'sigma_x': 0.0436, 'sigma_y': 0.0174},
            0.50: {'sigma_x': 0.0211, 'sigma_y': 0.0150},
            1.00: {'sigma_x': 0.0243, 'sigma_y': 0.0172},
            2.00: {'sigma_x': 0.0243, 'sigma_y': 0.0134},
            3.00: {'sigma_x': 0.0234, 'sigma_y': 0.0134},  
            8.00: {'sigma_x': 0.0247, 'sigma_y': 0.0173},
            'continuous': {'sigma_x': 0.0231, 'sigma_y': 0.0150}
        }
        ramp_times = [k for k in ramp_data.keys() if isinstance(k, (int, float))]
        closest_ramp = min(ramp_times, key=lambda x: abs(x - ramp_time))
        
        return ramp_data[closest_ramp]['sigma_x'], ramp_data[closest_ramp]['sigma_y']
    
    def system_parameter_effects(self, ideal_landing, ramp_time=3.0, 
                                     stroke_gain=0.10, pinch_diameter=37.4):
        stroke_x, stroke_y = self.stroke_gain_effect(stroke_gain)
        pinch_x, pinch_y = self.pinch_diameter_effect(pinch_diameter)  
        ramp_x, ramp_y = self.ramp_time_effect(ramp_time)
    
        combined_sigma_x = np.sqrt(stroke_x**2 + pinch_x**2 + ramp_x**2)
        combined_sigma_y = np.sqrt(stroke_y**2 + pinch_y**2 + ramp_y**2)
    
        noise_x = np.random.normal(0, combined_sigma_x)
        noise_y = np.random.normal(0, combined_sigma_y)
        noise_z = np.random.normal(0, 0.002)
        actual_landing = ideal_landing + np.array([noise_x, noise_y, noise_z])
        return actual_landing, {
        'sigma_x': combined_sigma_x,
        'sigma_y': combined_sigma_y,
        'individual_effects': {
            'stroke': (stroke_x, stroke_y),
            'pinch': (pinch_x, pinch_y), 
            'ramp': (ramp_x, ramp_y)
        }
    }

    def plot_3d_trajectories(self, shot_trajectories, hit_positions_list=None):
     fig = plt.figure(figsize=(12, 9))
     ax = fig.add_subplot(111, projection='3d')
     colors = plt.cm.tab10(np.linspace(0, 1, len(shot_trajectories)))
     for i, trajectory in enumerate(shot_trajectories):
        if len(trajectory) > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   color=colors[i], linewidth=2.5, label=f'Shot {i+1}', alpha=0.8)
     if hit_positions_list:
        for i, hit_positions in enumerate(hit_positions_list):
            if len(hit_positions) > 0:
                # Convert to numpy array for proper slicing
                hit_array = np.array(hit_positions)
                ax.scatter(hit_array[:, 0], hit_array[:, 1], hit_array[:, 2], 
                          color=colors[i], s=100, marker='o', zorder=5,
                          edgecolors='white', linewidth=2)
     table_length, table_width, table_height = 2.74, 1.525, 0.76
     xx, yy = np.meshgrid(np.linspace(0, table_length, 10), 
                         np.linspace(-table_width/2, table_width/2, 10))
     zz = np.full_like(xx, table_height)
     ax.plot_surface(xx, yy, zz, alpha=0.4, color='lightgray', 
                   edgecolor='black', linewidth=0.5)
     net_x = table_length / 2
     net_height = 0.1525
     ax.plot([net_x, net_x], [-table_width/2, table_width/2], 
           [table_height, table_height], 'k-', linewidth=4, label='Net Base')
     ax.plot([net_x, net_x], [-table_width/2, table_width/2], 
           [table_height + net_height, table_height + net_height], 
           'k-', linewidth=4, label='Net Top')
     for y in np.linspace(-table_width/2, table_width/2, 5):
        ax.plot([net_x, net_x], [y, y], 
               [table_height, table_height + net_height], 'k-', linewidth=1, alpha=0.7)
     ax.set_xlabel('X Position (m)', fontsize=12)
     ax.set_ylabel('Y Position (m)', fontsize=12)
     ax.set_zlabel('Z Position (m)', fontsize=12)
     ax.set_title('3D Ball Trajectories - Table Tennis Launcher Analysis', fontsize=14)
     ax.set_xlim(-0.5, table_length + 0.5)
     ax.set_ylim(-table_width/2 - 0.2, table_width/2 + 0.2)
     ax.set_zlim(0, 1.8)
     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
     ax.view_init(elev=20, azim=45)
     plt.tight_layout()
     plt.show()

    """def view_single_shot(self, phi, theta, rpm_tl, rpm_tr, rpm_bc, ramp_time, stroke_gain, pinch_diameter, shot_num):
      print(f"\n--- Viewing Shot {shot_num} ---")
    
      result = self.get_hitting_position_with_system_effects(
        phi, theta, rpm_tl, rpm_tr, rpm_bc, 
        ramp_time=ramp_time, stroke_gain=stroke_gain, pinch_diameter=pinch_diameter
    )
    
      if result['hit_found_table']:
        print(f"Shot {shot_num} landed at: ({result['hit_position_table'][0]:.3f}, {result['hit_position_table'][1]:.3f}, {result['hit_position_table'][2]:.3f})")
      else:
        print(f"Shot {shot_num} missed the table")
    
      self.run_3d_viewer(phi, theta, rpm_tl, rpm_tr, rpm_bc) """

def main():
    xml_path = 'balllauncher/balllaunch.xml'
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        sys.exit(1)
    
    try:
        launcher = SimpleTableTennisLauncher(xml_path)
        print("MuJoCo model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\n" + "="*70)
    print(" TABLE TENNIS BALL LAUNCHER - COMPLETE ANALYSIS SYSTEM")
    print("-" * 70)
    print("Input 5 launcher parameters:")
    print("• φ (phi): Horizontal angle in RADIANS (-0.52 to 0.52 rad)")
    print("• θ (theta): Vertical angle in RADIANS (0 to 0.79 rad)")
    print("• rpm_tl: Top-left wheel RPM ")
    print("• rpm_tr: Top-right wheel RPM ")
    print("• rpm_bc: Back-center wheel RPM ")
    print("="*70)
    
    while True:
        try:
            print("\nChoose option:")
            print("1. Analyze parameters (with plots)")
            print("2. Analyze + 3D viewer") 
            print("3. Quick analysis (no plots)")
            print("4. Test system parameter consistency (multiple shots)")
            print("5. Quit")
            
            choice = input("Enter choice (1-5): ").strip()

            if choice == '5' or choice.lower() == 'q':
                print("Exit")
                break
                
            if choice not in ['1', '2', '3', '4']:
                print("Invalid choice. Please enter 1-5.")
                continue
                
            print("\n" + "-"*50)
            print("Enter 5 launcher parameters:")
            phi = float(input("φ (horizontal angle in radians): ").strip())
            theta = float(input("θ (vertical angle in radians): ").strip())
            rpm_tl = float(input("rpm_tl (top-left wheel): ").strip())
            rpm_tr = float(input("rpm_tr (top-right wheel): ").strip())
            rpm_bc = float(input("rpm_bc (back-center wheel): ").strip())
            
            rpm_tl_int = int(round(rpm_tl))
            rpm_tr_int = int(round(rpm_tr))
            rpm_bc_int = int(round(rpm_bc))
            
            if choice == '1':
                launcher.analyze_parameters(phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int, 
                                           show_plot=True, show_3d=False,use_system_effects=False)
            elif choice == '2':
                launcher.analyze_parameters(phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int, 
                                           show_plot=True, show_3d=False,use_system_effects=True)
                launcher.run_3d_viewer(phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int)
                
            elif choice == '3':
                launcher.analyze_parameters(phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int, 
                                           show_plot=False, show_3d=False,use_system_effects=False)
            elif choice == '4':
              num_shots = int(input("Number of shots to test (default 10): ").strip() or "10")
              print("\nSimulation approach:")
              print("1. Noise-based variations")
              print("2. Realistic Variation")
              sim_choice = input("Choose approach ( default 1): ").strip() or "1"
              use_physics = (sim_choice == '2')
    
              use_custom = input("Use custom system parameters? (y/n, default n): ").strip().lower()
              if use_custom == 'y':
                ramp_time = float(input("Ramp time (default 3.0): ").strip() or "3.0")
                stroke_gain = float(input("Stroke gain (default 0.10): ").strip() or "0.10")
                pinch_diameter = float(input("Pinch diameter (default 37.4): ").strip() or "37.4")
              else:
                ramp_time, stroke_gain, pinch_diameter = 3.0, 0.10, 37.4
              landing_positions = launcher.test_system_parameter_consistency(
        phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int, 
        num_shots=num_shots, use_physics_variations=use_physics,
        ramp_time=ramp_time, stroke_gain=stroke_gain, pinch_diameter=pinch_diameter,
        plot_results=True  
    )
              if landing_positions and len(landing_positions) > 1:
                print("\n" + "="*50)
                print("TRAJECTORY VISUALIZATION")
                print("="*50)
                print("Generating 3D trajectory visualization")
                if use_physics:
                  trajectories = []
                  hit_positions_list = []

                  for shot in range(min(5, len(landing_positions))):
                    result = launcher.get_hitting_position(
                phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int,
                ramp_time=ramp_time, stroke_gain=stroke_gain, pinch_diameter=pinch_diameter,use_system_effects=True
            )
                    trajectories.append(result['trajectory'])
                    if result['hit_found_table']: 
                      hit_pos = result['hit_position_table']
                      hit_positions_list.append([hit_pos])

                  launcher.plot_3d_trajectories(trajectories, hit_positions_list)

              """view_3d = input("\nView 3D trajectory for a specific shot? (y/n, default n): ").strip().lower()
               if view_3d == 'y':
                shot_num = int(input(f"Enter shot number (1-{len(landing_positions)}): "))
                print(f"Launching 3D viewer for shot {shot_num}...")
                launcher.view_single_shot(phi, theta, rpm_tl_int, rpm_tr_int, rpm_bc_int, 
                                    ramp_time, stroke_gain, pinch_diameter, shot_num) """

        except ValueError as e:
            print(f" Invalid input: Please enter numeric values")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()

