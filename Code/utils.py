import scipy.io
import cv2
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_world_coordinates(tag_ids):
    '''
    all dimensions are in SI units
    '''
    # Define dimensions of the tags and grid
    tag_width = 0.152 
    column_spacing = [0, 0.152, 0.152, 0.178, 0.152, 0.152, 0.178, 0.152, 0.152]
    rows, cols = 12, 9

    world_pts_list = []

    for tag_id in tag_ids:
        row = tag_id % rows
        col = tag_id // rows

        # Calculate x, y, and z coordinates for each corner of the tag
        x_offset = tag_width * row
        y_offset = sum(column_spacing[:col + 1])

        img_pts = np.array(
            [
                [row * tag_width + 0, col * tag_width + 0, 0.0],
                [row * tag_width + tag_width, col * tag_width + 0, 0.0],
                [row * tag_width + tag_width, col * tag_width + tag_width, 0.0],
                [row * tag_width + 0, col * tag_width + tag_width, 0.0],
            ]
        )

        world_pts = img_pts + np.array([x_offset, y_offset, 0.0])
        world_pts_list.extend(map(tuple, world_pts))

    return world_pts_list

def estimate_pose(data):
    '''
    estimates the pose using the solvePnP method
    '''
    # extract img pts
    p1 = data['p4']
    p2 = data['p1']
    p3 = data['p2']
    p4 = data['p3']

    tag_ids =data['id']
    ts=[]

    # reshaping img points for int input
    if isinstance(tag_ids, int):
        tag_ids=np.array([tag_ids])
        p1=p1.reshape(2,1)
        p2=p2.reshape(2,1)
        p3=p3.reshape(2,1)
        p4=p4.reshape(2,1)

    # if no tags, set to default 
    if(len(tag_ids)==0):
        pos=np.nan
        ori=np.nan
        ts = np.nan
        return pos, ori ,ts

    img_pts = []
    for i in range(p1.shape[1]):
        img_pts.append((p1[0, i], p1[1, i]))
        img_pts.append((p2[0, i], p2[1, i]))
        img_pts.append((p3[0, i], p3[1, i]))
        img_pts.append((p4[0, i], p4[1, i]))

    world_pts = get_world_coordinates(tag_ids)

    world_pts=np.array(world_pts)
    img_pts=np.array(img_pts)

    camera_matrix = np.array([[314.1779 , 0 , 199.4848],
                              [0 , 314.2218 , 113.7838],
                              [0, 0, 1]])

    dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])

    _, rvec, tvec = cv2.solvePnP(world_pts, img_pts, camera_matrix, dist_coeffs)

    ts = data['t']

    # Convert rotation vector to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rvec)

    T_cam_world = np.hstack((rot_mat, tvec))
    
    T_cam_world = np.vstack((T_cam_world, [0, 0, 0, 1]))

    rotation_z = np.array([
                    [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                    [0, 0, 1] 
                ])
    
    rotation_x = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
    
    rotation = rotation_x @ rotation_z 
    # print(rotation)

    T_cam_imu = np.hstack([rotation,np.array([-0.04,0,-0.03]).reshape(3,1)])

    T_cam_imu = np.vstack((T_cam_imu, [0, 0, 0, 1]))

    T_world_imu = np.linalg.inv(T_cam_world) @ T_cam_imu

    # Converting rotation matrix to euler angles 
    pos = T_world_imu[:3, 3]

    r = Rotation.from_matrix(T_world_imu[:3, :3])
    roll, pitch, yaw = r.as_euler('xyz')

    ori = np.array([roll, pitch, yaw])


    return pos, ori, ts

def plotter(filename):

    file_path = 'data\\' + filename

    student_data = scipy.io.loadmat(file_path, simplify_cells=True)

    # Extract motion capture data
    vicon = student_data['vicon']
    vicon_data = np.array(vicon).T
    vicon_time = student_data['time']

    # Initialize arrays to store estimated and ground truth data
    est_pos = []
    est_ori = []
    est_ts=[]
    gt_pos = vicon_data[:, :3]

    i = 0
    for data in student_data['data']:
        pos, ori, ts = estimate_pose(data)
        if not np.isnan(pos).any() and not np.isnan(ori).any() and not np.isnan(ts).any():
          est_pos.append(pos)
          est_ori.append(ori)
          est_ts.append(ts)
          i += 1

    # Convert lists to numpy arrays for plotting
    est_pos = np.array(est_pos)
    est_ori = np.array(est_ori)

    # # Plot 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], label='Estimated', color='orange')
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label='Ground Truth', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Trajectory Plot: Ground Truth vs Estimates")
    fig.savefig(f"Outputs\\{filename}_traj.png", dpi=300)  # Save the 3D plot
    plt.tight_layout()

    # Plot euler angles
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
    axs[0].plot(est_ts, est_ori[:, 0], label='Estimated', color='red')
    axs[0].plot(vicon_time, vicon_data[:, 3], label='Ground Truth', color='blue')
    axs[0].set_ylabel('Phi')
    axs[0].legend()

    axs[1].plot(est_ts, est_ori[:, 1], label='Estimated', color='red')
    axs[1].plot(vicon_time, vicon_data[:, 4], label='Ground Truth', color='blue')
    axs[1].set_ylabel('Theta')
    axs[1].legend()

    axs[2].plot(est_ts, est_ori[:, 2], label='Estimated', color='red')
    axs[2].plot(vicon_time, vicon_data[:, 5], label='Ground Truth', color='blue')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Psi)')
    axs[2].legend()

    plt.tight_layout()
    fig.savefig(f"Outputs\\{filename}_ori.png")  # Save the ori plot
    # plt.show()

def estimate_covariances(filename):
    # Load .mat file
    file_path = 'data\\' + filename

    student_data = scipy.io.loadmat(file_path, simplify_cells=True)

    # Extract motion capture data
    vicon = student_data['vicon']
    gt_ts = student_data['time']
    vicon_data = np.array(vicon).T

    # Extract ground truth positions and orientations
    gt_pos = vicon_data[:, :3]
    gt_ori = vicon_data[:, 3:6]

    # Extract observation model data
    est_pos = []
    est_ori = []
    est_ts = []

    # estimate_pose is a function that returns pos, ori, and ts
    for data in student_data['data']:
        pos, ori, ts = estimate_pose(data)
        if not np.isnan(pos).any() and not np.isnan(ori).any() and not np.isnan(ts).any():
            est_pos.append(pos)
            est_ori.append(ori)
            est_ts.append(ts)

    n = len(student_data['data'])

    # Convert lists to numpy arrays
    est_pos = np.array(est_pos)
    est_ori = np.array(est_ori)
    est_ts = np.array(est_ts)

    # Create estimated state space matrix
    est_pose = np.hstack((est_pos, est_ori))

    # Aligning and extracting aligned data
    aligned_idx = []
    for est_timestamp in est_ts:
        closest_idx = np.argmin(np.abs(gt_ts - est_timestamp))
        aligned_idx.append(closest_idx)

    aligned_gt_pos = gt_pos[aligned_idx]
    aligned_gt_ori = gt_ori[aligned_idx]
    aligned_gt_pose = np.hstack((aligned_gt_pos, aligned_gt_ori))

    obs_noise = aligned_gt_pose - est_pose

    # covariance matrix
    R = obs_noise.T @ obs_noise / (n - 1)

    return R

def plotter_2(filename,filter_pos):
    # Load .mat file
    
    file_path = 'data\\' + filename

    student_data = scipy.io.loadmat(file_path, simplify_cells=True)

    # Extract motion capture data
    vicon = student_data['vicon']
    vicon_time = student_data['time']
    vicon_data = np.array(vicon).T

    # Extract ground truth positions
    gt_pos = vicon_data[:, :3]
    gt_pos = np.array(gt_pos)

    # Extract observation model data
    est_pos = []

    for data in student_data['data']:

        pos, ori, ts = estimate_pose(data)

        if not np.isnan(pos).any() and not np.isnan(ori).any() and not np.isnan(ts).any():
            est_pos.append(pos)
            # est_ori.append(ori)

    # Convert lists to numpy arrays
    est_pos = np.array(est_pos)
    filter_pos = np.array(filter_pos)
    print("Filtered pos shape", filter_pos.shape)

    # Plot ground truth, estimated, and filtered positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth pos
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label='Ground Truth', color='blue')

    # Plot estimated pos
    # ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], label='Observation Model Result', color='green')

    # Plot filtered pos
    ax.scatter(filter_pos[:, 0], filter_pos[:, 1], filter_pos[:, 2], label='EKF', color='green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ground Truth vs EKF Results for {filename}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"Outputs\\{filename}_filtered.png", dpi =300)  # Save the 3D plot
    # plt.show()


def visualize_filter_results(mat_file, filtered_positions, highest_filtered_positions, weighted_average_filtered_positions):
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_file, simplify_cells=True)

    # Extract motion capture data
    vicon = mat_data['vicon']
    ground_truth_timestamps = mat_data['time']
    vicon_data = np.array(vicon).T

    # Extract ground truth positions
    ground_truth_position = vicon_data[:, :3]
    ground_truth_position = np.array(ground_truth_position)

    # Extract observation model data
    estimated_positions = []

    # Assuming estimate_pose is a function that returns position, orientation, and timestamp
    for data_sample in mat_data['data']:
        position, orientation, timestamp = estimate_pose(data_sample)
        if not np.isnan(position).any() and not np.isnan(orientation).any() and not np.isnan(timestamp).any():
            estimated_positions.append(position)
            # estimated_orientations.append(orientation)

    # Convert lists to numpy arrays
    estimated_positions = np.array(estimated_positions)
    filtered_positions = np.array(filtered_positions)
    highest_filtered_positions = np.array(highest_filtered_positions)
    weighted_average_filtered_positions = np.array(weighted_average_filtered_positions)

    # Plot ground truth, estimated, and filtered positions
    fig, axes = plt.subplots(3, 1, figsize=(10, 20), subplot_kw={'projection': '3d'})

    # Plot ground truth position in each plot
    for ax in axes:
        ax.plot(ground_truth_position[:, 0], ground_truth_position[:, 1], ground_truth_position[:, 2], label='Ground Truth', color='blue')

    # Plot observation model position in each plot
    for ax in axes:
        ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Observation Model Result', color='green')

    # Plot filtered positions in each plot
    axes[0].plot(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2], label='PF Filtered', color='red')
    axes[1].plot(highest_filtered_positions[:, 0], highest_filtered_positions[:, 1], highest_filtered_positions[:, 2], label='Highest Filtered', color='orange')
    axes[2].plot(weighted_average_filtered_positions[:, 0], weighted_average_filtered_positions[:, 1], weighted_average_filtered_positions[:, 2], label='Weighted Average Filtered', color='purple')

    # Set labels and title for each plot
    for i, ax in enumerate(axes):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Ground Truth vs Estimated vs Filtered Position ({mat_file})')
        ax.legend()

    plt.show()

def plot_rmse(RMSE_plot_w_a, RMSE_plot_h, RMSE_plot_a, student_number):
    fig, ax = plt.subplots()
    ax.plot(range(len(RMSE_plot_w_a)), RMSE_plot_w_a, label='Weighted Average RMSE', color='blue')
    ax.plot(range(len(RMSE_plot_h)), RMSE_plot_h, label='Highest RMSE', color='green')
    ax.plot(range(len(RMSE_plot_a)), RMSE_plot_a, label='Average RMSE', color='red')

    ax.set_xlabel('Time')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Root Mean Square Error (RMSE) over Time for 750 particle size for the Student Data ({student_number})')
    ax.legend()

    plt.grid(True)
    plt.show()


rmse_data = []

def Calculate_RMSE(filename,highest_filtered_positions, highest_filtered_orientations, weighted_average_filtered_positions, weighted_average_filtered_orientations, average_filtered_positions, average_filtered_orientations, filtered_timestamps, student_number,particles):
    # Load .mat file
    mat_data = scipy.io.loadmat(filename, simplify_cells=True)

    # Extract motion capture data
    vicon = mat_data['vicon']
    ground_truth_timestamps = mat_data['time']
    vicon_data = np.array(vicon).T

    # Extract ground truth positions and orientations
    ground_truth_positions = vicon_data[:, :3]
    ground_truth_orientations = vicon_data[:, 3:6]

    highest_filtered_positions = np.array(highest_filtered_positions)
    highest_filtered_orientations = np.array(highest_filtered_orientations)
    weighted_average_filtered_positions = np.array(weighted_average_filtered_positions)
    weighted_average_filtered_orientations = np.array(weighted_average_filtered_orientations)
    average_filtered_positions = np.array(average_filtered_positions)
    average_filtered_orientations = np.array(average_filtered_orientations)

    n = len(mat_data['data'])

    # Match timestamps between ground truth and estimated data
    matched_indices = []
    for filtered_timestamp in filtered_timestamps:
      closest_idx = np.argmin(np.abs(ground_truth_timestamps - filtered_timestamp))
      matched_indices.append(closest_idx)

    # Extract matched ground truth positions and orientations
    matched_ground_truth_positions = ground_truth_positions[matched_indices]
    matched_ground_truth_orientations = ground_truth_orientations[matched_indices]
    matched_ground_truth_states = np.hstack((matched_ground_truth_positions, matched_ground_truth_orientations))

    # Calculate squared differences for each estimate
    highest_filtered_position_errors = np.sqrt(np.sum((highest_filtered_positions - matched_ground_truth_positions)**2,axis=1))

    highest_filtered_orientation_errors = np.sqrt(np.sum((highest_filtered_orientations - matched_ground_truth_orientations)**2,axis=1))

    weighted_average_filtered_position_errors = np.sqrt(np.sum((weighted_average_filtered_positions - matched_ground_truth_positions)**2,axis=1))
    weighted_average_filtered_orientation_errors = np.sqrt(np.sum((weighted_average_filtered_orientations - matched_ground_truth_orientations)**2,axis=1))

    average_filtered_position_errors = np.sqrt(np.sum((average_filtered_positions - matched_ground_truth_positions)**2,axis=1))
    average_filtered_orientation_errors = np.sqrt(np.sum((average_filtered_orientations - matched_ground_truth_orientations)**2,axis=1))

    # Calculate total RMSE for positions and orientations for each method
    RMSE_plot_w_a = np.sqrt(weighted_average_filtered_position_errors**2 + weighted_average_filtered_orientation_errors**2 )
    RMSE_plot_h = np.sqrt(highest_filtered_position_errors**2 + highest_filtered_orientation_errors**2 )
    RMSE_plot_a = np.sqrt(average_filtered_position_errors**2 + average_filtered_orientation_errors**2 )

    # Calculate mean squared errors
    highest_filtered_position_rmse = np.sqrt(np.mean(highest_filtered_position_errors**2)).round(decimals=3)
    highest_filtered_orientation_rmse = np.sqrt(np.mean(highest_filtered_orientation_errors**2)).round(decimals=3)

    weighted_average_filtered_position_rmse = np.sqrt(np.mean(weighted_average_filtered_position_errors**2)).round(decimals=3)
    weighted_average_filtered_orientation_rmse = np.sqrt(np.mean(weighted_average_filtered_orientation_errors**2)).round(decimals=3)

    average_filtered_position_rmse = np.sqrt(np.mean(average_filtered_position_errors**2)).round(decimals=3)
    average_filtered_orientation_rmse = np.sqrt(np.mean(average_filtered_orientation_errors**2)).round(decimals=3)

    rmse_average_values = [highest_filtered_position_rmse , highest_filtered_orientation_rmse, weighted_average_filtered_position_rmse, weighted_average_filtered_orientation_rmse, average_filtered_position_rmse,average_filtered_orientation_rmse]

    rmse_values = {
        'Student Number': student_number,
        'Number of Particles': particles,
        'Weighted Average Position RMSE': weighted_average_filtered_position_rmse,
        'Weighted Average Orientation RMSE': weighted_average_filtered_orientation_rmse,
        'Highest Position RMSE': highest_filtered_position_rmse,
        'Highest Orientation RMSE': highest_filtered_orientation_rmse,
        'Average Position RMSE': average_filtered_position_rmse,
        'Average Orientation RMSE': average_filtered_orientation_rmse
    }

    rmse_data.append(rmse_values)

    return RMSE_plot_w_a ,RMSE_plot_h ,RMSE_plot_a