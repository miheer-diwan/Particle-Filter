import numpy as np
import scipy.io
from utils import estimate_pose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ParticleFilter:
    def __init__(self, num_particles,filename):
        self.filepath = 'data\\' + filename
        self.num_particles = num_particles

        self.R  = np.diag([0.0067, 0.0048, 0.0088, 0.0041, 0.0064, 0.0011])

        self.Q = np.diag([0.01] * 3 + [0.01] * 3 + [0.01] * 3 + [0.001] * 3 + [0.001] * 3) * 500

        student_data = scipy.io.loadmat(self.filepath, simplify_cells=True)
        self.data= student_data["data"]

        self.avg_filter_pos = []
        self.avg_filter_ori = []
        self.high_filter_pos = []
        self.high_filter_ori = []
        self.wa_fil_pos = []
        self.wa_fil_ori = []
        self.aligned_ts = []


    def spawn_particles(self):
        n = self.num_particles
        particles = np.zeros((n, 15))

        # pos ranges
        x_range = (0, 3)
        y_range = (0, 3)
        z_range = (0, 2)

        # Angle range (-pi/2 to pi/2)
        angle_range = (-np.pi/2, np.pi/2)

        # randomly generating particles
        x_particles = np.random.uniform(low=x_range[0], high=x_range[1], size=n)
        y_particles = np.random.uniform(low=y_range[0], high=y_range[1], size=n)
        z_particles = np.random.uniform(low=z_range[0], high=z_range[1], size=n)

        # Generate particles for angles
        angle_particles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=(n, 3))

        particles[:, :3] = np.column_stack((x_particles, y_particles, z_particles))
        particles[:, 3:6] = angle_particles
        particles[:, 6:9] = np.random.uniform(low=-0.5, high=0.5, size=(self.num_particles, 3))
        particles[:, 9:15] = np.random.uniform(low=-0.5, high=0.5, size=(self.num_particles, 6))

        return particles

    def predict_step(self, particles, u, del_t):

        u_w = u[0:3]
        u_a =  u[3:6]
        phi_list, theta_list, psi_list = particles[:,3] , particles[:,4] ,particles[:,5]

        g = np.array([0, 0, -9.81]).reshape(1, 3, 1)

        x_dot = np.zeros_like(particles)
        u_w = np.tile(u[:3], (self.num_particles, 1, 1)) + particles[:,9:12].reshape(self.num_particles,3,1) # noisy u_a
        u_a = np.tile(u[3:], (self.num_particles, 1, 1)) + particles[:,12:15].reshape(self.num_particles,3,1) # noisy u_a

        # Compute G_q_inv for all particles
        G_q_inv = np.zeros((self.num_particles, 3, 3))
        G_q_inv[:, 0, 0] = np.cos(theta_list)
        G_q_inv[:, 0, 2] = np.sin(theta_list)
        G_q_inv[:, 1, 0] = np.sin(phi_list) * np.sin(theta_list) / np.cos(phi_list)
        G_q_inv[:, 1, 1] = 1.0
        G_q_inv[:, 1, 2] = -np.cos(theta_list) * np.sin(phi_list) / np.cos(phi_list)
        G_q_inv[:, 2, 0] = -np.sin(theta_list) / np.cos(phi_list)
        G_q_inv[:, 2, 2] = np.cos(theta_list) / np.cos(phi_list)

        # Compute R_q for each particle
        R_q = np.zeros((self.num_particles, 3, 3))
        R_q[:, 0, 0] = np.cos(psi_list) * np.cos(theta_list) - np.sin(phi_list) * np.sin(phi_list) * np.sin(theta_list)
        R_q[:, 0, 1] = -np.cos(phi_list) * np.sin(psi_list)
        R_q[:, 0, 2] = np.cos(psi_list) * np.sin(theta_list) + np.cos(theta_list) * np.sin(phi_list) * np.sin(psi_list)
        R_q[:, 1, 0] = np.cos(theta_list) * np.sin(psi_list) + np.cos(psi_list) * np.sin(phi_list) * np.sin(theta_list)
        R_q[:, 1, 1] = np.cos(phi_list) * np.cos(psi_list)
        R_q[:, 1, 2] = np.sin(psi_list) * np.sin(theta_list) - np.cos(psi_list) * np.cos(theta_list) * np.sin(phi_list)
        R_q[:, 2, 0] = -np.cos(phi_list) * np.sin(theta_list)
        R_q[:, 2, 1] = np.sin(phi_list)
        R_q[:, 2, 2] = np.cos(phi_list) * np.cos(theta_list)


        x_dot[:,0:3] = particles[:,6:9]
        x_dot[:,3:6] = (G_q_inv @ u_w).reshape((self.num_particles,3))
        x_dot[:,6:9] = (g + R_q @ u_a).reshape((self.num_particles,3))
        noise = noise = np.random.multivariate_normal(np.zeros(15), self.Q, size=self.num_particles)
        particles+= (x_dot + noise) * del_t

        return particles

    def est_wts(self, particles, z):
        """
        Calculates the wts for each particle based on the measurement likelihood.

        """

        wts = np.zeros((self.num_particles, 1))

        z = z.reshape(6, 1)
        
        # denominator
        denom = 1.0 / ((2 * np.pi) ** (15 / 2) * np.linalg.det(self.R) ** 0.5)

        # Loop over all particles to compute their wts
        for i in range(self.num_particles):
            # vectorize
            x = particles[i, :].reshape((15, 1))

            err = z - x[:6]

            # Compute ith particle wt using multivariate Gaussian formula
            wts[i] = np.exp(-0.5 * np.dot(np.dot(err.T, np.linalg.inv(self.R)), err)) * denom
            
            # # Check if weight_i is NaN or infinity

            # if np.isnan(weight_i) or np.isinf(weight_i):
            #     # Assign a very small weight to avoid division by zero or NaN wts
            #     wts[i] = 1e-6
            # else:
            #     wts[i] = weight_i    

        # Normalize the wts so that they sum to 1
        return wts / np.sum(wts)


    def update_step(self, particles, wts):

        # Find the index of the particle with the highest weight
        max_wt_idx = np.argmax(wts)
        updated_est_high = particles[max_wt_idx]

        updated_est_avg = np.mean(particles, axis=0)

        # Weighted average of all particles
        wts_norm = wts / np.sum(wts)

        # Calculate the weighted average using dot product
        updated_est_wa = np.dot(wts_norm.T, particles)
        updated_est_wa = updated_est_wa.reshape(15)


        return updated_est_high,  updated_est_avg , updated_est_wa


    def Low_Variance_Resampling(self, particles, wts):

        resampled_particles = np.zeros_like(particles)

        n = self.num_particles
        # Step size
        step = 1.0 / n

        # Random start index
        r = np.random.rand() * step

        # Initialize the cumulative sum of wts
        c = wts[0]

        i = 0

        # Resampling loop
        for m in range(n):
            # Move along the weight distribution until we find the particle
            # to resample for the m-th new particle
            U = r + m * step
            while U > c:
                i = i + 1
                c = c + wts[i]
            resampled_particles[m, :] = particles[i, :]

        return resampled_particles


    def iterator(self, student_number):
        """
        Estimate pose at each time stamp.
        """
        i = 0
        student_data = scipy.io.loadmat(self.filepath, simplify_cells=True)

        particles = self.spawn_particles()
        prev_t = 0

        for data in student_data['data']:
            i = i + 1
            del_t = prev_t - data['t']
            prev_t = data['t']

            # End condition for the for loop
            if student_number == 0:
                omg = np.array(data['drpy']).reshape(-1, 1)
            else:
                omg = np.array(data['omg']).reshape(-1, 1)

            acc = np.array(data['acc']).reshape(-1, 1)

            u = np.concatenate((omg, acc))

            pos, ori, ts = estimate_pose(data)

            if not np.isnan(pos).any() and not np.isnan(ori).any() and not np.isnan(ts).any():
                # If you get a measurement perform PF
                particles = self.predict_step(particles, u, del_t)

                z = np.concatenate((pos.reshape(-1, 1), ori.reshape(-1, 1)))
                wts = self.est_wts(particles, z)

                updated_est_high, updated_est_avg, updated_est_wa = self.update_step(particles, wts)
                particles = self.Low_Variance_Resampling(particles, wts)

                self.high_filter_pos.append([updated_est_high[0], updated_est_high[1], updated_est_high[2]])
                self.high_filter_ori.append([updated_est_high[3], updated_est_high[4], updated_est_high[5]])

                self.wa_fil_pos.append([updated_est_wa[0], updated_est_wa[1], updated_est_wa[2]])
                self.wa_fil_ori.append([updated_est_wa[3], updated_est_wa[4], updated_est_wa[5]])

                self.avg_filter_pos.append([updated_est_avg[0], updated_est_avg[1], updated_est_avg[2]])
                self.avg_filter_ori.append([updated_est_avg[3], updated_est_avg[4], updated_est_avg[5]])
                self.aligned_ts.append(data['t'])

                i += 1

        return self.high_filter_pos, self.high_filter_ori, self.wa_fil_pos, self.wa_fil_ori, self.avg_filter_pos, self.avg_filter_ori, self.aligned_ts

def plotter(filename, filter_pos, high_filter_pos, wa_fil_pos):
    # Load .mat file
    filepath = 'data\\'+ filename
    student_data = scipy.io.loadmat(filepath, simplify_cells=True)

    # Extract motion capture data
    vicon = student_data['vicon']
    vicon_time = student_data['time']
    vicon_data = np.array(vicon).T

    # Extract ground truth positions
    gt_pos = vicon_data[:, :3]
    gt_pos = np.array(gt_pos)

    # Extract observation model data
    est_pos = []

    # Assuming estimate_pose is a function that returns pos, ori, and ts
    for data in student_data['data']:
        pos, ori, ts = estimate_pose(data)
        if not np.isnan(pos).any() and not np.isnan(ori).any() and not np.isnan(ts).any():
            est_pos.append(pos)
            # est_orientations.append(ori)

    # Convert lists to numpy arrays
    est_pos = np.array(est_pos)
    filter_pos = np.array(filter_pos)
    high_filter_pos = np.array(high_filter_pos)
    wa_fil_pos = np.array(wa_fil_pos)

    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot ground truth pos
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label='Vicon', color='blue')

    # Plot observation model pos
    ax1.scatter(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], label='Observation Model', color='green')

    # Plot filtered positions
    ax1.scatter(filter_pos[:, 0], filter_pos[:, 1], filter_pos[:, 2], label='Filtered', color='red')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Vicon vs Estimated vs Filtered pos ({filename})')
    ax1.legend()
    fig1.savefig(f'Outputs\\{filename}_filtered.png')

    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot gt pos
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label='Vicon', color='blue')

    # Plot observation model pos
    ax2.scatter(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], label='Observation Model', color='green')

    # Plot highest filtered positions
    ax2.scatter(high_filter_pos[:, 0], high_filter_pos[:, 1], high_filter_pos[:, 2], label='Highest', color='orange')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Vicon vs Estimated vs Highest Pos ({filename})')
    ax2.legend()
    fig2.savefig(f'Outputs\\{filename}_highest.png')


    fig3 = plt.figure(figsize=(10, 7))
    ax3 = fig3.add_subplot(111, projection='3d')

    # Plot gt pos
    ax3.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label='Vicon', color='blue')

    # Plot observation model pos
    ax3.scatter(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], label='Observation Model', color='green')

    # Plot weighted average positions
    ax3.scatter(wa_fil_pos[:, 0], wa_fil_pos[:, 1], wa_fil_pos[:, 2], label='Weighted Average', color='purple')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Vicon vs Estimated vs Weighted Average Pos ({filename})')
    ax3.legend()
    fig3.savefig(f'Outputs/{filename}_wt_avg.png')

    plt.tight_layout()
    # plt.show()

def plot_rmse(RMSE_plot_w_a, RMSE_plot_h, RMSE_plot_a, filename):
    fig, ax = plt.subplots()
    ax.plot(range(len(RMSE_plot_w_a)), RMSE_plot_w_a, label='Wt. Avg RMSE', color='blue')
    ax.plot(range(len(RMSE_plot_h)), RMSE_plot_h, label='Highest RMSE', color='magenta')
    ax.plot(range(len(RMSE_plot_a)), RMSE_plot_a, label='Avg RMSE', color='red')

    ax.set_xlabel('Time')
    ax.set_ylabel('RMSE')
    ax.set_title(f'RMSE Vs Time for {filename} | p_size = 1000')
    ax.legend()

    plt.grid(True)
    fig.savefig(f'Outputs/{filename}_RSME.png')
    # plt.show()

def Calculate_RMSE(filename,high_filter_pos, high_filter_ori, wa_fil_pos, wa_fil_ori, avg_filter_pos, avg_filter_ori, aligned_ts, particles):
    # Load .mat file
    file_path = 'data\\' + filename

    student_data = scipy.io.loadmat(file_path, simplify_cells=True)

    vicon = student_data['vicon']
    vicon_time = student_data['time']
    vicon_data = np.array(vicon).T

    gt_pos = vicon_data[:, :3]
    gt_ori = vicon_data[:, 3:6]

    high_filter_pos = np.array(high_filter_pos)
    high_filter_ori = np.array(high_filter_ori)
    wa_fil_pos = np.array(wa_fil_pos)
    wa_fil_ori = np.array(wa_fil_ori)
    avg_filter_pos = np.array(avg_filter_pos)
    avg_filter_ori = np.array(avg_filter_ori)

    n = len(student_data['data'])

    # Match timestamps
    aligned_idx = []
    for aligned_ts in aligned_ts:
        closest_idx = np.argmin(np.abs(vicon_time - aligned_ts))
        aligned_idx.append(closest_idx)

    #  match ground truth positions and orientations
    aligned_gt_pos = gt_pos[aligned_idx]
    aligned_gt_ori = gt_ori[aligned_idx]
    aligned_gt_state = np.hstack((aligned_gt_pos, aligned_gt_ori))

    highest_pos_err = np.sqrt(np.sum((high_filter_pos - aligned_gt_pos)**2,axis=1))

    highest_ori_err = np.sqrt(np.sum((high_filter_ori - aligned_gt_ori)**2,axis=1))

    wa_pos_err = np.sqrt(np.sum((wa_fil_pos - aligned_gt_pos)**2,axis=1))
    wa_ori_err = np.sqrt(np.sum((wa_fil_ori - aligned_gt_ori)**2,axis=1))

    avg_pos_err = np.sqrt(np.sum((avg_filter_pos - aligned_gt_pos)**2,axis=1))
    avg_ori_err = np.sqrt(np.sum((avg_filter_ori - aligned_gt_ori)**2,axis=1))

    # total rmse for positions and orientations for each method
    RMSE_plot_w_a = np.sqrt(wa_pos_err**2 + wa_ori_err**2 )
    RMSE_plot_h = np.sqrt(highest_pos_err**2 + highest_ori_err**2 )
    RMSE_plot_a = np.sqrt(avg_pos_err**2 + avg_ori_err**2 )

    # Calculate mse
    highest_pos_rmse = np.sqrt(np.mean(highest_pos_err**2)).round(decimals=3)
    highest_ori_rmse = np.sqrt(np.mean(highest_ori_err**2)).round(decimals=3)

    wa_pos_rmse = np.sqrt(np.mean(wa_pos_err**2)).round(decimals=3)
    wa_ori_rmse = np.sqrt(np.mean(wa_ori_err**2)).round(decimals=3)

    avg_pos_rmse = np.sqrt(np.mean(avg_pos_err**2)).round(decimals=3)
    avg_ori_rmse = np.sqrt(np.mean(avg_ori_err**2)).round(decimals=3)

    rmse_avg_vals = [highest_pos_rmse , highest_ori_rmse, wa_pos_rmse, wa_ori_rmse, avg_pos_rmse, avg_ori_rmse]

    rmse_vals = {
        'Filename': filename,
        'Particel Size': particles,
        'Highest Pos rmse': highest_pos_rmse,
        'Highest Ori rmse': highest_ori_rmse,
        'Wt Avg Pos rmse': wa_pos_rmse,
        'Wt Avg Ori rmse': wa_ori_rmse,
        'Avg Pos rmse': avg_pos_rmse,
        'Avg Ori rmse': avg_ori_rmse 
    }

    return RMSE_plot_w_a ,RMSE_plot_h ,RMSE_plot_a, rmse_vals


def Calculate_RMSE_EKF(filename, filter_pos, filter_ori, aligned_ts):
    """
    Calculates the RMSE and returns a tuple of 3 arrays
    """
    file_path = 'data\\' + filename

    # Load data 
    student_data = scipy.io.loadmat(file_path, simplify_cells=True)

    vicon = student_data['vicon']
    vicon_time = student_data['time']
    vicon_data = np.array(vicon).T

    gt_pos = vicon_data[:, :3]
    gt_ori = vicon_data[:, 3:6]

    filter_pos = np.array(filter_pos, dtype=float)
    filter_ori = np.array(filter_ori, dtype=float)


    aligned_idx = []
    for aligned_ts in aligned_ts:
        closest_idx = np.argmin(np.abs(vicon_time - aligned_ts))
        aligned_idx.append(closest_idx)

    aligned_gt_pos = gt_pos[aligned_idx]
    aligned_gt_ori = gt_ori[aligned_idx]

    # Calculate rmse 
    filter_pos_err = np.sqrt(np.sum((filter_pos - aligned_gt_pos)**2, axis=1))
    filter_ori_err = np.sqrt(np.sum((filter_pos - aligned_gt_ori)**2, axis=1))

    RMSE_plot = np.sqrt(filter_pos_err**2 + filter_ori_err**2)
    filter_pos_rmse = np.sqrt(np.mean(filter_pos_err**2)).round(decimals=3)
    filter_ori_rmse = np.sqrt(np.mean(filter_ori_err**2)).round(decimals=3)

    # Organize rmse values into a dictionary and append to the global list
    rmse_ekf_vals = {
        'Filename': filename,
        ' EKF Pos rmse': filter_pos_rmse,
        ' EKF Ori rmse': filter_ori_rmse,

    }

    return RMSE_plot, rmse_ekf_vals 


