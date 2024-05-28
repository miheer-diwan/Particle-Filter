import numpy as np
from pf_functions import ParticleFilter, Calculate_RMSE, plot_rmse, plotter

filenames = [
    # 'studentdata0.mat',
    'studentdata1.mat',
    'studentdata2.mat',
    'studentdata3.mat',
    'studentdata4.mat',
    'studentdata5.mat',
    'studentdata6.mat',
    'studentdata7.mat'
]

## Task 1 particles sizes
p_sizes = [1000]

name_var = 1

# Iterate over each file and particle size
for filename in filenames:

    for p_size in p_sizes:
        print(f"Processing: {filename} with particle size: {p_size}")

        pf = ParticleFilter(p_size, filename)

        # running pf
        high_fil_pos, high_fil_ori, wa_fl_pos, wa_fl_ori, avg_fil_pos, avg_fil_ori, aligned_ts = pf.iterator(name_var)

        # get rmse
        rmse_plot_wa, rmse_plot_h, rmse_plot_a, rmse_vals = Calculate_RMSE(filename, high_fil_pos, high_fil_ori,
                                                                            wa_fl_pos, wa_fl_ori, avg_fil_pos,
                                                                            avg_fil_ori, aligned_ts, p_size)

        plotter(filename, avg_fil_pos, high_fil_pos, wa_fl_pos)

        # Plot rmse results.
        plot_rmse(rmse_plot_wa, rmse_plot_h, rmse_plot_a, filename)

    name_var += 1
