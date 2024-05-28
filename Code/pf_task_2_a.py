from pf_functions import ParticleFilter, Calculate_RMSE
import pandas as pd

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

## Task 2 p sizes
p_sizes = [200, 500, 750, 1000, 2000, 3000, 4000, 5000]
# p_sizes = [ 200, 500]

name_var = 1

rmse_data = []
ekf_rmse_data = []

for filename in filenames:
    for p_size in p_sizes:
        print(f"Filtering {filename} with p size: {p_size}")

        pf = ParticleFilter(p_size, filename)

        # running pf
        highest_pos, highest_ori, wa_pos, wa_ori, avg_pos, avg_ori, aligned_ts = pf.iterator(name_var)

        # Calculate RMSE for different estimation strategies.
        RMSE_plot_w_a, RMSE_plot_h, RMSE_plot_a, rmse_vals = Calculate_RMSE(
            filename, highest_pos, highest_ori,
            wa_pos, wa_ori,
            avg_pos, avg_ori,
            aligned_ts, name_var, p_size
        )

        rmse_data.append(rmse_vals)

    name_var += 1

rmse_df = pd.DataFrame(rmse_data)
rmse_df.to_csv('final_rmse_result.csv', index=True)
