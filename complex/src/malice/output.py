import os

import pandas as pd

from malice.reporter import CompLEx_Report


def make_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_output_files(optimizer, confidence, gvs, residues, fname_prefix,
                        output_dir, config, lam, performance, fit_points):
    # Generate a CSV with confidence intervals for all of the delta_w's and
    # trajectories if available
    alpha = 100.0*(1-confidence)/2
    deltaw_df = pd.DataFrame({'residue': residues,
                              'delta_w': optimizer.delta_ws(),
                              'conf_limit_'+str(round(alpha, 1)): optimizer.lower_conf_limits[gvs:],
                              'conf_limit_'+str(round(100-alpha, 1)): optimizer.upper_conf_limits[gvs:],
                              'estimated_theta': optimizer.thetas,
                              'theta_F_stat': optimizer.theta_F,
                              'uncorrected_theta_p_value': optimizer.theta_up})
    deltaw_df['corrected_theta_p_value'] = deltaw_df.uncorrected_theta_p_value * len(residues) / deltaw_df.uncorrected_theta_p_value.rank(ascending=False)
    optimizer.deltaw_df = deltaw_df

    # Print out data
    csv_name = os.path.join(output_dir, fname_prefix + '_CompLEx_deltaw.csv')
    txt_name = os.path.join(output_dir, fname_prefix+'_CompLEx_deltaw.txt')
    deltaw_df.to_csv(csv_name, index=False)
    deltaw_df[['residue', 'delta_w']].to_csv(txt_name, index=False, header=False)

    # Record the fit points to a file
    fit_points_name = os.path.join(output_dir, fname_prefix+'_CompLEx_fit_points.csv')
    fit_points.to_csv(fit_points_name, index=False)

    # Make a folder for figures
    image_dir = os.path.join(output_dir, 'images')
    make_output_dir(image_dir)

    # Generate summary PDF
    summary_pdf = CompLEx_Report(optimizer, config, performance, lam, image_dir)
    summary_pdf_name = os.path.join(output_dir, fname_prefix + '_CompLEx_summary.pdf')
    summary_pdf.output(summary_pdf_name)
