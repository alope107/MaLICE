import datetime
import os
import pathlib
import sys

from fpdf import fpdf, FPDF
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
import nmrglue as ng
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

from malice.resources import fonts
from malice.seeds import get_base_seed


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class CompLEx_PDF(FPDF):
 
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 22/72, txt='CompLEx-NMR v1', ln=0, align='L')
        self.cell(0, 22/72, txt='Job # blahdeblah', ln=1, align='R')
        #self.ln(1)
 
    def footer(self):
        self.set_y(-0.3)
        self.set_font('Arial', 'I', 8)
 
        # Add a page number
        page = 'Page ' + str(self.page_no())
        self.cell(0, 0, page, 0, 0, 'C')

def generate_sim_spectrum(optimizer, fit_peaks, larmor, residue, titrant_concs, color_palette, background_palette, figure_path):
    # Sample sparky dictionary
    udic = {'ndim': 2,
            0: {'car': larmor*0.10137*119,
                'complex': False,
                'encoding': 'states',
                'freq': True,
                'label': '15N',
                'obs': larmor*0.10137,
                'size': 512,
                'sw': larmor*0.10137*38,
                'time': False},
            1: {'car': 4.784*larmor,
                'complex': False,
                'encoding': 'direct',
                'freq': True,
                'label': '1H',
                'obs': larmor,
                'size': 2048,
                'sw': larmor*14,
                'time': False}
    }
    
    dic = ng.sparky.create_dic(udic)
    shape = (udic[0]['size'], udic[1]['size'])
    data = np.empty(shape ,dtype='float32')
    lineshapes = ('l','l')
    
    # convert the peak list from PPM to points
    uc_15N = ng.sparky.make_uc(dic, None, 0)
    uc_1H = ng.sparky.make_uc(dic, None, 1)
    
    #set contour levels
    #contour_start = optimizer.data.intensity.min()/5
    contour_start = optimizer.data.intensity.median()/20
    #contour_start = optimizer.reference.I_ref.min()/5
    contour_num = 15
    contour_factor = 1.2
    contour_levels = contour_start * contour_factor ** np.arange(contour_num)
    contour_linewidth = 0.2
    
    
    #max_csp = float(max([fit_peaks.csp.max(), fit_peaks.csfit.max()]))
    max_csp = float(optimizer.deltaw_df.delta_w.max())
    
    ## Overwrite with max_dw
    if max_csp < 0.15:   max_csp = 0.15

    h_ref = float(optimizer.reference.loc[optimizer.reference.residue == residue, '1H_ref'])
    n_ref = float(optimizer.reference.loc[optimizer.reference.residue == residue, '15N_ref'])
    amplitude_scaler = optimizer.ml_model[3]
    ## Initialize the figure
    fig, ax = plt.subplots(figsize=(3.1,3.1))
    ax.set(xlim=(h_ref+1.6*max_csp, h_ref-1.6*max_csp),
           ylim=(n_ref+1.6*max_csp/optimizer.nh_scale, n_ref-1.6*max_csp/optimizer.nh_scale))
    ax.set_title('Simulated $^{15}$N-HSQC',fontsize=10)
    ax.set_xlabel('$^1$H (ppm)', fontsize=8)
    ax.set_ylabel('$^{15}$N (ppm)', fontsize=8)
    ax.tick_params(labelsize=7)
    
    ## 200325 Want to change the way that I draw the peaks all together
    ## Instead of simulated peaks (which confuses everyone), now going to draw "real" spectra with linewidths
    ## just inferred from the user input data. But let's try drawing the entire titration series first using
    ## a grey color series, and then layer on the colored focal peaks (making the grey for that residue invisible).
    
    # Draw background
    zorder = 0
    background_colors = dict(zip(titrant_concs,background_palette))
    for titrant_conc in titrant_concs:
        background_params = []
        #data_at_conc = fit_peaks.copy()[fit_peaks.titrant == titrant_conc]
        background_intensities = []
        for bg_residue in list(fit_peaks[fit_peaks.titrant == titrant_conc].residue.unique()):
            bg_focal_peak = fit_peaks.copy()[(fit_peaks.titrant == titrant_conc) & (fit_peaks.residue == bg_residue)]
            if not np.isnan(bg_focal_peak.intensity.iloc[0]):
                bg_focal_intensity = float(bg_focal_peak.intensity.mean())
                background_intensities.append(bg_focal_intensity)
                n_loc = float(bg_focal_peak['15N'])
                pts_15N = uc_15N.f(n_loc, 'ppm')
                h_loc = float(bg_focal_peak['1H'])
                pts_1H = uc_1H.f(h_loc, 'ppm')

                lw_Hz = amplitude_scaler/bg_focal_intensity
                lw_1H = lw_Hz/udic[1]['sw']*udic[1]['size']
                lw_15N = lw_Hz/udic[0]['sw']*udic[0]['size']*optimizer.nh_scale

                background_params.append([(pts_15N, lw_15N), (pts_1H, lw_1H)])

        #background_intensities = list(data_at_conc.intensity)
        simulated_background_data = ng.linesh.sim_NDregion(shape,
                                                           lineshapes,
                                                           background_params,
                                                           background_intensities)
        
        # Make ppm scales
        background_uc_1h = ng.sparky.make_uc(dic, simulated_background_data, dim=1)
        background_ppm_1h = background_uc_1h.ppm_scale()
        background_ppm_1h_0, background_ppm_1h_1 = background_uc_1h.ppm_limits()
        background_uc_15n = ng.sparky.make_uc(dic, simulated_background_data, dim=0)
        background_ppm_15n = background_uc_15n.ppm_scale()
        background_ppm_15n_0, background_ppm_15n_1 = background_uc_15n.ppm_limits()

        zorder += 1

        ax.contour(simulated_background_data, contour_levels, 
                   colors = [background_colors[titrant_conc]],
                   extent=(background_ppm_1h_0, background_ppm_1h_1, 
                           background_ppm_15n_0, background_ppm_15n_1),
                   linewidths=contour_linewidth, zorder=zorder)
    
    #Draw focal residue
    focal_peaks = fit_peaks.copy()[fit_peaks.residue == residue].sort_values('titrant', ascending=True)
    #focal_peaks['15N_ref'] = float(optimizer.reference.loc[optimizer.reference.residue == residue,'15N_ref'])
    #focal_peaks['1H_ref'] = float(optimizer.reference.loc[optimizer.reference.residue == residue,'1H_ref'])
    titrant_colors = dict(zip(titrant_concs,color_palette))
    for titrant_conc in titrant_concs:
        focal_peak = focal_peaks.copy()[focal_peaks.titrant == titrant_conc]
        focal_intensity = [float(focal_peak.intensity.mean())]
        
        pt_15N = uc_15N.f(float(focal_peak['15N']), 'ppm')
        pt_1H = uc_1H.f(float(focal_peak['1H']), 'ppm')
        
        lw_Hz = amplitude_scaler/float(focal_peak.intensity)
        lw_1H = lw_Hz/udic[1]['sw']*udic[1]['size']
        lw_15N = lw_Hz/udic[0]['sw']*udic[0]['size']*optimizer.nh_scale
        
        focal_param = [[(pt_15N, lw_15N), (pt_1H, lw_1H)]]

        simulated_focal_data = ng.linesh.sim_NDregion(shape,
                                                      lineshapes,
                                                      focal_param,
                                                      focal_intensity)
        
        # make ppm scales
        focal_uc_1h = ng.sparky.make_uc(dic, simulated_focal_data, dim=1)
        focal_ppm_1h = focal_uc_1h.ppm_scale()
        focal_ppm_1h_0, focal_ppm_1h_1 = focal_uc_1h.ppm_limits()
        focal_uc_15n = ng.sparky.make_uc(dic, simulated_focal_data, dim=0)
        focal_ppm_15n = focal_uc_15n.ppm_scale()
        focal_ppm_15n_0, focal_ppm_15n_1 = focal_uc_15n.ppm_limits()

        zorder+=1

        ax.contour(simulated_focal_data, contour_levels, 
                   colors = [titrant_colors[titrant_conc]],
                   extent=(focal_ppm_1h_0, focal_ppm_1h_1, focal_ppm_15n_0, focal_ppm_15n_1),
                   linewidths=contour_linewidth, zorder=zorder)
    
    

    # Add the CompLEx estimated values to plot
    theta = float(optimizer.deltaw_df.estimated_theta[optimizer.deltaw_df.residue == residue])
    zorder+=1
    focal_peaks['15N_fit'] = focal_peaks['15N_ref'] + focal_peaks.csfit*np.sin(theta)/optimizer.nh_scale
    focal_peaks['1H_fit'] = focal_peaks['1H_ref'] + focal_peaks.csfit*np.cos(theta)

    ## Based on my current tests, the added points look very good with s=48 and linewidth=1.2
    ## if max_csp = 0.15, so let's define the sizes based on that
    scatter_point_size = 7.2 / max_csp
    scatter_point_lw = 0.18 / max_csp
    ax.scatter('1H_fit', '15N_fit', data=focal_peaks, c=color_palette, s=scatter_point_size, 
               zorder=zorder, edgecolors='black', linewidth=scatter_point_lw)
    
    ## If the delta_w is sufficiently large, draw a dotted line around where the end point
    ## should end up being
    
    focal_delta_w = float(optimizer.deltaw_df.delta_w[optimizer.deltaw_df.residue == residue])
    focal_theta_p = float(optimizer.deltaw_df.corrected_theta_p_value[optimizer.deltaw_df.residue == residue])
    if focal_theta_p < 0.05 or focal_delta_w >= 0.05:
        zorder+=1
        ax.add_artist(Ellipse((h_ref,n_ref), 2*focal_delta_w, 2*focal_delta_w/optimizer.nh_scale,
                              linestyle='dashed', fill=0, color='black', zorder=zorder))
    
    ## Add a trajectory line in theta was estimated to be significant
    
    if focal_theta_p < 0.05:
        H_traj = focal_peaks['1H_ref'].mean() + np.linspace(0, focal_delta_w, 1000)*np.cos(theta)
        N_traj = focal_peaks['15N_ref'].mean() + np.linspace(0, focal_delta_w, 1000)*np.sin(theta)/optimizer.nh_scale

        zorder+=1
        ax.plot(H_traj, N_traj, linestyle='solid', c='black', linewidth=1, zorder=zorder)
    
    fig.tight_layout(pad=0.3)
    
    file_name = 'simulated_hsqc_residue_'+str(residue)+'.jpg'
    
    fig.savefig(os.path.join(figure_path, file_name), dpi=300, quality=80)
    plt.close(fig)
    
    return file_name


# Sets the directory fpdf searches for fonts
def set_font_dir():
     font_dir = pathlib.Path(fonts.__file__).parent.absolute()
     fpdf.FPDF_FONT_DIR = font_dir

def CompLEx_Report(optimizer, config, performance, lam, image_dir):
    pdf = CompLEx_PDF(orientation='portrait', unit='in', format='letter')
    pdf.set_margins(0.5,0.5,0.5)
    set_font_dir()
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
    pdf.add_font('DejaVu', 'I', 'DejaVuSansCondensed-Oblique.ttf', uni=True)
    pdf.add_font('DejaVu', 'BI', 'DejaVuSansCondensed-BoldOblique.ttf', uni=True)
    fixed_width = "DejaVu"
    pdf.add_page()
    
    pdf.set_font('Arial','B',16)
    pdf.cell(0,20/72,txt='Report of CompLEx Results',ln=1,align='C')
    pdf.ln(0.1)
    
    pdf.set_font('Arial','B',12)
    pdf.cell(5, 12/72, txt='', ln=0)
    pdf.cell(1, 12/72, txt='Concentrations (uM)', ln=1, align='C')
    pdf.cell(1.8, 12/72, txt='NMR visible protein: ', ln=0, align='R')
    pdf.set_font('Arial','',12)
    pdf.cell(3.2, 12/72, txt=config.visible, ln=0, align='L')
    if len(optimizer.data.visible.unique()) == 1:
        pdf.cell(1, 12/72, txt=str(int(optimizer.data.visible[0])), ln=1, align='C')
    else:
        pdf.cell(1, 12/72, txt=str(round(optimizer.data.visible.min()))+' - '+str(round(optimizer.data.visible.max())), ln=1, align='C')
    pdf.set_font('Arial','B',12)
    pdf.cell(1.8, 12/72, txt='Titrant: ', ln=0, align='R')
    pdf.set_font('Arial','',12)
    pdf.cell(3.2, 12/72, txt=config.titrant, ln=0, align='L')
    pdf.cell(1, 12/72, txt=str(round(optimizer.data.titrant.min()))+' - '+str(round(optimizer.data.titrant.max())), ln=1, align='C')
    pdf.ln(0.2)
    pdf.set_font('Arial','B',12)
    pdf.cell(1.8, 12/72, txt='Data file: ', ln=0, align='R')
    pdf.set_font('Arial','',12)
    pdf.cell(5, 12/72, txt=config.input_file.split('/')[-1], ln=1)
    pdf.ln(0.2)
    
    init_x, init_y = pdf.get_x(), pdf.get_y()
    
    fixed_params_text = [('# of global variables', optimizer.gvs, 0),
                         ('15N-1H scaler', optimizer.nh_scale, 2), 
                         ('Larmor frequency (Hz)', optimizer.larmor, 0)]
    pdf.set_font(fixed_width,'BI',12)
    pdf.cell(3,14/72,txt='Fixed parameters:',ln=1,border=0,align='L')
    for name, var, round_digits in fixed_params_text:
        pdf.set_font(fixed_width,'I',10)
        pdf.cell(2.2, 10/72, txt='    '+name+':', ln=0)
        pdf.set_font(fixed_width,'B',10)
        pdf.cell(0.5, 10/72, txt=str(round(var,round_digits)), ln=1, align='R')
    
    pdf.ln(0.2)
    
    opt_params_text = [('L1 λ', format(lam, '.3f')),
                       ('Phase 1 islands', config.phase1_islands),
                       ('Phase 1 generations', config.phase1_generations),
                       ('Phase 1 rounds of evolution', config.phase1_evo_rounds),
                       ('Phase 2 islands', config.phase2_islands),
                       ('Phase 2 generations', config.phase2_generations),
                       ('Phase 2 rounds of evolution', config.phase2_evo_rounds),
                       ('Population size', config.pop_size),
                       ('MCMC walks', config.mcmc_walks),
                       ('MCMC steps', config.mcmc_steps),
                       ('Founder seed', get_base_seed()),
                       ('PyGMO tolerance', format(config.tolerance, '.1g')),
                       ('Least squares max iterations', format(config.least_squares_max_iter,'.1g'))]
    pdf.set_font(fixed_width,'BI',12)
    pdf.cell(3,14/72,txt='Optimization settings:',ln=1,border=0,align='L')
    for name, var in opt_params_text:
        pdf.set_font(fixed_width,'I',10)
        pdf.cell(2.2, 10/72, txt='    '+name+':', ln=0)
        pdf.set_font(fixed_width,'B',10)
        pdf.cell(0.5, 10/72, txt=str(var), ln=1, align='R')
    
    
    curr_x = init_x + 3.1
    pdf.set_xy(curr_x, init_y)
    
    ## Insert model performance results here
    pdf.set_font(fixed_width,'BI',12)
    pdf.cell(3,14/72,txt='Model performance:',ln=1,border=0,align='L')
    
    performance_text = (('L1 Model Penalized -logL', format(performance['l1_model_score'],'.1f')),
                        ('L1 Model Unpenalized -logL', format(performance['l1_unpenalized_score'],'.1f')),
                        ('Reference optimized -logL', format(performance['opt_ref_score'],'.1f')),
                        ('Scaled Δω optimized -logL', format(performance['scaled_dw_score'],'.1f')),
                        ('Final model -logL', format(performance['ml_model_score'],'.1f')),
                        ('L1 optimization runtime', str(datetime.timedelta(seconds=performance['phase1_time'])).split('.')[0]),
                        ('Reference optimization runtime', str(datetime.timedelta(seconds=performance['phase2_time'])).split('.')[0]),
                        ('ML optimization runtime', str(datetime.timedelta(seconds=performance['phase3_time'])).split('.')[0]),
                        ('MCMC runtime', str(datetime.timedelta(seconds=performance['phase4_time'])).split('.')[0]),
                        ('Total runtime', str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0]))
    
    for name, var in performance_text:
        pdf.set_font(fixed_width,'I',10)
        pdf.set_x(curr_x)
        pdf.cell(2.5, 10/72, txt='    '+name+':', ln=0)
        pdf.set_font(fixed_width,'B',10)
        pdf.cell(0.5, 10/72, txt=var, ln=1, align='R')
    
    pdf.ln(0.1)
    
    ## Insert table of global variable and estimates
    #width = 0.85
    width = 1.0
    
    pdf.set_line_width(1/72)
    #pdf.set_fill_color(255, 255, 255)
    pdf.rect(curr_x, pdf.get_y()-0.02, 4*width+0.02,(12+7*10)/72+0.10, 'D')

    global_variables_text = (('Kd (uM)', format(np.power(10.0,optimizer.ml_model[0]),'.1f'), format(np.power(10.0,optimizer.lower_conf_limits[0]),'.1f'),
                              format(np.power(10.0,optimizer.upper_conf_limits[0]),'.1f')),
                             ('koff (Hz)', format(np.power(10.0,optimizer.ml_model[1]),'.1f'), format(np.power(10.0,optimizer.lower_conf_limits[1]),'.1f'),
                              format(np.power(10.0,optimizer.upper_conf_limits[1]),'.1f')),
                             ('ΔR2 (Hz)', format(optimizer.ml_model[2],'.2f'), format(optimizer.lower_conf_limits[2],'.2f'), format(optimizer.upper_conf_limits[2],'.2f')),
                             ('Amplitude scaler', format(optimizer.ml_model[3],'.2g'), format(optimizer.lower_conf_limits[3],'.2g'), format(optimizer.upper_conf_limits[3],'.2g')),
                             ('Intensity ε', format(optimizer.ml_model[4],'.2g'), format(optimizer.lower_conf_limits[4],'.2g'), format(optimizer.upper_conf_limits[4],'.2g')),
                             ('CS ε (Hz)', format(optimizer.ml_model[5],'.2f'), format(optimizer.lower_conf_limits[5],'.2f'), format(optimizer.upper_conf_limits[5],'.2f')))  
    pdf.set_font(fixed_width,'BI',12)
    pdf.set_x(curr_x)
    pdf.cell(3,14/72,txt='Global variables:',ln=1,border=0,align='L')
    pdf.ln(0.05)
    pdf.set_font(fixed_width,'BIU',10)
    pdf.set_x(curr_x)
    pdf.cell(width, 10/72, txt='Variable', ln=0, align='L')
    pdf.set_x(curr_x+width)
    pdf.cell(width, 10/72, txt='ML Estimate', ln=0, align='C')
    pdf.set_x(curr_x+2*width)
    alpha_l = (1-config.confidence)/2
    alpha_u = 1-alpha_l
    pdf.cell(width, 10/72, txt=format(alpha_l,'.1%')+' CL', ln=0, align='C')
    pdf.set_x(curr_x+3*width)
    pdf.cell(width, 10/72, txt=format(alpha_u,'.1%')+ 'CL', ln=1, align='C')
    #for name, mle, stderr, lower_cl, upper_cl in global_variables_text:
    for name, mle, lower_cl, upper_cl in global_variables_text:
        pdf.set_x(curr_x)
        pdf.set_font(fixed_width,'I',10)
        pdf.cell(width, 10/72, txt = name, ln=0, align='L')
        pdf.set_font(fixed_width,'B',10)
        pdf.set_x(curr_x+width)
        pdf.cell(width, 10/72, txt = str(mle), ln=0, align='C')
        pdf.set_x(curr_x+2*width)
        pdf.cell(width, 10/72, txt = str(lower_cl), ln=0, align='C')
        pdf.set_x(curr_x+3*width)
        pdf.cell(width, 10/72, txt = str(upper_cl), ln=1, align='C')
    
    pdf.ln(0.1)
    
    ## Add a couple of lines about other output files that have been generated
    fname_prefix = config.input_file.split('/')[-1].split('.')[0]
    file_text = (('Reference peak optimization', fname_prefix + '_reference_peaks.csv'),
                 ('Δω spreadsheeet', fname_prefix + '_CompLEx_fits.csv'),
                 ('Δω text file', fname_prefix + '_CompLEx_deltaw.txt'))
    pdf.set_font(fixed_width,'BI',12)
    pdf.cell(3, 12/72, txt='Output files:', ln=1, align='L')
    for name, var in file_text:
        pdf.set_font(fixed_width,'',10)
        pdf.cell(3, 10/72, txt='    '+name+':', ln=0, align='L')
        pdf.set_font(fixed_width,'B',10)
        pdf.cell(3, 10/72, txt=var, ln=1, align='L')
    
    pdf.ln(0.2)
    
    ## Add plot of the per-residue delta_w fits
    
    rainbow_cmap = truncate_colormap(plt.get_cmap('jet'), 0.18, 0.92, n=1024)
    
    deltaw_df = pd.DataFrame({'residue':optimizer.residues,
                              'dw':optimizer.ml_model[optimizer.gvs:]/optimizer.larmor})
    deltaw_df['color'] = deltaw_df.residue/deltaw_df.residue.max()
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    dw_error_bars = [(np.array(optimizer.ml_model[optimizer.gvs:]) - np.array(optimizer.lower_conf_limits[optimizer.gvs:]))/optimizer.larmor,
                     (np.array(optimizer.upper_conf_limits[optimizer.gvs:]) - np.array(optimizer.ml_model[optimizer.gvs:]))/optimizer.larmor]
    ax.errorbar('residue', 'dw', data=deltaw_df, yerr=dw_error_bars,
                color='black', fmt='none', s=32, zorder=1)
    ax.scatter('residue', 'dw', data=deltaw_df, c=deltaw_df.residue, cmap=rainbow_cmap, edgecolors='black', linewidths=0.5, s=48, zorder=2)
    ax.set(title='Per-residue estimates of Δω', xlabel='Residue No.', ylabel='Δω (ppm)', 
           xlim=(deltaw_df.residue.min()-1,deltaw_df.residue.max()+1),
           ylim = (-0.02, 1.05*max(optimizer.upper_conf_limits[optimizer.gvs:])/optimizer.larmor))
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(image_dir,'deltaw_plot.jpg'), dpi=300, quality=80)
    plt.close(fig)
    
    pdf.image(os.path.join(image_dir,'deltaw_plot.jpg'), x = pdf.get_x(), y=pdf.get_y(),
              w = 7.5, h = 4.5)
    
    ## End of page 1, now to add residue-specific stuff to subsequent pages
    fit_points = optimizer.pfitter()
    optimizer.mode = 'lfitter'
    regression = optimizer.enhanced_df()
    
    lower_cl_csp_params = [optimizer.upper_conf_limits[0]] + list(optimizer.lower_conf_limits[1:])
    lower_cl_int_params = list(optimizer.lower_conf_limits[:2]) + [optimizer.upper_conf_limits[2], optimizer.lower_conf_limits[3]] + list(optimizer.upper_conf_limits[4:])
    lower_cl_csp_regression = optimizer.enhanced_df(lower_cl_csp_params)
    lower_cl_int_regression = optimizer.enhanced_df(lower_cl_int_params)
    
    
    upper_cl_csp_params = [optimizer.lower_conf_limits[0]] + list(optimizer.upper_conf_limits[1:])
    upper_cl_int_params = list(optimizer.upper_conf_limits[:2]) + [optimizer.lower_conf_limits[2], optimizer.upper_conf_limits[3]] + list(optimizer.lower_conf_limits[4:])
    upper_cl_csp_regression = optimizer.enhanced_df(upper_cl_csp_params)
    upper_cl_int_regression = optimizer.enhanced_df(upper_cl_int_params)
    
    optimizer.mode = 'simulated_peak_generation'
    regression_at_titrant_concs = optimizer.enhanced_df()
    
    ## Set up the color scheme for plotting over titrant range
    titrant_concs = list(fit_points.titrant.unique())
    titrant_concs.sort()
    
    plasma_short = truncate_colormap(cm.plasma, minval=0.2, maxval=0.8, n=256)
    grey_short = truncate_colormap(cm.Greys_r, minval=0.5, maxval=0.9, n=256)
    viridis_short = truncate_colormap(cm.viridis, minval=0.1, maxval=0.4, n=256)
    winter_short = truncate_colormap(cm.winter, minval=0.3, maxval=0.7, n=256)
    cividis_short = truncate_colormap(cm.cividis, minval=0.05, maxval=0.35, n=256)

    plasma_titrant = plasma_short( np.array(titrant_concs)/max(titrant_concs)  )
    grey_titrant = grey_short( np.array(titrant_concs)/max(titrant_concs) )
    viridis_titrant = viridis_short( np.array(titrant_concs)/max(titrant_concs) )
    winter_titrant = winter_short( np.array(titrant_concs)/max(titrant_concs) )
    cividis_titrant = cividis_short( np.array(titrant_concs)/max(titrant_concs) )
    
    residue_dw_df = pd.DataFrame({'residue':optimizer.residues,
                                  'dw':optimizer.ml_model[optimizer.gvs:],
                                  'lower':optimizer.lower_conf_limits[optimizer.gvs:],
                                  'upper':optimizer.upper_conf_limits[optimizer.gvs:]})
    
    
    
    i = 0
    for residue in optimizer.residues:
        if i%3 == 0:    
            pdf.add_page()
            pdf.ln(0.1)
        pdf.set_font(fixed_width,'B',12)
        pdf.cell(3, 12/72, txt='Residue '+str(residue), ln=1, align='L')
        
        residue_data = optimizer.data.loc[optimizer.data.residue == residue,['15N','1H']]
        residue_ref = optimizer.reference[optimizer.reference.residue == residue]
        
        residue_fits = fit_points.copy()[fit_points.residue == residue]
        residue_regression = regression[regression.residue == residue]

        residue_lower_csp_regression = lower_cl_csp_regression[lower_cl_csp_regression.residue == residue]
        residue_lower_int_regression = lower_cl_int_regression[lower_cl_int_regression.residue == residue]
        residue_upper_csp_regression = upper_cl_csp_regression[upper_cl_csp_regression.residue == residue]
        residue_upper_int_regression = upper_cl_int_regression[upper_cl_int_regression.residue == residue]
        
        
        # Add the regression lines for both the intensity and csp fits
        titrant_rng = fit_points.titrant.max()-fit_points.titrant.min()
        xl = [np.min(fit_points.titrant)-0.05*titrant_rng, np.max(fit_points.titrant)+0.05*titrant_rng]  
        csp_rng = np.max(fit_points.csp)-np.min(fit_points.csp)
        yl_csp = [np.min(fit_points.csp)-0.08*csp_rng, np.max(fit_points.csp)+0.08*csp_rng]
        int_rng = np.max(fit_points.intensity)-np.min(fit_points.intensity)
        yl_int = [np.min(fit_points.intensity)-0.08*int_rng, np.max(fit_points.intensity)+0.08*int_rng]
        
        
        yaxes = [yl_csp, yl_int]
        ylabels = ['CSP (ppm)', 'Intensity']
        points = ['csp', 'intensity']
        lines = ['csfit', 'ifit']
        residue_lower_conf_data = [residue_lower_csp_regression, residue_lower_int_regression]
        residue_upper_conf_data = [residue_upper_csp_regression, residue_upper_int_regression]
        errors = [optimizer.ml_model[5]/optimizer.larmor, optimizer.ml_model[4]]
        file_names = ['cs_regression_residue_', 'intensity_regression_residue_']

        conf_draw_flag_cs = 0
        conf_draw_flag_int = 0

        fig, ax = plt.subplots(nrows=2, figsize=(2.5,3.1))
        color_index = [plasma_titrant[titrant_concs.index(x)] for x in residue_fits.titrant]
        for i in range(2):
            #ax[i].scatter('tit', points[i], data=residue_fits, color='black', s=10, zorder=3)
            ax[i].scatter('titrant', points[i], data=residue_fits, c=color_index, s=32, linewidths=1, edgecolors='black', zorder=5)
            ax[i].errorbar('titrant', points[i], data=residue_fits, yerr=errors[i], color='black', fmt='none', s=16, zorder=4)
            ax[i].plot('titrant', lines[i], data=residue_regression, color='black', zorder=3)
            # Draw confidence intervals if they are consistently above/below the regression line
            if np.all(residue_regression[lines[i]] >= residue_lower_conf_data[i][lines[i]]) and np.all(residue_regression[lines[i]] <= residue_upper_conf_data[i][lines[i]]):
                ax[i].plot('titrant',lines[i], data=residue_lower_conf_data[i], color='black', linestyle='dotted', zorder=2)
                ax[i].plot('titrant',lines[i], data=residue_upper_conf_data[i], color='black', linestyle='dotted', zorder=2)
                ax[i].fill_between(residue_lower_conf_data[i].titrant, residue_lower_conf_data[i][lines[i]], residue_upper_conf_data[i][lines[i]], color=[cm.Greys_r(0.7)], zorder=1)
            else:
                # Make an output warning of potentially slow exchange
                ## TO DO: ADD CODE TO ACTUALLY OUTPUT THIS SOMEWHERE IN THE REPORT
                if i == 0:  conf_draw_flag_cs = 1
                else:   conf_draw_flag_int = 1
            ax[i].set(xlim=xl, ylim=yaxes[i])
            ax[i].set_xlabel('Titrant (μM)', fontsize=8)
            ax[i].set_ylabel(ylabels[i], fontsize=8)
            ax[i].yaxis.offsetText.set_fontsize(7)
            ax[i].tick_params(labelsize=7)
            
            # Calculate R^2
            #R2 = (np.var(residue_fits[points[i]]) - np.var(residue_fits[points[i]]-residue_fits[lines[i]]))/np.var(residue_fits[points[i]])
            #ss_total = np.sum( np.square( residue_fits[points[i]] - np.mean(residue_fits[points[i]]) ) )
            #ss_residual = np.sum( np.square(residue_fits[points[i]]-residue_fits[lines[i]]) )
            #R2 = 1 - ss_residual/ss_total
            slope, intercept, r_value, p_value, std_err = stats.linregress(residue_fits[points[i]],residue_fits[lines[i]])
            ax[i].text(xl[0]+0.05*(xl[1]-xl[0]), yaxes[i][1]-0.15*(yaxes[i][1]-yaxes[i][0]), 'r = '+format(r_value,'.2f'),fontsize=7)
        fig.tight_layout(pad=0.3,h_pad=0.1)
        #fig_path = os.path.join(image_dir,file_names[i]+str(residue)+'.png')
        fig_path = os.path.join(image_dir,'regression_lines_residue_'+str(residue)+'.png')
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
            
        pdf.image(fig_path, x=pdf.get_x(), y=pdf.get_y(), w=2.5, h=3.1)

        # Add the simulated spectrum figure
        figure_name = generate_sim_spectrum(optimizer, 
                                            regression_at_titrant_concs, 
                                            optimizer.larmor, 
                                            residue,
                                            #theta_est,
                                            #theta_p_value,
                                            titrant_concs,
                                            plasma_titrant,
                                            cividis_titrant, 
                                            image_dir)
        pdf.image(os.path.join(image_dir,figure_name), x = pdf.get_x()+2.5, y=pdf.get_y(),
              w = 3.1, h = 3.1)
        
        
        ## Add relevant text detailing fits of each residue
        pdf.ln(0.2)
        pdf.set_x(pdf.get_x()+5.7)
        pdf.set_font(fixed_width,'I',8)
        pdf.cell(1.5, 8/72, txt = '# of observed titration points:', ln=0, align='L')
        pdf.set_font(fixed_width,'B',8)
        pdf.cell(0.2, 8/72, txt = str(len(residue_fits)), ln=1, align='R')
        pdf.ln(10/72)
        
        res_data = optimizer.data.copy()[optimizer.data.residue == residue]
        original_ref = res_data[res_data.titrant == np.min(res_data.titrant)]
        current_ref = optimizer.reference.loc[optimizer.reference.residue == residue]
        reference_text = ( ('15N (ppm)', format(float(current_ref['15N_ref']),'.3f'),
                            format(float(original_ref['15N'])-float(current_ref['15N_ref']),'.3f')),
                           ('1H (ppm)', format(float(current_ref['1H_ref']),'.3f'),
                            format(float(original_ref['1H'])-float(current_ref['1H_ref']),'.3f')),
                           ('Intensity', format(float(current_ref['I_ref']),'.2g'),
                            format(float(original_ref['intensity'])-float(current_ref['I_ref']),'.2g')) )
        
        pdf.set_x(pdf.get_x()+5.7)
        pdf.set_font(fixed_width,'I',9)
        pdf.cell(2.8, 10/72, txt = 'Reference peak:', ln=1, align='L')
        for name, value, change in reference_text:
            pdf.set_x(pdf.get_x()+5.7)
            pdf.set_font(fixed_width,'I',7)
            pdf.cell(0.55, 7/72, txt = '    '+name, ln=0, align='L')
            pdf.set_font(fixed_width,'B',7)
            pdf.cell(0.6, 7/72, txt = value, ln=0, align='R')
            if np.sign(float(change)) > 0: sign = '+'
            else:   sign = ''
            pdf.cell(0.65, 7/72, txt = '('+sign+change+')', ln=1, align='R')
        pdf.ln(10/72)
        
        
        
        pdf.set_x(pdf.get_x()+5.7+0.34+0.47)
        pdf.set_font(fixed_width,'I',8)
        pdf.cell(0.47, 9/72, txt = format(alpha_l,'.1%')+' CL', ln=0, align='C')
        pdf.cell(0.47, 9/72, txt = format(alpha_u,'.1%')+' CL', ln=1, align='C')
        pdf.set_x(pdf.get_x()+5.7)
        pdf.cell(0.34, 9/72, txt = 'Δω (Hz):', ln=0, align='L')
        pdf.set_font(fixed_width,'B',8)
        pdf.cell(0.47, 9/72, txt = format(float(residue_dw_df.loc[residue_dw_df.residue == residue, 'dw']),'.1f'), ln=0, align='R')
        pdf.cell(0.47, 9/72, txt = format(float(residue_dw_df.loc[residue_dw_df.residue == residue, 'lower']),'.1f'), ln=0, align='C')
        pdf.cell(0.47, 9/72, txt = format(float(residue_dw_df.loc[residue_dw_df.residue == residue, 'upper']),'.1f'), ln=1, align='C')
        
        pdf.ln(1.8)
            
        i+=1
    
    return pdf
