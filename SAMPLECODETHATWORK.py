import streamlit as st
import numpy as np
from scipy.optimize import least_squares
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from io import BytesIO
from matplotlib.figure import Figure
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Self-Potential Inversion Tool",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .stDownloadButton button {
        width: 100%;
    }
    h1, h2, h3 {
        padding-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Self-Potential Data Inversion Tool")
st.markdown("""
This application performs inversion of self-potential data using different forward models.
Upload your data, select a model type, configure inversion parameters, and visualize the results.
""")

# Define forward modeling functions for different models
def forward_spherical(par, x_inp):
    """Spherical model forward function"""
    var_x0 = par[0]
    var_alpha = par[1]
    var_h = par[2]
    var_k = par[3]
    var_sp = []
    for i in x_inp:
        var_up = (i - var_x0) * np.cos(var_alpha * np.pi / 180) - var_h * np.sin(var_alpha * np.pi / 180)
        var_down = ((i - var_x0) ** 2 + var_h ** 2) ** (3 / 2)
        var = var_k * (var_up / var_down)
        var_sp.append(var)
    return np.array(var_sp)

def forward_cylinder(par, x_inp):
    """Horizontal and vertical cylinder model forward function"""
    var_x0 = par[0]
    var_alpha = par[1]
    var_h = par[2]
    var_k = par[3]
    var_sp = []
    for i in x_inp:
        # Horizontal cylinder component
        var_up = (i - var_x0) * np.cos(var_alpha * np.pi / 180) - var_h * np.sin(var_alpha * np.pi / 180)
        var_down = ((i - var_x0)**2 + var_h**2) ** (1.0)
        
        # Vertical cylinder component
        var_up1 = (i - var_x0) * np.cos(var_alpha * np.pi / 180) - var_h * np.sin(var_alpha * np.pi / 180)
        var_down1 = ((i - var_x0)**2 + var_h**2) ** (0.5)
        
        # Combined effect
        var = var_k * ((var_up / var_down)+(var_up1 / var_down1))
        var_sp.append(var)
    return np.array(var_sp)

def forward_fault(par, x_inp):
    """Inclined fault model forward function"""
    var_x0 = par[0]
    var_alpha = par[1]
    var_h = par[2]
    var_k = par[3]
    a = par[4]  # Half-width of the fault
    
    b = a * np.cos(var_alpha * np.pi / 180)
    d = a * np.sin(var_alpha * np.pi / 180)
    
    var_sp = []
    for i in x_inp:
        var_up = ((i - var_x0) + b) * ((i - var_x0) + b) + (var_h - d) * (var_h - d)
        var_down = ((i - var_x0) - b) * ((i - var_x0) - b) + (var_h + d) * (var_h + d)
        
        # Avoid division by zero or negative values in logarithm
        ratio = np.clip(var_up / var_down, 1e-10, None)
        var = var_k * np.log(ratio)
        var_sp.append(var)
    
    return np.array(var_sp)

# Define the misfit equation
def pers(var_m, x_inp, y, model_type, fault_half_width=None):
    """Calculate residuals between model and observed data"""
    if model_type == "Spherical":
        return forward_spherical(var_m, x_inp) - y
    elif model_type == "Horizontal Cylindrical" or model_type == "Vertical Cylindrical":
        return forward_cylinder(var_m, x_inp) - y
    elif model_type == "Fault":
        # For fault model, we need to pass 5 parameters including half-width
        par = list(var_m) + [fault_half_width]
        return forward_fault(par, x_inp) - y

# Define the log-likelihood function
def log_likelihood(params, x_inp, y, model_type, fault_half_width=None):
    """Calculate log-likelihood for MCMC"""
    residuals = pers(params, x_inp, y, model_type, fault_half_width)
    return -0.5 * np.sum(residuals ** 2)

# Define the log-prior function
def log_prior(params, bounds):
    """Prior distribution for MCMC (uniform within bounds)"""
    lb, ub = bounds
    if all(l <= p <= u for p, l, u in zip(params, lb, ub)):
        return 0.0  # Within bounds
    else:
        return -np.inf  # Outside bounds

# Define the log-posterior function
def log_posterior(params, x_inp, y, model_type, bounds, fault_half_width=None):
    """Posterior distribution for MCMC"""
    log_prior_val = log_prior(params, bounds)
    if np.isinf(log_prior_val):
        return log_prior_val
    else:
        log_likelihood_val = log_likelihood(params, x_inp, y, model_type, fault_half_width)
        return log_prior_val + log_likelihood_val

# Function to run the inversion
def run_inversion(position, sp_data, model_type, bounds, nwalkers=20, nsteps=1000, fault_half_width=None):
    """Run MCMC inversion followed by least squares refinement"""
    # Number of parameters depends on the model
    ndim = 4  # Default for most models
    
    # Initialize walkers randomly within the parameter bounds
    lb, ub = bounds
    initial_positions = np.random.uniform(low=lb, high=ub, size=(nwalkers, ndim))
    
    # Set up progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Set up the sampler with appropriate arguments based on model
    if model_type == "Fault":
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, 
            args=(position, sp_data, model_type, bounds, fault_half_width)
        )
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, 
            args=(position, sp_data, model_type, bounds)
        )
    
    # Run the sampler with progress updates
    for i, _ in enumerate(sampler.sample(initial_positions, iterations=nsteps, progress=False)):
        progress_bar.progress((i + 1) / nsteps)
        status_text.text(f"Running MCMC: {i+1}/{nsteps} iterations completed")
        if (i+1) % 100 == 0:
            time.sleep(0.1)  # Small delay to allow UI to update
    
    # Get the samples from the sampler
    samples = sampler.get_chain()
    
    # Calculate the log posterior probabilities for each sample
    log_probs = sampler.get_log_prob()
    
    # Find the best parameters across all walkers
    best_params = None
    max_log_prob = -np.inf
    
    for walker in range(nwalkers):
        walker_log_probs = log_probs[:, walker]
        best_index = np.argmax(walker_log_probs)
        
        if walker_log_probs[best_index] > max_log_prob:
            max_log_prob = walker_log_probs[best_index]
            best_params = samples[best_index, walker, :]
    
    status_text.text("Running Least Squares optimization to refine solution...")
    
    # Run least squares with the best parameters to refine the solution
    if model_type == "Fault":
        # For fault model, define a wrapper function to handle the fault_half_width
        def pers_wrapper(params):
            return pers(params, position, sp_data, model_type, fault_half_width)
        result = least_squares(pers_wrapper, best_params, method='lm')
    else:
        result = least_squares(
            lambda params: pers(params, position, sp_data, model_type), 
            best_params, 
            method='lm'
        )
    
    # Calculate the SP data using the optimized parameters
    if model_type == "Spherical":
        sp_calculation = forward_spherical(result.x, position)
    elif model_type == "Horizontal Cylindrical" or model_type == "Vertical Cylindrical":
        sp_calculation = forward_cylinder(result.x, position)
    elif model_type == "Fault":
        sp_calculation = forward_fault(list(result.x) + [fault_half_width], position)
    
    # Calculate error
    error = result.optimality * 100
    
    # Clear progress bar and status text
    progress_bar.empty()
    status_text.empty()
    
    return result.x, sp_calculation, error, samples

# Create sidebar for file upload and model configuration
with st.sidebar:
    st.header("Data Input")
    
    # File upload option
    uploaded_file = st.file_uploader("Upload data file (TXT/CSV)", type=["txt", "csv"])
    
    # Sample data option
    use_sample_data = st.checkbox("Use sample data", value=not bool(uploaded_file))
    
    # Data format info
    with st.expander("Data Format Information"):
        st.info("""
        Expected data format:
        - Two-column data: Position (x) and SP values
        - Can be TXT or CSV format
        - If CSV, ensure proper delimiter is selected
        """)
    
    # Delimiter selection for CSV files
    delimiter = st.selectbox(
        "Column Delimiter", 
        options=[",", " ", "\t"], 
        index=1,  # Default to space delimiter
        help="Select the delimiter used in your data file"
    )
    
    st.header("Model Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Select Forward Model",
        ["Spherical", "Horizontal Cylindrical", "Vertical Cylindrical", "Fault"]
    )
    
    # Fault-specific parameters
    if model_type == "Fault":
        fault_half_width = st.number_input(
            "Fault Half-Width (m)",
            value=25.0,
            min_value=1.0,
            help="Half-width parameter for the fault model"
        )
    
    # MCMC configuration
    st.subheader("Inversion Settings")
    nwalkers = st.slider("Number of Walkers", 10, 50, 20)
    nsteps = st.slider("Number of Steps", 100, 5000, 1000)
    
    # Parameter bounds
    st.subheader("Parameter Bounds")
    
    # Common bounds
    x0_min = st.number_input("Min X0 (m)", value=2.0)
    x0_max = st.number_input("Max X0 (m)", value=100.0)
    
    alpha_min = st.number_input("Min Alpha (degrees)", value=2.0)
    alpha_max = st.number_input("Max Alpha (degrees)", value=360.0)
    
    h_min = st.number_input("Min Depth (m)", value=2.0)
    h_max = st.number_input("Max Depth (m)", value=40.0)
    
    k_min = st.number_input("Min K (contrast)", value=-1000.0)
    k_max = st.number_input("Max K (contrast)", value=1000.0)
    
    # Run button
    run_inversion_button = st.button("Run Inversion", use_container_width=True)

# Main content area
# Load and display data
# Only the modified data loading section is shown here - replace this section in your code

# In the "Load and display data" section, replace the try-except block with this:
if uploaded_file is not None:
    try:
        # Get file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            data = pd.read_csv(uploaded_file, delimiter=delimiter, header=None).values
        else:  # txt or other
            try:
                # First method: try using numpy loadtxt with given delimiter
                data = np.loadtxt(io.StringIO(uploaded_file.getvalue().decode('utf-8')), delimiter=delimiter)
            except Exception as e1:
                try:
                    # Second method: try pandas with the given delimiter
                    data = pd.read_csv(
                        io.StringIO(uploaded_file.getvalue().decode('utf-8')), 
                        delimiter=delimiter, 
                        header=None
                    ).values
                except Exception as e2:
                    # Third method: manual parsing
                    content = uploaded_file.getvalue().decode('utf-8')
                    lines = content.strip().split('\n')
                    data = []
                    for line in lines:
                        # Try various splitting methods
                        if '\t' in line:
                            parts = line.strip().split('\t')
                        elif ' ' in line:
                            parts = line.strip().split()
                        elif ',' in line:
                            parts = line.strip().split(',')
                        else:
                            continue  # Skip lines that can't be split
                        
                        if len(parts) >= 2:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                data.append([x, y])
                            except ValueError:
                                continue  # Skip lines that can't be converted to float
                    
                    if not data:
                        raise ValueError("Could not parse any valid data rows from the file")
                    
                    data = np.array(data)
        
        position = data[:, 0]
        sp_data = data[:, 1]
        
        st.success(f"Data loaded successfully: {len(position)} data points")
        
        # Display raw data plot
        fig_raw, ax_raw = plt.subplots(figsize=(10, 6))
        ax_raw.plot(position, sp_data, 'b.', label='Field data')
        ax_raw.set_xlabel('Position (m)')
        ax_raw.set_ylabel('SP data (mV)')
        ax_raw.set_title('Raw Self-Potential Data')
        ax_raw.grid(True)
        ax_raw.legend()
        st.pyplot(fig_raw)
        
        # Create download button for raw data plot
        raw_img = BytesIO()
        fig_raw.savefig(raw_img, format='png', bbox_inches='tight', dpi=300)
        raw_img.seek(0)
        download_raw = st.download_button(
            label="Download Raw Data Plot (PNG)",
            data=raw_img,
            file_name="sp_raw_data.png",
            mime="image/png",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please check your file format and delimiter settings.")
        st.error("Attempting to display the first few lines of your file:")
        
        try:
            # Show the first few lines of the file to help diagnose
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')[:5]  # First 5 lines
            for i, line in enumerate(lines):
                st.code(f"Line {i+1}: {line}")
        except:
            st.error("Could not display file contents.")
        
        use_sample_data = True

elif use_sample_data:
    # Generate synthetic data for demonstration
    position = np.linspace(0, 100, 50)
    
    # Parameters: x0, alpha, h, K
    if model_type == "Spherical":
        true_params = [50, 30, 15, 500]
        sp_data = forward_spherical(true_params, position)
    elif model_type == "Horizontal Cylindrical" or model_type == "Vertical Cylindrical":
        true_params = [50, 30, 15, 500]
        sp_data = forward_cylinder(true_params, position)
    elif model_type == "Fault":
        true_params = [50, 30, 15, 500, fault_half_width]
        sp_data = forward_fault(true_params, position)
    
    # Add noise
    noise = np.random.normal(0, 0.1 * np.max(np.abs(sp_data)), len(sp_data))
    sp_data = sp_data + noise
    
    st.info("Using synthetic sample data for demonstration")
    
    # Display sample data plot
    fig_sample, ax_sample = plt.subplots(figsize=(10, 6))
    ax_sample.plot(position, sp_data, 'b.', label='Sample data')
    ax_sample.set_xlabel('Position (m)')
    ax_sample.set_ylabel('SP data (mV)')
    ax_sample.set_title('Sample Self-Potential Data')
    ax_sample.grid(True)
    ax_sample.legend()
    st.pyplot(fig_sample)
    
    # Create download button for sample data plot
    sample_img = BytesIO()
    fig_sample.savefig(sample_img, format='png', bbox_inches='tight', dpi=300)
    sample_img.seek(0)
    download_sample = st.download_button(
        label="Download Sample Data Plot (PNG)",
        data=sample_img,
        file_name="sp_sample_data.png",
        mime="image/png",
        use_container_width=True
    )

# Create two columns for results
col1, col2 = st.columns([1, 1])

# Run inversion if button is clicked and data is available
if run_inversion_button and ((uploaded_file is not None) or use_sample_data):
    with st.spinner("Running inversion... This may take several minutes."):
        # Prepare bounds for inversion
        lower_bounds = [x0_min, alpha_min, h_min, k_min]
        upper_bounds = [x0_max, alpha_max, h_max, k_max]
        bounds = (lower_bounds, upper_bounds)
        
        # Run the inversion
        if model_type == "Fault":
            best_params, sp_calculation, error, samples = run_inversion(
                position, sp_data, model_type, bounds, nwalkers, nsteps, fault_half_width
            )
        else:
            best_params, sp_calculation, error, samples = run_inversion(
                position, sp_data, model_type, bounds, nwalkers, nsteps
            )
        
        # Display results in the first column
        with col1:
            st.header("Inversion Results")
            
            # Display parameters
            st.subheader("Best-fit Parameters")
            params_df = pd.DataFrame({
                'Parameter': ['X0 (m)', 'Alpha (degrees)', 'Depth (m)', 'K (contrast)'],
                'Value': best_params
            })
            
            if model_type == "Fault":
                params_df = pd.concat([
                    params_df, 
                    pd.DataFrame({
                        'Parameter': ['Half-Width (m)'],
                        'Value': [fault_half_width]
                    })
                ], ignore_index=True)
            
            st.table(params_df)
            
            # Display error
            st.metric("Inversion Error", f"{error:.6f}%")
            
            # Display summary table
            results_df = pd.DataFrame({
                'Parameter': ['X0 (m)', 'Alpha (degrees)', 'Depth (m)', 'K (contrast)', 'Error (%)'],
                'Value': [*best_params, error]
            })
            
            if model_type == "Fault":
                # Add half-width to the table for fault model
                fault_row = pd.DataFrame({
                    'Parameter': ['Half-Width (m)'],
                    'Value': [fault_half_width]
                })
                # Insert the fault parameter at position 4 (before Error)
                results_df = pd.concat([
                    results_df.iloc[:4], 
                    fault_row, 
                    results_df.iloc[4:].reset_index(drop=True)
                ]).reset_index(drop=True)
            
            st.table(results_df)
        
        # Display plots in the second column
        with col2:
            st.header("Visualization")
            
            # Plot data and model fit
            fig_results, ax_results = plt.subplots(figsize=(10, 6))
            ax_results.plot(position, sp_data, 'r*', label='Field data')
            ax_results.plot(position, sp_calculation, 'g-', label='Calculated')
            ax_results.set_xlabel('Position (m)')
            ax_results.set_ylabel('SP data (mV)')
            ax_results.set_title(f'SP Data vs Inversion Result ({model_type} Model)')
            ax_results.grid(True)
            ax_results.legend()
            st.pyplot(fig_results)
            
            # Create download button for results plot
            results_img = BytesIO()
            fig_results.savefig(results_img, format='png', bbox_inches='tight', dpi=300)
            results_img.seek(0)
            download_results = st.download_button(
                label="Download Results Plot (PNG)",
                data=results_img,
                file_name=f"sp_inversion_results_{model_type}.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Plot MCMC samples for each parameter
            st.subheader("MCMC Parameter Convergence")
            parameter_names = ['X0', 'Alpha', 'Depth', 'K']
            fig_mcmc, axes = plt.subplots(len(parameter_names), 1, figsize=(10, 10), sharex=True)
            
            for i, (ax, name) in enumerate(zip(axes, parameter_names)):
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_ylabel(name)
                ax.yaxis.set_label_coords(-0.1, 0.5)
            
            axes[-1].set_xlabel("Step number")
            plt.tight_layout()
            st.pyplot(fig_mcmc)
            
            # Create download button for MCMC convergence plot
            mcmc_img = BytesIO()
            fig_mcmc.savefig(mcmc_img, format='png', bbox_inches='tight', dpi=300)
            mcmc_img.seek(0)
            download_mcmc = st.download_button(
                label="Download MCMC Convergence Plot (PNG)",
                data=mcmc_img,
                file_name=f"sp_mcmc_convergence_{model_type}.png",
                mime="image/png",
                key="mcmc_plot",  # Add unique key to avoid conflict with other download buttons
                use_container_width=True
            )
            
            # Create additional figures for analysis
            
            # 1. Create parameter distribution histograms
            flat_samples = samples.reshape(-1, ndim)
            fig_hist, axes_hist = plt.subplots(2, 2, figsize=(10, 8))
            axes_hist = axes_hist.flatten()
            
            for i, (ax, name) in enumerate(zip(axes_hist, parameter_names)):
                ax.hist(flat_samples[:, i], bins=30, alpha=0.7)
                ax.set_xlabel(name)
                ax.set_ylabel("Frequency")
                
                # Add vertical line for best value
                ax.axvline(best_params[i], color='r', linestyle='--')
                
            plt.tight_layout()
            st.subheader("Parameter Distributions")
            st.pyplot(fig_hist)
            
            # Create download button for histogram plot
            hist_img = BytesIO()
            fig_hist.savefig(hist_img, format='png', bbox_inches='tight', dpi=300)
            hist_img.seek(0)
            download_hist = st.download_button(
                label="Download Parameter Distributions (PNG)",
                data=hist_img,
                file_name=f"sp_parameter_distributions_{model_type}.png",
                mime="image/png",
                key="hist_plot",  # Add unique key
                use_container_width=True
            )

# Add explanatory information at the bottom
st.markdown("""
---
### About the Models

**Spherical Model**: Represents a spherical anomaly such as a mineralized body or intrusion.

**Horizontal/Vertical Cylindrical Model**: Represents cylindrical bodies with different orientations.

**Fault Model**: Represents a geological fault structure.

### Inversion Parameters

- **X0**: Horizontal position of the anomaly center (m)
- **Alpha**: Inclination angle (degrees)
- **Depth**: Depth to the top of the anomaly (m)
- **K**: Physical property contrast

### How to Use
1. Upload your data file or use the sample data
2. Select a model type that best represents your geological scenario
3. Configure the inversion settings and parameter bounds
4. Click "Run Inversion" to start the process
5. View and download the results
""")

# Footer
st.markdown("---")
st.markdown("© 2025 MCMC-Based Self-Potential Inversion Tool")