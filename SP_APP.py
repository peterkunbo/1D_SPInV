import streamlit as st
import numpy as np
from scipy.optimize import least_squares
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import io
from io import BytesIO
import time

# Set page configuration
st.set_page_config(
    page_title="MCMC-Based Self-Potential Inversion Tool",
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
st.title("MCMC-Based Self-Potential Data Inversion Tool")
st.markdown("""
This application performs inversion of self-potential data using different forward models. On the left tab, 
Upload your CSV data, select a model type, configure inversion parameters, and visualize the results.
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
    
    # File upload option - CSV only
    uploaded_file = st.file_uploader("Upload CSV data file", type=["csv"])
    
    # Sample data option
    use_sample_data = st.checkbox("Use sample data", value=not bool(uploaded_file))
    
    # Data format info
    with st.expander("Data Format Information"):
        st.info("""
        Expected data format:
        - Two-column CSV: Position (x) and SP values
        - No header row
        - Comma separated values
        """)
    
    # Delimiter for CSV files
    delimiter = st.selectbox(
        "CSV Delimiter", 
        options=[",", ";"], 
        index=0,  # Default to comma delimiter
        help="Select the delimiter used in your CSV file"
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
    
    k_min = st.number_input("Min K", value=-1000.0)
    k_max = st.number_input("Max K", value=1000.0)
    
    # Run button
    run_inversion_button = st.button("Run Inversion", use_container_width=True)

# Main content area
# Load and display data
if uploaded_file is not None:
    try:
        # Read CSV file using pandas
        data = pd.read_csv(uploaded_file, delimiter=delimiter, header=None).values
        
        position = data[:, 0]
        sp_data = data[:, 1]
        
        st.success(f"Data loaded successfully: {len(position)} data points")
        
        # Display raw data plot
        fig_raw, ax_raw = plt.subplots(figsize=(12, 6))
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
        use_sample_data = True

# Replace the entire sample data generation section with this:

elif use_sample_data:
    # Generate synthetic bell-shaped data from spherical model
    # Using a clearly defined spherical model with simple parameters
    position = np.linspace(0, 100, 50)
    
    # Simple parameters for spherical model: x0, alpha, h, K
    true_params = [50, 0, 20, 500]  # Center at 50m, vertical (alpha=0), depth=20m, K=500
    
    # Generate clean data
    clean_data = forward_spherical(true_params, position)
    
    # Add exactly 10% Gaussian noise
    noise_level = 0.1 * np.max(np.abs(clean_data))
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, noise_level, len(clean_data))
    sp_data = clean_data + noise
    
    st.info(f"""
    You can upload your data or use this synthetic spherical model data with 10% noise:
    - X0 = {true_params[0]} m (center position)
    - Alpha = {true_params[1]}° (vertical orientation)
    - Depth = {true_params[2]} m
    - K = {true_params[3]}
    """)
    
    # Display sample data plot
    fig_sample, ax_sample = plt.subplots(figsize=(12, 6))
    
    # Plot both clean data and noisy data
    #ax_sample.plot(position, clean_data, 'k-', linewidth=2, label='Clean spherical model')
    ax_sample.plot(position, sp_data, 'bo', markersize=5, label='Noisy data (10% noise)')
    
    ax_sample.set_xlabel('Position (m)')
    ax_sample.set_ylabel('SP data (mV)')
    ax_sample.set_title('Sample Spherical Model Data with 10% Noise')
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
        
        # Replace the parameter table display section with this simpler approach:

        # Display results - Parameter table first
        st.header("Best-Fit Parameters")

        # # Create a simple text-based display of results
        # st.markdown("**Parameters for best-fit model:**")
        # st.markdown(f"**X0:** {best_params[0]:.6f} m")
        # st.markdown(f"**Alpha:** {best_params[1]:.6f} degrees")
        # st.markdown(f"**Depth:** {best_params[2]:.6f} m")
        # st.markdown(f"**K:** {best_params[3]:.6f}")

        # if model_type == "Fault":
        #     st.markdown(f"**Half-Width:** {fault_half_width:.6f} m")

        # st.markdown(f"**Error:** {error:.6f}%")

        # Create a HTML table as an alternative to dataframe
        html_table = """
        <table style="width:100%; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid black;">
            <th style="text-align: left; padding: 8px;">Parameter</th>
            <th style="text-align: right; padding: 8px;">Value</th>
        </tr>
        """

        # Add parameters to the table
        html_table += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="text-align: left; padding: 8px;">X0 (m)</td>
            <td style="text-align: right; padding: 8px;">{best_params[0]:.6f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="text-align: left; padding: 8px;">Alpha (degrees)</td>
            <td style="text-align: right; padding: 8px;">{best_params[1]:.6f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="text-align: left; padding: 8px;">Depth (m)</td>
            <td style="text-align: right; padding: 8px;">{best_params[2]:.6f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="text-align: left; padding: 8px;">K</td>
            <td style="text-align: right; padding: 8px;">{best_params[3]:.6f}</td>
        </tr>
        """

        # Add fault parameter if applicable
        if model_type == "Fault":
            html_table += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="text-align: left; padding: 8px;">Half-Width (m)</td>
            <td style="text-align: right; padding: 8px;">{fault_half_width:.6f}</td>
        </tr>
        """

        # Add error row
        html_table += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="text-align: left; padding: 8px;">Error (%)</td>
            <td style="text-align: right; padding: 8px;">{error:.6f}</td>
        </tr>
        </table>
        """

        st.markdown(html_table, unsafe_allow_html=True)
        
        # Display large visualization plot
        st.header("Inversion Results Visualization")
        
        # Create a large figure for visualization
        fig_results, ax_results = plt.subplots(figsize=(16, 10))
        
        # Plot observed data
        ax_results.plot(position, sp_data, 'ro', markersize=6, label='Observed Data')
        
        # Plot calculated data
        ax_results.plot(position, sp_calculation, 'b-', linewidth=3, label='Calculated (Best-Fit Model)')
        
        # Add model parameters as text on the plot
        param_text = f"Model: {model_type}\n"
        param_text += f"X0 = {best_params[0]:.2f} m\n"
        param_text += f"Alpha = {best_params[1]:.2f}°\n"
        param_text += f"Depth = {best_params[2]:.2f} m\n"
        param_text += f"K = {best_params[3]:.2f}\n"
        
        if model_type == "Fault":
            param_text += f"Half-Width = {fault_half_width:.2f} m\n"
            
        param_text += f"Error = {error:.6f}%"
        
        # Position the text box in the top right corner
        ax_results.text(0.98, 0.98, param_text, 
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax_results.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax_results.set_xlabel('Position (m)', fontsize=14)
        ax_results.set_ylabel('SP data (mV)', fontsize=14)
        ax_results.set_title(f'Self-Potential Data Inversion Results ({model_type} Model)', fontsize=16)
        ax_results.grid(True, linestyle='--', alpha=0.7)
        ax_results.legend(fontsize=12)
        
        # Display the plot
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

# Add explanatory information at the bottom
st.markdown("""
---
### About the Models

- **Spherical Model**: Represents a spherical anomaly such as a mineralized body or intrusion.
- **Horizontal Cylindrical Model**: Represents geologic structure with different orientations such as channel or ore body with eloganted shapes.
- **Vertical Cylindrical Model**: Represents geologic structure with different orientations such as collapsed sinkholes or igneous intrusion.
- **Inclined Model**: Represents a geological fault structure.

### Inversion Parameters

- **X0**: Horizontal position of the anomaly center (m)
- **Alpha**: Inclination angle (degrees)
- **Depth**: Depth to the top of the anomaly (m)
- **K**:     Represents the strength of the electrical current source. 
### How to Use
1. Upload your CSV data file or use the sample data
2. Select a model type that best represents your geological scenario
3. Configure the inversion settings and parameter bounds
4. Click "Run Inversion" to start the process
5. View and download the results
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Peter Adetokunbo-MCMC-Based Self-Potential Inversion Tool")
