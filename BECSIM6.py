# BEC_GPE_2D_Streamlit.py
import streamlit as st
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from io import BytesIO
import time

st.set_page_config(layout="wide", page_title="BEC & Quantum Droplet 2D — GPE solver")
st.title("BEC & Quantum Droplet 2D — GPE (imaginary time)")

# -------------------- utilities --------------------
def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y, dx, dy

def ksq_array(Nx, Ny, dx, dy):
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx2 = kx[:, None]**2
    ky2 = ky[None, :]**2
    return kx2 + ky2

def density_to_pil(density, cmap_name='viridis', vmin=None, vmax=None):
    if vmin is None: vmin = density.min()
    if vmax is None: vmax = density.max()
    if vmax - vmin < 1e-16:
        vmax = vmin + 1e-12
    norm = (density - vmin) / (vmax - vmin)
    rgba = cm.get_cmap(cmap_name)(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)

# -------------------- numeric solver: split-step imaginary time --------------------
def imag_time_evolve(psi, V, g, k2, dtau, nsteps, log_term_params=None, renorm_to_N=None):
    """
    psi: complex ndarray (Nx x Ny)
    V: potential (Nx x Ny) (real)
    g: scalar interaction coefficient
    k2: kx^2 + ky^2 array (Nx x Ny)
    dtau: time step (imag. time)
    nsteps: number of imaginary-time steps to perform
    log_term_params: None or dict with keys {'alpha', 'beta'} for -beta*ln(alpha^3*|psi|^2)
    renorm_to_N: if provided, scale psi so integral |psi|^2 dx dy = renorm_to_N at the end
    returns psi (normalized to 1 or to renorm_to_N if provided)
    """
    Nx, Ny = psi.shape
    dx = 1.0  # solver assumes uniform grid and scaling; actual dx used only for normalization/scaling of N externally
    for _ in range(nsteps):
        # half-step potential + nonlinear
        dens = np.abs(psi)**2 + 1e-14
        nonlinear = g * dens
        if log_term_params is not None:
            alpha = log_term_params.get('alpha', 1.0)
            beta = log_term_params.get('beta', 0.0)
            # energy density correction: -beta * ln(alpha^3 * dens)
            # in imaginary-time exponential factor, use + (since eq multiplied by -1)
            # We include these consistently as a potential-like term
            nonlinear = nonlinear - beta * np.log((alpha**3) * dens + 1e-30)
        psi = psi * np.exp(-0.5 * dtau * (V + nonlinear))

        # kinetic step in Fourier space
        psi_hat = fft.fft2(psi)
        psi_hat = psi_hat * np.exp(-0.5 * dtau * k2)  # factor includes 1/2 from Laplacian with ħ=m=1
        psi = fft.ifft2(psi_hat)

        # half-step potential + nonlinear
        dens = np.abs(psi)**2 + 1e-14
        nonlinear = g * dens
        if log_term_params is not None:
            alpha = log_term_params.get('alpha', 1.0)
            beta = log_term_params.get('beta', 0.0)
            nonlinear = nonlinear - beta * np.log((alpha**3) * dens + 1e-30)
        psi = psi * np.exp(-0.5 * dtau * (V + nonlinear))

        # normalize to 1 each step (imag-time method)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dx)
        if norm != 0:
            psi /= norm

    # final renormalization to requested N (if provided) — we keep psi norm such that integral = N
    if renorm_to_N is not None:
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dx)
        if norm != 0:
            psi *= np.sqrt(renorm_to_N) / norm

    return psi

# -------------------- UI layout with tabs --------------------
tabs = st.tabs(["📘 Teoría", "🌡️ Simulación BEC (GPE)", "💧 Gota Cuántica (GPE mod.)"])

# -------------------- Tab 1: Theory --------------------
with tabs[0]:
    st.header("Teoría: MOT → BEC → Gotas cuánticas (resumen)")
    st.title("🧊 Enfriamiento Láser, Trampa Magneto-Óptica y Condensados de Bose–Einstein")

    st.markdown("""
    ## 1. Enfriamiento Láser
    El **enfriamiento láser** se basa en el principio de Doppler.  
    Cuando un átomo se mueve hacia un haz de luz con frecuencia ligeramente menor a una transición resonante, el efecto Doppler
    lo hace ver más cercano a la resonancia, por lo que **absorbe fotones** y recibe pequeños impulsos en la dirección opuesta a su movimiento.
    Al emitir fotones en direcciones aleatorias, el momento promedio de emisión es cero,
    resultando en una pérdida neta de momento lineal del átomo.

    La temperatura límite del enfriamiento Doppler es:
    $$
    T_D = \\frac{\\hbar \\Gamma}{2 k_B},
    $$
    donde $\\Gamma$ es el ancho natural de la línea de emisión del átomo.

    En la práctica, mediante técnicas de **enfriamiento sub-Doppler**, se puede alcanzar temperaturas del orden de **microkelvins**.
    """)

    st.markdown("""
    ## 2. Trampa Magneto-Óptica (MOT)
    La **trampa magneto-óptica** combina campos magnéticos no uniformes con haces láser contrapropagantes polarizados circularmente.
    Los campos magnéticos generan un desplazamiento Zeeman dependiente de la posición, lo que ajusta la frecuencia de resonancia
    localmente. De esta forma, los átomos son empujados hacia el centro de la trampa, donde el campo magnético es cero.

    La fuerza de atrapamiento total puede escribirse como:
    $$
    \\mathbf{F} = -\\kappa \\mathbf{r} - \\beta \\mathbf{v},
    $$
    donde $\\kappa$ representa la constante de resorte óptico (confinamiento espacial) y $\\beta$ la fricción viscosa óptica.
    """)

    st.markdown("""
    ## 3. Transición de Fase: Condensado de Bose–Einstein
    Al seguir enfriando el gas atrapado hasta temperaturas del orden de **nanokelvins**, los bosones ocupan colectivamente
    el **estado fundamental** del sistema. Este fenómeno macroscópico se describe mediante la **ecuación de Gross–Pitaevskii (GPE)**:
    $$
    i\\hbar \\frac{\\partial \\Psi}{\\partial t} =
    \\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r}) + g |\\Psi|^2\\right]\\Psi,
    $$
    donde $V(\\mathbf{r})$ es el potencial de trampa armónica y $g = \\frac{4\\pi \\hbar^2 a_s}{m}$ describe las interacciones
    entre partículas mediante la longitud de dispersión $a_s$.
    
    En la pestaña de simulación BEC, podrás visualizar cómo un gas bosónico se enfría y colapsa hacia el estado condensado
    en una trampa armónica.
    """)

    st.markdown("""
    ## 4. Gotas Cuánticas Bosónicas
    Al modificar las interacciones entre los átomos (por ejemplo, mediante resonancias de Feshbach), se pueden lograr
    condiciones donde la presión cuántica y las interacciones atractivas se balancean. En este régimen aparece una nueva fase:
    la **gota cuántica autoestabilizada**.

    Esta fase se describe por una ecuación de Gross–Pitaevskii **modificada con un término logarítmico**:
    $$
    i\\hbar \\frac{\\partial \\Psi}{\\partial t} =
    \\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r}) + g|\\Psi|^2 - \\beta \\ln(\\alpha^3|\\Psi|^2)\\right]\\Psi.
    $$
    El término logarítmico $-\\beta \\ln(\\alpha^3|\\Psi|^2)$ modela las **correcciones cuánticas de densidad** que estabilizan
    el sistema frente al colapso.  
    En la pestaña de **gota cuántica**, observarás cómo el condensado se transforma en una fase autoconfinada, incluso sin trampa.
    """)

    st.info("""
    💡 **Resumen conceptual:**
    1. Los átomos son enfriados mediante láseres hasta microkelvins.  
    2. El MOT atrapa los átomos en el centro mediante fuerzas ópticas y magnéticas.  
    3. Al alcanzar nanokelvins, surge el BEC, una fase cuántica macroscópica.  
    4. Ajustando las interacciones, el BEC puede evolucionar hacia una gota cuántica autoestabilizada.
    """)

    st.markdown("""
    ---
    🔬 **Referencia recomendada:**
    - C. J. Pethick and H. Smith, *Bose–Einstein Condensation in Dilute Gases*, Cambridge University Press (2008).
    - F. Dalfovo et al., *Theory of Bose–Einstein Condensation in Trapped Gases*, Rev. Mod. Phys. 71, 463 (1999).
    - D. S. Petrov, *Quantum mechanical stabilization of a collapsing Bose–Bose mixture*, Phys. Rev. Lett. 115, 155302 (2015).
    """)
    st.write("Pulsa las otras pestañas para ejecutar las simulaciones numéricas (2D, tiempo imaginario).")

# -------------------- Shared simulation parameters (sidebar) --------------------
with st.sidebar:
    st.subheader("Grid & numerical")
    Nx = st.selectbox("Nx = Ny (grid points)", options=[64, 96, 128, 160], index=2)
    L = st.number_input("Physical window L (arb. units)", value=10.0, min_value=1.0, max_value=40.0)
    dx = L / Nx
    dtau = st.number_input("Imag time step dtau", value=0.002, step=0.0005)
    inner_steps_per_frame = st.slider("Imag steps per frame (convergence)", 5, 400, 40, 5)
    frames = st.slider("Number of frames for animation", 6, 40, 12)

# -------------------- Tab 2: BEC (standard GPE) --------------------
with tabs[1]:
    st.header("Simulación: formación de BEC (GPE estándar)")

    col_params, col_run = st.columns([1,1])
    with col_params:
        st.subheader("Physical / model params")
        N_atoms = st.number_input("Total atom number N (arb units)", value=1e4, format="%.0f")
        g = st.number_input("Interaction g (dimensionless)", value=0.01, step=0.001, format="%.5f")
        omega_x = st.number_input("Trap ω_x (arb)", value=1.0, step=0.1)
        omega_y = st.number_input("Trap ω_y (arb)", value=1.0, step=0.1)
        trap_off_x = st.checkbox("Turn off trap X (ω_x = 0)")
        trap_off_y = st.checkbox("Turn off trap Y (ω_y = 0)")
        show_phase = st.checkbox("Show phase map (last frame)", value=False)

    with col_run:
        run_bec = st.button("Run BEC evolution")
        st.write("Notes: solver uses split-step imaginary-time (dimensionless units).")

    if run_bec:
        # build grid and potential
        Nx_loc = Nx; Ny_loc = Nx
        X, Y, dx_grid, dy_grid = make_grid(Nx_loc, Ny_loc, L, L)
        k2 = ksq_array(Nx_loc, Ny_loc, dx_grid, dy_grid)
        use_omega_x = 0.0 if trap_off_x else omega_x
        use_omega_y = 0.0 if trap_off_y else omega_y
        V = 0.5 * (use_omega_x**2 * X**2 + use_omega_y**2 * Y**2)

        # initial psi: small gaussian + noise
        sigma0 = L/6.0
        psi = np.exp(-(X**2 + Y**2)/(2*sigma0**2)).astype(np.complex128)
        psi *= (1.0 + 0.01*(np.random.rand(*psi.shape)-0.5))
        # normalize
        norm0 = np.sqrt(np.sum(np.abs(psi)**2) * dx_grid * dy_grid)
        psi /= norm0

        frames_density = []
        t0 = time.time()
        # progressive evolution: for each animation frame perform inner_steps_per_frame of imag-time
        for f in range(frames):
            psi = imag_time_evolve(psi, V, g, k2, dtau, inner_steps_per_frame, log_term_params=None, renorm_to_N=None)
            # scale to N_atoms for visualization
            norm = np.sqrt(np.sum(np.abs(psi)**2) * dx_grid * dy_grid)
            psi_vis = psi * (np.sqrt(N_atoms) / (norm + 1e-16))
            dens = np.abs(psi_vis)**2
            frames_density.append(dens)

        elapsed = time.time() - t0

        # Convert numeric frames to PIL images with consistent vmin/vmax
        vmin = min([d.min() for d in frames_density])
        vmax = max([d.max() for d in frames_density])
        imgs = [density_to_pil(d, cmap_name='viridis', vmin=vmin, vmax=vmax) for d in frames_density]

        # write GIF to BytesIO via Pillow
        buf = BytesIO()
        imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:], loop=0, duration=200)
        buf.seek(0)

        st.success(f'Finished BEC imag-time evolution in {elapsed:.2f}s ({len(frames_density)} frames).')
        st.image(buf.getvalue(), caption="BEC formation (imaginary-time)", use_container_width=True)

        # Show last density & optionally phase
        colA, colB = st.columns([1,1])
        with colA:
            st.subheader("Final density (last frame)")
            fig, ax = plt.subplots()
            im = ax.imshow(np.log10(frames_density[-1] + 1e-16), origin='lower', extent=[-L/2,L/2,-L/2,L/2], cmap='inferno')
            ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, label='log10 density')
            st.pyplot(fig)
        with colB:
            if show_phase:
                st.subheader("Phase of psi (last frame)")
                last_psi = psi * (np.sqrt(N_atoms) / (np.sqrt(np.sum(np.abs(psi)**2)*dx_grid*dy_grid) + 1e-16))
                phase = np.angle(last_psi)
                fig2, ax2 = plt.subplots()
                ax2.imshow(phase, origin='lower', extent=[-L/2,L/2,-L/2,L/2], cmap='twilight')
                ax2.set_title("Phase (radians)")
                st.pyplot(fig2)

# -------------------- Tab 3: Quantum droplet (GPE modified) --------------------
with tabs[2]:
    st.header("Gota cuántica: GPE modificada con término logarítmico")

    col_params, col_run = st.columns([1,1])
    with col_params:
        st.subheader("Model params")
        N_atoms_d = st.number_input("N (droplet sim)", value=1e4, format="%.0f")
        g_d = st.number_input("g (cubic interaction)", value=0.005, step=0.001, format="%.5f")
        alpha = st.number_input("alpha (scale inside log)", value=1.0, step=0.1)
        beta = st.number_input("beta (log strength)", value=0.12, step=0.01, format="%.4f")
        omega_x_d = st.number_input("Trap ω_x (droplet)", value=0.6, step=0.1)
        omega_y_d = st.number_input("Trap ω_y (droplet)", value=0.6, step=0.1)
        turn_off_x_d = st.checkbox("Turn off trap X (droplet)", key="toffx")
        turn_off_y_d = st.checkbox("Turn off trap Y (droplet)", key="toffy")
        show_profiles = st.checkbox("Show radial profile evolution", value=True)

    with col_run:
        run_drop = st.button("Run droplet evolution")
        st.write("The solver includes the log term: -β ln(α^3 |ψ|^2) in the effective potential.")

    if run_drop:
        Nx_loc = Nx; Ny_loc = Nx
        X, Y, dx_grid, dy_grid = make_grid(Nx_loc, Ny_loc, L, L)
        k2 = ksq_array(Nx_loc, Ny_loc, dx_grid, dy_grid)
        use_omega_x = 0.0 if turn_off_x_d else omega_x_d
        use_omega_y = 0.0 if turn_off_y_d else omega_y_d
        V = 0.5 * (use_omega_x**2 * X**2 + use_omega_y**2 * Y**2)

        sigma0 = L/6.0
        psi = np.exp(-(X**2 + Y**2)/(2*sigma0**2)).astype(np.complex128)
        psi *= (1.0 + 0.01*(np.random.rand(*psi.shape)-0.5))
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx_grid * dy_grid)

        frames_density = []
        radial_profiles = []
        t0 = time.time()
        log_params = {'alpha': alpha, 'beta': beta}

        for f in range(frames):
            psi = imag_time_evolve(psi, V, g_d, k2, dtau, inner_steps_per_frame, log_term_params=log_params, renorm_to_N=None)
            # scale to N_atoms_d for visibility
            norm = np.sqrt(np.sum(np.abs(psi)**2) * dx_grid * dy_grid)
            psi_vis = psi * (np.sqrt(N_atoms_d) / (norm + 1e-16))
            dens = np.abs(psi_vis)**2
            frames_density.append(dens)
            if show_profiles:
                # radial average
                R = np.sqrt(X**2 + Y**2)
                r_bins = np.linspace(0, L/2, 100)
                digitized = np.digitize(R.ravel(), r_bins)
                prof = np.array([dens.ravel()[digitized == i].mean() if np.any(digitized==i) else 0.0 for i in range(1, len(r_bins))])
                radial_profiles.append((r_bins[1:], prof))

        elapsed = time.time() - t0

        # Convert frames to images
        vmin = min(d.min() for d in frames_density)
        vmax = max(d.max() for d in frames_density)
        imgs = [density_to_pil(d, cmap_name='magma', vmin=vmin, vmax=vmax) for d in frames_density]
        buf = BytesIO()
        imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:], loop=0, duration=200)
        buf.seek(0)

        st.success(f'Droplet evolution done in {elapsed:.2f}s ({len(frames_density)} frames).')
        st.image(buf.getvalue(), caption="Droplet formation (imaginary-time)", use_container_width=True)

        # show final density and radial profile
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Final density (log scale)")
            fig1, ax1 = plt.subplots()
            ax1.imshow(np.log10(frames_density[-1] + 1e-16), origin='lower', extent=[-L/2,L/2,-L/2,L/2], cmap='inferno')
            ax1.set_xlabel("x"); ax1.set_ylabel("y")
            st.pyplot(fig1)
        with col2:
            if show_profiles and radial_profiles:
                st.subheader("Radial profile evolution")
                fig2, ax2 = plt.subplots()
                for i, (r, prof) in enumerate(radial_profiles[::max(1,len(radial_profiles)//6)]):
                    ax2.plot(r, prof, label=f"frame {i*max(1,len(radial_profiles)//6)}")
                ax2.set_xlabel("r"); ax2.set_ylabel("density")
                ax2.legend()
                st.pyplot(fig2)

        st.download_button("Download droplet GIF", data=buf.getvalue(), file_name="droplet_formation.gif", mime="image/gif")
