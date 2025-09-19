import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.fft import fft, fftfreq
from scipy import signal as sp_signal

# Set matplotlib style for better aesthetics
plt.style.use('dark_background')

# Check for audio dependencies. If they're not available, disable the recorder tab.
try:
    import sounddevice as sd
    import soundfile as sf
    can_record = True
except (ImportError, OSError):
    can_record = False

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎶 Fourier Series Visualization", layout="wide")

# Custom CSS for a futuristic/modern look
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap" rel="stylesheet">
<style>
body {
    background-color: #0f0f0f;
    font-family: 'Orbitron', sans-serif;
}
h1, h2, h3 {
    color: #00ffe1; /* Bright Cyan */
}
.stTabs [data-baseweb="tab"] {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 10px;
    margin-right: 4px;
    border: 1px solid #333;
}
.stTabs [aria-selected="true"] {
    background-color: #00ffe1;
    color: black;
    font-weight: bold;
}
.stButton>button {
    border: 2px solid #00ffe1;
    border-radius: 8px;
    background-color: transparent;
    color: #00ffe1;
    padding: 10px 24px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #00ffe1;
    color: black;
    box-shadow: 0 0 15px #00ffe1;
}
.st-emotion-cache-1c5c56d.eqr7sfw1 {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎶 Fourier Series Visualization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Record, Generate, Analyze & Understand Sound with Fourier Series</p>", unsafe_allow_html=True)

# ---------- SESSION STATE INITIALIZATION ----------
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None
if "fs" not in st.session_state:
    st.session_state["fs"] = 44100
if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False
if "signal_source" not in st.session_state:
    st.session_state["signal_source"] = None

# ---------- CORE FUNCTIONS ----------
def record_audio(duration=5, fs=44100):
    """Record audio for a given duration"""
    st.info("🎙 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(recording)

def play_audio(data, fs):
    """Return WAV bytes for streamlit audio player"""
    bio = io.BytesIO()
    sf.write(bio, data, fs, format="WAV")
    return bio.getvalue()

# --- Wave Generation Functions ---
def generate_sine_wave(amp, freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return t, amp * np.sin(2 * np.pi * freq * t)

def generate_cosine_wave(amp, freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return t, amp * np.cos(2 * np.pi * freq * t)

def generate_square_wave(amp, freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return t, amp * sp_signal.square(2 * np.pi * freq * t)

def generate_triangular_wave(amp, freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 0.5 width for a symmetric triangular wave
    return t, amp * sp_signal.sawtooth(2 * np.pi * freq * t, 0.5) # type: ignore

def generate_impulse(duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    impulse = np.zeros_like(t)
    impulse[0] = 1.0  # A single point impulse at the beginning
    return t, impulse

# --- Fourier Calculation Functions ---
def trig_coeff(signal, n_max, T, t):
    """Calculate the coefficients for the trigonometric Fourier series."""
    a0 = np.mean(signal)
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    w0 = 2 * np.pi / T
    # Ensure dt is calculated correctly even for single-period slices
    if len(t) > 1:
        dt = t[1] - t[0]
    else:
        dt = T / len(signal) # Fallback for single sample
    for n in range(1, n_max + 1):
        a[n] = (2.0 / T) * np.sum(signal * np.cos(n * w0 * t)) * dt
        b[n] = (2.0 / T) * np.sum(signal * np.sin(n * w0 * t)) * dt
    return a0, a, b

def trig_recon(t, a0, a, b, T):
    """Reconstruct a signal from trigonometric Fourier series coefficients."""
    w0 = 2 * np.pi / T
    s = (a0 / 2) * np.ones_like(t)
    for n in range(1, len(a)):
        s += a[n] * np.cos(n * w0 * t) + b[n] * np.sin(n * w0 * t)
    return s

def exp_coeff(signal, n_max, T, t):
    """Calculate the coefficients for the exponential Fourier series."""
    coeffs = np.zeros(2 * n_max + 1, dtype=complex)
    w0 = 2 * np.pi / T
    if len(t) > 1:
        dt = t[1] - t[0]
    else:
        dt = T / len(signal)
    for i, n in enumerate(range(-n_max, n_max + 1)):
        coeffs[i] = (1.0 / T) * np.sum(signal * np.exp(-1j * n * w0 * t)) * dt
    return coeffs

# ---------- TABS LAYOUT ----------
tab_info, tab_recorder, tab_generator, tab_analysis, tab_comparison = st.tabs(["ℹ Information", "🎤 Recorder", "🎹 Function Generator", "📊 Fourier Analysis", "⚖ Comparison"])

# ---------- TAB 1: INFORMATION ----------
with tab_info:
    st.subheader("ℹ About This Fourier Visualization Project")
    st.markdown("""
    This application is an interactive tool designed to help you understand the fascinating concept of the Fourier Series. It demonstrates how any periodic signal—including the sound you record or a mathematically perfect waveform—can be represented as a sum of simple sine and cosine waves.

    ### What is a Fourier Series?
    The Fourier Series is a mathematical tool that decomposes a periodic function or signal into a sum of simpler oscillating functions, namely sines and cosines (or complex exponentials). The core idea, pioneered by Joseph Fourier, is that even the most complex periodic waveforms can be built by adding up enough of these basic "harmonics."

    Each harmonic is a sine or cosine wave with a frequency that is an integer multiple of the original signal's fundamental frequency. The "recipe" for reconstructing the original signal is given by the coefficients of these harmonics.

    ### How This App Works
    The application is divided into four main functional parts: Recorder, Function Generator, Fourier Analysis, and Comparison.

    #### Trigonometric Fourier Series
    This form represents the signal s(t) as a sum of sines and cosines:
    $$
    s(t) = \\frac{a_0}{2} + \sum_{n=1}^{\infty} [a_n \cos(n \omega_0 t) + b_n \sin(n \omega_0 t)]
    $$
    Where:
    - $\omega_0 = 2\pi / T$ is the fundamental angular frequency of the signal.
    - $a_0$ is the DC offset (the average value of the signal).
    - $a_n$ and $b_n$ are the coefficients that determine the amplitude of each cosine and sine harmonic.

    #### Exponential Fourier Series
    This is a more compact form using complex exponentials:
    $$
    s(t) = \sum_{n=-\infty}^{\infty} c_n e^{j n \omega_0 t}
    $$
    Where $c_n$ are the complex Fourier coefficients. This app visualizes the real and imaginary parts of the $c_n$ coefficients, showing the signal's frequency spectrum.

    ### Applications of This Visualization Tool
    This tool serves several practical and educational purposes:
    - Educational Tool: For students of engineering, physics, and mathematics, this app provides an intuitive, hands-on way to grasp how the Fourier series works. Seeing the reconstruction improve as more terms are added solidifies the concept.
    - Audio Synthesis: The core principle of additive synthesis is building complex sounds by adding sine waves. This tool is a basic additive synthesizer; by analyzing a sound, you can see the 'recipe' of sine/cosine waves needed to recreate it.
    - Understanding Audio Effects: Visualizing the frequency spectrum (via exponential coefficients) helps in understanding how filters and equalizers work. An EQ boosts or cuts the amplitudes of specific frequency components ($c_n$).
    - Signal Analysis: For any recorded signal, you can immediately see its harmonic content. For example, you can see the rich harmonics of a square wave versus the single pure tone of a sine wave.
    """)

# ---------- TAB 2: RECORDER ----------
with tab_recorder:
    st.subheader("🎙 Audio Recorder")
    
    if can_record:
        dur = st.slider("Recording Duration (seconds)", 1, 10, 5)
        
        if st.button("▶ Start Recording"):
            audio = record_audio(duration=dur, fs=st.session_state["fs"])
            st.session_state["audio_data"] = audio
            st.session_state["fs"] = 44100 # Reset fs to default for recording
            st.session_state["signal_source"] = "recorder"
            if "fundamental_freq" in st.session_state:
                del st.session_state["fundamental_freq"]
            st.success("✅ Recording finished and ready for analysis!")

        if st.session_state["audio_data"] is not None and st.session_state.get("signal_source") == "recorder":
            st.audio(play_audio(st.session_state["audio_data"], st.session_state["fs"]), format="audio/wav")
            
            st.markdown("### 📈 Recorded Waveform")
            fig, ax = plt.subplots(figsize=(10, 4))
            data = st.session_state["audio_data"]
            fs = st.session_state["fs"]
            t = np.linspace(0, len(data)/fs, len(data), endpoint=False)
            ax.plot(t, data, color="#00ffe1")
            ax.set_title("Audio Signal")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("⚠️ Recording is not available in this environment.")
        st.info("This Streamlit app is running on a server that does not have access to a microphone. Please use the 'Function Generator' tab to create a signal for analysis.")

# ---------- TAB 3: FUNCTION GENERATOR ----------
with tab_generator:
    st.subheader("🎹 Function Generator")
    st.markdown("<p style='color:#888;'>Create a perfect waveform for analysis</p>", unsafe_allow_html=True)

    waveform_type = st.radio(
        "Select Waveform Type",
        ("Sine", "Cosine", "Square", "Triangular", "Impulse"),
        horizontal=True
    )

    col1, col2 = st.columns(2)
    with col1:
        gen_amp = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1, key="gen_amp")
        gen_freq = st.slider("Frequency (Hz)", 50, 2000, 440, 10, key="gen_freq")
    with col2:
        gen_dur = st.slider("Duration (s)", 1, 5, 2, 1, key="gen_dur")
        gen_fs = st.select_slider("Sampling Rate (Hz)", options=[8000, 16000, 44100, 48000], value=44100, key="gen_fs")

    if st.button("🎹 Generate Signal"):
        func_map = {
            "Sine": generate_sine_wave,
            "Cosine": generate_cosine_wave,
            "Square": generate_square_wave,
            "Triangular": generate_triangular_wave,
            "Impulse": generate_impulse
        }
        if waveform_type == "Impulse":
             t, generated_signal = func_map[waveform_type](gen_dur, gen_fs)
             if "fundamental_freq" in st.session_state:
                 del st.session_state["fundamental_freq"]
        else:
            t, generated_signal = func_map[waveform_type](gen_amp, gen_freq, gen_dur, gen_fs)
            st.session_state["fundamental_freq"] = gen_freq

        st.session_state["audio_data"] = generated_signal
        st.session_state["fs"] = gen_fs
        st.session_state["signal_source"] = "generator"
        
        st.markdown(f"### 📈 Generated {waveform_type} Wave")
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_samples = min(len(t), int(0.05 * gen_fs))
        ax.plot(t[:plot_samples], generated_signal[:plot_samples], color="#00ffe1")
        ax.set_title(f"Generated {waveform_type} Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        st.pyplot(fig)
        
        if can_record:
            st.audio(play_audio(generated_signal, gen_fs), format="audio/wav")
        st.success("✅ Signal generated and ready for analysis!")


# ---------- TAB 4: FOURIER ANALYSIS ----------
with tab_analysis:
    st.subheader("📊 Fourier Analysis")

    if st.session_state["audio_data"] is None:
        st.info("First, 🎤 record audio or 🎹 generate a signal.")
    else:
        data = st.session_state["audio_data"]
        fs = st.session_state["fs"]
        
        if st.session_state.get("signal_source") == "generator" and "fundamental_freq" in st.session_state:
            T = 1.0 / st.session_state["fundamental_freq"]
        else:
            T = len(data) / fs
        
        t_full = np.linspace(0, len(data)/fs, len(data), endpoint=False)
        plot_samples = min(len(t_full), int(0.05 * fs)) 

        st.markdown("### 📈 Input Signal Waveform")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_full[:plot_samples], data[:plot_samples], color="#00ffe1")
        ax.set_title("Input Signal (Zoomed In)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        view_mode = st.radio(
            "Select Fourier Analysis Type:",
            ("Trigonometric (Reconstruction)", "Exponential (Coefficients)"),
            horizontal=True
        )

        st.markdown("### 🎛 Parameters")
        N = st.slider("Number of Fourier Terms (N)", 1, 200, 20)
        
        if view_mode == "Trigonometric (Reconstruction)":
            st.markdown("### 📉 Trigonometric Fourier Series Reconstruction")
            a0, a, b = trig_coeff(data, N, T, t_full)
            recon = trig_recon(t_full, a0, a, b, T)
            w0 = 2 * np.pi / T

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            ax1.plot(t_full[:plot_samples], data[:plot_samples], 'w--', label="Original Signal", alpha=0.7)
            ax1.plot(t_full[:plot_samples], recon[:plot_samples], color="#ff4081", label=f"Reconstruction (N={N})")
            ax1.set_title("Original Signal vs. Fourier Reconstruction")
            ax1.set_ylabel("Amplitude")
            ax1.legend()
            ax1.grid(True, alpha=0.2)
            
            num_harmonics_to_show = min(N, 5)
            colors = plt.cm.viridis(np.linspace(0, 1, num_harmonics_to_show)) # type: ignore
            for n in range(1, num_harmonics_to_show + 1):
                harmonic = a[n] * np.cos(n * w0 * t_full) + b[n] * np.sin(n * w0 * t_full)
                ax2.plot(t_full[:plot_samples], harmonic[:plot_samples], label=f'Harmonic {n}', color=colors[n-1], alpha=0.9)
            ax2.set_title(f"First {num_harmonics_to_show} Harmonic Components")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            ax2.legend()
            ax2.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig2)

        elif view_mode == "Exponential (Coefficients)":
            st.markdown("### 🧠 Exponential Fourier Series Coefficients")
            coeffs = exp_coeff(data, N, T, t_full)
            n_vals = np.arange(-N, N + 1)
            
            col_real, col_imag = st.columns(2)
            with col_real:
                figR, axR = plt.subplots(figsize=(6, 4))
                axR.stem(n_vals, np.real(coeffs), linefmt='#00ffe1', markerfmt='o', basefmt=" ")
                axR.set_title("$Re(c_n)$")
                axR.set_xlabel("n")
                axR.set_ylabel("Amplitude")
                plt.tight_layout()
                st.pyplot(figR)
            
            with col_imag:
                figI, axI = plt.subplots(figsize=(6, 4))
                axI.stem(n_vals, np.imag(coeffs), linefmt='#ff4081', markerfmt='o', basefmt=" ")
                axI.set_title("$Im(c_n)$")
                axI.set_xlabel("n")
                axI.set_ylabel("Amplitude")
                plt.tight_layout()
                st.pyplot(figI)

# ---------- TAB 5: COMPARISON ----------
with tab_comparison:
    st.subheader("⚖ Compare Fourier Representations")
    st.markdown("<p style='color:#888;'>See how a different number of Fourier terms affects the analysis.</p>", unsafe_allow_html=True)

    # --- Comparison Setup ---
    compare_waveform = st.selectbox(
        "Select a waveform to compare:",
        ("Sine", "Cosine", "Square", "Triangular"),
        key="compare_select"
    )
    
    compare_view_mode = st.radio(
        "Select Analysis Type for Comparison:",
        ("Trigonometric (Reconstruction)", "Exponential (Coefficients)"),
        horizontal=True, key="compare_radio"
    )

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        compare_N1 = st.slider("Number of Terms (N1)", 1, 100, 5, key="compare_N1")
    with col_n2:
        compare_N2 = st.slider("Number of Terms (N2)", 1, 100, 25, key="compare_N2")

    # --- Comparison Execution ---
    # Define standard parameters for a fair comparison
    comp_amp = 1.0
    comp_freq = 10
    comp_dur = 1
    comp_fs = 2000
    comp_T = 1.0 / comp_freq
    
    func_map = {
        "Sine": generate_sine_wave, "Cosine": generate_cosine_wave,
        "Square": generate_square_wave, "Triangular": generate_triangular_wave
    }

    t, signal = func_map[compare_waveform](comp_amp, comp_freq, comp_dur, comp_fs)
    plot_samples = int(2 * comp_fs / comp_freq) # Plot two periods

    col1, col2 = st.columns(2)

    # --- Column 1 Plot (N1) ---
    with col1:
        st.markdown(f"#### Analysis with N = {compare_N1}")
        if compare_view_mode == "Trigonometric (Reconstruction)":
            a0, a, b = trig_coeff(signal, compare_N1, comp_T, t)
            recon = trig_recon(t, a0, a, b, comp_T)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(t[:plot_samples], signal[:plot_samples], 'w--', alpha=0.7, label='Original')
            ax.plot(t[:plot_samples], recon[:plot_samples], color="#00ffe1", label=f'N={compare_N1}')
            ax.set_title(f"{compare_waveform} (N={compare_N1})")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else: # Exponential
            coeffs = exp_coeff(signal, compare_N1, comp_T, t)
            n_vals = np.arange(-compare_N1, compare_N1 + 1)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.stem(n_vals, np.abs(coeffs), linefmt='#00ffe1', markerfmt='o', basefmt=" ")
            ax.set_title(f"Magnitude $|c_n|$ (N={compare_N1})")
            st.pyplot(fig)

    # --- Column 2 Plot (N2) ---
    with col2:
        st.markdown(f"#### Analysis with N = {compare_N2}")
        if compare_view_mode == "Trigonometric (Reconstruction)":
            a0, a, b = trig_coeff(signal, compare_N2, comp_T, t)
            recon = trig_recon(t, a0, a, b, comp_T)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(t[:plot_samples], signal[:plot_samples], 'w--', alpha=0.7, label='Original')
            ax.plot(t[:plot_samples], recon[:plot_samples], color="#ff4081", label=f'N={compare_N2}')
            ax.set_title(f"{compare_waveform} (N={compare_N2})")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else: # Exponential
            coeffs = exp_coeff(signal, compare_N2, comp_T, t)
            n_vals = np.arange(-compare_N2, compare_N2 + 1)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.stem(n_vals, np.abs(coeffs), linefmt='#ff4081', markerfmt='o', basefmt=" ")
            ax.set_title(f"Magnitude $|c_n|$ (N={compare_N2})")
            st.pyplot(fig)

    st.markdown("---")
    # --- Explanation Section ---
    with st.expander("📖 Analysis and Interpretation"):
        st.markdown(f"""
        ### Waveform Efficiency and Accuracy vs. Number of Terms (N)

        When we analyze how well a Fourier series represents a signal, *efficiency* refers to how few terms (a small 'N') are needed to get an *accurate* reconstruction. The key factor determining this is the *smoothness* of the waveform.

        Here's a ranking from most to least efficient:

        - *Sine/Cosine Waves (Most Efficient & Accurate):*
            - A pure sine or cosine wave is perfectly and completely represented by just *one* Fourier term (N=1). They are the fundamental building blocks of the series itself, so they are spectrally pure and require the minimum possible terms for perfect accuracy.

        - *Triangular Wave (Moderately Efficient & Accurate):*
            - This wave is continuous, but its slope changes abruptly at the peaks (its derivative is discontinuous). Because it's smoother than a square wave, its Fourier coefficients decrease more quickly. This means you can get a reasonably accurate approximation with a moderate number of terms (e.g., N=10 to 20).

        - *Square Wave (Least Efficient & Accurate):*
            - This is the least efficient of the group because it has instantaneous jumps (discontinuities). To approximate these sharp vertical edges, the Fourier series needs a large number of high-frequency harmonics.
            - For any given 'N', the square wave reconstruction will be the least accurate. You'll notice that even with a high 'N', there are persistent overshoots and "ringing" at the corners. This famous artifact is known as the *Gibbs Phenomenon*.

        *Conclusion:* For a fixed number of terms 'N', the accuracy of the reconstruction is highest for the smoothest signals. Conversely, to achieve a certain level of accuracy, a smoother signal requires a much smaller 'N' than a signal with sharp corners or jumps.

        ### Trigonometric vs. Exponential Fourier Analysis
        Both forms of the Fourier series are mathematically equivalent and describe the same phenomenon, but they offer different perspectives.

        - *Trigonometric Series (Reconstruction):*
            - Pros: Highly intuitive. It directly shows how the signal is built from sine and cosine waves, which are easy to visualize. The coefficients ($a_n, b_n$) are real numbers representing the amplitude of each component. It's excellent for educational purposes and understanding the concept of superposition.
            - Cons: Can be mathematically cumbersome with two sets of coefficients to manage.
        
        - *Exponential Series (Coefficients):*
            - Pros: Mathematically elegant and compact. Each harmonic is represented by a single complex number ($c_n$) that contains both amplitude and phase information. It is the standard form used in advanced signal processing, control systems, and communications because the math is often simpler. It provides a direct view of the signal's frequency spectrum.
            - Cons: Less intuitive for beginners, as it involves complex numbers and negative frequencies.

        *Which is better?* Neither is universally "better"—the best choice depends on the application. For visualizing how a signal is built from real sinusoids, the *Trigonometric* form is superior. For mathematical analysis, filtering, and understanding the full frequency/phase spectrum, the *Exponential* form is the professional standard.
        """)
