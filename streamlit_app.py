import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sounddevice as sd
import soundfile as sf
import io
from scipy import signal as sp_signal

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
    return t, amp * sp_signal.sawtooth(2 * np.pi * freq * t, 0.5)

def generate_impulse(duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    impulse = np.zeros_like(t)
    impulse[0] = 1.0
    return t, impulse

# --- Fourier Calculation Functions ---
def trig_coeff(signal, n_max, T, t):
    a0 = np.mean(signal)
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    w0 = 2 * np.pi / T
    dt = t[1] - t[0] if len(t) > 1 else T / len(signal)
    for n in range(1, n_max + 1):
        a[n] = (2.0 / T) * np.sum(signal * np.cos(n * w0 * t)) * dt
        b[n] = (2.0 / T) * np.sum(signal * np.sin(n * w0 * t)) * dt
    return a0, a, b

def trig_recon(t, a0, a, b, T):
    w0 = 2 * np.pi / T
    s = (a0 / 2) * np.ones_like(t)
    for n in range(1, len(a)):
        s += a[n] * np.cos(n * w0 * t) + b[n] * np.sin(n * w0 * t)
    return s

def exp_coeff(signal, n_max, T, t):
    coeffs = np.zeros(2 * n_max + 1, dtype=complex)
    w0 = 2 * np.pi / T
    dt = t[1] - t[0] if len(t) > 1 else T / len(signal)
    for i, n in enumerate(range(-n_max, n_max + 1)):
        coeffs[i] = (1.0 / T) * np.sum(signal * np.exp(-1j * n * w0 * t)) * dt
    return coeffs

# --- Plotting Functions ---
def plot_waveform(t, data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=data, mode='lines', line=dict(color='#00ffe1')))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_stem_plot(n_vals, coeffs_part, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_vals, y=coeffs_part,
        mode='markers',
        marker=dict(color=color, size=6)
    ))
    for i in range(len(n_vals)):
        fig.add_shape(
            type='line',
            x0=n_vals[i], y0=0, x1=n_vals[i], y1=coeffs_part[i],
            line=dict(color=color, width=2)
        )
    fig.update_layout(
        title=title,
        xaxis_title="n",
        yaxis_title="Amplitude",
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

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
    
    #### Exponential Fourier Series
    This is a more compact form using complex exponentials:
    $$
    s(t) = \sum_{n=-\infty}^{\infty} c_n e^{j n \omega_0 t}
    $$
    Where $c_n$ are the complex Fourier coefficients. This app visualizes the real and imaginary parts of the $c_n$ coefficients, showing the signal's frequency spectrum.

    ### Applications of This Visualization Tool
    This tool serves several practical and educational purposes:
    - *Educational Tool:* For students of engineering, physics, and mathematics, this app provides an intuitive, hands-on way to grasp how the Fourier series works.
    - *Audio Synthesis:* The core principle of additive synthesis is building complex sounds by adding sine waves. This tool is a basic additive synthesizer.
    - *Understanding Audio Effects:* Visualizing the frequency spectrum helps in understanding how filters and equalizers (EQ) work.
    - *Signal Analysis:* For any recorded signal, you can immediately see its harmonic content.
    """)

# ---------- TAB 2: RECORDER ----------
with tab_recorder:
    st.subheader("🎙 Audio Recorder")
    dur = st.slider("Recording Duration (seconds)", 1, 10, 5)
    
    if st.button("▶ Start Recording"):
        audio = record_audio(duration=dur, fs=st.session_state["fs"])
        st.session_state["audio_data"] = audio
        st.session_state["fs"] = 44100 # Reset fs to default
        st.session_state["signal_source"] = "recorder"
        if "fundamental_freq" in st.session_state:
            del st.session_state["fundamental_freq"]
        st.success("✅ Recording finished and ready for analysis!")

    if st.session_state.get("audio_data") is not None and st.session_state.get("signal_source") == "recorder":
        st.audio(play_audio(st.session_state["audio_data"], st.session_state["fs"]), format="audio/wav")
        
        st.markdown("### 📈 Recorded Waveform")
        data = st.session_state["audio_data"]
        fs = st.session_state["fs"]
        t = np.linspace(0, len(data)/fs, len(data), endpoint=False)
        fig = plot_waveform(t, data, "Recorded Audio Signal")
        st.plotly_chart(fig, use_container_width=True)


# ---------- TAB 3: FUNCTION GENERATOR ----------
with tab_generator:
    st.subheader("🎹 Function Generator")
    st.markdown("<p style='color:#888;'>Create a perfect waveform for analysis</p>", unsafe_allow_html=True)

    waveform_type = st.radio("Select Waveform Type", ("Sine", "Cosine", "Square", "Triangular", "Impulse"), horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        gen_amp = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1, key="gen_amp")
        gen_freq = st.slider("Frequency (Hz)", 50, 2000, 440, 10, key="gen_freq")
    with col2:
        gen_dur = st.slider("Duration (s)", 1, 5, 2, 1, key="gen_dur")
        gen_fs = st.select_slider("Sampling Rate (Hz)", options=[8000, 16000, 44100, 48000], value=44100, key="gen_fs")

    if st.button("🎹 Generate Signal"):
        func_map = {"Sine": generate_sine_wave, "Cosine": generate_cosine_wave, "Square": generate_square_wave, "Triangular": generate_triangular_wave, "Impulse": generate_impulse}
        
        if waveform_type == "Impulse":
            t, generated_signal = func_map[waveform_type](gen_dur, gen_fs)
            if "fundamental_freq" in st.session_state: del st.session_state["fundamental_freq"]
        else:
            t, generated_signal = func_map[waveform_type](gen_amp, gen_freq, gen_dur, gen_fs)
            st.session_state["fundamental_freq"] = gen_freq

        st.session_state["audio_data"] = generated_signal
        st.session_state["fs"] = gen_fs
        st.session_state["signal_source"] = "generator"
        
        st.markdown(f"### 📈 Generated {waveform_type} Wave")
        plot_samples = min(len(t), int(0.05 * gen_fs))
        fig = plot_waveform(t[:plot_samples], generated_signal[:plot_samples], f"Generated {waveform_type} Signal")
        st.plotly_chart(fig, use_container_width=True)
        
        st.audio(play_audio(generated_signal, gen_fs), format="audio/wav")
        st.success("✅ Signal generated and ready for analysis!")


# ---------- TAB 4: FOURIER ANALYSIS ----------
with tab_analysis:
    st.subheader("📊 Fourier Analysis")

    if st.session_state.get("audio_data") is None:
        st.info("First, 🎤 record audio or 🎹 generate a signal.")
    else:
        data = st.session_state["audio_data"]
        fs = st.session_state["fs"]
        
        T = 1.0 / st.session_state["fundamental_freq"] if st.session_state.get("signal_source") == "generator" and "fundamental_freq" in st.session_state else len(data) / fs
        
        t_full = np.linspace(0, len(data)/fs, len(data), endpoint=False)
        plot_samples = min(len(t_full), int(0.05 * fs))

        st.markdown("### 📈 Input Signal Waveform")
        fig = plot_waveform(t_full[:plot_samples], data[:plot_samples], "Input Signal (Zoomed In)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        view_mode = st.radio("Select Fourier Analysis Type:", ("Trigonometric (Reconstruction)", "Exponential (Coefficients)"), horizontal=True)
        N = st.slider("Number of Fourier Terms (N)", 1, 200, 20)
        
        if view_mode == "Trigonometric (Reconstruction)":
            st.markdown("### 📉 Trigonometric Fourier Series Reconstruction")
            a0, a, b = trig_coeff(data, N, T, t_full)
            recon = trig_recon(t_full, a0, a, b, T)
            w0 = 2 * np.pi / T

            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.1, subplot_titles=("Original Signal vs. Fourier Reconstruction", f"First {min(N, 5)} Harmonic Components"))
            
            fig2.add_trace(go.Scatter(x=t_full[:plot_samples], y=data[:plot_samples], mode='lines', name='Original', line=dict(color='white', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=t_full[:plot_samples], y=recon[:plot_samples], mode='lines', name=f'Reconstruction (N={N})', line=dict(color='#ff4081')), row=1, col=1)

            num_harmonics_to_show = min(N, 5)
            for n in range(1, num_harmonics_to_show + 1):
                harmonic = a[n] * np.cos(n * w0 * t_full) + b[n] * np.sin(n * w0 * t_full)
                fig2.add_trace(go.Scatter(x=t_full[:plot_samples], y=harmonic[:plot_samples], name=f'Harmonic {n}', mode='lines', line=dict(width=2)), row=2, col=1)
            
            fig2.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=60, b=20), height=600)
            fig2.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig2.update_yaxes(title_text="Amplitude", row=2, col=1)
            fig2.update_xaxes(title_text="Time (s)", row=2, col=1)
            st.plotly_chart(fig2, use_container_width=True)

        elif view_mode == "Exponential (Coefficients)":
            st.markdown("### 🧠 Exponential Fourier Series Coefficients")
            coeffs = exp_coeff(data, N, T, t_full)
            n_vals = np.arange(-N, N + 1)
            
            col_real, col_imag = st.columns(2)
            with col_real:
                figR = create_stem_plot(n_vals, np.real(coeffs), "$Re(c_n)$", '#00ffe1')
                st.plotly_chart(figR, use_container_width=True)
            with col_imag:
                figI = create_stem_plot(n_vals, np.imag(coeffs), "$Im(c_n)$", '#ff4081')
                st.plotly_chart(figI, use_container_width=True)

# ---------- TAB 5: COMPARISON ----------
with tab_comparison:
    st.subheader("⚖ Compare Fourier Representations")
    st.markdown("<p style='color:#888;'>See how a different number of Fourier terms affects the analysis.</p>", unsafe_allow_html=True)

    compare_waveform = st.selectbox("Select a waveform to compare:", ("Sine", "Cosine", "Square", "Triangular"), key="compare_select")
    compare_view_mode = st.radio("Select Analysis Type for Comparison:", ("Trigonometric (Reconstruction)", "Exponential (Coefficients)"), horizontal=True, key="compare_radio")

    col_n1, col_n2 = st.columns(2)
    with col_n1: compare_N1 = st.slider("Number of Terms (N1)", 1, 100, 5, key="compare_N1")
    with col_n2: compare_N2 = st.slider("Number of Terms (N2)", 1, 100, 25, key="compare_N2")

    comp_amp, comp_freq, comp_dur, comp_fs = 1.0, 10, 1, 2000
    comp_T = 1.0 / comp_freq
    
    func_map = {"Sine": generate_sine_wave, "Cosine": generate_cosine_wave, "Square": generate_square_wave, "Triangular": generate_triangular_wave}
    t, signal = func_map[compare_waveform](comp_amp, comp_freq, comp_dur, comp_fs)
    plot_samples = int(2 * comp_fs / comp_freq)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Analysis with N = {compare_N1}")
        if compare_view_mode == "Trigonometric (Reconstruction)":
            a0, a, b = trig_coeff(signal, compare_N1, comp_T, t)
            recon = trig_recon(t, a0, a, b, comp_T)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t[:plot_samples], y=signal[:plot_samples], mode='lines', name='Original', line=dict(color='white', dash='dash')))
            fig.add_trace(go.Scatter(x=t[:plot_samples], y=recon[:plot_samples], mode='lines', name=f'N={compare_N1}', line=dict(color='#00ffe1')))
            fig.update_layout(title=f"{compare_waveform} (N={compare_N1})", template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            coeffs = exp_coeff(signal, compare_N1, comp_T, t)
            n_vals = np.arange(-compare_N1, compare_N1 + 1)
            fig = create_stem_plot(n_vals, np.abs(coeffs), f"Magnitude $|c_n|$ (N={compare_N1})", '#00ffe1')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### Analysis with N = {compare_N2}")
        if compare_view_mode == "Trigonometric (Reconstruction)":
            a0, a, b = trig_coeff(signal, compare_N2, comp_T, t)
            recon = trig_recon(t, a0, a, b, comp_T)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t[:plot_samples], y=signal[:plot_samples], mode='lines', name='Original', line=dict(color='white', dash='dash')))
            fig.add_trace(go.Scatter(x=t[:plot_samples], y=recon[:plot_samples], mode='lines', name=f'N={compare_N2}', line=dict(color='#ff4081')))
            fig.update_layout(title=f"{compare_waveform} (N={compare_N2})", template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            coeffs = exp_coeff(signal, compare_N2, comp_T, t)
            n_vals = np.arange(-compare_N2, compare_N2 + 1)
            fig = create_stem_plot(n_vals, np.abs(coeffs), f"Magnitude $|c_n|$ (N={compare_N2})", '#ff4081')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    with st.expander("📖 Analysis and Interpretation"):
        st.markdown("""
        ### Waveform Efficiency vs. Number of Terms (N)
        Efficiency refers to how few terms (a small 'N') are needed to get an accurate reconstruction. The key factor is the smoothness of the waveform.
        - *Sine/Cosine Waves (Most Efficient):* Perfectly represented by just one Fourier term (N=1).
        - *Triangular Wave (Moderately Efficient):* Smoother than a square wave, its coefficients decrease quickly. A moderate 'N' gives a good approximation.
        - *Square Wave (Least Efficient):* Has sharp jumps (discontinuities) that require a large number of high-frequency harmonics to approximate. Even with a high 'N', you see overshoots known as the Gibbs Phenomenon.

        ### Trigonometric vs. Exponential Fourier Analysis
        - *Trigonometric Series (Reconstruction):* Highly intuitive. It directly shows how the signal is built from sine and cosine waves. Excellent for educational purposes.
        - *Exponential Series (Coefficients):* Mathematically elegant and compact. It uses a single complex number ($c_n$) for each harmonic, containing both amplitude and phase. It is the standard for advanced signal processing and provides a direct view of the signal's frequency spectrum.
        
        *Which is better?* For visualizing how a signal is built, the *Trigonometric* form is superior. For mathematical analysis and understanding the full frequency/phase spectrum, the *Exponential* form is the professional standard.
        """)
