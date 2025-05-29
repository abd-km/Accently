import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="Accently - English Accent Analyzer", layout="wide")

import tempfile
import os
import shutil
from moviepy.editor import VideoFileClip
import yt_dlp
import requests # For direct MP4 download fallback
import torch # SpeechBrain dependency
# Updated import based on SpeechBrain 1.0 recommendation
from speechbrain.inference import EncoderClassifier
import time
import gc  # Garbage collection for better memory management

# --- Configuration & Model Loading ---
# Use Streamlit's caching to load the model only once and reuse it.
@st.cache_resource
def load_accent_classifier():
    try:
        # Use print instead of st.write for logging during model loading
        print("Initializing Accent Classifier (ECAPA-TDNN model)... This may take a moment on first run as model files are downloaded.")
        # Define a persistent directory for model saving relative to the app's root
        savedir = os.path.join(os.getcwd(), "pretrained_models", "accent-id-commonaccent_ecapa")
        os.makedirs(savedir, exist_ok=True) # Ensure the directory exists
        
        classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir=savedir, # SpeechBrain will save/load model files here
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"} # Use GPU if available
        )
        
        # Try to use model quantization for CPU inference to speed up processing
        if not torch.cuda.is_available() and hasattr(torch, 'quantization'):
            try:
                print("Applying quantization to model for faster CPU inference...")
                classifier.mods.encoder = torch.quantization.quantize_dynamic(
                    classifier.mods.encoder, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("Model quantization applied successfully.")
            except Exception as quant_e:
                print(f"Model quantization failed (non-critical): {quant_e}")
        
        print("Accent Classifier loaded successfully.")
        return classifier
    except Exception as e:
        print(f"Fatal Error: Could not load SpeechBrain Accent Classifier: {e}")
        print("This might be due to network issues preventing model download, or compatibility problems.")
        return None

accent_classifier = load_accent_classifier()

# Display loading status after model is loaded
if accent_classifier is not None:
    st.success("Accent Classifier loaded successfully.")
else:
    st.error("The accent classifier model could not be loaded. This might be due to network issues or compatibility problems.")
    st.error("Please check your internet connection and try refreshing. If the problem persists, the model source or dependencies might need attention.")

# Supported accents by the model (for user information)
SUPPORTED_ACCENTS = [
    "african", "australia", "bermuda", "canada", "england", "hongkong",
    "indian", "ireland", "malaysia", "newzealand", "philippines",
    "scotland", "singapore", "southatlandtic", "us", "wales"
]

# --- Helper Functions ---

def cleanup_temp_dir(temp_dir_path):
    if temp_dir_path and os.path.exists(temp_dir_path):
        try:
            shutil.rmtree(temp_dir_path)
        except Exception as e:
            st.warning(f"Could not clean up temporary directory {temp_dir_path}: {e}")

@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid redownloading the same video
def cached_download_video(url):
    """Cached version of video download to improve performance for repeated URLs"""
    temp_dir = tempfile.mkdtemp(prefix="video_dl_")
    output_path = os.path.join(temp_dir, "video.mp4")
    
    # Set file size limit to 300MB (300 * 1024 * 1024 bytes)
    file_size_limit = 300 * 1024 * 1024
    
    # Simpler options that are more likely to work
    ydl_opts = {
        'format': 'best[filesize<300M][ext=mp4]/best[filesize<300M]',  # Limit file size
        'outtmpl': output_path,
        'noplaylist': True,
        'quiet': True,  # Less verbose output
        'max_filesize': file_size_limit  # Set maximum file size
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Check if file exists and has reasonable size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:  # At least 10KB
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return output_path, temp_dir, file_size_mb
        else:
            cleanup_temp_dir(temp_dir)
            return None, None, 0
            
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        return None, None, 0

def download_video(url):
    """Wrapper for cached download with UI feedback"""
    st.write(f"Downloading video from {url}...")
    start_time = time.time()
    
    video_path, temp_dir, file_size_mb = cached_download_video(url)
    
    if video_path:
        download_time = time.time() - start_time
        st.write(f"Video downloaded successfully! File size: {file_size_mb:.2f} MB (in {download_time:.1f}s)")
        return video_path, temp_dir
    else:
        st.error("Failed to download video. Please check the URL and ensure it's publicly accessible.")
        return None, None

def extract_audio(video_path, max_duration=60):
    """Extract audio from video, limiting to first max_duration seconds to improve performance"""
    if not video_path or not os.path.exists(video_path):
        st.error("Video file not found for audio extraction.")
        return None, None
    
    temp_audio_dir = tempfile.mkdtemp(prefix="audio_ext_")
    # SpeechBrain expects 16kHz, mono. MoviePy can resample if needed.
    audio_path = os.path.join(temp_audio_dir, "audio.wav")

    try:
        st.write(f"Extracting audio from video...")
        start_time = time.time()
        
        with VideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                st.error("Video file does not contain an audio track.")
                cleanup_temp_dir(temp_audio_dir)
                return None, None
            
            # Limit to first max_duration seconds for faster processing
            duration = min(max_duration, video_clip.duration)
            if duration < video_clip.duration:
                st.info(f"⏱️ Using only the first {duration} seconds of audio for faster analysis.")
                video_clip = video_clip.subclip(0, duration)
            
            # Extract as WAV, mono, 16kHz (SpeechBrain default)
            video_clip.audio.write_audiofile(
                audio_path,
                codec='pcm_s16le', # Standard WAV codec
                fps=16000,         # Target 16kHz sampling rate
                nbytes=2,          # 16-bit
                ffmpeg_params=["-ac", "1", "-q:a", "9"] # Mono channel, lower quality for faster processing
            )
        
        extraction_time = time.time() - start_time
        st.write(f"Audio extracted in {extraction_time:.1f}s")
        return audio_path, temp_audio_dir
    except Exception as e:
        st.error(f"Error extracting audio: {type(e).__name__} - {e}")
        cleanup_temp_dir(temp_audio_dir)
        return None, None

def analyze_accent_speechbrain(audio_path, classifier):
    if not classifier:
        return "Error", 0, "Accent classifier model not loaded."
    if not audio_path or not os.path.exists(audio_path):
        return "Error", 0, "Audio file missing for analysis."

    try:
        st.write("Analyzing accent with SpeechBrain model...")
        start_time = time.time()
        
        # The classify_file method handles loading and preprocessing the audio
        result = classifier.classify_file(audio_path)
        
        # Debug info for advanced users
        with st.expander("Debug Info (Advanced)", expanded=False):
            st.text(f"Result type: {type(result)}, Length: {len(result)}")
            for i, item in enumerate(result):
                st.text(f"Item {i}: Type: {type(item)}, Shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
        
        # Unpack result based on its length and structure
        try:
            if len(result) >= 4:
                out_prob, score, index, text_lab = result
            elif len(result) == 3:
                # Some versions return 3 elements
                out_prob, index, text_lab = result
                score = torch.tensor([0.0])  # Default score
            elif len(result) == 2:
                # Handle potential 2-element return
                out_prob, index = result
                text_lab = ["Unknown"]
                score = torch.tensor([0.0])
            elif len(result) == 1:
                # Handle single tensor return (unlikely but possible)
                out_prob = result[0]
                index = torch.argmax(out_prob) if isinstance(out_prob, torch.Tensor) else torch.tensor([0])
                text_lab = ["Unknown"]
                score = torch.tensor([0.0])
            else:
                # Empty result (shouldn't happen)
                raise ValueError("Empty result from classifier")
                
            # Handle tensor with multiple elements
            if isinstance(out_prob, torch.Tensor) and out_prob.numel() > 1:
                # If out_prob is a multi-element tensor, handle it differently
                if out_prob.dim() > 1:
                    # If it's a multi-dimensional tensor, flatten it
                    out_prob = out_prob.squeeze()
                    
                # Find the max probability and its index
                max_prob, max_idx = torch.max(out_prob, dim=0)
                confidence_percentage = max_prob.item() * 100
                
                # Extract accent from text_lab if available, otherwise use SUPPORTED_ACCENTS
                if isinstance(text_lab, list) and len(text_lab) > 0:
                    detected_accent = text_lab[0]
                elif hasattr(classifier.hparams.label_encoder, 'ind2lab') and max_idx.item() in classifier.hparams.label_encoder.ind2lab:
                    detected_accent = classifier.hparams.label_encoder.ind2lab[max_idx.item()]
                elif max_idx.item() < len(SUPPORTED_ACCENTS):
                    detected_accent = SUPPORTED_ACCENTS[max_idx.item()]
                else:
                    detected_accent = f"Accent_{max_idx.item()}"
            else:
                # Handle the case when out_prob is a scalar tensor or not a tensor
                if hasattr(out_prob, 'item'):
                    confidence_percentage = out_prob.item() * 100
                else:
                    confidence_percentage = float(out_prob) * 100 if isinstance(out_prob, (int, float)) else 0.0
                
                # Extract accent
                if isinstance(text_lab, list) and len(text_lab) > 0:
                    detected_accent = text_lab[0]
                else:
                    detected_accent = "Unknown"
                    
            # Prepare a more detailed summary
            summary_details = "Top accent probabilities:\n"
            
            # Try to get sorted probabilities if possible
            try:
                if isinstance(out_prob, torch.Tensor) and out_prob.numel() > 1:
                    # Squeeze any extra dimensions and sort
                    all_probs_sorted = torch.sort(out_prob.squeeze(), descending=True)
                    top_n = min(5, out_prob.numel())
                    
                    for i in range(top_n):
                        prob = all_probs_sorted.values[i].item()
                        idx = all_probs_sorted.indices[i].item()
                        
                        # Try to get the label name safely
                        try:
                            if hasattr(classifier.hparams.label_encoder, 'ind2lab') and idx in classifier.hparams.label_encoder.ind2lab:
                                label_name = classifier.hparams.label_encoder.ind2lab[idx]
                            elif idx < len(SUPPORTED_ACCENTS):
                                label_name = SUPPORTED_ACCENTS[idx]
                            else:
                                label_name = f"Accent_{idx}"
                        except Exception as label_err:
                            # If label can't be determined, use a placeholder
                            label_name = f"Accent_{idx}"
                            
                        summary_details += f"- {label_name}: {prob*100:.2f}%\n"
                else:
                    # If there's only one probability or none, just show the detected accent
                    summary_details += f"- {detected_accent}: {confidence_percentage:.2f}%\n"
                    
            except Exception as detail_e:
                # If we can't generate detailed probabilities, provide basic info
                summary_details += f"- {detected_accent}: {confidence_percentage:.2f}%\n"
                summary_details += f"(Detailed probabilities unavailable: {str(detail_e)})\n"
            
            analysis_time = time.time() - start_time
            summary = f"The primary detected accent is '{detected_accent}'.\n" + summary_details
            summary += f"\nAnalysis completed in {analysis_time:.2f} seconds."
            
            st.write(f"Accent analysis complete in {analysis_time:.2f}s")
            return detected_accent, confidence_percentage, summary
            
        except Exception as unpack_e:
            import traceback
            print(f"Error unpacking results: {unpack_e}")
            print(f"Traceback: {traceback.format_exc()}")
            return "Error", 0, f"Analysis failed: Unable to process model output - {unpack_e}"

    except Exception as e:
        import traceback
        st.error(f"Error during SpeechBrain accent analysis: {type(e).__name__} - {e}")
        print(f"Full error: {traceback.format_exc()}")
        return "Error", 0, f"Analysis failed: {e}"

# --- Streamlit UI ---
st.image("https://remwaste.com/wp-content/uploads/2023/08/REM-Waste-Logo-Dark.png", width=200)
st.title("🗣️ Accently - English Accent Analyzer")

st.markdown(f"""
This tool analyzes a speaker's English accent from a video URL using a pre-trained AI model.
It can identify among the following **{len(SUPPORTED_ACCENTS)} English accents**: `{(", ".join(SUPPORTED_ACCENTS))}`.
""")

# Add file upload option alongside URL
st.write("### Upload or link to a video")
tab1, tab2 = st.tabs(["Video URL", "Upload Video"])

with tab1:
    video_url = st.text_input("Enter public video URL (YouTube, Loom, direct MP4, etc.):", 
                              placeholder="e.g., https://www.youtube.com/watch?v=...")
    analyze_from_url = st.button("Analyze URL", type="primary", use_container_width=True)
    
    if video_url and analyze_from_url:
        if accent_classifier is None:
            st.error("The accent classification model could not be loaded. The application cannot proceed.")
            st.stop()
            
        video_path, video_temp_dir = None, None
        audio_path, audio_temp_dir = None, None
        
        try:
            with st.spinner("Step 1/3: Downloading video... This may take a moment."):
                video_path, video_temp_dir = download_video(video_url)

            if video_path:
                st.video(video_path)
                with st.spinner("Step 2/3: Extracting audio..."):
                    audio_path, audio_temp_dir = extract_audio(video_path)

                if audio_path:
                    st.audio(audio_path, format='audio/wav')
                    with st.spinner("Step 3/3: Analyzing accent with AI model..."):
                        accent, confidence, summary = analyze_accent_speechbrain(audio_path, accent_classifier)

                    st.subheader("📊 Accent Analysis Results")
                    if accent not in ["Unknown", "Error"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Detected English Accent", value=accent.upper())
                        with col2:
                            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                        
                        with st.expander("View Detailed Analysis", expanded=False):
                            st.text_area("Analysis Details", summary, height=200)
                            
                        if confidence < 60:
                            st.warning("The confidence score is moderate. This could be due to mixed accent features, audio clarity issues, or an accent not well-represented in the model's training data.")
                    elif accent == "Error":
                        st.error(f"Accent analysis failed. Reason: {summary}")
                    else: # Unknown
                        st.warning("Could not determine a specific English accent from the supported categories.")
                        with st.expander("View Analysis Log", expanded=True):
                            st.text_area("Analysis Details", summary, height=200)
                else:
                    st.error("❌ Failed to extract audio from the video.")
            else:
                st.error("❌ Failed to download the video. Please check the URL and ensure it's publicly accessible.")
        
        finally:
            # Cleanup
            cleanup_temp_dir(audio_temp_dir)
            cleanup_temp_dir(video_temp_dir)
            # Force garbage collection to free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        st.warning("⚠️ Please enter a video URL to analyze.")

with tab2:
    uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, AVI, etc.)", type=["mp4", "mov", "avi", "mkv", "webm"])
    analyze_from_upload = st.button("Analyze Upload", type="primary", use_container_width=True)
    
    if uploaded_file is not None and analyze_from_upload:
        if accent_classifier is None:
            st.error("The accent classification model could not be loaded. The application cannot proceed.")
            st.stop()
            
        video_temp_dir = tempfile.mkdtemp(prefix="upload_")
        video_path = os.path.join(video_temp_dir, "uploaded_video.mp4")
        audio_path, audio_temp_dir = None, None
        
        try:
            # Save the uploaded file
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.video(video_path)
            with st.spinner("Step 1/2: Extracting audio..."):
                audio_path, audio_temp_dir = extract_audio(video_path)

            if audio_path:
                st.audio(audio_path, format='audio/wav')
                with st.spinner("Step 2/2: Analyzing accent with AI model..."):
                    accent, confidence, summary = analyze_accent_speechbrain(audio_path, accent_classifier)

                st.subheader("📊 Accent Analysis Results")
                if accent not in ["Unknown", "Error"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Detected English Accent", value=accent.upper())
                    with col2:
                        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                    
                    with st.expander("View Detailed Analysis", expanded=False):
                        st.text_area("Analysis Details", summary, height=200)
                        
                    if confidence < 60:
                        st.warning("The confidence score is moderate. This could be due to mixed accent features, audio clarity issues, or an accent not well-represented in the model's training data.")
                elif accent == "Error":
                    st.error(f"Accent analysis failed. Reason: {summary}")
                else: # Unknown
                    st.warning("Could not determine a specific English accent from the supported categories.")
                    with st.expander("View Analysis Log", expanded=True):
                        st.text_area("Analysis Details", summary, height=200)
            else:
                st.error("❌ Failed to extract audio from the video.")
        
        finally:
            # Cleanup
            cleanup_temp_dir(audio_temp_dir)
            cleanup_temp_dir(video_temp_dir)
            # Force garbage collection to free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

st.markdown("---")
st.markdown("""
**Disclaimer:** Accent classification is complex. This tool provides an estimation based on a SpeechBrain ECAPA-TDNN model trained on the CommonAccent dataset. 
Results depend on audio quality, speaker clarity, background noise, and model limitations. This tool is a proof-of-concept.
""")

# Add footer with GitHub link
st.markdown("""
<div style="text-align: center; margin-top: 20px;">
    <p>
        <a href="https://github.com/abd-km/Accently" target="_blank">View on GitHub</a> | 
        Built with SpeechBrain & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
