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
import traceback  # For error logging

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

# Model status section
status_container = st.container()
with status_container:
    if accent_classifier is not None:
        st.success("**Accent Classifier Ready** - AI model loaded successfully and ready for analysis")
    else:
        st.error("**Model Loading Failed** - The accent classifier could not be initialized")
        st.warning("""
        **Possible causes:**
        - Network connectivity issues preventing model download
        - Insufficient system resources
        - Dependency compatibility problems
        
        **Please try:**
        - Refreshing the page
        - Checking your internet connection
        - Waiting a moment for the model to download
        """)
        st.stop()

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
            # Check if this is a persistent directory we want to keep
            if ".streamlit/temp" in str(temp_dir_path):
                # Don't delete persistent cache directories
                return
            shutil.rmtree(temp_dir_path)
        except Exception as e:
            st.warning(f"Could not clean up temporary directory {temp_dir_path}: {e}")

@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid redownloading the same video
def cached_download_video(url):
    """Cached version of video download to improve performance for repeated URLs"""
    # Use a more stable temp directory within the Streamlit cache
    temp_dir = os.path.join(os.getcwd(), ".streamlit", "temp", f"video_{hash(url) % 10000}")
    os.makedirs(temp_dir, exist_ok=True)
    
    output_path = os.path.join(temp_dir, "video.mp4")
    
    # If the file already exists and is recent (within 1 hour), use it
    if os.path.exists(output_path) and (time.time() - os.path.getmtime(output_path)) < 3600:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        return output_path, temp_dir, file_size_mb
    
    # Set file size limit to 300MB (300 * 1024 * 1024 bytes)
    file_size_limit = 300 * 1024 * 1024
    
    # Updated options with better format selection and cache management
    ydl_opts = {
        'format': 'best[height<=720][filesize<300M]/best[filesize<300M]/best[height<=480]/best',  # More flexible format selection
        'outtmpl': output_path,
        'noplaylist': True,
        'quiet': True,  # Less verbose output
        'max_filesize': file_size_limit,  # Set maximum file size
        'no_cache_dir': True,  # Prevent caching issues that cause format errors
        'extract_flat': False,
        'ignoreerrors': False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Clear any existing cache before download (if available)
            if hasattr(ydl, 'cache') and hasattr(ydl.cache, 'remove'):
                try:
                    ydl.cache.remove()
                except:
                    pass  # Ignore if cache removal fails
            
            ydl.download([url])
        
        # Check if file exists and has reasonable size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:  # At least 10KB
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return output_path, temp_dir, file_size_mb
        else:
            cleanup_temp_dir(temp_dir)
            return None, None, 0
            
    except Exception as e:
        print(f"yt-dlp download error: {e}")
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
                st.info(f"Using only the first {duration} seconds of audio for faster analysis.")
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
            print(f"Error unpacking results: {unpack_e}")
            print(f"Traceback: {traceback.format_exc()}")
            return "Error", 0, f"Analysis failed: Unable to process model output - {unpack_e}"

    except Exception as e:
        st.error(f"Error during SpeechBrain accent analysis: {type(e).__name__} - {e}")
        print(f"Full error: {traceback.format_exc()}")
        return "Error", 0, f"Analysis failed: {e}"

# --- Streamlit UI ---
st.title("🎯 Accently")
st.subheader("AI-Powered English Accent Analysis")

# Hero section with app description
with st.container():
    st.markdown("""
    **Analyze English accents from video content using advanced AI technology.**
    
    This application uses a state-of-the-art SpeechBrain ECAPA-TDNN model to identify speakers' English accents from video or audio content.
    """)

# Supported accents section
with st.expander("Supported Accent Regions", expanded=False):
    cols = st.columns(4)
    accent_display = [accent.replace('_', ' ').title() for accent in SUPPORTED_ACCENTS]
    
    for i, accent in enumerate(accent_display):
        with cols[i % 4]:
            st.write(f"• {accent}")

st.divider()

# Add file upload option alongside URL
st.header("Choose Your Input Method")
st.markdown("Select how you'd like to provide your video content:")

tab1, tab2 = st.tabs(["Video URL", "Upload Video File"])

with tab1:
    st.markdown("**Enter a public video URL from platforms like YouTube, Vimeo, or direct MP4 links**")
    
    with st.container():
        video_url = st.text_input(
            "Video URL", 
            placeholder="https://www.youtube.com/watch?v=example",
            help="Paste any public video URL here. The video will be processed automatically."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_from_url = st.button(
                "Analyze Video from URL", 
                type="primary", 
                use_container_width=True,
                disabled=not video_url
            )
    
    if video_url and analyze_from_url:
        if accent_classifier is None:
            st.error("The accent classification model could not be loaded. Please refresh the page and try again.")
            st.stop()
            
        video_path, video_temp_dir = None, None
        audio_path, audio_temp_dir = None, None
        
        progress_container = st.container()
        
        try:
            with progress_container:
                # Step 1: Download
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Downloading video...")
                progress_bar.progress(10)
                video_path, video_temp_dir = download_video(video_url)

            if video_path:
                # Display video preview with robust error handling
                st.subheader("Video Preview")
                try:
                    # First verify the file exists and is readable
                    with open(video_path, 'rb') as f:
                        # Read just the first few bytes to confirm it's accessible
                        _ = f.read(1024)
                    
                    # Now display it
                    st.video(video_path)
                except Exception as video_err:
                    st.error(f"Could not display video preview: {video_err}")
                    # Try to continue with audio extraction even if preview fails
                
                with progress_container:
                    # Step 2: Extract audio
                    status_text.text("Extracting audio...")
                    progress_bar.progress(40)
                    audio_path, audio_temp_dir = extract_audio(video_path)

                if audio_path:
                    # Display audio preview
                    st.subheader("Extracted Audio")
                    st.audio(audio_path, format='audio/wav')
                    
                    with progress_container:
                        # Step 3: Analyze
                        status_text.text("Analyzing accent with AI...")
                        progress_bar.progress(70)
                        accent, confidence, summary = analyze_accent_speechbrain(audio_path, accent_classifier)
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")

                    # Results section
                    st.divider()
                    st.header("Analysis Results")
                    
                    if accent not in ["Unknown", "Error"]:
                        # Success case - show metrics in a nice layout
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            st.metric(
                                label="Detected Accent", 
                                value=accent.replace('_', ' ').title(),
                                help="The most likely English accent identified by the AI model"
                            )
                        
                        with result_col2:
                            confidence_color = "normal" if confidence >= 70 else "inverse"
                            st.metric(
                                label="Confidence Level", 
                                value=f"{confidence:.1f}%",
                                help="How confident the model is in this prediction"
                            )
                        
                        # Confidence interpretation
                        if confidence >= 80:
                            st.success("**High Confidence**: The model is very confident in this accent classification.")
                        elif confidence >= 60:
                            st.info("**Moderate Confidence**: Good prediction, but some uncertainty remains.")
                        else:
                            st.warning("**Lower Confidence**: Results should be interpreted with caution.")
                        
                        # Detailed analysis in expandable section
                        with st.expander("View Detailed Analysis", expanded=False):
                            st.markdown("**Full Analysis Report:**")
                            st.text_area("", summary, height=200, disabled=True)
                            
                    elif accent == "Error":
                        st.error("**Analysis Failed**")
                        st.error(f"**Reason:** {summary}")
                        
                    else: # Unknown
                        st.warning("**Uncertain Result**")
                        st.warning("Could not determine a specific English accent from the supported categories.")
                        with st.expander("View Analysis Details", expanded=True):
                            st.text_area("Analysis Log", summary, height=150, disabled=True)
                else:
                    st.error("**Audio Extraction Failed** - Could not extract audio from the video.")
            else:
                st.error("**Download Failed** - Please verify the URL is accessible and try again.")
        
        finally:
            # Cleanup
            cleanup_temp_dir(audio_temp_dir)
            cleanup_temp_dir(video_temp_dir)
            # Force garbage collection to free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear progress indicators
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
                
    elif not video_url and analyze_from_url:
        st.warning("Please enter a video URL before analyzing.")

with tab2:
    st.markdown("**Upload a video file directly from your device**")
    
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Supported formats: MP4, MOV, AVI, MKV, WebM (max 300MB recommended)"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_from_upload = st.button(
                "Analyze Uploaded Video", 
                type="primary", 
                use_container_width=True,
                disabled=uploaded_file is None
            )
    
    if uploaded_file is not None and analyze_from_upload:
        if accent_classifier is None:
            st.error("The accent classification model could not be loaded. Please refresh the page and try again.")
            st.stop()
            
        video_temp_dir = tempfile.mkdtemp(prefix="upload_")
        video_path = os.path.join(video_temp_dir, "uploaded_video.mp4")
        audio_path, audio_temp_dir = None, None
        
        # Create a progress container
        progress_container = st.container()
        
        try:
            # Save the uploaded file
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display video preview
            st.subheader("Uploaded Video Preview")
            st.video(video_path)
            
            with progress_container:
                # Step 1: Extract audio
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Extracting audio...")
                progress_bar.progress(30)
                audio_path, audio_temp_dir = extract_audio(video_path)

            if audio_path:
                # Display audio preview
                st.subheader("Extracted Audio")
                st.audio(audio_path, format='audio/wav')
                
                with progress_container:
                    # Step 2: Analyze
                    status_text.text("Analyzing accent with AI...")
                    progress_bar.progress(70)
                    accent, confidence, summary = analyze_accent_speechbrain(audio_path, accent_classifier)
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                # Results section
                st.divider()
                st.header("Analysis Results")
                
                if accent not in ["Unknown", "Error"]:
                    # Success case - show metrics in a nice layout
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            label="Detected Accent", 
                            value=accent.replace('_', ' ').title(),
                            help="The most likely English accent identified by the AI model"
                        )
                    
                    with result_col2:
                        st.metric(
                            label="Confidence Level", 
                            value=f"{confidence:.1f}%",
                            help="How confident the model is in this prediction"
                        )
                    
                    # Confidence interpretation
                    if confidence >= 80:
                        st.success("**High Confidence**: The model is very confident in this accent classification.")
                    elif confidence >= 60:
                        st.info("**Moderate Confidence**: Good prediction, but some uncertainty remains.")
                    else:
                        st.warning("**Lower Confidence**: Results should be interpreted with caution.")
                    
                    # Detailed analysis in expandable section
                    with st.expander("View Detailed Analysis", expanded=False):
                        st.markdown("**Full Analysis Report:**")
                        st.text_area("", summary, height=200, disabled=True)
                        
                elif accent == "Error":
                    st.error("**Analysis Failed**")
                    st.error(f"**Reason:** {summary}")
                    
                else: # Unknown
                    st.warning("**Uncertain Result**")
                    st.warning("Could not determine a specific English accent from the supported categories.")
                    with st.expander("View Analysis Details", expanded=True):
                        st.text_area("Analysis Log", summary, height=150, disabled=True)
            else:
                st.error("**Audio Extraction Failed** - Could not extract audio from the video.")
        
        finally:
            # Cleanup
            cleanup_temp_dir(audio_temp_dir)
            cleanup_temp_dir(video_temp_dir)
            # Force garbage collection to free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear progress indicators
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

st.divider()

# Information section
st.header("About This Tool")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.subheader("Technology")
    st.markdown("""
    - **AI Model**: SpeechBrain ECAPA-TDNN
    - **Training Data**: CommonAccent Dataset
    - **Processing**: Real-time audio analysis
    - **Accuracy**: Research-grade classification
    """)

with info_col2:
    st.subheader("Usage Tips")
    st.markdown("""
    - **Audio Quality**: Clear speech works best
    - **Duration**: 30-60 seconds is optimal
    - **Background**: Minimize background noise
    - **Languages**: English speech only
    """)

# Disclaimer section
with st.expander("Important Disclaimers", expanded=False):
    st.markdown("""
    **Please Note:**
    
    - **Research Tool**: This is a proof-of-concept application for educational and research purposes
    - **Accuracy Limitations**: Results depend on audio quality, speaker clarity, and model training data
    - **Bias Considerations**: AI models may have inherent biases based on their training data
    - **Privacy**: Videos are processed temporarily and not stored permanently
    - **Internet Required**: Model downloads and video processing require internet connectivity
    
    **Not suitable for:**
    - Critical decision-making processes
    - Official language assessment
    - Commercial accent evaluation services
    """)

st.divider()

# Footer
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style="text-align: center;">
        <p><strong>Developed with ❤ by Abdullah Mostafa</strong></p>
    </div>
    """, unsafe_allow_html=True)
