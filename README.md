# Accently - English Accent Analyzer


Accently is a web application that automatically analyzes English accents from videos using AI. Simply provide a video URL (YouTube, Loom, etc.) or upload a video file, and the application will identify the speaker's accent from among 16 different English accent categories.

## 🔍 Features

- **Accent Detection**: Identifies 16 different English accents with confidence scores
- **Multiple Input Methods**: Supports video URLs or direct file uploads
- **Fast Processing**: Optimized for cloud deployment with quick processing times
- **Detailed Analysis**: Provides probability breakdown for different accent possibilities

## 🌐 Live Demo

You can try Accently at: [https://accently.streamlit.app](https://accently.streamlit.app)

## 💻 Supported Accents

The application can identify the following English accents:
- African
- Australian
- Bermudian
- Canadian
- English (UK)
- Hong Kong
- Indian
- Irish
- Malaysian
- New Zealand
- Philippine
- Scottish
- Singaporean
- South Atlantic
- American (US)
- Welsh

## 🛠️ Technology Stack

- **Frontend & Backend**: Streamlit
- **Speech Processing**: SpeechBrain (ECAPA-TDNN model)
- **Video Processing**: MoviePy
- **Video Download**: yt-dlp
- **ML Framework**: PyTorch

## 🚀 Deployment

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/abd-km/Accently.git
   cd Accently
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Fork this repository
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select your forked repository
4. Deploy the app using the following settings:
   - Main file path: `app.py`
   - Python version: 3.9 or newer

## 📋 Requirements

- Python 3.9+
- SpeechBrain 1.0+
- MoviePy 1.0.3
- Streamlit 1.32.0+
- PyTorch 1.9.0+
- yt-dlp

## 📝 Notes

- The first run may take some time as the SpeechBrain model files are downloaded
- Accuracy depends on audio quality, background noise, and speaker clarity
- Video files are processed in the cloud and then deleted - no personal data is stored

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
