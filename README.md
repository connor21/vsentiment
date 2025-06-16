# Video Sentiment Analysis (vsentiment)

A Streamlit application that analyzes sentiment from video recordings by:
1. Extracting audio from video
2. Transcribing speech to text using OpenAI Whisper
3. Performing sentiment analysis using German BERT model
4. Visualizing sentiment changes over time

## Features
- Video upload interface
- High-quality speech-to-text transcription (Whisper)
- German language sentiment analysis (oliverguhr/german-sentiment-bert)
- Facial expression analysis (DeepFace)
- Combined time-series visualization of:
  - Text sentiment
  - Facial emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- Automatic cleanup of temporary files

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download Whisper model (will happen automatically on first run)

## Container Deployment (Docker)
1. Build the container image:
```bash
docker build -t vsentiment .
```
2. Run the container:
```bash
docker run -p 8501:8501 vsentiment
```
3. Access the application at:
```bash
http://localhost:8501
```

Note: For GPU support, install NVIDIA Container Toolkit and add `--gpus all` to the docker run command.

## Usage
1. Run the application:
```bash
streamlit run app.py
```
2. Open http://localhost:8501 in your browser
3. Upload a video file (MP4 or MOV)
4. View analysis results including:
   - Full transcript
   - Sentiment timeline chart
   - Overall sentiment score

## Technical Details
- Speech Recognition: OpenAI Whisper (base model)
- Sentiment Analysis: German BERT model (oliverguhr/german-sentiment-bert)
- Visualization: Matplotlib with Streamlit
- Audio Processing: MoviePy

## Project Structure
```
vsentiment/
├── app.py              # Main MCP server implementation
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── PLANNING.md         # Project planning
├── TASK.md             # Task tracking
└── .streamlit/         # Streamlit config
    └── config.toml
```

## Code Standards
- Follows PEP8 standards
- Uses type hints for all functions
- Google-style docstrings for documentation
- Formatted with Black
- Uses Pydantic for data validation

## Future Improvements
- Frame-level video segmentation for more granular analysis
- Multi-language support
- Whisper model upgrades (small/medium)
- Facial expression analysis integration
