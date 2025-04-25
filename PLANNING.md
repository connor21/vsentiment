# vsentiment - Project Planning

## Project Overview
 This project is an app that anlyzes a video and shows a sentiment anlysis of a person in the video taling into the camera. The analysis checks the changing mood of the person during the video and creates a graph to show the changes over time.

## Architecture
- **Framework**: Streamlit
- **Video Processing**: opencv-python
- **Audio Extraction**: MoviePy
- **Audio to Text**:openai-whisper
- **Text Sentiment analysis**: transformer
- **facial expression analysis**: deepface

## Components
1. **APP**
   - Offers a UI to upload a video
   - Anlyzes the snetiment of the person in the video
   - Shows a report of the analysis

## Environment Configuration

## File Structure
```
vsentiment/
├── app.py              # Main MCP server implementation
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── PLANNING.md            # Project planning (this file)
└── TASK.md                # Task tracking
```

## Style Guidelines
- Follow PEP8 standards
- Use type hints for all functions
- Document functions with Google-style docstrings
- Format code with Black
- Use Pydantic for data validation

## Dependencies
- streamlit
- Opencv
- whisper
- moviepie
- transformers