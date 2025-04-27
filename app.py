import streamlit as st
import cv2
from moviepy import VideoFileClip
import whisper
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from deepface import DeepFace
from typing import Dict, List

@dataclass
class AnalysisResult:
    timestamps: list[float]
    sentiments: list[float]
    facial_expressions: List[Dict[str, str]]  # {'time': float, 'emotion': str}
    transcript: str

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video file using MoviePy"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio to text using Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"[Could not transcribe audio; {e}]"

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text using German BERT model"""
    sentiment_pipeline = pipeline(
        "text-classification",
        model="oliverguhr/german-sentiment-bert"
    )
    result = sentiment_pipeline(text)[0]
    # Map labels to scores: positive=1, neutral=0, negative=-1
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    return label_map[result['label']] * result['score']

def process_video(video_path: str) -> AnalysisResult:
    """Main processing pipeline for video analysis"""
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)
    
    # Split transcript into sentences for time-series analysis
    sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    
    # Calculate approximate timestamps based on sentence count
    video = VideoFileClip(video_path)
    duration = video.duration
    segment_duration = duration / max(1, len(sentences))
    
    # Analyze sentiment for each sentence
    timestamps = []
    sentiments = []
    for i, sentence in enumerate(sentences):
        sentiment = analyze_sentiment(sentence)
        timestamps.append(i * segment_duration)
        sentiments.append(sentiment)
    
    # Analyze facial expressions frame by frame and aggregate by second
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_emotions = {}
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze every 10th frame for performance
        if i % 10 == 0:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(results, list):
                    results = results[0]
                
                # Get dominant emotion (highest score)
                emotions = results['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                second = int(i/fps)
                
                # Track emotion scores per second
                if second not in frame_emotions:
                    frame_emotions[second] = {'count': 0, 'scores': {e: [] for e in emotions}}
                frame_emotions[second]['count'] += 1
                for e, score in emotions.items():
                    frame_emotions[second]['scores'][e].append(score)
            except Exception as e:
                st.warning(f"Could not analyze frame {i}: {e}")
    
    cap.release()
    
    # Aggregate to emotion with highest average probability per second
    facial_expressions = []
    for second, data in frame_emotions.items():
        if data['count'] > 0:
            # Calculate average probability for each emotion
            avg_scores = {
                e: sum(scores)/len(scores) 
                for e, scores in data['scores'].items() 
                if scores
            }
            # Get emotion with highest average probability
            dominant_emotion = max(avg_scores.items(), key=lambda x: x[1])[0]
            facial_expressions.append({
                'time': second,
                'emotion': dominant_emotion
            })
    
    return AnalysisResult(
        timestamps=timestamps,
        sentiments=sentiments,
        facial_expressions=facial_expressions,
        transcript=transcript
    )

def main():
    st.title("Video Sentiment Analysis")
    st.write("Upload a video to analyze the speaker's sentiment over time")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])
    
    if uploaded_file is not None:
        with st.spinner("Analyzing video..."):
            try:
                # Save uploaded file temporarily
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process video
                result = process_video("temp_video.mp4")
                
                # Display results
                st.subheader("Analysis Results")
                st.write(f"Transcript: {result.transcript}")
                
                # Plot sentiment and facial expressions over time
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Plot sentiment
                ax1.plot(result.timestamps, result.sentiments)
                ax1.set(xlabel='Time (s)', ylabel='Sentiment', 
                       title='Sentiment Analysis Over Time')
                
                # Plot facial expressions if available
                if result.facial_expressions:
                    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                    time_points = [fe['time'] for fe in result.facial_expressions]
                    dominant_emotions = [fe['emotion'] for fe in result.facial_expressions]
                    
                    # Create numeric values for each emotion for plotting
                    emotion_map = {e: i for i, e in enumerate(emotions)}
                    numeric_emotions = [emotion_map[e] for e in dominant_emotions]
                    
                    ax2.scatter(time_points, numeric_emotions, c=numeric_emotions, cmap='viridis')
                    ax2.set_yticks(range(len(emotions)))
                    ax2.set_yticklabels(emotions)
                    ax2.set(xlabel='Time (s)', ylabel='Dominant Emotion',
                           title='Facial Expression Analysis (Highest Average Probability per Second)')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Create and display text sentiment analysis table
                st.subheader("Text Sentiment Analysis")
                sentences = [s.strip() for s in result.transcript.split('.') if s.strip()]
                table_data = []
                for i, (time, sentiment) in enumerate(zip(result.timestamps, result.sentiments)):
                    if i < len(sentences):
                        table_data.append({
                            "Timestamp (s)": f"{time:.1f}",
                            "Text": sentences[i],
                            "Sentiment Score": f"{sentiment:.2f}",
                            "Sentiment Label": "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                        })
                
                # Display table sorted by timestamp
                table_data.sort(key=lambda x: float(x["Timestamp (s)"]))
                st.table(table_data)
            finally:
                # Clean up temporary files
                import os
                for temp_file in ["temp_video.mp4", "temp_audio.wav"]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {temp_file}: {e}")

if __name__ == "__main__":
    main()
