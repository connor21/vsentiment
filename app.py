import os
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
import librosa
import librosa.display
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

@dataclass
class AnalysisResult:
    timestamps: list[float]
    sentiments: list[float]
    facial_expressions: List[Dict[str, str]]  # {'time': float, 'emotion': str}
    emotion_examples: List[Dict[str, any]]  # {'time': float, 'emotion': str, 'image': np.array}
    transcript: str
    audio_sentiment: Dict[str, any]  # {'timestamps': list[float], 'scores': list[float], 'features': dict}

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video file using MoviePy"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        video = VideoFileClip(video_path)
        if not video.audio:
            raise ValueError("Video file has no audio track")
            
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}") from e

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

def analyze_audio_sentiment(audio_path: str) -> Dict[str, List[float]]:
    """Analyze audio features to determine voice sentiment"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Extract features
        features = {
            'pitch': librosa.yin(y, fmin=50, fmax=500),  # Fundamental frequency
            'energy': librosa.feature.rms(y=y),  # Root mean square energy
            'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y),
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        }
        
        # Calculate sentiment indicators
        # Higher pitch and energy generally indicate more positive/excited sentiment
        avg_pitch = np.mean(features['pitch'][features['pitch'] > 0])  # Filter out invalid pitches
        avg_energy = np.mean(features['energy'])
        
        # Normalize and combine into sentiment score (-1 to 1)
        pitch_score = np.clip((avg_pitch - 150) / 100, -1, 1)  # Normalize around typical speech pitch
        energy_score = np.clip((avg_energy - 0.05) * 20, -1, 1)  # Normalize around typical speech energy
        
        # Combine scores with weights
        sentiment_score = (pitch_score * 0.6 + energy_score * 0.4)
        
        return {
            'features': features,
            'sentiment_score': sentiment_score,
            'timestamps': np.linspace(0, len(y)/sr, len(features['energy'][0]))
        }
    except Exception as e:
        st.warning(f"Audio analysis failed: {e}")
        return {
            'features': {},
            'sentiment_score': 0,
            'timestamps': []
        }

def process_video(video_path: str) -> AnalysisResult:
    """Main processing pipeline for video analysis"""
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)
    
    # Split transcript into proper sentences using NLTK
    sentences = nltk.sent_tokenize(transcript, language='german')
    
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
    emotion_examples = []
    
    # Re-open video to capture example frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
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
            
            # Capture example frame for this emotion
            frame_pos = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                emotion_examples.append({
                    'time': second,
                    'emotion': dominant_emotion,
                    'image': frame_rgb
                })
            
            facial_expressions.append({
                'time': second,
                'emotion': dominant_emotion
            })
    
    cap.release()
    
    # Analyze audio sentiment features
    audio_sentiment = analyze_audio_sentiment(audio_path)
    
    return AnalysisResult(
        timestamps=timestamps,
        sentiments=sentiments,
        facial_expressions=facial_expressions,
        emotion_examples=emotion_examples,
        transcript=transcript,
        audio_sentiment=audio_sentiment
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
                
                # Plot sentiment, facial expressions and audio features over time
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
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
                
                # Plot audio sentiment if available
                if hasattr(result, 'audio_sentiment') and result.audio_sentiment and \
                   'timestamps' in result.audio_sentiment and len(result.audio_sentiment['timestamps'])>0 and \
                   'features' in result.audio_sentiment and 'energy' in result.audio_sentiment['features']:
                    
                    # Aggregate pitch values per second
                    timestamps = result.audio_sentiment['timestamps']
                    pitch = result.audio_sentiment['features']['pitch']
                    energy = result.audio_sentiment['features']['energy'][0]
                    
                    # Create second-level bins
                    seconds = np.floor(timestamps).astype(int)
                    unique_seconds = np.unique(seconds)
                    avg_pitch = [np.mean(pitch[seconds == s]) for s in unique_seconds]
                    
                    # Create separate plots for pitch and energy
                    fig2, (ax_pitch, ax_energy) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Plot pitch with proper scale (Hz)
                    ax_pitch.plot(unique_seconds, avg_pitch, color='blue')
                    ax_pitch.set(xlabel='Time (s)', ylabel='Pitch (Hz)',
                               title='Average Pitch per Second')
                    ax_pitch.grid(True)
                    
                    # Plot energy with proper scale
                    ax_energy.plot(timestamps, energy, color='orange')
                    ax_energy.set(xlabel='Time (s)', ylabel='Energy (RMS)',
                                title='Energy Over Time')
                    ax_energy.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)

                    # Visualize MFCC features
                    if 'mfcc' in result.audio_sentiment['features']:
                        mfcc = result.audio_sentiment['features']['mfcc']
                        fig3, ax = plt.subplots(figsize=(10, 4))
                        img = librosa.display.specshow(
                            mfcc,
                            x_axis='time',
                            sr=result.audio_sentiment['features'].get('sr', 22050),
                            ax=ax
                        )
                        fig3.colorbar(img, ax=ax)
                        ax.set(title='MFCC Features')
                        st.pyplot(fig3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display audio sentiment statistics
                if result.audio_sentiment['features']:
                    st.subheader("Audio Sentiment Analysis")
                    st.write(f"Overall Audio Sentiment Score: {result.audio_sentiment['sentiment_score']:.2f}")
                    st.write("Feature Statistics:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Average Pitch: {np.mean(result.audio_sentiment['features']['pitch']):.1f} Hz")
                        st.write(f"Average Energy: {np.mean(result.audio_sentiment['features']['energy']):.4f}")
                    with col2:
                        st.write(f"Max Pitch: {np.max(result.audio_sentiment['features']['pitch']):.1f} Hz")
                        st.write(f"Max Energy: {np.max(result.audio_sentiment['features']['energy']):.4f}")
                
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
                
                # Display emotion examples table if available
                if result.emotion_examples:
                    st.subheader("Emotion Examples")
                    for example in sorted(result.emotion_examples, key=lambda x: x['time']):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(example['image'], 
                                    caption=f"Time: {example['time']:.1f}s",
                                    width=150)
                        with col2:
                            st.markdown(f"""
                            **Emotion**: {example['emotion'].capitalize()}
                            """)
                        st.markdown("---")
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
