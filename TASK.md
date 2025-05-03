# vsentiment - Task List

## ✅ Initial tasks
- [x] Create initial project structure
- [x] implement video upload
- [x] implement audio extraction
- [x] Implement audio to text transcription (using Whisper)
- [x] Implement text sentiment analysis
- [x] Implement sentiment report with time-series analysis (sentence-level)
- [x] Cleanup temporary files
- [x] Replaced SpeechRecognition with OpenAI Whisper (base model)
- [x] Use model 'oliverguhr/german-sentiment-bert' for sentiment analysis
- [x] Update README.md with comprehensive documentation
- [x] Implement frame level analysis of facial expression
- [x] Implement facial expression report
- [x] Create pod description for podman to deploy the app
- [x] Create table for text analysis with timestamps, detected emotions, and source text
- [x] Create a table showing example images of detected emotions with tags and timestamps

## ✅ Audio sentiment analysis
- [x] Implement an audio and voice sentiment analysis with `librosa`
- [x] Add a report about voice sentiment analysis
- [x] Use different diagrams for pitch and energy with proper scales
- [x] Aggregate pitch values per second to the average

## ✅ Optimizations
- [x] Split the text into complete sentences for the textual sentiment analysis (using nltk.sent_tokenize)
- [x] Visualize MFCC extracted from audio analysis with `librosa.display.specshow`
- [ ] Implement emotion detection based on audio data with `pyAudioAnalysis`
- [ ] Visualize results of emotion detection with `pyAudioAnalysis`in report
