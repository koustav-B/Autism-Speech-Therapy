#----------------for displaying output in the terminal only----------------------------------------------
import whisper
import numpy as np
import pyaudio
import wave
import librosa
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.chunk import ne_chunk
from nltk import pos_tag
from pydub import AudioSegment
import pyttsx3
import language_tool_python
from joblib import Memory

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


# --------------------- Constants ---------------------
AUDIO_PARAMS = {
    'chunk': 4096,
    'sample_format': pyaudio.paInt16,
    'channels': 1,
    'rate': 16000,
    'record_seconds': 5,
    'output_filename': "recorded_audio.wav"
}

# --------------------- Caching Setup ---------------------
memory = Memory("cache_directory", verbose=0)

# --------------------- Load Whisper Model ---------------------
@memory.cache
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# --------------------- Download NLTK Resources ---------------------
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)

# --------------------- Initialize TTS Engine ---------------------
engine = pyttsx3.init()

# --------------------- Initialize Grammar Tool ---------------------
tool = language_tool_python.LanguageTool('en-US')

# --------------------- Record and Transcribe Speech ---------------------

def record_and_transcribe():
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_PARAMS['sample_format'], channels=AUDIO_PARAMS['channels'],
                    rate=AUDIO_PARAMS['rate'], input=True, frames_per_buffer=AUDIO_PARAMS['chunk'])

    print("Recording... Please speak into the microphone.")
    engine.say("Recording... Please speak into the microphone.")
    engine.runAndWait()

    frames = [stream.read(AUDIO_PARAMS['chunk']) for _ in range(0, int(AUDIO_PARAMS['rate'] / AUDIO_PARAMS['chunk'] * AUDIO_PARAMS['record_seconds']))]

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(AUDIO_PARAMS['output_filename'], "wb") as wf:
        wf.setnchannels(AUDIO_PARAMS['channels'])
        wf.setsampwidth(p.get_sample_size(AUDIO_PARAMS['sample_format']))
        wf.setframerate(AUDIO_PARAMS['rate'])
        wf.writeframes(b"".join(frames))

    print("Recording complete. Transcribing...")
    engine.say("Recording complete. Transcribing.")
    engine.runAndWait()

    audio, _ = librosa.load(AUDIO_PARAMS['output_filename'], sr=AUDIO_PARAMS['rate'])
    audio = whisper.audio.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    print(f"Transcribed Text: {result.text}")
    engine.say(f"Transcribed Text: {result.text}")
    engine.runAndWait()
    return result.text

# --------------------- Grammar Checking ---------------------
def check_grammar(text):
    matches = tool.check(text)
    return [match.message for match in matches] or ["No grammatical errors found."]

# --------------------- Basic NLP Analysis ---------------------
def analyze_text(text):
    feedback = []
    tokens = word_tokenize(text)
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    if sentiment['compound'] < 0:
        feedback.append("Consider improving the tone of your speech.")

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    if len(filtered_tokens) < 5:
        feedback.append("Your sentence is quite short. Try to elaborate more.")

    tagged_tokens = pos_tag(filtered_tokens)
    named_entities = ne_chunk(tagged_tokens)
    named_entity_count = sum(1 for entity in named_entities if isinstance(entity, nltk.Tree))

    if named_entity_count == 0:
        feedback.append("Add some named entities (people, places, organizations) to make your speech more engaging.")

    n_grams = ngrams(filtered_tokens, 2)
    ngram_freq = FreqDist(n_grams)

    if len(ngram_freq) < 2:
        feedback.append("Consider using more varied sentence structures.")

    return feedback or ["Speech is clear and accurate!"]

# --------------------- Audio and Text Visualization ---------------------
def visualize_audio_features(audio):
    plt.figure(figsize=(10, 4))
    plt.title("Audio Waveform")
    plt.plot(audio)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=2, noverlap=512)
    plt.title("Audio Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=AUDIO_PARAMS['rate'], n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=AUDIO_PARAMS['rate'])
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# --------------------- Word Cloud Generation ---------------------
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# --------------------- Interactive Feedback and Suggestions ---------------------
def provide_feedback(predicted_text):
    feedback = analyze_text(predicted_text)
    print(f"\nFeedback: {feedback}")
    engine.say(f"Feedback: {feedback}")
    engine.runAndWait()
    print(f"Transcribed Text: {predicted_text}")
    engine.say(f"Transcribed Text: {predicted_text}")
    engine.runAndWait()
    return "Your speech seems clear and accurate!" if not feedback else "Try to improve the following aspects: " + ", ".join(feedback)

# --------------------- Main Function ---------------------
def main():
    while True:
        print("Welcome to the Advanced Speech Therapy System!")
        engine.say("Welcome to the Advanced Speech Therapy System!")
        engine.runAndWait()

        print("Please speak into the microphone. The system will transcribe your speech and provide feedback.\n")
        engine.say("Please speak into the microphone. The system will transcribe your speech and provide feedback.")
        engine.runAndWait()

        text = record_and_transcribe()
        grammar_errors = check_grammar(text)
        print(f"Grammar Check: {grammar_errors}")
        engine.say(f"Grammar Check: {grammar_errors}")
        engine.runAndWait()

        audio, _ = librosa.load(AUDIO_PARAMS['output_filename'], sr=AUDIO_PARAMS['rate'])
        visualize_audio_features(audio)

        feedback = provide_feedback(text)
        generate_wordcloud(text)

        print("\nDo you want to continue? Say 'yes' or 'no'.")
        engine.say("Do you want to continue? Say 'yes' or 'no'.")
        engine.runAndWait()

        response = input("Do you want to continue? (yes/no): ").strip().lower()
        if response != "yes":
            print("Exiting the system.")
            engine.say("Exiting the system.")
            engine.runAndWait()
            break

if __name__ == "__main__":
    main()



