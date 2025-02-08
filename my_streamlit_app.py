import streamlit as st
import whisper
import numpy as np
import pyaudio
import wave
import librosa
import librosa.display
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.chunk import ne_chunk
from nltk import pos_tag
import pyttsx3
import language_tool_python
from joblib import Memory
import tempfile
import os

# ------------------ Initialize Streamlit ------------------
st.title("🎙️ Autism Speech Therapy System")

# ------------------ Caching Setup ------------------
memory = Memory("cache_directory", verbose=0)

# ------------------ Load Whisper Model ------------------
@memory.cache
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# ------------------ Initialize NLP Tools ------------------
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)

sia = SentimentIntensityAnalyzer()
tool = language_tool_python.LanguageTool('en-US')

# ------------------ Constants ------------------
AUDIO_PARAMS = {
    'chunk': 4096,
    'sample_format': pyaudio.paInt16,
    'channels': 1,
    'rate': 16000,
    'record_seconds': 10,
}

# ------------------ Recording Audio ------------------
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_PARAMS['sample_format'], channels=AUDIO_PARAMS['channels'],
                    rate=AUDIO_PARAMS['rate'], input=True, frames_per_buffer=AUDIO_PARAMS['chunk'])

    frames = []
    for _ in range(0, int(AUDIO_PARAMS['rate'] / AUDIO_PARAMS['chunk'] * AUDIO_PARAMS['record_seconds'])):
        frames.append(stream.read(AUDIO_PARAMS['chunk']))

    stream.stop_stream()
    stream.close()
    p.terminate()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(AUDIO_PARAMS['channels'])
        wf.setsampwidth(p.get_sample_size(AUDIO_PARAMS['sample_format']))
        wf.setframerate(AUDIO_PARAMS['rate'])
        wf.writeframes(b"".join(frames))

    return temp_file.name

# ------------------ Transcription ------------------
def transcribe_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=AUDIO_PARAMS['rate'])
    audio = whisper.audio.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text

# ------------------ Grammar Check ------------------
def check_grammar(text):
    matches = tool.check(text)
    return [match.message for match in matches] or ["No grammatical errors found."]

# ------------------ Sentiment Analysis ------------------
def analyze_text(text):
    feedback = []
    tokens = word_tokenize(text)
    sentiment = sia.polarity_scores(text)

    if sentiment['compound'] < 0:
        feedback.append("⚠️ Consider improving the tone of your speech.")

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    if len(filtered_tokens) < 5:
        feedback.append("⚠️ Your sentence is quite short. Try to elaborate more.")

    tagged_tokens = pos_tag(filtered_tokens)
    named_entities = ne_chunk(tagged_tokens)
    named_entity_count = sum(1 for entity in named_entities if isinstance(entity, nltk.Tree))

    if named_entity_count == 0:
        feedback.append("⚠️ Add some named entities (people, places, organizations) to make your speech more engaging.")

    return feedback or ["✅ Speech is clear and accurate!"]

# ------------------ Word Cloud ------------------
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ------------------ Audio Visualization ------------------
def visualize_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=AUDIO_PARAMS['rate'])

    # Waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(audio, color='blue')
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
    ax.set_title("Audio Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Pitch Contour
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.max(pitches, axis=0), color='green')
    ax.set_title("Pitch Contour")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Energy Plot
    energy = np.array([sum(abs(audio[i:i+100])) for i in range(0, len(audio), 100)])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(energy, color='red')
    ax.set_title("Speech Energy Over Time")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Energy")
    st.pyplot(fig)

# ------------------ Text-to-Speech Feedback ------------------
def text_to_speech(feedback_text):
    engine = pyttsx3.init()
    engine.say(feedback_text)
    engine.runAndWait()
#----1️⃣ Text Complexity Analysis---------
from textstat import flesch_reading_ease, gunning_fog

def analyze_text_complexity(text):
    reading_ease = flesch_reading_ease(text)
    fog_index = gunning_fog(text)

    st.subheader("📖 Readability Analysis")
    st.write(f"**Flesch Reading Ease Score:** {reading_ease:.2f}")
    st.write(f"**Gunning Fog Index:** {fog_index:.2f}")

    if reading_ease < 50:
        st.warning("Your speech is quite complex. Consider simplifying your sentences.")
    elif reading_ease > 70:
        st.success("Your speech is easy to understand. Great job!")

    if fog_index > 12:
        st.warning("Your speech has a high fog index. Try using simpler words.")
#--------2️⃣ Text Coherence & Sentence Structure Analysis------
import spacy

nlp = spacy.load("en_core_web_sm")

def analyze_sentence_structure(text):
    doc = nlp(text)
    sentence_lengths = [len(sent.text.split()) for sent in doc.sents]

    st.subheader("📌 Sentence Structure Analysis")
    st.write(f"**Total Sentences:** {len(sentence_lengths)}")
    st.write(f"**Average Sentence Length:** {np.mean(sentence_lengths):.2f} words")

    if np.mean(sentence_lengths) > 25:
        st.warning("Your sentences are quite long. Consider breaking them into shorter ones.")

    pos_counts = Counter([token.pos_ for token in doc])
    st.write("**Part of Speech Distribution:**")
    st.write(pos_counts)

    if pos_counts.get("VERB", 0) < 3:
        st.warning("Consider using more verbs to make your speech dynamic.")
#3️⃣ Named Entity Recognition (NER---
def analyze_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    st.subheader("📌 Named Entity Recognition")
    if not entities:
        st.warning("No named entities detected. Try adding names, locations, or organizations.")
    else:
        for entity, label in entities:
            st.write(f"🔹 **{entity}** → *{label}*")
#4️⃣ Word Variation & Redundancy Detection
def analyze_word_variation(text):
    words = word_tokenize(text.lower())
    fdist = FreqDist(words)

    st.subheader("📝 Word Variation Analysis")
    st.write("**Most Common Words:**")
    common_words = fdist.most_common(10)
    for word, freq in common_words:
        st.write(f"- {word}: {freq} times")

    if common_words[0][1] > 5:
        st.warning(f"You use the word '{common_words[0][0]}' quite often. Consider using synonyms.")
#5️⃣ Passive Voice Detection
def detect_passive_voice(text):
    doc = nlp(text)
    passive_sentences = [sent.text for sent in doc.sents if "be" in sent.text and "by" in sent.text]

    st.subheader("⚠️ Passive Voice Detector")
    if passive_sentences:
        st.warning("Consider rewriting these passive sentences in active voice:")
        for sent in passive_sentences:
            st.write(f"- {sent}")
    else:
        st.success("No passive sentences detected. Well done!")
#6️⃣ Complex Word Usage & Synonym Suggestions
from nltk.corpus import wordnet

def suggest_simpler_words(text):
    words = word_tokenize(text)
    complex_words = [word for word in words if len(word) > 7]

    st.subheader("🧐 Complex Word Suggestions")
    if complex_words:
        for word in complex_words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                st.write(f"🔹 **{word}** → Try using '{synonyms[0].lemmas()[0].name()}' instead.")
    else:
        st.success("Your vocabulary is clear and concise!")
#7️⃣ Grammar Rule-Based Checks
def custom_grammar_check(text):
    rules = tool.check(text)
    errors = [rule.message for rule in rules if "Spelling" not in rule.ruleId]

    st.subheader("📏 Advanced Grammar Check")
    if errors:
        for error in errors:
            st.warning(f"- {error}")
    else:
        st.success("No major grammar issues detected!")
#8️⃣ POS Tagging Visualization
import seaborn as sns

def visualize_pos_tags(text):
    doc = nlp(text)
    pos_counts = Counter([token.pos_ for token in doc])

    st.subheader("📊 Part of Speech Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()), ax=ax)
    ax.set_xlabel("POS Tag")
    ax.set_ylabel("Count")
    st.pyplot(fig)
#9️⃣ Speech Speed & Pauses Analysis
def analyze_speech_speed(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    st.subheader("🎵 Speech Speed Analysis")
    st.write(f"**Estimated Speech Rate:** {float(tempo):.2f} words per minute")

    if tempo < 100:
        st.warning("Your speech is quite slow. Try to speak more naturally.")
    elif tempo > 160:
        st.warning("You're speaking too fast. Slow down a bit.")
    else:
        st.success("Your speech speed is well-paced!")



if st.button("🎤 Start Recording"):
    st.write("Recording... Speak now!")
    audio_file = record_audio()
    st.success("Recording Complete!")

    st.write("Transcribing...")
    transcribed_text = transcribe_audio(audio_file)
    st.subheader("📝 Transcribed Text")
    st.write(transcribed_text)

    # Grammar & NLP Enhancements
    check_grammar(transcribed_text)
    analyze_text_complexity(transcribed_text)
    analyze_sentence_structure(transcribed_text)
    analyze_named_entities(transcribed_text)
    analyze_word_variation(transcribed_text)
    detect_passive_voice(transcribed_text)
    suggest_simpler_words(transcribed_text)
    custom_grammar_check(transcribed_text)

    # Audio Features
    visualize_pos_tags(transcribed_text)
    #analyze_speech_speed(audio_file)

    # Visualization
    visualize_audio(audio_file)
    generate_wordcloud(transcribed_text)
    
    os.remove(audio_file)
