import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import os
import tempfile
import ffmpeg


# Fonction : Transcrire l'audio d'une vidéo avec Whisper small
def transcribe_audio(video_path):
    """
    Transcrit l'audio d'une vidéo MP4 en utilisant le modèle Whisper 'small'.
    :param video_path: Chemin du fichier vidéo.
    :return: Texte transcrit et segments temporels.
    """
    # Charger le modèle Whisper le plus sophistiqué
    model = whisper.load_model("small")

    # Extraire l'audio de la vidéo
    clip = VideoFileClip(video_path)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    clip.audio.write_audiofile(temp_audio.name)

    # Transcrire l'audio avec Whisper
    result = model.transcribe(temp_audio.name)

    # Supprimer le fichier audio temporaire
    os.unlink(temp_audio.name)

    # Retourner le texte transcrit et les segments
    return result["text"], result["segments"]


# Fonction : Diviser le texte en segments plus courts 
def split_text_into_chunks(text, max_length=500):
    """Divise le texte en chunks ne dépassant pas max_length tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # Ajouter 1 pour l'espace
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = len(word) + 1
        current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Fonction : Traduire un texte
def translate_text(text, src_lang="en", tgt_lang="fr", max_length=500):
    """Traduit le texte en morceaux pour éviter les erreurs de longueur."""
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    chunks = split_text_into_chunks(text, max_length)
    translated_chunks = []

    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)

    return " ".join(translated_chunks)


# Fonction : Générer un audio avec synthèse vocale
def generate_audio(text, language, output_path):
    model_name = f"tts_models/{language}/css10/vits"  # Ajustez selon la langue
    tts = TTS(model_name)
    tts.tts_to_file(text=text, file_path=output_path)

# Fonction : Remplacer l'audio dans une vidéo
def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Fonction principale : Traduire une vidéo
def process_video(video_path, src_lang, tgt_lang, output_video_path, output_audio_path=None):
    # Étape 1 : Transcription
    st.write("Transcription de la vidéo...")
    transcription, _ = transcribe_audio(video_path)

    # Étape 2 : Traduction
    st.write("Traduction en cours...")
    translated_text = translate_text(transcription, src_lang, tgt_lang)

    # Étape 3 : Génération audio
    st.write("Génération de l'audio traduit...")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    generate_audio(translated_text, tgt_lang, temp_audio.name)

    # Étape 4 : Remplacement audio
    st.write("Remplacement de l'audio dans la vidéo...")
    replace_audio_in_video(video_path, temp_audio.name, output_video_path)

    # Optionnel : Extraction de l'audio en MP3
    if output_audio_path:
        audio = AudioFileClip(temp_audio.name)
        audio.write_audiofile(output_audio_path)

    os.unlink(temp_audio.name)


# Interface utilisateur avec Streamlit
st.title("Traducteur de vidéos MP4")
st.write("Téléchargez une vidéo MP4, choisissez la langue d'origine et la langue cible, puis cliquez sur **Run** pour générer une version traduite.")

# Téléchargement de la vidéo par l'utilisateur
uploaded_video = st.file_uploader("Téléchargez une vidéo MP4", type=["mp4"])
src_lang = st.selectbox("Langue d'origine", ["en", "fr", "es", "de"])
tgt_lang = st.selectbox("Langue cible", ["en", "fr", "es", "de"])
output_option = st.radio("Format de sortie", ["Vidéo traduite (MP4)", "Audio seulement (MP3)", "Les deux (MP4 et MP3)"])

if st.button("Run"):
    if uploaded_video:
        # Sauvegarder temporairement la vidéo uploadée
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            temp_video_path = temp_video.name

        # Chemins de sortie
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        output_audio_path = None
        if output_option in ["Audio seulement (MP3)", "Les deux (MP4 et MP3)"]:
            output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

        # Traiter la vidéo
        try:
            process_video(temp_video_path, src_lang, tgt_lang, output_video_path, output_audio_path)

            # Télécharger les fichiers générés
            if output_option in ["Vidéo traduite (MP4)", "Les deux (MP4 et MP3)"]:
                st.success("Vidéo traduite générée avec succès !")
                with open(output_video_path, "rb") as f:
                    st.download_button("Télécharger la vidéo traduite (MP4)", f, file_name="translated_video.mp4")

            if output_option in ["Audio seulement (MP3)", "Les deux (MP4 et MP3)"]:
                st.success("Audio traduit généré avec succès !")
                with open(output_audio_path, "rb") as f:
                    st.download_button("Télécharger l'audio traduit (MP3)", f, file_name="translated_audio.mp3")

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

        # Nettoyer les fichiers temporaires
        os.unlink(temp_video_path)
        if output_video_path:
            os.unlink(output_video_path)
        if output_audio_path:
            os.unlink(output_audio_path)
    else:
        st.error("Veuillez uploader une vidéo avant de lancer le processus.")