from openai import OpenAI
from gtts import gTTS
from io import BytesIO
import pygame
import sounddevice as sd
from time import sleep
import numpy as np
import speech_recognition as sr

SAMPLE_RATE = 44100
SILENCE_THRESHOLD = 1500
WAIT_FOR_SOUND_SEC = 5
LLM_CLIENT_URL = "http://localhost:1234/v1"
LLM_CLIENT_API_KEY = "lm-studio"
LLM_CLIENT_MODEL = "LM Studio Community/Phi-3-mini-4k-instruct-GGUF"
LLM_CLIENT_CONTEXT = "You are a drive-through attendant at Starbucks. You do not speak for user."

class AudioRecorder:
    @staticmethod
    def record():
        initial_audio = AudioRecorder._record_with_defaults(WAIT_FOR_SOUND_SEC)

        while True:
            audio_batch = AudioRecorder._record_with_defaults(1)
            combined_audio = np.concatenate((initial_audio, audio_batch))
            
            if np.max(audio_batch) < SILENCE_THRESHOLD:
                break

        return sr.AudioData(combined_audio.flatten().tobytes(), SAMPLE_RATE, 2)

    @staticmethod
    def _record_with_defaults(wait_time_seconds):
        audio = sd.rec(
            int(wait_time_seconds * SAMPLE_RATE),
            samplerate = SAMPLE_RATE,
            channels = 1,
            dtype = "int16")
        
        sd.wait()

        return audio

class AudioPlayer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self._music_player = pygame.mixer.music

    def queue_sound(self, audio_file):
        if self._music_player.get_busy():
            self._music_player.queue(audio_file)
        else:
            self._music_player.load(audio_file)
            self._music_player.play()
    
    def wait_til_audio_ends(self):
        while self._music_player.get_busy():
            continue

class TTSProvider:
    def text_to_speech(self, text):
        tts = gTTS(text = text, lang="en")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes

class STTProvider:
    def __init__(self):
        self._google_audio_recognizer = sr.Recognizer()

    def speech_to_text(self, audio_bytes):
        # TODO: Google may revoke this functionality at any time.
        # Move to an implementation that doesn't rely on a Web API
        text = ""
        try:
            text += self._google_audio_recognizer.recognize_google(audio_bytes)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        return text

class LLMClient:
    def __init__(self):
        self._client = OpenAI(base_url=LLM_CLIENT_URL, api_key=LLM_CLIENT_API_KEY)
        self._history = [
            {"role": "system", "content": LLM_CLIENT_CONTEXT},
        ]

    def stream_response(self, prompt, assistant_response = ""):
        if assistant_response != "":
            self._history.append({"role": "assistant", "content": assistant_response})
        
        self._history.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model = LLM_CLIENT_MODEL,
            messages = self._history,
            temperature = 0.0,
            stream = True,
        )

        return response

class Main:
    def __init__(self):
        self.llm_client = LLMClient()
        self.audio_player = AudioPlayer()
        self.tts_provider = TTSProvider()
        self.stt_provider = STTProvider()
    
    def run(self):
        user_prompt = "Hello."
        assistant_response = ""

        while True:
            response_stream = self.llm_client.stream_response(user_prompt, assistant_response)

            assistant_response = ""
            current_sentence = ""
            for bulk in response_stream:
                current_content = bulk.choices[0].delta.content
                if current_content:
                    found_mark = False
                    for mark in [". ", "? ", "! ", ".", "?", "!"]:
                        if mark in current_content:
                            the_list = current_content.split(mark)
                            full_words = mark.join(the_list[:-1]) + mark
                            remainder = the_list[-1]
                            current_sentence += full_words

                            audio_bytes = self.tts_provider.text_to_speech(current_sentence)
                            self.audio_player.queue_sound(audio_bytes)

                            current_sentence = remainder
                            found_mark = True
                            break
                    if not found_mark:
                        current_sentence += current_content
                
                    print(current_content, end="", flush=True)
                    assistant_response += current_content

            self.audio_player.wait_til_audio_ends()

            print()

            audio_bytes = AudioRecorder.record()
            user_prompt = self.stt_provider.speech_to_text(audio_bytes)

            print(user_prompt)

Main().run()
