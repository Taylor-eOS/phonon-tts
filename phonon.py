import os
import shutil
import subprocess
import time
import wave
import numpy as np
import ptts
from utils import write_log
import settings

MODEL_TEMPLATE = "hf://gradium/phonon-40m-{lang}"
LANGUAGES = ("en", "de", "es")
LANG = LANGUAGES[0]
VOICE = ""
QUANT = ""
SAMPLE_RATE = 24000
INPUT_FILE = "input.txt"
OUTPUT_FILE = ""
MP3_FILE = ""
MP3_BITRATE = "192k"
GAP_SECONDS = 0.4
tts = None

def choose_language():
    global LANG
    LANG = input(f"Language {'/'.join(LANGUAGES)} ({LANGUAGES[0]}): ").strip().lower() or LANGUAGES[0]
    while LANG not in LANGUAGES:
        LANG = input("Unsupported language, try again: ").strip().lower()

def choose_voice():
    global VOICE
    options = [(group, name) for group, names in settings.VOICES[LANG].items() for name in names]
    names = {name.lower(): name for _, name in options}
    for number, (group, name) in enumerate(options, 1):
        print(f"{number:2d}. {name} ({group})")
    while True:
        answer = input("Voice number/name (1): ").strip().lower() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            VOICE = options[int(answer) - 1][1]
            return
        if answer in names:
            VOICE = names[answer]
            return
        print(f"Unsupported voice: {answer}")

def load_sentences(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    segments = [[]]
    for line in lines:
        if line.startswith("#"):
            continue
        if line:
            segments[-1].append(line)
        elif segments[-1]:
            segments.append([])
    return [segment for segment in segments if segment]

def load_model():
    global tts
    kwargs = {"lang": LANG}
    if QUANT:
        kwargs["quant"] = QUANT
    tts = ptts.TTS(MODEL_TEMPLATE.format(lang=LANG), **kwargs)

def set_output_paths(index, total):
    global OUTPUT_FILE, MP3_FILE
    suffix = "" if total == 1 else f"_{index:0{max(2, len(str(total)))}d}"
    OUTPUT_FILE = f"output{suffix}.wav"
    MP3_FILE = f"output{suffix}.mp3"

def write_wav(path, samples):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())

def synthesize(text):
    start = time.perf_counter()
    samples = np.asarray(tts.synth(text, voice=VOICE), dtype=np.float32).reshape(-1)
    elapsed = time.perf_counter() - start
    duration = samples.size / SAMPLE_RATE
    print(f"{duration:.2f}s audio in {elapsed:.3f}s {duration / elapsed:.1f}x")
    return samples

def convert_to_mp3():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found, install it with: sudo apt install ffmpeg")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUTPUT_FILE, "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, MP3_FILE]
    subprocess.run(command, check=True)
    os.remove(OUTPUT_FILE)
    print(f"Converted to {MP3_FILE} ({os.path.getsize(MP3_FILE) / 1024:.0f} KiB)")

def process_segment(sentences):
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    all_samples = []
    for i, text in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] {text}")
        all_samples.extend([synthesize(text), gap])
    full = np.concatenate(all_samples)
    write_wav(OUTPUT_FILE, full)
    print(f"Synthesized {full.size / SAMPLE_RATE:.2f} s of audio")
    convert_to_mp3()
    write_log(sentences, MP3_FILE, LANG, VOICE)

def main():
    choose_language()
    choose_voice()
    segments = load_sentences(INPUT_FILE)
    load_model()
    for index, sentences in enumerate(segments, 1):
        print(f"Segment {index}/{len(segments)}")
        set_output_paths(index, len(segments))
        process_segment(sentences)

if __name__ == "__main__":
    print("This requires exporting the Hugging Face token!")
    main()
