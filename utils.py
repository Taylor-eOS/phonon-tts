import os
import time

LOG_FILE = "speaker.log"
LOG_WORDS = 6

def write_log(sentences, mp3_file, lang, voice):
    preview = " ".join(" ".join(sentences).split()[:LOG_WORDS])
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{os.path.basename(mp3_file)}\t{lang}\t{voice}\t{preview}\n")
