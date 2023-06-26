import util.music_util as music_util
import hdc.hdclearn as hdclib

midiPath = "data/simple.mid"
song = list(music_util.load_midi(midiPath))[0]

N = 10000
itemmem = hdclib.learn_song(N,song)
accs = hdclib.predict_song(itemmem,song)

