
from music21 import converter, corpus, instrument, midi, note, chord, pitch, roman, stream
import mido
from mido import Message, MidiFile, MidiTrack
from sklearn.metrics import jaccard_score

from miditoolkit import MidiFile



class Song:

    def __init__(self):
        self.notes = []

    def add_note(self,nt):
        self.notes.append(nt)

    @classmethod
    def summarize_note(cls,nt):
        if isinstance(nt,note.Note):
            print("note=%s offset=%d octave=%d duration=%f" % (nt.name,nt.offset,nt.octave,nt.duration.quarterLength))
        elif isinstance(nt,chord.Chord):
            print("chord=%s class=%s offset=%d duration=%f" % (str(nt), nt.commonName,nt.offset,nt.duration.quarterLength))


    def window(self,stride):
        for i in range(stride, len(self.notes)-stride):
            yield self.notes[i:i+stride]


def load_midi(midiPath):
    print("loading")
    mf = midi.MidiFile()
    mf.open(midiPath)
    mf.read()
    mf.close()
    base_midi = midi.translate.midiFileToStream(mf)

    for midi_part in base_midi.parts:
        song = Song()
        for nt in midi_part.flat.notes:        
            Song.summarize_note(nt)
            if isinstance(nt, note.Note):
                song.add_note(nt)
            elif isinstance(nt, chord.Chord):
                song.add_note(nt)
            else:
                raise Exception("unsupported element: %s" % nt)
    yield song


