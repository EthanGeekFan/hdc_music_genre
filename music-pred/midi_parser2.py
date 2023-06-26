import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pprint
import os
import operator 
import math
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

from music21 import converter, corpus, instrument, midi, note, chord, pitch, roman, stream
import mido
from mido import Message, MidiFile, MidiTrack
from sklearn.metrics import jaccard_score

from miditoolkit import MidiFile

def extract_notes(midi_part):
    parent_element = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, note.Note):
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                parent_element.append(nt)
    
    return parent_element

class BasisHypervectorBuilder:

    def __init__(self,N,relative_octaves=False):
        self.n_octaves = 7
        self.notes = ["A","B","B-","C","D","E","E-","F","F#","G","G#"]
        self.note_hvs = {}
        self.octave_hvs = {}
        self.N = N 

        self.build_bases(relative_octaves=relative_octaves)

    def level_based(self,l,h,count):
        hv = h[0:count] + l[count:]
        assert(len(hv) == len(h))
        return hv

    def random_hypervector(self):
        return list(np.random.binomial(1,0.5,self.N))

    def is_binary(self,hv):
        assert(all(map(lambda v: v == 0 or v == 1, hv)))

    def bundle(self,els):
        if len(els) == 1:
            return els[0]

        sums = list(map(lambda t: sum(t), zip(*els)))
        if len(els) % 2 == 1:
            k = len(els) / 2.0
            hv = list(map(lambda s: int(math.floor(s/(k+0.1))) , sums))
            self.is_binary(hv)
            return hv
        else:
            k = len(els) / 2.0
            sums = list(map(lambda t: sum(t), zip(*els)))
            hv = list(map(lambda s: np.random.binomial(1,0.5) if s == k else int(math.floor(s/(k+0.1))), sums))
            self.is_binary(hv)
            return hv

    def bind(self,els):
        assert(len(els) >= 2)
        hv = list(map(lambda t: reduce(operator.xor, t), zip(*els)))
        return hv

    def dist(self,a,b):
        return sum(self.bind([a,b]))/self.N




    def permute(self,hv,k):
        if k == 1:
            return hv[k:] + [hv[0]]
        else:
            return hv[k:] + hv[:k]

    def sequence(self,lst):
        n = len(lst)
        els = []
        for i in range(0,n):
            j = n - i - 1
            els.append(self.permute(lst[j],j))
        return self.bundle(els)


    def build_bases(self,relative_octaves=True):
        if relative_octaves:

            for octv in range(1,14):
                self.octave_hvs[octv-6] = self.random_hypervector()

            '''
            self.octave_hvs[0] = self.random_hypervector()
            self.octave_hvs[6] = self.random_hypervector()
            self.octave_hvs[13] = self.random_hypervector()
            level_size = int(self.N/7)
            for octv in range(1,6):
                self.octave_hvs[-octv] = self.level_based(self.octave_hvs[0],self.octave_hvs[6],level_size*octv)
            for octv in range(1,13):
                self.octave_hvs[octv] = self.level_based(self.octave_hvs[6],self.octave_hvs[13],level_size*octv)
            '''


        else:
            self.octave_hvs[0] = self.random_hypervector()
            self.octave_hvs[6] = self.random_hypervector()
            level_size = int(self.N/7) 
            for octv in range(1,6):
                self.octave_hvs[octv] = self.level_based(self.octave_hvs[0],self.octave_hvs[6],level_size*octv)

        for note in self.notes:
            self.note_hvs[note] = self.random_hypervector()


    def chord_to_hv(self,chord,relative_octave=None):
        hvs= []
        for note in chord:
            notehv = self.note_to_hv(note,relative_octave=relative_octave)
            hvs.append(notehv)

        return self.bundle(hvs)

    def note_to_hv(self,note,relative_octave=None):
        if relative_octave is None:
            octave_hv = self.octave_hvs[note.octave]
        else:
            octave_hv = self.octave_hvs[note.octave - relative_octave]

        note_hv = self.note_hvs[note.name]
        return self.bind([octave_hv,note_hv])

    def key_hypervectors(self):
        for octave,oct_hv in self.octave_hvs.items():
            for note, note_hv in self.note_hvs.items():
                yield octave,note,self.bind([oct_hv,note_hv])

def get_prediction(hvdb, memory, window):
    hv = hvdb.sequence(window)
    cmp = hvdb.bind([hv,memory])
    scores = []
    keys = []
    for octave,note,key_hv in hvdb.key_hypervectors():
        d = hvdb.dist(cmp,key_hv)
        keys.append((octave,note))
        scores.append(d)

    indices = np.argsort(scores)
    for i in indices[:5]:
        print(scores[i], keys[i])


def generate_hypervectors(midi):
    np.random.seed(seed=143431)
    hvdb = BasisHypervectorBuilder(10000,relative_octaves=True)

    print("transcribing notes")
    hvs = []
    history = None
    memory = None
    window = 4
    relative_octave = None
    for i in range(len(midi.parts)):
        top = midi.parts[i].flat.notes                  
        song = extract_notes(top)
        for idx,nt in enumerate(song):
            if isinstance(nt, note.Note):
                hv = hvdb.note_to_hv(nt,relative_octave=relative_octave)
                relative_octave = nt.octave 
            elif isinstance(nt, chord.Chord):
                hv = hvdb.chord_to_hv(list(nt.notes), relative_octave=relative_octave)
                ch_octave = min(map(lambda cn: cn.octave, nt.notes))
                relative_octave = ch_octave 

            if idx > window:
                next_note = hv
                mem_item = hvdb.bind([next_note, history])
                if memory is None:
                    memory = mem_item
                else:
                    memory = hvdb.bundle([memory, mem_item])

                history = hvdb.bundle([hvdb.permute(history,1),next_note])

            else:
                history = list(hv)


            if idx > window:
                pred = get_prediction(hvdb,memory,hvs[len(hvs)-window:])
                print("idx=%d] %s prediction=%s" % (idx,nt,pred))
                input()
            else:
                print("idx=%d] %s" % (idx,nt))
            
            hvs.append(hv)

            #hvs.append(hv)


    print("building note database")

# Focusing only on 6 first measures to make it easier to understand.

matplotlib.use("Cairo")
#midiPath = "ladispute.mid"
midiPath = "converted.mid"
# read in midi file
wholeMid = mido.MidiFile(midiPath, clip=True)

mf = midi.MidiFile()
mf.open(midiPath)
mf.read()
mf.close()
base_midi = midi.translate.midiFileToStream(mf)
generate_hypervectors(base_midi)