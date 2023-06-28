import hdc.basis as basislib
import hdc.hdc as hdclib
import math

class BasicEncoding:

    def __init__(self,N,stride):
        self.stride = stride
        self.N = N
        self.hdc = hdclib.HDC()

    def create_basis_hypervectors(self):
        resolution = 0.05
        self.basis_numer_notes = basislib.BasisHVsNumerNotes(self.N, self.N*resolution)
        self.basis_melody = basislib.BasisHVsMelody(self.N, self.N*resolution)
        self.basis_durations = basislib.BasisHVsDuration(self.N)
        self.basis_numer_durations = basislib.BasisHVsNumerDuration(self.N,self.N*resolution)
        self.basis_rests = basislib.BasisHVsRests(self.N,self.N*resolution)
        self.basis_notes = basislib.BasisHVsNotes(self.N) 
        self.basis_octaves = basislib.BasisHVAbsOctaves(self.N, self.N*resolution) 

    def encode_rhythm_history(self,notes):
        win = []
        timestamp = notes[0].offset
        for note in notes:
            rest,hold = self.encode_rhythm(timestamp,note)
            win.append(rest)
            win.append(hold)
            timestamp = note.offset

        histhv = self.hdc.bind(self.hdc.sequence(win))
        return histhv


    def encode_note_history(self,notes):
        win = []
        for note in notes[1:]:
            notehv = self.encode_relative_note(notes[0],note)
            win.append(notehv)

        histhv = self.hdc.bind(self.hdc.sequence(win))
        notehv = self.encode_note(notes[0])
        return self.hdc.bind([histhv, notehv])


    def encode_note(self,note):
        #notehv = self.basis_numer_notes.get(note.pitch.midi)
        if note.isChord:
            hvs = []
            octv = self.basis_octaves.get(note.notes[0].octave)

            for n in note.notes:
                note_key = self.basis_notes.enum.from_pitch(n.pitch.name)
                notehv = self.basis_notes.get(note_key)
                hvs.append(notehv)

            #return self.hdc.bind([octv]+hvs)
            return self.hdc.bind(hvs)


        else:
            octv = self.basis_octaves.get(note.octave)
            note_key = self.basis_notes.enum.from_pitch(note.pitch.name)
            notehv = self.basis_notes.get(note_key)
            #return self.hdc.bind([octv,notehv])
            return notehv 

    def encode_rhythm(self,timestamp, note):
        #dur_key = self.basis_durations.enum.from_duration(note.duration)
        #durhv = self.basis_durations.get(dur_key)
        durhv = self.basis_numer_durations.get_value(note.duration)
        timehv = self.basis_rests.get_value(note.offset - timestamp)
        return timehv,durhv

    def encode_relative_note(self,base_note,note):
        if base_note.isChord:
            base_note_id = base_note.notes[0].pitch.midi 
        else:
            base_note_id = base_note.pitch.midi 


        if note.isChord:
            nhvs = []
            for n in note.notes:
                note_id = n.pitch.midi
                diff = note_id - base_note_id
                notehv = self.basis_melody.get(diff)
                nhvs.append(notehv)
            return self.hdc.bind(nhvs)
        else:
            note_id = note.pitch.midi
            diff = note_id - base_note_id
            notehv = self.basis_melody.get(diff)
            return notehv

    def encode_note_key(self,note):
        if note.isChord:
            return note.notes[0].pitch.midi
        else:
            return note.pitch.midi

    def encode_rhythm_key(self,timestamp,note):
        dur_key = note.duration.quarterLength 
        time_diff = (note.offset - timestamp)  
        return (time_diff,dur_key)