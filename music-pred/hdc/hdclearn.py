import hdc.encoding as hdcencode
import hdc.hdc as hdclib
import music21.note as notelib
#from music21 import converter, corpus, instrument, midi, note, chord, pitch, roman, stream
import numpy as np
from tqdm import tqdm

stride = 5

def learn_song(size,song):
    enc = hdcencode.BasicEncoding(N=size,stride=stride)
    note_itemmem = hdclib.ItemMemory(enc)
    rhy_itemmem = hdclib.ItemMemory(enc)
    enc.create_basis_hypervectors()

    note_dict = {}
    rhythm_dict = {}
    n_windows = len(list(song.window(stride+1)))

    for notes in tqdm(song.window(stride+1), total=n_windows):
        note_hist = enc.encode_note_history(notes[:stride])
        note_key = enc.encode_note_key(notes[stride])

        rhythm_hist = enc.encode_rhythm_history(notes[:stride])
        rhythm_key = enc.encode_rhythm_key(notes[stride-1].offset, notes[stride])

        if not note_key in note_dict:
            note_dict[note_key] = []
        note_dict[note_key].append(note_hist)

        if not rhythm_key in rhythm_dict:
            rhythm_dict[rhythm_key] = []

        rhythm_dict[rhythm_key].append(rhythm_hist)

    for note_key,items in note_dict.items():
        note_itemmem.store(note_key,enc.hdc.bundle(items))

    for rhy_key,items in rhythm_dict.items():
        rhy_itemmem.store(rhy_key,enc.hdc.bundle(items))


    return note_itemmem,rhy_itemmem 


def generate_song(mems,song,top_k=1,n_user_notes=3,n_gen_notes=1):
    def random_sample(top):
        keys = list(top.keys())
        indices = list(range(len(keys)))
        weights = list(map(lambda k: 1/(top[k]+0.01), keys))
        total = sum(weights)
        probs = list(map(lambda w: w/total, weights))
        idx= np.random.choice(indices,size=1,replace=False, p=probs)[0]
        return keys[idx]

    def note_match(nt1,nt2):
        if nt2.isChord:
            nt2 = nt2.notes[0]
        if nt1.isChord:
            nt1 = nt1.notes[0]
        return nt.pitch.midi == nt2.pitch.midi

    def rhythm_match(prevnt,nt1,nt2):
        rest1 = nt1.offset - prevnt.offset
        rest2 = nt2.offset - prevnt.offset
        restMatch = (rest1 == rest2)
        return (nt1.duration.quarterLength == nt2.duration.quarterLength) and restMatch

    note_itemmem,rhy_itemmem = mems
    encoder = note_itemmem.encoder

    total =0
    gen = [] 
    k = top_k
    notebuf = list(song.notes)
    num_els = len(notebuf)-2*stride
    # generate entire song
    if n_gen_notes < 0:
        n_gen_notes = len(notebuf) - stride

    idx = stride
    pbar = tqdm(total=len(notebuf))
    correct_note_pred = 0
    correct_rhythm_pred = 0
    correct_both_pred = 0
    total_pred = 0
    while idx <= len(notebuf):
        for j in range(0,n_user_notes):
            idx += 1
            pbar.update(1)

        for j in range(0,n_gen_notes):
            if stride + idx + 1 >= len(notebuf):
                continue
            notes = song.notes[idx-stride:idx]
            note_hist = encoder.encode_note_history(notes)
            rhy_hist = encoder.encode_rhythm_history(notes)

            top_notes = note_itemmem.lookup(note_hist, K=k)
            top_rhys = rhy_itemmem.lookup(rhy_hist, K=k)
            top_note = random_sample(top_notes)
            offset, dur = random_sample(top_rhys)
            nt = notelib.Note(top_note)
            prevnt = notebuf[idx-1]
            orig_note = notebuf[idx]

            nt.offset = prevnt.offset + offset
            nt.duration = notelib.Duration(quarterLength=dur)
            notebuf[idx] = nt

            # update predictions 
            total_pred += 1
            correct_note_pred += 1 if note_match(orig_note, nt) else 0
            correct_rhythm_pred += 1 if rhythm_match(prevnt,orig_note, nt) else 0
            correct_both_pred += 1 if note_match(orig_note,nt) and rhythm_match(prevnt,orig_note,nt) else 0
            idx += 1
            pbar.update(1)
            pbar.set_description("[top=%d] total=%d both=%f notes=%f (dist=%f) rhythms=%f (dist=%f)" % \
                (k, total_pred, correct_both_pred/total_pred*100.0, \
                correct_note_pred/total_pred*100.0, \
                top_notes[top_note],
                correct_rhythm_pred/total_pred*100.0, \
                top_rhys[(offset,dur)]))
        



        

    return notebuf


def predict_song(mems,song,k=3):

    print("[initializing]")
    note_itemmem,rhy_itemmem = mems
    encoder = note_itemmem.encoder
    history = None

    correct = 0
    topk = 0
    notes_topk = 0
    rhythms_topk = 0

    total =0 
    n_windows = len(list(song.window(stride+1)))
    for notes in (pbar := tqdm(song.window(stride+1), total=n_windows)):
        note_hist = encoder.encode_note_history(notes[:stride])
        rhy_hist = encoder.encode_rhythm_history(notes[:stride])
        # print("note_hist=")
        # print(note_hist)
        # print("rhy_hist=")
        # print(rhy_hist)
        # print("notes=")
        # print(notes)

        note_key = encoder.encode_note_key(notes[stride])
        rhy_key = encoder.encode_rhythm_key(notes[stride-1].offset, notes[stride])
        # print("note_key=")
        # print(note_key)
        # print("rhy_key=")
        # print(rhy_key)
        # exit()

        top_note = note_itemmem.lookup(note_hist, K=k)
        top_rhy = rhy_itemmem.lookup(rhy_hist, K=k)
        #print("top-note=%s note=%s (%s)" % (top_note, note_key, notes[stride]))
        #print("top-rhy=%s rhy=%s" % (top_rhy, rhy_key))
        #print("")
        if note_key in top_note.keys() and rhy_key in top_rhy.keys():
            topk += 1
        if note_key in top_note.keys():
            notes_topk += 1

        if rhy_key in top_rhy.keys():
            rhythms_topk += 1

        total += 1
        if total % 5 == 0: 
            pbar.set_description("[top=%d] total=%d both=%f notes=%f rhythms=%f" % \
            (k, total, topk/total*100.0, notes_topk/total*100.0, rhythms_topk/total*100.0))
        


    print("=== correct note in top-%d ===" % (k))
    print("both %f" % (topk/total*100.0))
    print("  notes %f" % (notes_topk/total*100.0))
    print("  rhythms %f" % (rhythms_topk/total*100.0))

    pass