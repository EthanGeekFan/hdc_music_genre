import hdc.encoding as hdcencode
import hdc.hdc as hdclib

stride = 5

def learn_song(size,song):
    enc = hdcencode.BasicEncoding(N=size,stride=stride)
    note_itemmem = hdclib.ItemMemory(enc)
    rhy_itemmem = hdclib.ItemMemory(enc)
    enc.create_basis_hypervectors()

    note_dict = {}
    rhythm_dict = {}
    for notes in song.window(stride+1):
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

def predict_song(mems,song):

    note_itemmem,rhy_itemmem = mems
    encoder = note_itemmem.encoder
    history = None

    correct = 0
    top3 = 0
    notes_top3 = 0
    rhythms_top3 = 0

    total =0 
    for notes in song.window(stride+1):
        note_hist = encoder.encode_note_history(notes[:stride])
        rhy_hist = encoder.encode_rhythm_history(notes[:stride])

        note_key = encoder.encode_note_key(notes[stride])
        rhy_key = encoder.encode_rhythm_key(notes[stride-1].offset, notes[stride])

        top_note = note_itemmem.lookup(note_hist, K=3)
        top_rhy = rhy_itemmem.lookup(rhy_hist, K=3)
        print("top-note=%s note=%s" % (top_note, note_key))
        print("top-rhy=%s rhy=%s" % (top_rhy, rhy_key))

        if note_key in top_note.keys() and rhy_key in top_rhy.keys():
            top3 += 1
        if note_key in top_note.keys():
            notes_top3 += 1

        if rhy_key in top_rhy.keys():
            rhythms_top3 += 1

        total += 1

    print("both top-3=%f" % (top3/total*100.0))
    print("  notes top-3=%f" % (notes_top3/total*100.0))
    print("  rhythms top-3=%f" % (rhythms_top3/total*100.0))

    pass