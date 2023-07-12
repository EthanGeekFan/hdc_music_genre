def encode_song(song):
    ntarr = []
    n_notes = 0
    for note in song.notes:
        if note.isChord:
            ntarr.append(len(note.notes))
            for n in note.notes:
                ntarr.append(n.pitch.midi)
        else:
            ntarr.append(1)
            ntarr.append(note.pitch.midi)
            # print()
            # print(note.pitch.name)
            # print(note.pitch.midi)
            # print((note.pitch.midi + 3) % 12)
        n_notes += 1

    # encode to c array
    code = "int song[] = {"
    for i in range(len(ntarr)):
        code += str(ntarr[i])
        if i < len(ntarr)-1:
            code += ","
    code += "};"
    code += "\n"
    code += "unsigned int song_len = " + str(len(ntarr)) + ";"
    code += "\n"
    code += "unsigned int n_notes = " + str(n_notes) + ";"
    return code
