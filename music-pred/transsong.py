def encode_song(song):
    ntarr = []
    for note in song.notes:
        if note.isChord:
            ntarr.append(len(note.notes))
            for n in note.notes:
                ntarr.append(n.pitch.midi)
        else:
            ntarr.append(1)
            ntarr.append(note.pitch.midi)

    # encode to c array
    code = "unsigned int song[] = {"
    for i in range(len(ntarr)):
        code += str(ntarr[i])
        if i < len(ntarr)-1:
            code += ","
    code += "};"
    code += "\n"
    code += "unsigned int song_len = " + str(len(ntarr)) + ";"
    return code
