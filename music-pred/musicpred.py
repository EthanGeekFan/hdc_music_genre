import util.music_util as music_util
import hdc.hdclearn as hdclib
import sys
import argparse

midiPath = sys.argv[1]
#midiPath = "data/simple.mid"
print("-> loading %s" % midiPath)
basename = midiPath.split("/")[1].split(".mid")[0]

tracks = music_util.load_midi(midiPath)
N = 10000
n_gen_notes = 5
n_user=6
for idx,(name,song) in enumerate(tracks.items()):
    print("-----------------------")
    print("-> learning track %s" % name)
    itemmem = hdclib.learn_song(N,song)

   
    print("-> generating %s" % name)
    for k in range(1,4):
        for j in range(1,n_gen_notes+1):
            gen_notes = hdclib.generate_song(itemmem,song,top_k=k,n_user_notes=n_user, n_gen_notes=j)
            new_path = "output/%s-track%d-top%d-user%d-gen%d.mid" % (basename,idx,k,n_user,j)
            status = music_util.write_song(gen_notes, new_path, ghost_track=song)
    
        gen_notes = hdclib.generate_song(itemmem,song,top_k=k,n_user_notes=n_user, n_gen_notes=-1)
        new_path = "output/%s-track%d-top%d-gen-all.mid" % (basename,idx,k)
        status = music_util.write_song(gen_notes, new_path)
        

    print("-> predicting %s" % name)
    accs = hdclib.predict_song(itemmem,song,k=1)
    accs = hdclib.predict_song(itemmem,song,k=3)

    #print("-> wrote generated song <%s>" % status)
    print("")

