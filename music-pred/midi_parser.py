from mido import MidiFile
import py_midicsv as pm

fPath = "ladispute.txt"
rawFile = open(fPath, 'r')

uniqueTypes = [] 
controllerTypes = []

processedCSV = []

for line in rawFile:
    processedLine = line.split(" ")
    
    # processedLine[0] is time in ms
    
    # processedLine[1] is event type
    processedLine[1] = str(int(processedLine[1], 0))
    
    if processedLine[1] not in uniqueTypes:
        uniqueTypes.append(processedLine[1])
    
    #    MIDI EVENT TYPES
    #    0x80 (hex)      128 (decimal)    Note Off
    #    0x90 (hex)      144 (decimal)    Note On
    #    0xB0 (hex)      176 (decimal)    Continuous Controller
    if processedLine[1] == "128":
        processedLine[1] = "Note_off_c"
    if processedLine[1] == "144":
        processedLine[1] = "Note_on_c"
    if processedLine[1] == "176":
        processedLine[1] = "Control_c"
    
    # processedLine[2] is note
    processedLine[2] = str(int(processedLine[2], 0))
    
    # Controller event
    if processedLine[1] == "176":
        if processedLine[2] not in controllerTypes:
            controllerTypes.append(processedLine[2])
    
    # processedLine[3] is velocity
    processedLine[3] = str(int(processedLine[3],0))
    
    # Track value = 1 (MIDI Data)
    # Expects data in the following format:
    # Track, Time, Note_on_c, Channel, Note, Velocity
    # Track, Time, Note_off_c, Channel, Note, Velocity
    # Track, Time, Control_c, Channel, Control_num, Value
    processedLine.insert(0, "1")
    processedLine.insert(3, "1")
    processed = ", ".join(processedLine)
    print(processed)
    processedCSV.append(processed)

processedCSV.insert(0, "1, 0, Start_track")

fileType = 0
clockPulses = 2400
# 0, 0, Header, format, nTracks, division
#   - format: the MIDI file type (0, 1, or 2)
#   - nTracks: the number of tracks in the file
#   - division: the number of clock pulses per quarter note. The Track and Time fields are always zero.
header = "0, 0, Header, " + str(fileType) + ", 1, " + str(clockPulses)
processedCSV.insert(0, header)


timeStamp = processedCSV[-1].split(", ")[1]
endTrack = "1, " + timeStamp + ", End_track"

processedCSV.append(endTrack)
processedCSV.append("0, 0, End_of_file")

with open("processed.csv", "w") as out: 
    for line in processedCSV:
        out.write(line + "\n")
    out.close()