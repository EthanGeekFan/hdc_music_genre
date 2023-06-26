from hdc.hdc import HDC
from enum import Enum
import math 


class BasisHVs:

    def __init__(self,N):
        self.N = N
        self.hvs = {}
        self.hdc = HDC()

    def from_enum(self,e):
        self.enum = e
        for val in e:
            self.add(val)

    def has(self,key):
        return key in self.hvs 

    def init(self,key,hv):
        self.hvs[key] = hv

    def add(self,key):
        self.hvs[key] = self.hdc.random_hypervector(self.N)

    def lookup(self,hvkey,K):
        dists = []
        keys = []
        for key,hv in self.hvs.items():
            dists.append(self.hdc.dist(hv,hvkey))
            keys.append(key)

        indices = np.argsort(dists)
        return list(map(lambda ind: keys[ind], indices))

    def get(self,key):
        return self.hvs[key]

     
class BasisHVsValues(BasisHVs):

    def __init__(self,N,R,minv,maxv):
        BasisHVs.__init__(self,N)
        self.maxv = maxv 
        self.minv = minv
        self.count = maxv-minv
        self.num_hvs = math.ceil(self.count*R/N)

        stride = math.ceil((maxv-minv)/self.num_hvs)
        for i in range(self.num_hvs):
            self.add(minv + i*stride)

        currv = minv
        for i in range(self.num_hvs):
            nbits = math.floor(N/stride)
            low = self.get(currv)
            high = self.get(currv + stride)
            for j in range(1,stride):
                hv = self.level_based(low,high,j)
                self.init(minv+i*stride+j, hv)

        for i in range(minv,maxv+1):
            if not (self.has(i)):
                raise Exception("value <%d> is not in basis vector set" % i)


    def level_based(self,l,h,count):
        hv = h[0:count] + l[count:]
        assert(len(hv) == len(h))
        return hv


class BasisHVsNotes(BasisHVs):
    class Pitches(Enum):
        A = "A"
        B = "B"
        Bf = "B-"
        C = "C"
        Cs = "C#"
        D = "D"
        E = "E"
        Ef = "E-"
        F = "F"
        Fs = "F#"
        G = "G"
        Gs = "G#"

        @classmethod
        def from_pitch(cls,pitch):
            vals = [e.value for e in cls]
            if pitch in vals:
                return cls(pitch) 
            else:
                raise Exception("unknown pitch <%s>" % pitch)


    def __init__(self,N):
        BasisHVs.__init__(self,N)
        self.from_enum(BasisHVsNotes.Pitches)



class BasisHVsDuration(BasisHVs):
    class Duration(Enum):
        Quarter = "1/4"
        Half = "1/2"
        Whole = "1"
        Eighth = "1/8"
        Complex = "cplx"

        def from_duration(dur):
            if dur.type == "quarter":
                return BasisHVsDuration.Duration.Quarter
            elif dur.type == "half":
                return BasisHVsDuration.Duration.Half
            elif dur.type == "whole":
                return BasisHVsDuration.Duration.Whole
            elif dur.type == "eighth":
                return BasisHVsDuration.Duration.Eighth
            elif dur.type == "complex":
                return BasisHVsDuration.Duration.Complex
            else:
                print(dur.type)
                raise NotImplementedError

    def __init__(self,N):
        BasisHVs.__init__(self,N)
        self.from_enum(BasisHVsDuration.Duration)

def BasisHVAbsOctaves(N,R):
    return BasisHVsValues(N,R,0,7) 

def BasisHVRelOctaves(N,R):
    return BasisHVsValues(N,R,-7,7) 

def BasisHVsRests(N,R):
    return BasisHVsValues(N,R,0,88)

def BasisHVsNumerNotes(N,R):
    return BasisHVsValues(N,R,0,88)

def BasisHVsMelody(N,R):
    melody_range = 44
    return BasisHVsValues(N,R,-melody_range,melody_range)

