from hdc.hdc import HDC
from enum import Enum
import math
import numpy as np


def c_gen_hv_to_hex_arr(hv: list[int]) -> str:
    # for each 32 ints, convert to a uint literal 0x00000000
    # then join them all together
    nums = []
    words = len(hv) // 32
    tail = len(hv) % 32
    for i in range(words):
        num = 0
        for j in range(32):
            num = num << 1
            num = num | (hv[i*32+j] & 1)
        nums.append(num)
    if tail > 0:
        num = 0
        for j in range(tail):
            num = num << 1
            num = num | (hv[words*32+j] & 1)
        num = num << (32-tail)
        nums.append(num)
    return "{" + ", ".join(map(lambda n: "0x%08x" % n, nums)) + "}"


class BasisHVs:

    def __init__(self,name,N):
        self.name = name
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
    
    def __str__(self) -> str:
        s = "{\n"
        for key,hv in self.hvs.items():
            s += "\t%s: dim(%s)\n" % (key,str(len(hv)))
        s += "}"
        return s
    
    def c_code(self, indent: str = "") -> str:
        code = "/* %s */\n" % self.name
        code += indent + "std::map<std::string, unsigned int *> basis_%s = {\n" % self.name
        for key,hv in self.hvs.items():
            code += indent + "    { \"%s\", new unsigned int[%d]%s },\n" % (key, self.N // 32, c_gen_hv_to_hex_arr(hv))
        code += indent + "};\n"
        return code

     
class BasisHVsValues(BasisHVs):

    def __init__(self,N,R,name,minv,maxv):
        BasisHVs.__init__(self,name,N)
        self.maxv = maxv 
        self.minv = minv
        self.count = maxv-minv
        self.num_hvs = max(2,math.ceil(self.count*R/N)+1)

        self.add("underflow")
        self.add("overflow")
        stride = math.ceil(self.count/(self.num_hvs-1))
        for i in range(self.num_hvs):
            self.add(minv + i*stride)

        currv = minv
        nbits = math.floor(N/stride)
        for i in range(self.num_hvs-1):
            assert(nbits >= R)
            low = BasisHVs.get(self,currv)
            high = BasisHVs.get(self,currv + stride)
            for j in range(1,stride):
                assert(j > 0 and j < stride)
                hv = self.level_based(low,high,j*nbits)
                self.init(currv+j, hv)
            currv += stride

        for i in range(minv,maxv+1):
            if not (self.has(i)):
                raise Exception("value <%d> is not in basis vector set" % i)
        
    def get(self,i):
        if i < self.minv:
            print("[warn] underflow in basis vector %s <%d>" % (self.name,i))
            input("")
            return BasisHVs.get(self, "underflow")
        elif i > self.maxv:
            print("[warn] overflow in basis vector %s <%d> max=%d" % (self.name,i,self.maxv))
            input("")
            return BasisHVs.get(self,"overflow")

        return BasisHVs.get(self, i)

    def get_value(self,i_):
        i = self.xform(i_)
        return self.get(i)

    def xform(self,i):
        return i
    
    def level_based(self,l,h,count):
        hv = h[0:count] + l[count:]
        assert(len(hv) == len(h))
        return hv
    
    def c_code(self, indent: str = "") -> str:
        code = "/* %s */\n" % self.name
        code += indent + "std::map<int, unsigned int *> basis_%s = {\n" % self.name
        for key,hv in self.hvs.items():
            if key == "overflow" or key == "underflow":
                continue
            code += indent + "    { %d, new unsigned int[%d]%s },\n" % (key, self.N // 32, c_gen_hv_to_hex_arr(hv))
        code += indent + "};\n"
        return code


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
        BasisHVs.__init__(self,"pitch",N)
        self.from_enum(BasisHVsNotes.Pitches)

    
    def c_code(self, indent: str = "") -> str:
        # write as an array [A,Bf,B,C,Cs,D,Ef,E,F,Fs,G,Gs]
        code = indent + "/* %s */\n" % self.name
        code += indent + "unsigned int basis_%s[%d][%d] = {\n" % (self.name,len(self.hvs),self.N // 32)
        keys = [self.Pitches.A, self.Pitches.Bf, self.Pitches.B, self.Pitches.C, self.Pitches.Cs, self.Pitches.D, self.Pitches.Ef, self.Pitches.E, self.Pitches.F, self.Pitches.Fs, self.Pitches.G, self.Pitches.Gs]
        for key in keys:
            code += indent + "    %s, /* %s */\n" % (self.hdc.c_code(self.hvs[key]), key.value)
        code += indent + "};\n"
        return code




class BasisHVsDuration(BasisHVs):
    class Duration(Enum):
        Quarter = "quarter"
        Half = "half"
        Whole = "whole"
        Breve = "breve"
        Eighth = "eighth"
        Sixteenth = "16th"
        Longa = "longa"
        Maxima = "Maxima"
        ThirtyTwo = "32nd"
        Complex = "complex"

        def from_duration(dur):
            if dur.type == "quarter":
                return BasisHVsDuration.Duration.Quarter
            elif dur.type == "half":
                return BasisHVsDuration.Duration.Half
            elif dur.type == "whole":
                return BasisHVsDuration.Duration.Whole
            elif dur.type == "eighth":
                return BasisHVsDuration.Duration.Eighth
            elif dur.type == "16th":
                return BasisHVsDuration.Duration.Sixteenth
            elif dur.type == "32nd":
                return BasisHVsDuration.Duration.ThirtyTwo
            elif dur.type == "complex":
                print(dur)
                raise Exception("complex?")
                return BasisHVsDuration.Duration.Complex
            elif dur.type == "breve":
                return BasisHVsDuration.Duration.Breve
            elif dur.type == "longa":
                return BasisHVsDuration.Duration.Longa
            elif dur.type == "maxima":
                return BasisHVsDuration.Duration.Maxima
            else:
                print(dur.type)
                raise NotImplementedError

    def __init__(self,N):
        BasisHVs.__init__(self,"duration",N)
        self.from_enum(BasisHVsDuration.Duration)

def BasisHVAbsOctaves(N,R):
    return BasisHVsValues(N,R,"octaves",0,7) 

def BasisHVRelOctaves(N,R):
    return BasisHVsValues(N,R,"reloctaves",-7,7) 

def BasisHVsNumerNotes(N,R):
    return BasisHVsValues(N,R,"notes",0,88)

class BasisHVsRests(BasisHVsValues):
 
    def __init__(self,N,R):
        BasisHVsValues.__init__(self,N,R,"numer_rests",0,88)

    def xform(self,rest):
        sc = rest*12
        int_sc = int(round(sc))
        assert(abs(sc-int_sc) < 1e-4)
        return int_sc

   
class BasisHVsNumerDuration(BasisHVsValues):

    def __init__(self,N,R):
        BasisHVsValues.__init__(self,N,R,"numer_duration",-8*self.scale(),8*self.scale())

    def scale(self):
        return 24

    def max_duration(self):
        return 10

    def xform(self,dur):
        sc = math.log2(dur.quarterLength)*self.scale()
        int_sc = int(round(sc))
        v1 = 2**(sc/self.scale())
        v2 = 2**(int_sc/self.scale())
        #print(sc,int_sc,v1,v2,v1-v2)
        assert(abs(v1-v2)/v1 < 5e-2)
        return int_sc

def BasisHVsMelody(N,R):
    melody_range = 44
    return BasisHVsValues(N,R,"melody",-melody_range,melody_range)

