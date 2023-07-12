import numpy as np
import functools
import operator 
import math

random = np.random.RandomState(0)

class BSC:

    def __init__(self):
        pass


    @classmethod
    def random_hypervector(cls,N):
        return list(random.binomial(1,0.5,N))

    @classmethod
    def is_binary(cls,hv):
        assert(all(map(lambda v: v == 0 or v == 1, hv)))

    @classmethod
    def bundle(cls,els):
        if len(els) == 1:
            return els[0]

        sums = list(map(lambda t: sum(t), zip(*els)))
        if len(els) % 2 == 1:
            k = len(els) / 2.0
            hv = list(map(lambda s: int(math.floor(s/(k+0.1))) , sums))
            cls.is_binary(hv)
            return hv
        else:
            k = len(els) / 2.0
            sums = list(map(lambda t: sum(t), zip(*els)))
            hv = list(map(lambda s: random.binomial(1,0.5) if s == k else int(math.floor(s/(k+0.1))), sums))
            cls.is_binary(hv)
            return hv

    @classmethod
    def bind(cls,els_):
        els = list(els_)
        assert(len(els) >= 2)
        hv = list(map(lambda t: functools.reduce(operator.xor, t), zip(*els)))
        return hv

    @classmethod
    def dist(cls,a,b):
        N = len(a)
        return sum(cls.bind([a,b]))/N



    @classmethod
    def permute(cls,hv,k):
        k *= 8
        if k == 1:
            return hv[k:] + [hv[0]]
        else:
            return hv[k:] + hv[:k]

    @classmethod
    def sequence(cls,lst):
        n = len(lst)
        els = []
        for i in range(0,n):
            j = n - i - 1
            yield cls.permute(lst[j],j)

    @classmethod
    def c_code(cls, hv: list[int]) -> str:
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

def HDC():
    return BSC


class ItemMemory:

    def __init__(self,encoder):
        self.mem = {}
        self.encoder = encoder

    @property
    def n(self):
        return len(self.mem.keys())

    def store(self,key,hd):
        self.mem[key] = hd

    def lookup(self,hd,K=1):
        indices = list(range(self.n))
        keys = list(self.mem.keys())
        dists = []
        for i in indices:
            dists.append(self.encoder.hdc.dist(self.mem[keys[i]],hd))
        
        top_inds = np.argsort(dists)
        scores = list(map(lambda i: (keys[i], dists[i]), top_inds[:K]))
        return dict(scores)
    
    def c_code(self, name: str = "ItemMemory", indent: str = "") -> str:
        code = "/* %s */\n" % name
        k_type = "int" if isinstance(list(self.mem.keys())[0], int) else "std::string"
        code += indent + "std::map<%s, unsigned int *> %s = {\n" % (k_type, name)
        for key,hv in self.mem.items():
            if k_type != "int":
                key = '"%s"' % (key,)
            code += indent + "    { %s, new unsigned int[%d]%s },\n" % (key, self.encoder.N // 32, self.encoder.hdc.c_code(hv))
        code += indent + "};\n"
        return code