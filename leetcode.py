import functools

class Solution:
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        bds = []
        for s,e,h in buildings:
            bds.append(CP(s,h,True))
            bds.append(CP(e,h,False))
        bds.sort(key=functools.cmp_to_key(comp_cp))
        maxheap = Maxheap([0])
        premax = 0
        res = []
        for bd in bds:
            x,h,t = bd.x,bd.h,bd.start
            if t:
                maxheap.append(h)
            else:
                maxheap.remove(h)
            curmax = maxheap.peek_max()
            if curmax != premax:
                res.append([x, curmax])
                premax = curmax
        return res

def comp_cp(cp1,cp2):
    if cp1.x!=cp2.x:
        return cp1.x-cp2.x
    else:
        return cp1.comp_code-cp2.comp_code

class CP:
    def __init__(self,x,h,start):
        self.x = x
        self.h = h
        self.start=start

    @property
    def comp_code(self):
        if self.start:
            return -self.h
        else:
            return self.h

class Maxheap:
    def __init__(self,array=None):
        if array is None:
            self.heap = []
            self.heap_size = 0
            self.ind = {}
            self.cnt = {}
        else:
            self.ind = {}
            self.cnt = {}
            for i,a in enumerate(array):
                self.cnt[a] = self.cnt.setdefault(a,0)+1
            array = list(set(array))
            self.heap_size=len(array)
            self.heap = array
            for i,a in enumerate(array):
                self.ind[a]=i
            self.heapify()

    def max_heapify(self,i):
        largest = i
        l = 2*i+1
        if l<self.heap_size and self.heap[l]>self.heap[i]:
            largest = l
        r = 2*i+2
        if r<self.heap_size and self.heap[r]>self.heap[largest]:
            largest = r
        if largest!=i:
            self.ind[self.heap[i]], self.ind[self.heap[largest]] = largest,i
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.max_heapify(largest)

    def heapify(self):
        for i in range(self.heap_size//2-1,-1,-1):
            self.max_heapify(i)

    def get_max(self):
        max = self.heap[0]
        self.cnt[max]-=1
        if self.cnt[max]==0:
            self.heap_size-=1
            if self.heap_size!=0:
                self.heap[0] = self.heap[self.heap_size]
                self.ind[self.heap[0]] = 0
                self.cnt.pop(max)
                self.ind.pop(max)
                self.max_heapify(0)
            else:
                self.cnt.pop(max)
                self.ind.pop(max)
        return max

    def peek_max(self):
        return self.heap[0]

    def float_up(self,ind):
        parent = (ind-1)//2
        while parent>-1 and self.heap[parent]<self.heap[ind]:
            self.ind[self.heap[parent]],self.ind[self.heap[ind]] = ind,parent
            self.heap[parent],self.heap[ind] = self.heap[ind],self.heap[parent]
            ind = parent
            parent = (ind-1)//2

    def append(self,val):
        if val in self.cnt:
            self.cnt[val]+=1
            return
        if self.heap_size==len(self.heap):
            self.heap.append(val)
        else:
            self.heap[self.heap_size]=val
        self.cnt[val]=1
        self.ind[val]=self.heap_size
        self.heap_size+=1
        self.float_up(self.heap_size-1)

    def remove(self,val):
        self.cnt[val]-=1
        if self.cnt[val]==0:
            self.cnt.pop(val)
            index = self.ind.pop(val)
            self.heap_size-=1
            if self.heap_size!=index:
                self.heap[index] = self.heap[self.heap_size]
                self.ind[self.heap[index]]=index
                self.max_heapify(index)

    @property
    def content(self):
        return self.heap[:self.heap_size]

if __name__ == '__main__':
    bds = [ [2, 9 ,10], [3 ,7, 15], [5, 12 ,12], [15 ,20, 10], [19, 24 ,8] ]
    print(Solution().getSkyline(bds))