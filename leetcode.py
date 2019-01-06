class Solution:
    def match(self,n):
        n = list(range(1,n+1))
        while len(n)>1:
            new_n = [(n[i],n[len(n)-i-1])for i in range(len(n)//2)]
            n = new_n
        print(str(n[0]))

if __name__=="__main__":
    Solution().match(16)