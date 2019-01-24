class Solution:
    def maxWidthRamp(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        return self.two_for_loops(A)

    def two_for_loops(self,A):
        """
        Time limit exceeded
        :param A:
        :return:
        """
        end_max = [A[-1]]*len(A)
        for j in range(len(A)-2,-1,-1):
            end_max[j] = max(A[j],end_max[j+1])
        max_length = 0
        for i,a in enumerate(A):
            if i+max_length+1<len(A) and a<=end_max[i+max_length+1]:
                for j in range(len(A)-1,i+max_length,-1):
                    if A[j]>=a:
                        max_length = max(max_length,j-i)
                        break
        return max_length

    def dp(self,start,end):
        """
        time limit exceeded
        :param start:
        :param end:
        :return:
        """
        if (start,end) in self.cache:
            return self.cache[(start,end)]
        if self.A[start]<=self.A[end]:
            self.cache[(start,end)]=end-start
            return end-start
        else:
            return max(self.dp(start+1,end),self.dp(start,end-1))


if __name__=="__main__":
    a =[3,1,2,4]
    print(Solution().maxWidthRamp(a))
