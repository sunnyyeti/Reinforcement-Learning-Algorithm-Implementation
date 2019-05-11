class Solution:
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        x_points = {}
        y_points = {}
        self.nbs = {}
        for x,y in stones:
            x_points.setdefault(x,[]).append((x,y))
            y_points.setdefault(y,[]).append((x,y))
        for x,y in stones:## Find all connected points
            nbset = self.nbs.setdefault((x,y),set())
            for p in x_points[x]:
                if p!=(x,y):
                    nbset.add(p)
            for p in y_points[y]:
                if p!=(x,y):
                    nbset.add(p)
        self.visited = set()
        cnt = 0
        for x,y in stones:
            if (x,y) not in self.visited:
                loopcnt = self.dfs((x,y))
                cnt+=loopcnt-1
        return cnt
    def dfs(self,p):
        self.visited.add(p)
        point_cnt = 1
        for next_point in self.nbs[p]:
            if next_point not in self.visited:
                point_cnt+=self.dfs(next_point)
        return point_cnt

if __name__=="__main__":
    print(Solution().removeStones([[3,2],[3,1],[4,4],[1,1],[0,2],[4,0]]))