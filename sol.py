from typing import List

# Spiral Matrix 2
def generateMatrix(A):
    t = 0
    b = A
    l = 0
    r = A
    d = 0
    mat = [[0 for i in range(A)] for j in range(A)]
    i = 1
    while t <= b and l <= r:
        if d == 0:
            for k in range(l, r):
                mat[t][k] = i
                i += 1
            t += 1
            d = 1
        if d == 1:
            for k in range(t, b):
                mat[k][r-1] = i
                i += 1
            r -= 1
            d = 2
        if d == 2:
            for k in range(r-1, l-1, -1):
                mat[b-1][k] = i
                i += 1
            b -= 1
            d = 3
        if d == 3:
            for k in range(b-1, t-1, -1):
                mat[k][l] = i
                i += 1
            l += 1
            d = 0
    return mat

# print(generateMatrix(3))

# Pascals Triangle


def solve(A):
    x = [[0 for i in range(A+1)] for j in range(A+1)]
    x[0][0] = 1
    for i in range(A+1):
        for j in range(A+1):
            if i == 0 and j == 0:
                x[i][j] = 1
            elif i > j:
                x[i][j] = x[i-1][j] + x[i-1][j-1]
    print(x)
    result = []
    for i in range(1, A+1):
        a = []
        for j in range(A+1):
            if x[i][j] != 0:
                a.append(x[i][j])
        result.append(a)
    print(result)


# solve(5)

# Pascals Triangle - kth row, where k is zero based.


def solve_for_k(A, k):
    x = [[0 for i in range(A+1)] for j in range(A+1)]
    x[0][0] = 1
    for i in range(A+1):
        for j in range(A+1):
            if i == 0 and j == 0:
                x[i][j] = 1
            elif i > j:
                x[i][j] = x[i-1][j] + x[i-1][j-1]
    result = []
    for i in range(1, A+1):
        a = []
        for j in range(A+1):
            if x[i][j] != 0:
                a.append(x[i][j])
        result.append(a)
    res = []
    for j in range(len(result[k])):
        res.append(result[k][j])
    print(res)


# solve_for_k(5, 3)

# Max non-negative sub array
def maxset(A):
    i = 0
    maxi = -1
    # index = 0
    a = []

    while i < len(A):
        while i < len(A) and A[i] < 0:
            i += 1
        l = []
        # index = i
        while i < len(A) and A[i] >= 0:
            l.append(A[i])
            i += 1

        if (sum(l) > maxi):
            a = l
            maxi = sum(l)

    return a

# maxset([1,2,3, -1, -4, -2, 1, 3, 5, 7, -1, 2])

# Prime sum of a given even number


def primesum(A):
    isPrime = [0] * (A+1)

    isPrime[0] = isPrime[1] = False
    for i in range(2, A+1):
        isPrime[i] = True

    p = 2
    while(p*p <= A):

        # If isPrime[p] is not changed,
        # then it is a prime
        if (isPrime[p] == True):

            # Update all multiples of p
            i = p*p
            while(i <= A):
                isPrime[i] = False
                i += p
        p += 1

    # Traversing all numbers to find
    # first pair
    for i in range(0, A):

        if (isPrime[i] and isPrime[A - i]):

            print([i, (A - i)])
            break


# primesum(1048574)


# Product except self : Array - leetcode 238
def productExceptSelf(nums: List[int]) -> List[int]:
    ans = [1]*len(nums)
    
    s = nums[0]
    for i in range(1, len(nums)):
        ans[i] *= s
        s *= nums[i]
    
    s = nums[-1]
    for i in range(len(nums)-2, -1, -1):
        ans[i] *= s
        s *= nums[i]
    
    return ans

# print(productExceptSelf([2, 4, 6 , 8]))

# Happy number -> if square sum of digits of number adds upto 1 - leetcode 202
def isHappy(n: int) -> bool:
    visited = {n}
    
    while True:
        prod = 0
        ls = list(map(int, str(n)))
        for a in ls:
            prod += (a*a)
        n = prod
        
        if n == 1:
            return True
        elif n in visited:
            return False
        else:
            visited.add(n)

# print(isHappy(536))

# Number of islands given a 2d array of 1's and 0's.
# count the number of 1's placed together and that becomes an island, count such distinct islands in the grid.
# Done using BFS traversal of the grid. - leetcode 200

def numIslands(grid: List[List[str]]) -> int:
        if len(grid) < 1: return 0
        counter = 0
        offset_locs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    grid[i][j] = '0'
                    counter += 1
                    queue = []
                    queue.append([i, j])
                    while queue:
                        row, col = queue.pop(0)
                        for offset in offset_locs:
                            if 0 <= row + offset[0] < rows and 0 <= col + offset[1] < cols and grid[row + offset[0]][col + offset[1]] == '1':
                                grid[row + offset[0]][col + offset[1]] = '0'
                                queue.append([row + offset[0], col + offset[1]])
        
        return counter

# island = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
# print(numIslands(island))

# Remove k digits from the given number, and then printing the smallest possible number of size len(num)-k - leetcode 402
def removeKdigits(num: str, k: int) -> str:

    if k == 0:
        return num
    if len(num) <= k:
        return '0'
    
    counter = 0
    stack_idx = 0
    while counter < k:
        counter += 1
        while stack_idx < len(num) - 1 and num[stack_idx] <= num[stack_idx + 1]:
            stack_idx += 1
        if stack_idx == len(num) - 1:
            num = num[:stack_idx]
        else:
            num = num[:stack_idx] + num[stack_idx + 1:]
        stack_idx -= 1
        if stack_idx == -1:
            i = 0
            while i < len(num) and num[i] == '0':
                    i += 1
            if i == len(num):
                return '0'
            else:
                num = num[i:]
        stack_idx = 0
    return num

# print(removeKdigits('10200', 1))

## LRU Cache - leetcode 146

class DLL:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.length = 0
        self.head = DLL(-1, -1)
        self.tail = self.head
        self.hash = {}

    def get(self, key: int) -> int:
        if key not in self.hash:
            return -1
        ret = self.hash[key]
        val = ret.value
        while ret.right:
            ret.left.right = ret.right
            ret.right.left = ret.left
            self.tail.right = ret
            ret.left = self.tail
            ret.right = None
            self.tail = ret
            
        return val

    def put(self, key: int, value: int) -> None:
        if key in self.hash:
            node = self.hash[key]
            node.value = value
            while node.right:
                node.left.right = node.right
                node.right.left = node.left
                self.tail.right = node
                node.left = self.tail
                node.right = None
                self.tail = node
            
        if key not in self.hash:
            node = DLL(key, value)
            node.left = self.tail
            self.tail.right = node
            self.tail = node
            self.hash[key] = node
            self.length += 1
            if self.length > self.capacity:
                rem = self.head.right
                self.head.right = rem.right
                rem.right.left = self.head
                del self.hash[rem.key]
                self.length -= 1

# test cases.
# cache = LRUCache(2)
# output = []
# output.append(cache.put(1, 1))
# output.append(cache.put(2, 2))
# output.append(cache.get(1))
# output.append(cache.put(3, 3))
# output.append(cache.get(2))
# output.append(cache.put(4, 4))
# output.append(cache.get(1))
# output.append(cache.get(3))
# output.append(cache.get(4))
# print(output)

# Missing number - leetcode 268
# With O(n) space complexity

# def missingNumber(nums: List[int]) -> int: 
#         numbers = {}
#         for num in nums:
#             if num not in numbers:
#                 numbers[num] = True
#         for i in range(0, len(nums) + 1):
#             if i not in numbers:
#                 return i

# With O(1) space complexity

def missingNumber(nums: List[int]) -> int: 
        n = len(nums)
        sum_of_n = n*(n+1)/2
        sum = 0
        for num in nums:
            sum += num
            
        return int(sum_of_n - sum)

# print(missingNumber([0, 1]))


# Populating next pointers for each node in an fully complete binary tree - Leetcode 116
# With O(1) space complexity.

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

def connect(root: 'Node') -> 'Node':
        cur_head = root
        while cur_head:
            temp = cur_head
            prev = None
            next_head = temp.left
            if not next_head:
                break
            while temp:
                if prev:
                    prev.next = temp.left
                temp.left.next = temp.right
                prev = temp.right
                temp = temp.next
            cur_head = next_head
        return root

def outputConnector():

    n4 = Node(4)
    n5 = Node(5)
    n6 = Node(6)
    n7 = Node(7)
    n3 = Node(3, n6, n7)
    n2 = Node(2, n4, n5)
    n1 = Node(1, n2, n3)

    root = connect(n1)

    # output = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.val)
        print(node.next.val if node.next != None else 'no node')
        if node.left:
            queue.append(node.left)
            print(node.left.val)
            print(node.left.next.val if node.left.next != None else 'no node')
        if node.right:
            queue.append(node.right)
            print(node.right.val)
            print(node.right.next.val if node.right.next != None else 'no node')


# output print is wrong, the original output should be [1,#,2,3,#,4,5,6,7,#]
# outputConnector()

# Circular Array Loop - leetcode 457
# To find whether the circular array contains any loop while traversing through the keys

def circularArrayLoop(nums: List[int]) -> bool:
    n = len(nums)
    if n <= 1: return False
    for i,v in enumerate(nums):
        head = i
        # print("head", head)
        dir = v
        while True:
            step = nums[i]
            # print("step", step)
            if step*dir < 0:
                break
            next_i = (i+step)%n
            # print("next",next_i)
            if next_i == i:
                break
            if next_i == head:
                return True
            # print("before", nums)
            # For cases where there is a loop but not starting from the head.
            # eg: 0 -> 1 -> 2 -> 1 -> 2 ....
            nums[i] = head - i + (n if dir > 0 else -n)
            # print("new nums[i]", i, nums[i])
            # print("After", nums)
            i = next_i
            # print("\n")
        # print("bleh")
    return False

# print(circularArrayLoop([2, -1, 1, 2, 2]))
# print(circularArrayLoop([3, 1, 2]))
# print(circularArrayLoop([-2, 1, -1, -2, -2]))


# Course Schedule II - Leetcode 210
# This is a program to find the correct order to complete the courses, such that no dependent courses are omitted.
# We use DFS topological sorting to solve this or else use the precedence of each courses are arrange them accordingly. 

from collections import defaultdict

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adjList = defaultdict(list)
        visit = [0]*numCourses
        output = []
        for course1, course2 in prerequisites:
            adjList[course1].append(course2)
        
        for course in range(numCourses):
            if self.topological_sort(course, adjList, visit, output):
                return []
        return output
    
    def topological_sort(self, course, adjList, visit, output):
        if visit[course] == -1:
            return True
        if visit[course] == 1:
            return False
        
        visit[course] = -1
        
        for depends in adjList[course]:
            if self.topological_sort(depends, adjList, visit, output):
                return True
        
        visit[course] = 1
        output.append(course)

# solution = Solution()
# print(solution.findOrder(2, [[0, 1]]))

# Generate Paranthesis - Leetcode 22
# This program is used to generate various combinations of paranthesis given a number of paranthesis.
# It is done using recursive calls to the same helper function inorder to create the various possibilities.

def generateParenthesis(n: int) -> List[str]:
    if n == 0:
        return [""]
    output = []
    
    def helper(comb, a, b):
        if b < a:
            return
        if a == 0:
            comb = comb + ')'*b
            output.append(comb)
            return
        
        helper(comb + '(', a-1, b)
        helper(comb + ')', a, b-1)
    
    helper('', n, n)
    return output

# print(generateParenthesis(3))

# Sliding window maximum - Leetcode 239
# This program is to find the maximum subarray from the given input by sliding window method, it passes a smaller frame and the max value in that frame is in the output.
# Done using heapq to attain max efficiency. Time complexity is linear in the case.
# Another method to do this is by attaining the max of each window using the max function. (This method is more useful for saving space but slower)
# For improvement, we can maybe try and generate the heapq method on our own to save time i.e. convert heapq functionality to simple python code.
import heapq

# def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    # if len(nums) == 0:
    #     return []
    # output = []
    # [output.append(max(nums[i:i+k])) for i in range(0, len(nums)-k+1)]
    # return output

def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    if len(nums) == 0:
        return []
    if k >= len(nums):
        return [max(nums)]
    heap = []
    output = []
    for i in range(0, k):
        heapq.heappush(heap, (-nums[i], i))
    output.append(max(nums[0:k]))
    for i in range(k, len(nums)):
        heapq.heappush(heap, (-nums[i], i))
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        output.append(-heap[0][0])
    return output

# print(maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))

# Most profit assigning work - Leetcode 826
# This program is used to find the maximum profit able to obtain by a list of workers depending on the difficulty level of the job as well as the profit of the job
# It is solved using two ways, heapq as well as using zip method and sorted to sort the most profitting jobs and solve accordingly.
# The zip method is faster compared to the heapq method.

# def maxProfitAssignment(difficulty: List[int], profit: List[int], worker: List[int]) -> int:
#     helper = []
#     heapq.heapify(helper)
#     for i in range(0, len(difficulty)):
#         heapq.heappush(helper, (-profit[i], difficulty[i]))
        
#     worker.sort()
#     maxProfit = 0
    
#     while worker and helper:
#         if worker[-1] >= helper[0][1]:
#             maxProfit -= helper[0][0]
#             worker.pop()
#         else:
#             heapq.heappop(helper)
    
#     return maxProfit

def maxProfitAssignment(difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        
    taskProfitPairs = sorted(list(zip(difficulty, profit)), key=lambda pair: pair[0])
    sortedTasks = [v[0] for v in taskProfitPairs]
    n = len(sortedTasks)
    money = 0
    lastExploredTask = 0
    lastProfit = 0
    for ability in sorted(worker):
        while lastExploredTask < n and ability >= sortedTasks[lastExploredTask]:
            currProfit = taskProfitPairs[lastExploredTask][1]
            if currProfit > lastProfit:
                lastProfit = currProfit
            lastExploredTask += 1
        money += lastProfit
    return money

# print(maxProfitAssignment([2,4,6,8,10], [10,20,30,40,50], [4,5,6,7]))

# Task Scheduler - Leetcode 621
# This program aims to find the least amount of time taken to run these number of tasks considering the amount of gap in running the same tasks again, specified by n.

def leastInterval(tasks: List[str], n: int) -> int:
    if len(tasks) <= 1 or n == 0:
        return len(tasks)
    d = dict()
    for task in tasks:
        if task in d:
            d[task] += 1
        else:
            d[task] = 1
    vals = []
    for i in d:
        vals.append(d[i])
    vals.sort(reverse=True)
    if len(tasks) / (n+1) >= vals[0]:
        return len(tasks)
    else:
        idle = 0
        for i in range(len(vals)):
            if vals[i] == vals[0]:
                idle += 1
        # This is a on-the-go formula to return the number of time cycles needed for runnning these tasks.
        return (vals[0] - 1)*(n + 1) + idle
    
# print(leastInterval(["A","A","A","B","B","B"], 2))

