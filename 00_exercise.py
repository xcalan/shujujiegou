# -*- coding:utf-8 -*-


class Solution:
    def maxInWindows(self, num, size):
        # write code here
        #num=[2,3,4,2,6,2,5,1]
        #size=3

        if num is None or size==0:
            return []
        if size > len(num):
            size = len(num)
        max_list = []
        for i in range(0, len(num)-size+1):
            l = []
            for j in range(i, i+size):
                l.append(num[j])
            max_list.append(max(l))
        return max_list


if __name__ == '__main__':
    s = Solution()
    li = s.maxInWindows([2,3,4,2,6,2,5,1], 3)
    print(li)