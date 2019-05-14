# coding:utf-8
# 归并排序


def merge_sort(alist):
    n = len(alist)
    if n <= 1:
        return alist
    mid = n//2

    left_li = merge_sort(alist[:mid])
    right_li = merge_sort(alist[mid:])

    left_pointer, right_pointer = 0, 0
    result = []

    while left_pointer < len(left_li) and right_pointer < len(right_li):
        if left_li[left_pointer] > right_li[right_pointer]:
            result.append(right_li[right_pointer])
            right_pointer += 1
        else:
            result.append(left_li[left_pointer])
            left_pointer += 1

    result += left_li[left_pointer:]
    result += right_li[right_pointer:]
    return result


if __name__ == '__main__':
    l = [54, 26, 93, 16, 77, 31, 44, 55, 20]
    res = merge_sort(l)
    print(res)