def quick_sort(li, start, end):
    print(list)
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = li[left]
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        while left < right and li[right] >= mid:
            right -= 1
        li[left] = li[right]
        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and li[left] < mid:
            left += 1
        li[right] = li[left]
    # while结束后，把mid放到中间位置，left=right
    li[left] = mid
    # 递归处理左边的数据
    quick_sort(li, start, left-1)
    # 递归处理右边的数据
    quick_sort(li, left+1, end)
def special_quicksort(list, start, end):
    print(list)
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = list[left]
    print(mid[1])
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        while left < right and list[right][1] >= mid[1]:
            right -= 1
        list[left] = list[right]
        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and list[left][1] < mid[1]:
            left += 1
        list[right] = list[left]
    # while结束后，把mid放到中间位置，left=right
    list[left] = mid
    # 递归处理左边的数据
    special_quicksort(list, start, left - 1)
    # 递归处理右边的数据
    special_quicksort(list, left + 1, end)
list = [152,134,38796,7438415,1,2272,34345,24,127]
# print(list)
quick_sort(list,0,list.__len__()-1)
print(list)


list = [[1,152],[2,134],[3,38796],[4,7438415],[5,1],[6,2272],[7,34345],[8,24],[9,127]]
special_quicksort(list,0,list.__len__()-1)
print(list)