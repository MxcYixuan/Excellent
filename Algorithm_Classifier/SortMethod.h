//
// Created by qisheng.cxw on 2021/5/27.
//
#ifndef SORTMETHODS_SORT_METHOD_H
#define SORTMETHODS_SORT_METHOD_H

#endif //SORTMETHODS_SORT_METHOD_H
#include <iostream>
#include <vector>

using namespace std;
// 归并排序
void merge(vector<int>& nums, int left, int mid, int right, vector<int>& temp) {
    int i = left;
    int j = mid + 1;
    int idx = 0;
    while(i <= mid && j <= right) {
        if (nums[i] < nums[j]) {
            temp[idx++] = nums[i++];
        } else {
            temp[idx++] = nums[j++];
        }
    }
    while(i <= mid) {
        temp[idx++] = nums[i++];
    }
    while(j <= right) {
        temp[idx++] = nums[j++];
    }
    // 将数据复制到数组中
    idx = 0;
    for(int i = left; i <= right; i++) {
        nums[i] = temp[idx++];
    }

}

void sort(vector<int>& nums, int left, int right, vector<int>& temp) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        sort(nums, left, mid, temp);
        sort(nums, mid + 1, right, temp);
        merge(nums, left, mid, right, temp);
    }
}

void merge_sort(vector<int>& nums) {
    vector<int> temp(nums.size(), 0);
    sort(nums, 0, nums.size() - 1, temp);
}


// 堆排序
void swap(vector<int>& nums, int a, int b) {
    int temp = nums[a];
    nums[a] = nums[b];
    nums[b] = temp;
}

void adjust(vector<int>& nums, int len, int index) {
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int max_id = index;
    if(left < len && nums[left] > nums[max_id])
        max_id = left;
    if (right < len && nums[right] > nums[max_id])
        max_id = right;
    if (max_id != index) {
        // swap(nums[max_id], nums[index]);
        swap(nums, max_id, index);
        adjust(nums, len, max_id);
    }
}

void heap_sort(vector<int>& nums) {
    // 构建大根堆，从最后一个非叶子节点向上
    int len = nums.size();
    for(int i = len / 2 - 1; i >= 0; i--) {
        adjust(nums, len, i);
    }
    // 调整大根堆
    for (int i = len - 1; i >= 1; i--) {
        swap(nums, 0, i);
        adjust(nums, i, 0);  // 将未完成排序的部分继续进行堆排序
    }
}

void print_nums(vector<int> nums) {
    for (auto num : nums) {
        cout << num << '\t';
    }
    cout << endl;
}

// 快速排序
int partition(vector<int>& nums, int start, int end) {
    if (start == end) return start;
    int temp = nums[start];
    int i = start;
    int j = end + 1;
    while(i < j) {
        while (nums[++i] < temp)
            if(i == end) break;
        while (nums[--j] > temp)
            if (j == start) break;
        if (i >= j) break;
        swap(nums, i, j);
    }
    swap(nums, j, start);
    return j;
}

void q_sort(vector<int>& nums, int start, int end) {
    if (start >= end) return;
    int pivot = partition(nums, start, end);
    q_sort(nums, start, pivot - 1);
    q_sort(nums, pivot + 1, end);
}
void quick_sort(vector<int>& nums) {
    int start = 0;
    int end = nums.size() - 1;
    q_sort(nums, start, end);
}
