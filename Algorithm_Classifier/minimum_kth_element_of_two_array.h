
// 题目：给定两个从小到大的正序数组，请找出这两个数组中的第k小的数
// 要求算法的时间复杂度小于O(k)
int getKth_1(vector<int>& arr1, vector<int>& arr2, int k) {
    if(arr1.size() + arr2.size() < k) {
        return 0;
    }
    int index = 0;
    int i = 0, j = 0;
    int res = 0;
    while(index < k) {
        if (arr1[i] < arr2[j]) {
            res = arr1[i++];
        } else {
            res = arr2[j++];
        }
        index++;
    }
    return res;
}

// 采用二分查找法 进行寻找第k小的数
// 标准版
int getKth(vector<int>& nums1, int start1, int end1, vector<int>& nums2, int start2, int end2, int k) {
    int len1 = end1 - start1 + 1;
    int len2 = end2 - start2 + 1;

    //让len1 < len2, 这样就能保证如果有数组空了，一定是len1
    if (len1 > len2)
        return getKth(nums2, start2, end2, nums1, start1, end1, k);
    if (len1 == 0)
        return nums2[start2 + k - 1];
    if (k == 1)
        return min(nums1[start1], nums2[start2]);

    int i = start1 + min(len1, k / 2) - 1;
    int j = start2 + min(len2, k / 2) - 1;

    if (nums1[i] > nums2[j]) {
        return getKth(nums1, start1, end1, nums2, j + 1, end2, k - min(len2, k / 2));
    }else {
        return getKth(nums1, i + 1, end1, nums2, start2, end2, k - min(len1, k / 2));
    }
}
int getKth_base(vector<int>& nums1, vector<int>& nums2,int k) {
    return getKth(nums1, 0, nums1.size() - 1, nums2, 0, nums2.size() - 1, k);
}

// 自己写的 二分查找版本
int build_getKth_2(vector<int>& arr1, int start1, int end1, vector<int>& arr2, int start2, int end2, int k) {
    // 计算每个数组 排查前面元素之后的长度
    int len1 = end1 - start1 + 1;
    int len2 = end2 - start2 + 1;
    // 让len1 < len2 这样如果数组为空，那肯定是Len1了
    if(len1 > len2) {
        build_getKth_2(arr2, start2, end2, arr1, start1, end1, k);
    }
    if (len1 == 0) {
        return arr2[start2 + k - 1];
    }
    if (k == 1) {
        return min(arr1[start1], arr2[start2]);
    }
    // 分别取两个数组的 k/2-1索引位置 进行判断
    int i = start1 + min(len1, k / 2) - 1;
    int j = start2 + min(len2, k / 2) - 1;
    if (arr1[i] < arr2[j]) {
        return build_getKth_2(arr1, i + 1, end1, arr2, start2, end2, k - min(len1, k / 2));
    } else {
        return build_getKth_2(arr1, start1, end1, arr2, j + 1, end2, k - min(len2, k / 2));
    }
}

int getKth_2(vector<int>& arr1, vector<int>& arr2, int k) {
    return build_getKth_2(arr1, 0, arr1.size() - 1, arr2, 0, arr2.size() - 1, k);
}


double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int n = nums1.size();
    int m = nums2.size();

    int left = (n + m + 1) / 2;
    int right = (n + m + 2) / 2;

    return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;
}
