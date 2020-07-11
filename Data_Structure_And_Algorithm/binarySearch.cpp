int binarySearch(vector<int> &nums, int target) {

    int low = 0, high = nums.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        cout << "mid: " << mid << endl;
        if (nums[mid] < target) {
            low = mid + 1;
        }
        if (nums[mid] > target) {
            high = mid - 1;
        }
        if (nums[mid] == target) {
            return mid;
        }

    }
    return low;
}


//二分查找，详细介绍
https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/er-fen-cha-zhao-suan-fa-xi-jie-xiang-jie-by-labula/

int binary_search(vector<int>& nums, int target) {
    if (nums.size() == 0) return -1;
    int left = 0, right = nums.size() - 1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] < target) {
            //[mid+1, right]
            left = mid + 1;
        }else if(nums[mid] > target) {
            //[left, mid-1]
            right = mid - 1;
        }else if(nums[mid] == target) {
            return mid;
        }
    }
    return -1;
}

int left_bound(vector<int>& nums, int target) {
    if (nums.size() == 0) return -1;
    int left = 0, right = nums.size() - 1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            //[mid+1, right]
            left = mid + 1;
        }else if (nums[mid] > target) {
            //[left, mid - 1]
            right = mid - 1;
        } else if (nums[mid] == target){
            // 别返回，收缩右侧边界
            right = mid - 1;
        }
    }
    if (left == nums.size() || nums[left] != target)
        return -1;
    return left;
}

int right_bound(vector<int>& nums, int target) {
    if (nums.size() == 0) return -1;
    int left = 0, right = nums.size() - 1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            //[mid + 1, right]
            left = mid + 1;
        }else if (nums[mid] > target) {
            //[left, mid - 1]
            right = mid - 1;
        }else if (nums[mid] == target) {
            // 别返回，收缩左侧边界
            left = mid + 1;
        }
    }
    if(right < 0 || nums[right] != target)
        return -1;
    return right;
}


//4. 寻找两个正序数组的中位数
/*给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。
请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。
*/
int getKth(vector<int>& nums1, int start1, int end1, vector<int>& nums2, int start2, int end2, int k) {
    int len1 = end1 - start1 + 1;
    int len2 = end2 - start2 + 1;

    //让len1 < len2, 这样就能保证如果有数组空了，一定是len1
    if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
    if (len1 == 0) return nums2[start2 + k - 1];

    if (k == 1) return min(nums1[start1], nums2[start2]);

    int i = start1 + min(len1, k / 2) - 1;
    int j = start2 + min(len2, k / 2) - 1;

    if (nums1[i] > nums2[j]) {
        return getKth(nums1, start1, end1, nums2, j + 1, end2, k - min(len2, k / 2));
    }else {
        return getKth(nums1, i + 1, end1, nums2, start2, end2, k - min(len1, k / 2));
    }
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int n = nums1.size();
    int m = nums2.size();

    int left = (n + m + 1) / 2;
    int right = (n + m + 2) / 2;
    
    return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;
}

int getKthElement2(const vector<int>& nums1, const vector<int>& nums2, int k) {
    /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
     * 这里的 "/" 表示整除
     * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
     * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
     * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
     * 这样 pivot 本身最大也只能是第 k-1 小的元素
     * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
     * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
     * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
     */

    int m = nums1.size();
    int n = nums2.size();
    int index1 = 0, index2 = 0;

    while (true) {
        // 边界情况
        if (index1 == m) {
            return nums2[index2 + k - 1];
        }
        if (index2 == n) {
            return nums1[index1 + k - 1];
        }
        if (k == 1) {
            return min(nums1[index1], nums2[index2]);
        }
        // 正常情况
        int newIndex1 = min(index1 + k / 2 - 1, m - 1);
        int newIndex2 = min(index2 + k / 2 - 1, n - 1);
        int pivot1 = nums1[newIndex1];
        int pivot2 = nums2[newIndex2];
        if (pivot1 <= pivot2) {
            k -= newIndex1 - index1 + 1;
            index1 = newIndex1 + 1;
        }
        else {
            k -= newIndex2 - index2 + 1;
            index2 = newIndex2 + 1;
        }
    }
}

double findMedianSortedArrays2(vector<int>& nums1, vector<int>& nums2) {
    int totalLength = nums1.size() + nums2.size();
    if (totalLength % 2 == 1) {
        return getKthElement2(nums1, nums2, (totalLength + 1) / 2);
    }
    else {
        return (getKthElement2(nums1, nums2, totalLength / 2) + getKthElement2(nums1, nums2, totalLength / 2 + 1)) / 2.0;
    }
}


double findMedianSortedArrays3(vector<int>& nums1, vector<int>& nums2) {
    if (nums1.size() > nums2.size()) {
        return findMedianSortedArrays(nums2, nums1);
    }

    int m = nums1.size();
    int n = nums2.size();
    int left = 0, right = m, ansi = -1;
    // median1：前一部分的最大值
    // median2：后一部分的最小值
    int median1 = 0, median2 = 0;

    while (left <= right) {
        // 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
        // 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
        int i = (left + right) / 2;
        int j = (m + n + 1) / 2 - i;

        // nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
        int nums_im1 = (i == 0 ? INT_MIN : nums1[i - 1]);
        int nums_i = (i == m ? INT_MAX : nums1[i]);
        int nums_jm1 = (j == 0 ? INT_MIN : nums2[j - 1]);
        int nums_j = (j == n ? INT_MAX : nums2[j]);

        if (nums_im1 <= nums_j) {
            ansi = i;
            median1 = max(nums_im1, nums_jm1);
            median2 = min(nums_i, nums_j);
            left = i + 1;
        }
        else {
            right = i - 1;
        }
    }

    return (m + n) % 2 == 0 ? (median1 + median2) / 2.0 : median1;
}

