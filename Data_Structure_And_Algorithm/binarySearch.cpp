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


//69. x 的平方根
/*实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
*/
int mySqrt(int x) {
    int left = 0, right = x;
    int res = 0.0;
    while(left <= right) {
        int mid = left + ((right - left) >> 1);
        if ((long long) mid * mid > x) {
            right = mid - 1;
        } else {
            left = mid + 1;
            res = mid;
        }
    }
    return res;
}
//50. Pow(x, n)
//实现 pow(x, n) ，即计算 x 的 n 次幂函数。
double quickMul(double x, int n) {
    if (n == 0)
        return 1;
    double y = quickMul(x, n / 2);
    return n % 2 == 0 ? y * y : y * y * x;
}
double myPow(double x, int n) {
    long long N = n;

    return N >= 0 ? quickMul(x, N) : 1 / quickMul(x, -N);
}

//74. 搜索二维矩阵
/*编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。
*/
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    if (m == 0) return false;
    int n = matrix[0].size();

    // 二分查找
    int left = 0, right = m * n - 1;
    int pivotIdx, pivotElement;
    while (left <= right) {
        int mid = (left + right) / 2;
        int value = matrix[mid / n][mid % n];
        if (target == value)
            return true;
        else if (value < target)
            left = mid + 1;
        else right = mid - 1;
    }
    return false;
}

//33. 搜索旋转排序数组
/*假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素。
你的算法时间复杂度必须是 O(log n) 级别。
*/
int search(vector<int>& nums, int target) {
    int n = nums.size();
    if (0 == n) return -1;
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        if (nums[mid] == target) return mid;
        //这里 nums[left]和nums[right]相等，和小于情况一致
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (nums[mid] <= target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}

//81. 搜索旋转排序数组 II
/*
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。
*/
bool search(vector<int>& nums, int target) {
    int len = nums.size();
    if(0 == len) return false;
    int left = 0, right = len - 1;
    while(left <= right) {
        int mid = left + ((right - left) >> 1);
        if (nums[mid] == target) {
            return true;
        } else if(nums[mid] > nums[left]) {
            if (nums[left] <= target && target <= nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else if (nums[mid] < nums[left]) {
            if (nums[mid] <= target && target <= nums[right]) {
                left = mid + 1;
            }else {
                right = mid - 1;
            }
        } else if (nums[mid] == nums[left]) {
            left++;
        }
    }
    return false;
}
//153. 寻找旋转排序数组中的最小值
//难度中等
/*假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
请找出其中最小的元素。
你可以假设数组中不存在重复元素。
*/
int findMin(vector<int>& nums) {
    int len = nums.size();
    if (len == 1) return nums[0];
    int left = 0, right = len - 1;

    while(left < right) {
        int mid = left + ((right - left) >> 1);
        if (nums[mid] > nums[right])
            left = mid + 1;
        else if (nums[mid] < nums[right])
            right = mid;
        //else
        //    right -= 1; 
    }
    return nums[left];
}

//154. 寻找旋转排序数组中的最小值 II
/*假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
请找出其中最小的元素。
注意数组中可能存在重复的元素。
*/

int findMin(vector<int>& nums) {
    int len = nums.size();
    if (len == 1) return nums[0];
    int left = 0, right = len - 1;

    while(left < right) {
        int mid = left + ((right - left) >> 1);
        if (nums[mid] > nums[right])
            left = mid + 1;
        else if (nums[mid] < nums[right])
            right = mid;
        else
            right -= 1; 
    }
    return nums[left];
}

//162. 寻找峰值
/*峰值元素是指其值大于左右相邻值的元素。
给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。
数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。
你可以假设 nums[-1] = nums[n] = -∞。
*/
int findPeakElement(vector<int>& nums) {
    int len = nums.size();
    int left = 0, right = len - 1;
    while(left < right) {
        int mid = left + ((right - left) >> 1);
        if (nums[mid] > nums[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

//167. 两数之和 II - 输入有序数组
//其实这道题解法是双指针法，但是二分查找也可以，二分查找是O(Nlog(N))
//而双指针法是：O(N)
/*给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
*/
vector<int> twoSum(vector<int>& numbers, int target) {
    int len = numbers.size();
    int left = 0, right = len - 1;
    while(left <= right) {
        int sum = numbers[left] + numbers[right];
        if (target == sum) {
            return {left + 1, right + 1};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return {-1, -1};
}


