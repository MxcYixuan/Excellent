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
