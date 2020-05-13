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
