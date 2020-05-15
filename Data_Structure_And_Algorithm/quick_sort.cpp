#include <iostream>
#include <vector>

using namespace std;

int partition_nums(vector<int> &nums, int left, int right) {
    int base = nums[left];
    int i = left ;
    int j = right;
    while(i != j) {
        while(nums[j] >= base && i < j) j--;
        while(nums[i] <= base && i < j) i++;
        if (i < j) {
            swap(nums[i], nums[j]);
        }
    }

    swap(nums[left], nums[j]);
    return j;
}

void quickSort(vector<int> &nums, int left, int right) {
    if (left > right) return;
    int mid = partition_nums(nums, left, right);
    quickSort(nums, left, mid - 1);
    quickSort(nums, mid + 1, right);

};

void main () {

    cout << "hi clion" << endl;

};
