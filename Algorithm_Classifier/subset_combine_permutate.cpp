#include <iostream>
#include <vector>

using namespace std;

// 子集问题
// 方法1: 数学归纳法
vector<vector<int>> subsets(vector<int>& nums) {
    // base case，返回一个空集
    if (nums.empty()) return {{}};
    // 取出最后一个元素
    int num = nums.back();
    nums.pop_back();
    vector<vector<int>> res = subsets(nums);
    int size = res.size();
    for(int i = 0; i < size; i++) {
        // 然后在之前的结果之上追加
        res.push_back(res[i]);
        res.back().push_back(num);
    }
    return res;
}

void print_2vec(vector<vector<int>> res) {
    int i = 0;
    cout << res.size() << endl;
    for (auto  nums : res) {
        cout << "the " << ++i << " set is : " << endl;
        if (nums.size() > 0)
        {
            for (auto num : nums) {
                cout << num << '\t';
            }
            cout << endl;
        }
    }
}
// 子集：方法2, 回溯法
vector<vector<int>> res;
void back_track(vector<int>& nums, int start, vector<int>& track) {
    // 前序遍历的位置
    res.push_back(track);
    // 从start 开始，防止产生重复的子集
    for(int i = start; i < nums.size(); i++) {
        track.push_back(nums[i]);
        back_track(nums, i + 1, track);
        track.pop_back();
    }
}
vector<vector<int>> subsets2(vector<int>& nums) {
    // 记录走过的路径
    vector<int> track;
    back_track(nums, 0, track);
    return res;
}


// 组合问题
vector<vector<int>> res;
void backtrack(int n, int k, int start, vector<int>& track) {
    // 到达树的底部
    if(k == track.size()) {
        res.push_back(track);
        return;
    }
    // 注意i 从 start 开始递增
    for(int i = start; i <= n; i++) {
        // 做选择
        track.push_back(i);
        backtrack(n, k, i + 1, track);
        // 撤销选择
        track.pop_back();
    }
}

vector<vector<int>> combine(int n, int k) {
    if(k <= 0 || n <= 0) return res;
    vector<int> track;
    backtrack(n, k, 1, track);
    return res;
}
