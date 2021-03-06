int coinChange(vector<int>& coins, int amount) {
	vector<int> dp(amount + 1, amount + 1);
	// base case 
	dp[0] = 0;
	// 外层for循环遍历所以状态的取值
	for(int i = 0; i < dp.size(); i++) {
		// 内层循环求所有选择的最小值
		for(int coin : coins) {
			// 子问题无解，跳过
			if(i - coin < 0) continue
			dp[i] = min(dp[i], 1 + dp[i - coin]);
		}
	}
	return (dp[amount] == amount + 1) ? -1 : dp[amount];
}



bool isValid(vector<string>& board, int row, int col) {
    int n = board.size();
    // 检查列中是否有皇后互相冲突
    for (int i = 0; i < row; i++) {
        if (board[i][col] == 'Q') 
            return false;
    }
    // 检查右上方是否有皇后互相冲突
    for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q') 
            return false;
    } 
    // 检查左上方是否有皇后互相冲突
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q')
            return false;
    }
    return true;
}
// 路径：board中小于row的那些行都已经成功放置了皇后
// 选择列表：第row行的所有列都是放置皇后的选择
// 结束条件: row超过board的最后一行, 说明棋盘放满了
void backtracking(vector<string>& board, int row) {
    // 触发结束条件
    if (row == board.size()) {
        res.push_back(board);
        return;
    } 
    int n = board[row].size();
    for (int col = 0; col < n; col++) {
        // 排除不合法的选择
        if (!isValid(board, row, col)) {
            continue;
        }
        // 做选择
        board[row][col] = 'Q';
        // 进入下一行选择
        backtracking(board, row + 1);
        // 撤销选择
        board[row][col] = '.';
    }
}
vector<vector<string>> solveNQueens(int n) {
    // '.'表示空, 'Q'表示皇后, 初始化空棋盘
    vector<string> board(n, string(n, '.'));
    backtracking(board, 0);
    return res;
}




/*
 Part1 滑动窗口，解题示例
*/
 // 滑动窗口算法框架
 void slidingWindow(string s, string t) {
 	unordered_map<char, int> need, window;
 	for (char c : t) 
 		need[c]++;

 	int left = 0, right = 0;
 	int valid = 0;
 	while(right < s.size()) {
 		// c 是将移入的字符
 		char c = s[right];
 		// 右移窗口
 		right++;
 		// 进行窗口内数据的一系列更新
 		...
 		printf("window: [%d, %d]\n", left, right);

 		// 判断左侧窗口是否要收缩
 		while（window needs shrink) {
			// d 是将移出的字符
			char d = s[left];
			//左移窗口
			left++;
			// 进行窗口内数据的一系列更新
			...
 		}
 	}
 }


// 示例1：最小覆盖子串，leet 76
// 问题：给你一个字符串S, 一个字符串T，请在字符串S里面找出： 包含T所有字母的最小子串。
/*
	示例：
	输入：s = "ADOBECODEBANC", T = "ABC"
	输出：”BANC“
*/
// 滑动窗口完整解法
string minWindow(string s, string t) {
	unordered_map<char, int> need, window;
	for (auto c : t) 
		need[c]++;
	int left = 0, right = 0;
	int valid = 0;
	// 记录最小覆盖子串的起始索引及长度
	int start = 0, len = INT_MAX;
	while(right < s.size()) {
		char c = s[right];
		right++;
		// 进行窗口的一系列更新
		if (need.count(c)) {
			window[c]++;
			if (window[c] == need[c])
				valid++;
		}

		// 判断是否要收缩
		while(valid == need.size()) {
			if (right - left < len) {
				start = left;
				len = right - left;
			}
			char d = s[left];
			left++;
			if (need.count(d)) {
				if (window[d] == need[d]) {
					valid--;
				}
				window[d]--;
			}
		}
	}
	return len == INT_MAX : "" ? s.substr(start, len);
}


// 示例2：字符串排列， leet 567
/*
 给定两个字符串s1和s2, 写一个函数来判断s2 是否包含s1的全排列
 换句话说，第一个字符串的排列之一是第二个字符串的子串
 示例1：s1 = "ab"  s2 = "eidbaooo"
 输出：True
 解释：s2包含s1的排列之一("ba")
 注意，输入的s1是可以包含重复字符的，所以这个题目难度不小
*/
 // 这道题明显是：滑动窗口算法，相当于给你一个S和T, 请问S中是否存在一个子串，包含T中所有字符且不包含其它字符

// 判断 s 中是否存在 t 的排列
 bool checkInclusion(string s, string t) {
 	unordered_map<char, int> need, window;
 	for (auto ch : t) 
 		need[ch]++;

 	int left = 0, right = 0;
 	int valid = 0;

 	while(right < s.size()) {
 		char c = s[right];
 		right++;
 		if (need.count(c)) {
 			window[c]++;
 			if (need[c] == window[c]) {
 				valid++;
 			}
 		}

 		// 判断是否要收缩
 		if (right - left >= t.size()) {
 			if (valid == need.size()) 
 				return true;
 			char d = s[left];
 			if (need.count(d)) {
 				if (need[c] == window[c]) {
 					valid--;
 				}
 				window[c]--;
 			}
 		}
 	}
 	return false;
 }


// 示例3：找所以字母的异位词 leet 438
/*
 给定一个字符串s和一个非空字符串p，找到s中所有是p的字母异位词的子串，并返回这些子串的起始索引
 字符串只包含小写英文字母，并且字符串s和p的长度都不超过20100
 说明：
 1.字母异位词 指字母相同，但排列不同的字符串
 2.不考虑答案输出的顺序
 
 示例1：
 输入：s="cbaebabacd" p="abc"
 输出：[0, 6]
*/
 // 异曲同工
vector<int> findAnagrams(string s, string t) {
	unordered_map<char, int> need, window;
	for (auto ch : t) 
		need[ch]++;

	int left = 0, right = 0;
	int valid = 0;
	std::vector<int> result;
	while(right < s.size()) {
		char c = s[right];
		right++;
		if (need.count(c)) {
			window[c]++;
			if (need[c] == window[c]) {
				valid++;
			}
		}
		// 判断窗口是否收缩
		if (right - left >= t.size()) {
			if (valid == need.size()) {
				res.push_back(left);
			}
			char d = s[left];
			if (need.count(d)) {
				if (need[d] == window[d]) 
					valid--;
				window[d]--;
			}
		}
	}
	return result;
}

// 示例4：最长无重复子串， leet 3
/*
  给定一个字符串，请你找出其中不含有重复字符的”最长子串“的长度
示例：1
输入：”abcabcbb“
输出：3

示例：2
输入：”pwwkew“
输出：3
*/
int lenghtOfLongestSubstring(string s) {
	unordered_map<char, int> window;
	int left = 0, right = 0;
	int res = 0;
	while(right < s.size()) {
		char c = s[right];
		right++;
		window[c]++;
		// 判断左窗口是否要收缩
		while(window[c] > 1) {
			char d = s[left];
			left++;
			window[d]--;
		}
		// 在这里更新答案
		res = max(res, right - left);
	}
	return res;
}


// 经典动态规划问题
// 打家劫舍 I
/*
 你是一个专业的盗贼，计划偷打劫的房屋，每间房间内藏有一定的现金，影响你的唯一制约因素是：
 相邻的房屋装有互相连通的防盗系统，如果两件相邻的房屋在同一晚上被盗，系统会自动报警。
 给定一个代表每个房屋存放全部的非负整数数组，计算你在不触动警报装置的情况下，能够偷盗的最高金额。
 示例1：
 输入：[1, 2, 3, 1]
 输出：4
 示例2：
 输入：[2, 7, 9, 3, 1]
 输出：12
 解释：偷盗 2， 9， 1
*/

// 方法1：自顶向下的动态规划解法，递归
vector<int> memo;
int dp(vector<int>& nums, int start) {
    if (start >= nums.size())
        return 0;
    if (memo[start]) return memo[start];
    int res = max(dp(nums, start + 1), nums[start] + dp(nums, start + 2));
    memo[start] = res;
    return res;
}
int rob2(vector<int>& nums) {
    int n = nums.size();
    memo.resize(n);
    return dp(nums, 0);
}
// dp[i]表示 前i个房屋能够偷盗的最高金额
int rob3(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    vector<int> dp(n, 0);
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    for (int i = 2; i < n; i++) {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    return dp[n - 1];
}
// 如果不用dp数组呢
int rob4(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    int dp_0 = nums[0];
    int dp_1 = max(nums[0], nums[1]);
    int dp = max(dp_0, dp_1);
    for (int i = 2; i < n; i++) {
        dp = max(dp_1, nums[i] + dp_0);
        dp_0 = dp_1;
        dp_1 = dp;
    } 
    return dp;
}
// 如果采用倒序呢？
// dp[i] = x 表示从第i个房间开始抢劫，能够抢到的最大的金额为x
int rob5(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    vector<int>dp(n + 2, 0);
    // base case dp[n] = 0;
    for (int i = n - 1; i >= 0; i--) {
        dp[i] = max(dp[i + 1], nums[i] + dp[i + 2]);
    }
    return dp[0];
}
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    int dp_i_1 = 0;
    int dp_i_2 = 0;
    int dp = 0;
    for (int i = n - 1; i >= 0; i--) {
        dp = max(dp_i_1, nums[i] + dp_i_2);
        dp_i_2 = dp_i_1;
        dp_i_1 = dp;
    } 
    return dp;
}

// 打家劫舍 II
/*
 你是一个专业的盗贼，计划偷打劫的房屋，每间房间内藏有一定的现金，这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的.
 同时，相邻的房屋装有互相连通的防盗系统，如果两件相邻的房屋在同一晚上被盗，系统会自动报警。
 给定一个代表每个房屋存放全部的非负整数数组，计算你在不触动警报装置的情况下，能够偷盗的最高金额。
 示例1：
 输入：nums = [2,3,2]
 输出：3
 示例2：
 输入：nums = [1,2,3,1]
 输出：4
*/

 // 方法1：
// dp[i] 前i个房间偷盗，能够得到的最大金额
int dp(vector<int>& nums, int start, int end) {
    if (start == end) return nums[start];
    int dp_0 = nums[start];
    int dp_1 = max(nums[start + 1], nums[start]);
    int dp = max(dp_0, dp_1);
    for (int i = start + 2; i <= end; i++) {
        dp = max(dp_1, dp_0 + nums[i]);
        dp_0 = dp_1;
        dp_1 = dp;
    }
    return dp;
}
int rob2(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    return max(dp(nums, 0, n - 2), dp(nums, 1, n - 1));
}
// 方法2：
// dp[i] = x 表示从i开始偷盗，能够得到的最大金额为x
int robRange(vector<int>& nums, int start, int end) {
    int dp_i_1 = 0;
    int dp_i_2 = 0;
    int dp = 0;
    for (int i = end; i >= start; i--) {
        dp = max(dp_i_1, nums[i] + dp_i_2);
        dp_i_2 = dp_i_1;
        dp_i_1 = dp;
    } 
    return dp;
}
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    return max(robRange(nums, 0, n - 2), robRange(nums, 1, n - 1));
}



// 打家劫舍 III
/*
 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 
 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
 计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
 示例1：
 输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.

 示例2：
 输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
*/

unordered_map<TreeNode*, int> memo;
int rob(TreeNode* root) {
    if (root == nullptr) return 0;
    if (memo.count(root)) return memo[root];
    // 抢，或者不抢
    int do_it1 =  (root->left == nullptr) ? 0 : rob(root->left->left) + rob(root->left->right); 
    int do_it2  = (root->right == nullptr) ? 0 : rob(root->right->left) + rob(root->right->right);
    int do_it = root->val + do_it1 + do_it2;
    // 不抢
    int not_do_it = rob(root->left) + rob(root->right);
    int res = max(do_it, not_do_it);
    memo[root] = res;
    return res; 
}   

// Part2 区间问题解法
/*
 1288. 删除被覆盖区间
 给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
 只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。
 在完成所有删除操作后，请你返回列表中剩余区间的数目。

 示例：
 输入：intervals = [[1,4],[3,6],[2,8]]
 输出：2
 解释：区间 [3,6] 被区间 [2,8] 覆盖，所以它被删除了。
*/
static bool cmp(vector<int>& a, vector<int>& b) {
    if (a[0] == b[0]) {
        return a[1] > b[1];
    }
    return a[0] < b[0];
}
int removeCoveredIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(), cmp);
    int remove = 0;
    int maxRight = intervals[0][1];
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][1] <= maxRight) {
            remove++;
        } else {
            maxRight = intervals[i][1];
        }
    }
    return intervals.size() - remove;
}


// leet 56 合并区间
// 给出一个区间的集合，请合并所有重叠的区间。
/*
示例 1:
输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
*/

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> result;
    for (auto interval : intervals) {
        int left = interval[0];
        int right = interval[1];
        if (result.empty() || result.back()[1] < left) {
            result.push_back({left, right});
        } else {
            result.back()[1] = max(right, result.back()[1]);
        }
    }
    return result;
}

// python 解法
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0]);
    result = []
    for interval in intervals:
        left = interval[0];
        right = interval[1];
        if not result or result[-1][1] < left:
            result.append([left, right])
        else:
            result[-1][1] = max(result[-1][1], right)
    
    return result



// 例题3，leet 986. 区间列表的交集
/*
 给定两个由一些闭区间组成的列表，每个区间列表都是成对不相交的，并且已经排序。
 返回这两个区间列表的交集。
 形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。
 两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。
 
 示例1:
 输入：A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
 输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
 
 示例2：
 输入：A = [[1,7]], B = [[3,10]]
 输出：[[3,7]]

*/

vector<vector<int>> intervalIntersection(vector<vector<int>>& A, vector<vector<int>>& B) {
    int i = 0, j = 0;
    vector<vector<int>> result;
    while(i < A.size() && j < B.size()) {
        int a1 = A[i][0], a2 = A[i][1];
        int b1 = B[j][0], b2 = B[j][1];
        if (a2 >= b1 && a1 <= b2) {
            int c1 = max(a1, b1);
            int c2 = min(a2, b2);
            result.push_back({c1, c2});
        }
        if (b2 >= a2) {
            i += 1;
        } else {
            j += 1;
        }
    }
    return result;
}


def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    res = []
    i, j = 0, 0
    while(i < len(A) and j < len(B)):
        a1, a2 = A[i][0], A[i][1]
        b1, b2 = B[j][0], B[j][1]
        if a2 >= b1 and a1 <= b2:
            c1 = max(a1, b1)
            c2 = min(a2, b2)
            res.append([c1, c2])
        if b2 > a2:
            i = i + 1
        else:
            j = j + 1

    return res;


// Part3 团灭 2sum, 3sum, 4sum, nsum问题
// leetcode 题目：two sum问题，返回给定数组两元素之和为target的元素
vector<int> twoSum2(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        // 根据sum和target的比较，移动左右指针
        if (sum < target) {
            left++;
        } else if (sum > target) {
            right--;
        } else {
            return {nums[left], nums[right]};
        }
    }
    return {};
}
// leetcode twosum问题进阶，nums中可能有多对元素之和等于target，请你的算法返回
// 所有和为target的元素对，其中不能出现重复
vector<vector<int>> twoSumTarget(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    int left = 0, right = nums.size() - 1;
    vector<vector<int>> res;
    while (left < right) {
        int sum = nums[left] + nums[right];
        int left_num = nums[left], right_num = nums[right];
        if (sum < target) {
            while (left < right && nums[left] == left_num) left++;
        }
        else if (sum > target) {
            while (left < right && nums[right] == right_num) right--;
        }
        else {
            res.push_back({left_num, right_num});
            while(left < right && nums[left] == left_num) left++;
            while(left < right && nums[right] == right_num) right--;
        }
    }
    return res;
}
// 增加一个start索引
vector<vector<int>> twoSumTarget(vector<int>& nums, int start, int target) {
    sort(nums.begin(), nums.end());
    int left = start, right = nums.size() - 1;
    vector<vector<int>> res;
    while (left < right) {
        int sum = nums[left] + nums[right];
        int left_num = nums[left], right_num = nums[right];
        if (sum < target) {
            while (left < right && nums[left] == left_num) left++;
        }
        else if (sum > target) {
            while (left < right && nums[right] == right_num) right--;
        }
        else {
            res.push_back({left_num, right_num});
            while(left < right && nums[left] == left_num) left++;
            while(left < right && nums[right] == right_num) right--;
        }
    }
    return res;
}

// leetcode 15， 3sum问题；
// 给定一个包含n个整数的数组nums, 判断nums中是否存在三个元素a,b,c 使得a + b + c = 0?
// 请找出所有满足条件，且不重复的三元组
vector<vector<int>> threeSum1(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    // 找出 a + b + c = 0
    // a = nums[i], b = nums[left], c = nums[right]
    for (int i = 0; i < nums.size(); i++) {
        // 排序之后，如果第一个元素大于0，那么组合肯定不成立
        if (nums[i] > 0) continue;
        // 正确去重方法
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        int left = i + 1;
        int right = nums.size() - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum < 0) left++;
            else if (sum > 0) right--;
            else {
                result.push_back({nums[i], nums[left], nums[right]});
                // 去重逻辑应该放在找到第一个三元组后
                while(left < right && nums[left] == nums[left + 1]) left++;
                while(left < right && nums[right] == nums[right - 1]) right--;
                // 找到答案时，双指针同时收缩
                left++;
                right--;
            }
        }
    }
    return result;
}

// leetcode 15， 3sum问题；dong哥解法
vector<vector<int>> threeSumTarget(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int>> res;
    // 穷举 threeSum 的第一个数
    for (int i = 0; i < n; i++) {
        // 对target - nums[i] 计算twosum
        vector<vector<int>> tuples = twoSumTarget(nums, i + 1, target - nums[i]);
        // 如果存在满足条件的二元组，再加上nums[i]就是结果三元组
        for (auto tuple : tuples) {
            tuple.push_back(nums[i]);
            res.push_back(tuple);
        }
        // 跳过第一个数字重复的情况，否则会出现重复结果
        while (i < n - 1 && nums[i] == nums[i + 1]) i++;
    }
    return res;
}

vector<vector<int>> threeSumTarget2(vector<int>& nums, int start, int target) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int>> res;
    // 穷举 threeSum 的第一个数
    for (int i = start; i < n; i++) {
        // 对target - nums[i] 计算twosum
        vector<vector<int>> tuples = twoSumTarget(nums, i + 1, target - nums[i]);
        // 如果存在满足条件的二元组，再加上nums[i]就是结果三元组
        for (auto tuple : tuples) {
            tuple.push_back(nums[i]);
            res.push_back(tuple);
        }
        // 跳过第一个数字重复的情况，否则会出现重复结果
        while (i < n - 1 && nums[i] == nums[i + 1]) i++;
    }
    return res;
}

// leetcode 18 四数之和
// 给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，
// 使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
vector<vector<int>> fourSum(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int>> res;
    // 穷举fourSum 的第一个数
    for (int i = 0; i < n; i++) {
        vector<vector<int>> triples = threeSumTarget2(nums, i + 1, target - nums[i]);
        for (auto triple : triples) {
            triple.push_back(nums[i]);
            res.push_back(triple);
        }
        while(i < n - 1 && nums[i] == nums[i + 1]) i++;
    }
    return res;

}

// leetcode 100sum, nsum问题
// 当n == 2时 是twosum的双指针解法，n > 2就是穷举第一个数字，然后递归的调用(n-1)Sum，组装答案
vector<vector<int>> nSumTarget(vector<int>& nums, int n, int start, int target) {
    int sz = nums.size();
    vector<vector<int>> res;
    // 至少是2sum，且数组大小不应该小于n
    if (n < 2 || sz < n) return res;
    // base case 为2sum
    if (n == 2) {
        int left = start, right = sz - 1;
        while (left < right) {
            int sum = nums[left] + nums[right];
            int left_num = nums[left], right_num = nums[right];
            if (sum < target) {
                while(left < right && nums[left] == left_num) left++;
            } else if (sum > target) {
                while(left < right && nums[right] == right_num) right--;
            } else {
                res.push_back({left_num, right_num});
                while(left < right && nums[left] == left_num) left++;
                while(left < right && nums[right] == right_num) right--;
            }
        }
    } else {
        // n > 2 时递归计算 (n-1)Sum结果
        for (int i = start; i < sz; i++) {
            vector<vector<int>> sub = nSumTarget(nums, n - 1, i + 1, target - nums[i]);
            for(auto arr : sub) {
                // (n-1)Sum 加上 nums【i】就是nSum
                arr.push_back(nums[i]);
                res.push_back(arr);
            }
            while(i < sz - 1 && nums[i] == nums[i + 1]) i++;
        }
    }
    return res;
}
vector<vector<int>> fourSum2(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    return nSumTarget(nums, 4, 0, target);
}


// Part4: 股票买卖问题
// 121. 买卖股票的最佳时机
/*
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

示例 1：
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票
*/
// 方法1：动态规划
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    vector<vector<int>> dp(n, vector<int>(2, 0));
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            dp[i][1] = -prices[i];
            dp[i][0] = 0;
        } else {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = max(dp[i - 1][1], - prices[i]); // 这里是-prices[i], 因为k只有1次
        }
    }
    return dp[n - 1][0];
}
// 方法2：动态规划，因为状态转移方程，新状态只和相邻的一个状态有关
// 其实不需要dp数组，只需要一个变量存储相邻的那个状态就足够了，空间复杂度可降低到O(1)
int maxProfit2(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    // base case
    int dp_i_0 = 0;
    int dp_i_1 = INT_MIN;
    for (int i = 0; i < n; i++) {
        // dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
        // dp[i][1] = max(dp[i - 1][1], - prices[i]);
        dp_i_1 = max(dp_i_1, -prices[i]);
    } 
    return dp_i_0;
}


// 122. 买卖股票的最佳时机 II
/*
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:

输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
*/

// 股票交易第二题，k = +infinity, 动态规划
// 如果k为正无穷，那么可以认为k和k-1是一样的
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
//             = max(dp[i-1][k][1], dp[i-1][k][0] - prices[i])
// 发现数组中的k 已经不会改变了，也就是不需要记录k 这个状态了
// dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
// dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    // base case
    int dp_i_0 = 0;
    int dp_i_1 = INT_MIN;
    for (int i = 0; i < n; i++) {
        int tmp = dp_i_0;
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = max(dp_i_1, tmp - prices[i]);
    }
    return dp_i_0;
}


// 123. 买卖股票的最佳时机 III
/*
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:

输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
*/

// 股票交易第五题，k = 2；这个题和前面的题有所不同，前面的题基本和k无关
// 原始的动态转移方程，没有可化简的地方
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
// 按照之前的写法，是错误的
/***
int k = 2;
int[][][] dp = new int[n][k + 1][2];
for (int i = 0; i < n; i++)
    if (i - 1 == -1) {  处理一下 base case }
    dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
    dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
}
return dp[n - 1][k][0];
***/
// 错误原因，主要是没有消掉k的影响，所以必须对k进行穷举；
// base case：
// dp[-1][k][0] = dp[i][0][0] = 0
// dp[-1][k][1] = dp[i][0][1] = -infinity
// 状态转移方程：
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])

int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    int max_k = 2;
    vector<vector<vector<long>>> dp(n, vector<vector<long>>(max_k + 1, vector<long>(2, 0)));
    for (int i = 0; i < n; i++) {
        for (int k = max_k; k >= 1; k--) {
            if (i == 0) {
                dp[i][k][0] = 0;
                dp[i][k][1] = INT_MIN;
                dp[i][0][0] = 0;
                dp[i][0][1] = INT_MIN;
                continue;
            }
            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
            dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
        }
    }
    return dp[n-1][max_k][0];
}


// 188. 买卖股票的最佳时机 IV
/*
给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1：
输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
*/

// 股票交易第六题，k 为任意整数；这个题和前面的题有所不同，前面的题基本和k无关
// 原始的动态转移方程，没有可化简的地方
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
// 按照之前的写法，是错误的
/***
int k = 2;
int[][][] dp = new int[n][k + 1][2];
for (int i = 0; i < n; i++)
    if (i - 1 == -1) {  处理一下 base case }
    dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
    dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
}
return dp[n - 1][k][0];
***/
// 错误原因，主要是没有消掉k的影响，所以必须对k进行穷举；
// base case：
// dp[-1][k][0] = dp[i][0][0] = 0
// dp[-1][k][1] = dp[i][0][1] = -infinity
// 状态转移方程：
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
// 这个代码不是所以的case 都能通过，需要再看下

int maxProfit_k_inf(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    // base case
    int dp_i_0 = 0;
    int dp_i_1 = INT_MIN;
    for (int i = 0; i < n; i++) {
        int tmp = dp_i_0;
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = max(dp_i_1, tmp - prices[i]);
    }
    return dp_i_0;
}

int maxProfit(int k, vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    int max_k = k;
    if (max_k > n / 2)
        return maxProfit_k_inf(prices);
    vector<vector<vector<long>>> dp(n, vector<vector<long>>(max_k + 1, vector<long>(2, 0)));
    for (int i = 0; i < n; i++) {
        for (int k = max_k; k >= 1; k--) {
            if (i == 0) {
                dp[i][k][0] = 0;
                dp[i][k][1] = INT_MIN;
                dp[i][0][0] = 0;
                dp[i][0][1] = INT_MIN;
                continue;
            }
            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
            dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
        }
    }
    return dp[n-1][max_k][0];
}

// 309. 最佳买卖股票时机含冷冻期
/*
给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

示例:
输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
*/
// 股票交易第三题，k = +infinity with cooldown
// 每次sell 之后，需要等一天才能继续交易，只需要把这个特点融入上一题的状态转移方程即可；
// dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
// dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
// 解释：第i天选择 buy的时候，需要从i-2的状态转移，而不是 i-1;
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;
    // base case
    int dp_i_0 = 0, dp_i_1 = INT_MIN;
    int dp_pre_0 = 0;  // 代表dp[i-2][0]
    for (int i = 0; i < n; i++) {
        int tmp = dp_i_0;
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = max(dp_i_1, dp_pre_0 - prices[i]);
        dp_pre_0 = tmp;
    }
    return dp_i_0;
}

// 714. 买卖股票的最佳时机含手续费
/*
给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。
你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
返回获得利润的最大值。
注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

示例 1:
输入: prices = [1, 3, 2, 8, 4, 9], fee = 2
输出: 8
解释: 能够达到的最大利润:  
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
*/

// 股票交易第四题，k =+infinity with fee
// dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
// dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i] - fee)
int maxProfit(vector<int>& prices, int fee) {
    int n = prices.size();
    if (n < 2) return 0;
    // base case
    long dp_i_0 = 0, dp_i_1 = INT_MIN;
    for (int i = 0; i < n; i++) {
        long tmp = dp_i_0;
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i] - fee);
        dp_i_1 = max(dp_i_1, tmp - prices[i]);
    }
    return dp_i_0;
}



// Part 5 二叉树系列
// 递归思维练习
// (系列1)

// 二叉树遍历的框架
void traverse(TreeNode* root) {
	// 前序遍历(这里的代码)
	traverse(root->left);
	// 中序遍历(这里的代码)
	traverse(root->right);
	// 后序遍历(这里的代码)
}

// 快速排序是二叉树的 前序遍历框架？ 要理解
void sort(vector<int>& nums, int low, int high) {
	/*********前序遍历位置**********/
	// 通过交换元素，构建分界点p
	int p = partition(nums, low, high);
	/*******************************/

	sort(nums, low, p - 1);
	sort(nums, p + 1, high);
}

// 归并排序是二叉树的后续遍历框架
void sort(vector<int>& nums， int low, int high) {
	int mid = low + (high - low) / 2;
	sort(nums, low, mid);
	sort(nums, mid + 1, high);

	/********二叉树的后续遍历位置********/
	// 合并两个排序好的子数组
	merge(nums, low, mid, high);
}

// 写递归算法的技巧
// 非常重要的点
// 关键在于要明确函数的定义是什么，然后相信这个定义，利用这个定义推导出最终结果，绝不要试图跳入递归。

// 示例1：计算一颗二叉树共有几个节点？
// 定义：count(root)返回以root为根的树有多少节点
int count(TreeNode* root) {
	// base case
	if (root == nullptr) return 0;
	// 自己加上，子树的节点，就是整棵树的节点
	return 1 + count(root->left) + count(root->right);
}

// 示例2：翻转二叉树
TreeNode* invertTree(TreeNode* root) {
	// base case
	if (root == nullptr) return nullptr;

	// 前序遍历位置
	// root 节点要交换它的左右子树
	TreeNode* tmp = root->left;
	root->left = root->right;
	root->right = tmp;

	// 让左右子节点 继续翻转它们的子节点
	invertTree(root->left);
	invertTree(root->right);

	return root;
}


// 示例3：填充二叉树节点的右侧指针
/* leet 116, 给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。
   二叉树定义如下：
   struct Node {
	int val;
	Node* left;
	Node* right;
	Node* next;
   } 
   请填充它的每一个next指针，让这个指针指向其下一个右侧节点，如果找不到下一个右侧节点，则设置为NULL
*/

// 主函数
Node* connect(Node* root) {
	if (root == nullptr) return nullptr;

	return root;
} 

//定义：输入两个节点，将它俩连接起来；
void connectTwoNode(Node* node1, Node* node2) {
	if (node1 == nullptr || node2 == nullptr) 
		return;

	// 前序遍历位置
	// 将传入的两个节点连接
	node1->next = node2;

	// 连接相同父节点的两个子节点
	connectTwoNode(node1->left, node1->right);
	connectTwoNode(node2->left, node2->right);

	// 连接跨越父节点的两个子节点
	connectTwoNode(node1->right, node2->left);
}

// 示例4：将二叉树展开为链表
// 给定一个二叉树，原地将它展开为一个单链表。
/*
  例如，给定二叉树

    1
   / \
  2   5
 / \   \
3   4   6

将其展开为：
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
*/

// 定义：将以root为根的树拉平为链表
void flatten(TreeNode* root) {
	// base case
	if (root == nullptr) return;
	flatten(root->left);
	flatten(root->right);

	// 后续遍历位置
	// 1.左子树已经被拉平为一条链表
	TreeNode* left = root->left;
	TreeNode* right = root->right;
	
	// 2.将左子树作为右子树
	root->left = nullptr;
	root->right = left;

	// 将原来的右子树 拼接到当前右子树的末端
	TreeNode* p = root;
	while(p->right != nullptr) {
		p = p->right;
	}
	p->right = right;
}
// this is 递归的魅力，


// 二叉树递归练习
// (系列2)


// 示例1：leet 654. 最大二叉树
/*给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。
*/

/* 将nums[low, high]构造成符合条件的树，返回根节点 */
TreeNode* build(vector<int>& nums, int left, int right) {
    // base case
    if (left > right) return nullptr;
    
    //找到数组中的最大值和对应的索引
    int index = -1, maxVal = INT_MIN;
    for (int i = left; i <= right; i++) {
        if (maxVal < nums[i]) {
            index = i;
            maxVal = nums[i];
        }
    }
    TreeNode* root = new TreeNode(maxVal);
    // 递归调用构造左右子树
    root->left = build(nums, left, index - 1);
    root->right = build(nums, index + 1, right);
    return root;
}
TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
    return build(nums, 0, nums.size() - 1);
}

// 示例2：leet 105, 从前序和中序遍历序列构造二叉树
// 主函数
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
	return build(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}

/*
 若前序遍历数组为 preorder[preStart, preEnd],
 中序遍历数组为 inorder[inStart, inEnd]
 构造二叉树，返回该二叉树的根节点
*/
TreeNode* build(vector<int>& preorder, int preStart, int preEnd, vector<int>& inorder, int inStart, int inEnd) {
	// root 节点对应的值就是前序遍历数组的第一个元素
	int rootVal = preorder[preStart];
	// rootVal 在中序数组中的索引
	int index = 0;
	for (int i = inStart; i < inEnd; i++) {
		if (inorder[i] == rootVal) {
			index = i;
			break;
		}
	}
	int left_size = index - inStart;
	// 先构造出当前根节点
	TreeNode* root = new TreeNode(rootVal);
	root->left = build(preorder, preStart + 1, preStart + left_size, inorder, inStart, index - 1);
	root->right = build(preorder, preStart + left_size + 1, preEnd, inorder, index + 1, inEnd);
	return root;
}


// 示例3： leet 106, 从中序与后序遍历序列狗仔二叉树
// 主函数
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
	return build(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
}

TreeNode* build(vector<int>& inorder, int inStart, int inEnd, vector<int>& postorder, int postStart, int postEnd) {
	if (inStart > inEnd) return nullptr;
	// root 节点对应的值，就是后序遍历数组的最后一个元素
	int rootVal = postorder[postEnd];
	// rootVal 在中序遍历数组中的索引
	int index = 0；
	for (int i = inStart; i <= inEnd; i++) {
		if (inorder[i] == rootVal) {
			index = i;
			break;
		}
	}
	// 左子树的节点个数
	int left_size = index - inStart;
	TreeNode* root = new TreeNode(rootVal);
	// 递归狗仔左右子树
	root->left = build(inorder, inStart, index - 1, postorder, postStart, postStart + left_size - 1);
	root->right = build(inorder, index + 1, inEnd, postorder, postStart + left_size, postEnd - 1);
	return root;
}



// 二叉树递归练习
// (系列3)
// leet 652 寻找重复子树
/*
给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。
两棵树重复是指它们具有相同的结构以及相同的结点值。

示例 1：

        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
下面是两个重复的子树：

      2
     /
    4
和

    4

*/

// 记录所有子树以及出现的次数
unordered_map<string, int> memo;
//记录重复的子树根节点
vector<TreeNode*> res;
// 主函数
vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
    traverse(root);
    return res;
}
/* 辅助函数 */
string traverse(TreeNode* root) {
    if (root == nullptr) 
        return "#";

    string left = traverse(root->left);
    string right = traverse(root->right);

    string subTree = left + "," + right + "," + to_string(root->val);
    int freq = memo[subTree];
    // 多次重复也只会被加入结果集一次
    if (freq == 1) {
        res.push_back(root);
    }
    // 给子树对应的次数加1
    memo[subTree] = freq + 1;
    return subTree;
}


// 二叉搜索递归练习
// (系列4) (Binary Search Tree BFS)
// BST的特性
/*
 1.对于 BST的每一个节点node, 左子树节点的值都比node的值要小，右子树节点的值都比node的值大
 2.对于BST的每个节点，它的左侧子树和右侧子树都是BST
 二叉搜索树并不算复杂，但是它构建起了数据结构领域的半壁江山，直接基于BST的数据结构有：
  AVL树，红黑树等等，拥有了自平衡性，可以提供logN级别的增删查改效率，
  还有B+树，线段树等结构都是基于BST的思想来设计的。
*/
 // 从做算法题的角度来看BST，除了它的定义，还有一个重要的性质：BST的中序遍历结果是有序的。
  void traverse(TreeNode* root) {
  	if (root == nullptr) return;
  	traverse(root->left);
  	// 中序遍历代码位置
  	print(root->val);
  	traverse(root->right);
  }

// 示例1：leet 230, 寻找第K小的元素
/*
给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。

说明：
你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。
示例 1:

输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 1
*/
// BST中序遍历
int kthSmallest(TreeNode* root, int k) {
	// 利用 BST的中序遍历提升
	traverse(root, k);
	return res;
}

int res = 0;
int rank = 0;
void traverse(TreeNode* root, int k) {
	if (root == nullptr) return;
	traverse(root->left, k);
	/* 中序遍历代码位置 */
	rank++;
	if (k == rank) {
		// 找到第k小元素
		res = root->val;
		return;
	}
	/********************/
	traverse(root->right, k);

}


// 例子2：BST 转化累加树
// leet 538
/*
给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。
*/
// 递归降序打印节点
void traverse(TreeNode* root) {
	if (root == nullptr) return;
	traverse(root->right);
	print(root->val);
	traverse(root->left);
} 

// 主函数
int sum = 0;
TreeNode* covertBST(TreeNode* root) {
	if (root == nullptr) return;
	traverse(root->right);
	// 维护累加和
	sum += root->val;
	// 将BST转化成累加树
	root->val = sum;
	traverse(root->left);
}

// 例子3：判断BTS的合法性
bool isValidBST(TreeNode* root) {
	return isValidBST(root, NULL, NULL);
}
/* 限定以 root 为根的子树节点必须满足max->val > root->var > min->val */
bool isValidBST(TreeNode* root, TreeNode* min, TreeNode* max) {
	// base case
	if (root == nullptr) return true;
	// 如果 root->val 不符合 max 和 min的限制，说明不是合法的BST
	if (min != nullptr && root->val <= min->val) return false;
	if (max != nullptr && root->val >= max_>val) return false;
	// 限定左子树的最大值为root->val, 右子树的最小值为 root->val
	return isValidBST(root->left, min, root) && isValidBST(root->right, root, max);
}
// 通过辅助函数，增加函数参数列表，在参数中携带额外信息，将这种约束传递给子树的所有节点，
// 这也是二叉树算法的一个小技巧吧。

// 例子4：在BST中搜索一个树
// 如果在二叉树中寻找元素，可以这样写代码：
bool isInBST(TreeNode* root, int target) {
	if (root == NULL) return false;
	if (root->val == target) return true;
	// 当前节点没找到，就递归地去左右子树寻找
	return isInBST(root->left, target) || isInBST(root->right, target);
	// 这样写完全正确，但是这段代码，相当于穷举了所有的节点，并没有用到BST这个特性。
}
// 我们稍微改动一下
bool isInBST(TreeNode* root, int target) {
	if (root == NULL) return false;
	if (root->val == target) return true;
	if (root->val < target) 
		return isInBST(root->right, target);
	if (root->val > target) 
		return isInBST(root->left, target);
}

// 重要的框架
// 重要的框架
// 重要的框架
// 于是，我们对原始框架改造，抽象出一套针对BST的遍历框架：
void BST(TreeNode* root, int target) {
	if (root->val == target)
		// 找到目标 做点什么
	if (root->val < target) {
		BST(root->right, target);
	}
	if (root->val > target) {
		BST(root->left, target);
	}
}

// 示例5： 在BST中插入一个数
// 对数据结构的操作无非就是 遍历+访问，遍历就是找，访问就是改。
// 具体到这个问题，插入一个数，就是先找到插入位置，然后进行插入操作。
// 套用上面的BST框架 
// 一旦涉及 改，函数要返回TreeNode 类型，并且对递归调用的返回值进行接收
TreeNode* insertIntoBST(TreeNode* root, int val) {
	// 找到空位置 插入新节点
	if(root == nullptr) return new TreeNode(val);
	if (root->val < val)
		root->right = insertIntoBST(root->right, val);
	if (root->val > val) 
		root->left = insertIntoBST(root->left, val);
	return root;
}

// 示例6：在BST中删除一个数
// 这个问题稍微复杂，跟插入操作类似，先找 再改，先把框架写出来再说。
/* 分三种情况
  情况1：如果删除节点A 是末端节点，那么当场可以删去了。
  情况2：如果A只有一个非空子节点，纳闷它要让这个孩子接替自己的位置
  情况3：A有两个子节点，为了不破坏BST性质，A必须找到左子树中最大的那个节点
        或者右子树中最小的囊额节点，来接替自己。
*/
TreeNode* deleteNode(TreeNode* root, int key) {
	if (root == NULL) return NULL;
	if (root->val == key) {
		// 这两个 if 把情况1 和 2 都正确处理了
		if (root->left == NULL) return root->right;
		if (root->right == NULL) return root->left;
		// 找到右子树最小的 节点
		TreeNode* minNode = getMin(root->right);
		root->val = minNode->val;
		root->right = deleteNode(root->right, minNode->val);

	} else if (root->val > key) {
		root->left = deleteNode(root->left, key);
	} else if (root->val < key) {
		root->right = deleteNode(root->right, key);
	}
	return root;
}

TreeNode* getMin(TreeNode* node) {
	// BST 最左边的就是最小值
	while(node->left != NULL) 
		node = node->left;
	return node;
}


// 二叉搜序列化练习
// (系列5) 序列化 序列化
// 假设有一颗用Java实现的二叉树，我想把它序列化字符串，
// 然后用C++ 读取这颗并还原这颗二叉树的结构，怎么办？
// 这就需要对二叉树进行 序列化 和 反序列化操作了

// 示例1：leet 297 [二叉树的序列化与反序列化]
// 描述：给你输入一颗二叉树的根节点 root, 要求你事先如下一个类：

class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        
    }
};
// 所谓序列化，不过就是把结构化的数据【打平】 其实就是考察二叉树的遍历方法
// 二叉树的遍历方法
// 递归遍历方法：前序、中序、后序；
// 迭代遍历方法：一般是层次遍历

// 代表分隔符的字符
string sep = " ";
// 代表 null 空指针的字符
string null = "#";
// 用于 拼接字符串
ostringstream out;


// 首先是前序遍历解法
// 主函数，将二叉树序列化为字符串
string serialize(TreeNode* root) {
	if (root == NULL) return "";
	ostringstream out;
	serialize(root, out);
	return out.str();
}

/* 辅助函数, 将二叉树存入 ostringstream */
void serialize(TreeNode* root, ostringstream& out) {
	if (root == NULL) {
		out << "# ";
		return;
	}
	/* 前序遍历位置 */
	out << to_string(root->val) << sep;
	/* 前序遍历结束 */

	serialize(root->left, out);
	serialize(root->right, out);
}

/* 主函数，将字符串反序列为二叉树结构 */
TreeNode* deserialize(string data) {
	if (data.empty()) return nullptr;

	istringstream in(data);
	return deserialize(in);
}

TreeNode* deserialize(istringstream& in) {
	string val;
	in >> val;
	if (val == "#") return nullptr;
	/*前序遍历位置*/
	auto root= new TreeNode(stoi(val));
	root->left = deserialize(in);
	root->right = deserialize(in);
	return root;
} 


// 后续遍历解法
void serialize(TreeNode* root, ostringstream& out) {
	if (root == nullptr) {
		out << "#" << sep;
	}
	serialize(root->left, out);
	serialize(root->right, out);

	/* 后续遍历位置 */
	out << to_string(root->val) << sep;
	/* 后续遍历结束 */
} 

// 反序列化，我们采用java实现，后续可以改为c++
TreeNode deserialize(String data) {
	LinkedList<String> nodes = new LinkedList<>():
	for (String s : data.split(sep)) {
		nodes.addLast(s);
	}
	return deserialize(nodes);
}
/* 辅助函数，通过nodes 列表构造二叉树 */
TreeNode deserialize(LinkedList<String> nodes) {
	if (nodes.isEmpty()) return null;
	// 从后往前取出元素
	String last = nodes.removeLast();
	if (last.equals(NULL)) return null;
	TreeNode root = new TreeNode(Integer.parseInt(last));
	// 先构造右子树，后构造左子树
	root.right = deserialize(nodes);
	root.left = deserialize(nodes);

	return root;
}

// 中序遍历解法，无法实现反序列化，原因是无法确定根节点

// 层侧遍历解法
// 层次遍历框架如下：
void traverse(TreeNode* root) {
	if (root == nullptr) return;
	// 初始化队列，将root加入队列
	queue<TreeNode*> q;
	q.push(root);

	while(!q.empty()) {
		TreeNode* cur = q.front();
		q.pop();
		/* 层次遍历代码位置 */
		cout << root->val << endl;
		/* ************** */
		if (cul->left != nullptr) {
			q.push(cur->left);
		}
		if (cur->right != null) {
			q.push(cur->right);
		}
	}
}

// 那么完整的层次遍历 实现序列化和反序列化如下
string sep = " ";
/* 将二叉树序列化为字符串 */
string serialize(TreeNode* root) {
	if (root == null) return "";
	ostringstream out;
	queue<TreeNode*> q;
	q.push(root);

	while(!q.empty()) {
		TreeNode* cur = q.front();
		q.pop();
		/* 层次遍历代码位置 */
		if (cur == nullptr) {
			out << "#" << sep;
			continue;
		} else {
			out << cur->val << sep;
			q.push(cur->left);
			q.push(cur->right);
		}
	}
	return out.str();
}

// 反序列化 写一个Java实现的版本
/* 将字符串反序列化为二叉树结构 */
TreeNode deserialize(String data) {
	if (data.isEmpty()) return null;
	String[] nodes = data.split(sep);
	// 第一个元素就是 root的值
	TreeNode root = new TreeNode(Integer.parseInt(nodes[0]));

	// 队列q 记录父节点，将root加入队列
	Query<TreeNode> q = new LinkedList<>();
	q.offer(root);

	for (int i = 1; i < nodes.length; ) {
		// 队列中的都是父节点
		TreeNode parent = q.poll();
		// 父节点对应的左侧子节点的值
		String left = nodes[i++];
		if (!left.equals(NULL)) {
			parent.left = new TreeNode(Integer.parseInt(left));
			q.offer(parent.left);
		} else {
			parent.left = null;
		}

		// 父节点对应的右侧子节点的值
		String right = nodes[i++]; 
		if (!right.equals(NULL)) {
			parent.right = new TreeNode(Integer.parseInt(right));
			q.offer(parent.right);
		} else {
			parent.right = null;
		}
	}
	return root;
}



// 二叉树练习
// (系列6) 嵌套列表迭代器 Leet 341
/*
给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。
列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。
示例 1:

输入: [[1,1],2,[1,1]]
输出: [1,1,2,1,1]
解释: 通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,1,2,1,1]。

示例 2:
输入: [1,[4,[6]]]
输出: [1,4,6]
解释: 通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,4,6]。
*/

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */
// Java 解法
public class NestedIterator implements Iterator<Integer> {

	private Integer<Integer> it;
    public NestedIterator(List<NestedInteger> nestedList) {
        // 存放 将 nestedList 打平的结果
        List<Integer> result = new LinkedList<>();
        for (NestedInteger node : nestedList) {
        	// 以每个节点为根 遍历
        	traverse(node, result);
        }
        // 得到 result 列表的迭代器
        this.it = result.iterator();
    }

    @Override
    public Integer next() {
        return it.next();
    }

    @Override
    public boolean hasNext() {
        return it.hasNext();
    }
    // 遍历以 root 为根的多叉树，将叶子节点的值 加入到result列表中
    private void traverse(NestedInteger root, List<Integer> result) {
    	if (root.isInteger()) {
    		// 达到叶子节点
    		result.add(root.getInteger());
    		return;
    	}
    	// 遍历框架
    	for (NestedInteger child : root.getList()) {
    		traverse(child, result);
    	}
     }
}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */

// 另外一种 比较好的做法，C++实现
// 原理：1. 时间均衡的栈
// 构造时仅仅扒一层皮就 逆向 堆入栈中，在用户调用 hasNext 时才做深入扒皮搜索。
// 这种做法比较时间均衡，如果用户搞了一个很长的列表，然后就取前边几个元素就不用了，那这种实现要高效的多。

class NestedIterator {
private: 
	stack<NestedInteger> st;
public:
    NestedIterator(vector<NestedInteger> &nestedList) {
        for(auto iter = nestedList.rbegin(); iter != nestedList.rend(); iter++) {
        	st.push(*iter);
        }
    }
    
    int next() {
    	auto t = st.top();
    	st.pop();
        return t.getInteger();
    }
    
    bool hasNext() {
        while(!st.empty()) {
        	auto cur = st.top();
        	if (cur.isInteger()) return true;
        	st.pop();
        	auto curList = cur.getList();
        	for (auto iter = curList.rbegin(); iter != curList.rend(); iter++) {
        		st.push(*iter);
        	}
        }
        return false;
    }
};




// 二叉树练习
// (系列7) 二叉树的公共祖先 leet 236 

/*给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，
最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]

示例 1:
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。

示例 2:
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
*/

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (root == nullptr) return nullptr;

	if (root == p || root == q) return root;

	TreeNode* left = lowestCommonAncestor(root->left, p, q);

	TreeNode* right = lowestCommonAncestor(root->right, p, q);

	if (left != nullptr && right != nullptr) 
		return root;
	if (left == nullptr && right == nullptr) 
		return nullptr;
	return (left == nullptr) ? right : left;
}



// 二叉树练习
// (系列8) 完全二叉树和满二叉树 计算节点数
// 普通二叉树 计算节点数
// 时间复杂度 O(N)
int conutNodes(TreeNode* root) {
	if (root == nullptr) 
		return 0;
	return 1 + conutNodes(root->left) + conutNodes(root->right);
}

// 满二叉树 计算节点数
// 时间复杂度 O(logN)
int conutNodes(TreeNode* root) {
	int height = 0;
	while(root != nullptr) {
		root = root->left;
		height++;
	}
	return int(pow(2, height)) - 1;
}


// 完全二叉树 计算节点数
// 时间复杂度 O(logN*logN)
// 原因：return 1 + conutNodes(root->left) + conutNodes(root->right);
// 这两个递归只有一个 会真的递归下去，另一个一定会触发 left_height = right_height 而立即返回
// 不会递归下去的
int conutNodes(TreeNode* root) {
	TreeNode* left_node = root; 
	TreeNode* right_node = root;
	int left_height = 0, right_height = 0;

	while(left_node != nullptr) {
		left_node = left_node->left;
		left_height++;
	}

	while(right_node != nullptr) {
		right_node = right_node->right;
		right_height++;
	} 

	if (left_height == right_height) {
		return int(pow(2, left_height)) - 1;
	}

	return 1 + conutNodes(root->left) + conutNodes(root->right);
}



// 章节1 经典动态规划问题
// Part1 动态规划基本技巧
// 思维框架，辅助你思考状态转移方程

明确[状态] -> 定义dp数组/函数的定义 -> 明确[选择] -> 明确 base case

// 示例1：
# 凑零钱问题
"""
给你k种面值的硬币，面值分别为c1,c2, ..., ck, 每种硬币的数量无限
再给你一个总金额amount，问你最少需要几枚硬币凑出这个金额，如果不可能凑出，算法返回-1
"""
"""方法1： 暴力递归"""
def coinChange(coins: List[int], amount: int):
	def dp(n):
		# base case
		if n == 0: return 0
		if n < 0: return -1
		# 求最小值，所以初始化为正无穷
		res = float('INF')
		for coin in coins:
			subproblem = dp(n - coin)
			if subproblem == -1:
				continue
			res = min(res, 1 + subproblem)

		return res if res != float('INF') else -1

	return dp(amount)


"""方法2： 带备忘录的递归"""
def coinChange2(coins: List[int], amount: int):
	# 备忘录
	memo = dict()
	def dp(n):
		if n in memo: return memo[n]
		# base case
		if n == 0: return 0
		if n < 0: return -1
		res = float('INF')
		for coin in coins:
			subproblem = dp(n - coin)
			if subproblem == -1:
				continue
			res = min(res, 1 + subproblem)

		memo[n] = res if res != float('INF') else -1
		return memo[n]

	return dp(amount)

"""方法3：dp数组的迭代解法"""
"dp数组的定义：当目标金额为i时，至少需要dp[i]枚硬币凑出"
int coinChange(vector<int>& coins, int amount) {
	vector<int> dp(amount + 1, amount + 1);
	// base case 
	dp[0] = 0;
	// 外层for循环遍历所以状态的取值
	for(int i = 0; i < dp.size(); i++) {
		// 内层循环求所有选择的最小值
		for(int coin : coins) {
			// 子问题无解，跳过
			if(i - coin < 0) continue
			dp[i] = min(dp[i], 1 + dp[i - coin]);
		}
	}
	return (dp[amount] == amount + 1) ? -1 : dp[amount];
}

// 示例2：
// 最优子结构 详解
/* 什么是最优子结构 和动态规划什么关心 */
/* 为什么动态规划遍历 dp数组的方式五花八门，有的正着遍历，有的倒着遍历，有的斜着遍历 */
// 子问题之间必须互相独立 

//dp数组的遍历方向
// 正向遍历
vector<vector<int>> dp(m, vector<int>(n));
for (int i = 0; i < m; i++) 
	for (int j = 0; j < n; j++)
		// 计算 dp[i][j]


// 反向遍历
for (int i = m - 1; i >= 0; i--) {
	for (int j = n - 1; j >= 0; j--) {
		// 计算 dp[i][j]
	}
}

// 斜向遍历
for (int l = 2; l <= n; l++) {
	for (int i = 0; i <= n - 1; i++) {
		int j = l + i - 1;
		// 计算dp[i][j]
	}
}

// 需要把握两点
// 1.遍历的过程中，所需的状态必须是已经计算出来的。
// 2.遍历的终点必须是存储结构的那个位置

// 示例3：
// 状态压缩：对动态规划进行降维打击
// 要看状态转移方程，如果计算状态dp[i][j]需要的都是dp[i][j]相邻的状态
// 那么就可以使用状态压缩技巧。

// 示例3：最长回文子序列
int longestPalindromesubseq(string s) {
	int n = s.size();
	vector<vector<int>> dp(n, vector<int>(n, 0));
	// base case
	for (int i = 0; i < n; i++) {
		dp[i][i] = 1;
	}
	// 反着遍历，保证正确的状态转移
	for (int i = n - 1; i >= 0; i--) {
		for (int j = i + 1; j < n; j++) {
			// 状态转移方程
			if (s[i] == s[j])
				dp[i][j] = dp[i + 1][j - 1] + 2;
			else 
				dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
		}
	}
	// 整个 s 的最长回文子串长度
	return dp[0][n - 1];
}

// 状态压缩的核心思路是：将二维数组投影到 一维数组中
// dp[i][j]压缩，一般是去掉i 这个维度，只剩下j这个维度
// 压缩后的一维数组就是 之前二维数组dp[i][...]那一行

// 完成降维之后的代码
int longestPalindromesubseq(string s) {
	int n = size();
	// base case : 一维 dp数组全部初始化为1 
	vector<int> dp(n, 1);

	for (int i = n - 1; i >= 0; i--) {
		int pre = 0;
		for (int j = i + 1; j < n; j++) {
			int temp = dp[j];
			// 状态转移方程
			if (s[i] == s[j])
				dp[j] = pre + 2;
			else 
				dp[j] = max(dp[j], dp[j - 1]);

			pre = temp;
		}
	}
	return dp[n - 1];
}

// 示例4：回溯算法与动态规划
// leet 494 目标和
/* 给定一个非负整数组，a1,a2,...,an,和一个目标数S。现在你有两个符号+和-，
   对于数组中的任意一个整数，你都可以从+或-中选择 一个符号添在前面。
   返回可以使最终数组和为目标数S的所有添加符号的方法数。
示例：输入：nums=[1, 1, 1, 1, 1], S : 3
     输出：5
*/ 
// 方法1：回溯算法思想
int result = 0;

/* 主函数 */
int findTargetSumWays(vector<int>& nums, int target) {
	if (nums.size() == 0) return 0;
	backtrack(nums, 0, target);
	return result;
}

/* 回溯算法模板 */
void backtrack(vector<int> nums, int i, int rest) {
	// base case 
	if (i == nums.size()) {
		if (rest == 0) {
			// 说明 恰好凑出 target
			result++;
		}
		return;
	}
	// 给nums[i] 选择 - 号
	rest += nums[i];
	// 穷举 nums[i + 1]
	backtrack(nums, i + 1, rest);
	// 撤销选择
	rest -= nums[i];

	// 给nums[i] 选择 + 号
	rest -= nums[i];
	// 穷举 nums[i + 1]
	backtrack(nums, i + 1, rest);
	// 撤销选择
	rest += nums[i];
}

// 以上是将target减到0
// 如果是加的话
void backtrack(vector<int> nums, int i, int sum, int target) {
	// base case
	if (i == nums.size()) {
		if (sum == target) {
			result++;
		}
		return;
	}
}
// 以上算法的时间复杂度为O(2^N), N为nums的大小
// 发现这个回溯算法就是个二叉树的遍历问题：
// 树的高度就是 nums的长度嘛，所以说时间复杂度就是这颗二叉树的节点数，为O(2^N)
// 其实是非常低效的。
void backtrack(vector<int> nums, int rest) {
	if (i == nums.size())
		return;
	backtrack(nums, i + 1, rest - nums[i]);
	backtrack(nums, i + 1, rest + nums[i]);
}

// 方法2： 消除重叠子问题
// 动态规划之所以比暴力算法快，是因为动态规划技巧消除了重叠子问题
// 如何发现重叠子问题？看是否可能出现重复的 ”状态“。
// 对于递归函数来说，函数参数中会变的参数就是 [状态]，对于backtrack
// 函数来说，会变的参数为 i 和 rest
// 因此(i, rest) 是可以用备忘录技巧进行优化的：

int findTargetSumWays(vector<int>& nums, int target) {
	if (nums.size() == 0) return 0;
	return dp(nums, 0, target);
}

// 备忘录
unordered_map<string, int> memo;
int dp(vector<int>& nums, int i, int rest) {
	// base case
	if (i == nums.size()) {
		if (rest == 0) return 1;
		return 0;
	}
	// 把它俩转成字符串 才能作为哈希表的键
	string key = i + "," + rest;
	// 避免重复计算
	if (memo.count(key)) {
		return memo[key];
	}

	// 还是穷举
	int result = dp(nums, i + 1, rest - nums[i]) + dp(nums, i + 1, rest + nums[i]);
	// 记入备忘录
	memo[key] = result;
	return result;
}
// 以前用python的元祖配合哈希表dict 来做备忘录，其它语言没有元组，可以用把[状态]转化为字符串作为哈希表的键，这是个常用技巧

// 方法3：动态规划
// 消除重叠子问题之后，算法的时间复杂度仍然为O(2^N)
// 这只能叫对回溯算法进行[剪枝]，提升了算法在某些情况下的效率
/* 其实，这个问题可以转化为一个子集划分问题，而子集划分问题又是一个典型的背包问题。
 首先，如果我们把nums划分成两个子集A和B，分别代表分配+的数，和分配-的数，
 那么他们和target存在如下关系：
 sum(A) - sum(B) = target
 sum(A) - target = sum(B)
 sum(A) + sum(A) = target + sum(B) + sum(A
 2 * sum(A) = target + sum(nums)
 sum(A) = (target + sum(nums)) / 2

 就是把原问题转化成：nums中存在几个子集 A, 使得A中元素和为 (target + sum(nums))/2 ?
 可以参考：经典背包问题：子集划分
*/
 /* 计算nums中有几个子集的和为sum */
 int subset(vector<int> nums, int sum) {

 }

 int findTargetSumWays(vector<int>& nums, int target) {
 	int sum = 0;
 	for (n : nums) sum += n;

 	// 这两种情况，不可能存在合法的子集划分
 	if(sum < target || (sum + target) % 2 == 1) {
 		return 0;
 	}
 	return subset(nums, (sum + target) / 2);
 }

/* 计算nums 中有几个子集和为 sum */
 int subset(vector<int>& nums, int sum) {
 	int n = nums.size();
 	// dp[i][j] = x 表示，若只在前i个物品中选择，若当前背包的容量为j, 则最多有x重方法可以恰好装满背包
 	vector<vector<int>> dp(n + 1, vector<int>(sum + 1, 0));

 	// base case
 	for (int i = 0; i <= n; i++) {
 		dp[i][0] = 1; // 金额为0，什么都不装也是一种装法
 	} 
 	for (int i = 1; i <= n; i++) {
 		for (int j = 0; j <= sum; j++) {
 			if (j >= nums[i - 1]) {
 				// 两种选择的结果之后
 				dp[i][j] = dp[i -1][j] + dp[i - 1][j - num[i - 1]];
 			} else {
 				// 背包的空间不足，只能选择不装物品i
 				dp[i][j] = dp[i - 1][j];
 			}
 		}
 	}
 	return dp[n][sum];
 }

// 发现这个 dp[i][j] 只和 前一行dp[i - 1][...]有关，
// 肯定可以优化成一维 dp:
/* 计算nums中 有几个子集的和 为 sum */
int subset(vector<int>& nums, int sum) {
	int n = nums.size();
	vector<int> dp(sum + 1, 0);
	// base case 
	dp[0] = 1;
	for (int i = 1; i <= n; i++) {
		// j 要从后往前遍历
		for(int j = sum; j >= 0; j--) {
			// 状态转移方程
			if (j >= nums[i - 1]) {
				dp[j] = dp[j] + dp[j - nums[i - 1]];
			} else {
				dp[j] = dp[j];
			}
		}
	}
	return dp[sum];
}
// 对照二维dp，只需要把 dp数组的第一个维度全都去掉就行了，唯一的区别是这里的j 要从后往前遍历




// 章节1 经典动态规划问题
// Part2 子序列类型问题
// 示例1：经典动态规划：编辑距离
/*
  题目：
  给定两个字符串s1和s2，计算出将s1转换成s2 所使用的最少操作数。
  你可以对一个字符串进行如下三种操作：
  1.插入一个字符
  2.删除一个字符
  3.替换一个字符
  例子：
  输入：s1 = ”horse“, s2 = "ros"
  解释：
  horse -> rorse
  rorse -> rose
  rose -> ros
*/ 


// 题目比较难，借鉴最长公共子序列思路，用两个指针i，j 分别指向两个字符串的最后，
// 然后一步步往前走，缩小问题的规模
/*
def minDistince(s1, s2):
	// dp(i, j)函数定义，返回s1[0...i] 和 s2[0...j]的最小编辑距离
	def dp(i, j):
		// base case
		if i == -1: return j + 1
		if j == -1: return i + 1

		if (s1[i] == s2[j]):
			return dp(i - 1, j - 1);
		else:
			return min(
				dp(i, j - 1) + 1,     // 插入操作
				dp(i - 1, j) + 1,     // 删除
				dp(i - 1, j - 1) + 1  // 替换
				)

	// i, j 初始化指向最后一个索引
	return dp(len(s1) - 1, len(s2) - 1)

// 动态规划优化，解决重叠子问题

def minDistince(s1, s2) -> int:
	memo = dict()

	def dp(i, j):
		if (i, j) in memo:
			return memo[(i, j)]
		if s1[i] == s2[j]:
			memo[(i, j)] = ...
		else:
			memo[(i, j)] = ...
		return memo[(i, j)]

	return dp(len(s1) - 1, len(s2) - 1)

*/

// 动态规划 dp数组解法
int minDistince(string s1, string s2) {
	int m = s1.size();
	int n = s2.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	// base case
	for (int i = 1; i <= m; i++) {
		dp[i][0] = i;
	}
	for (int j = 1; j <= n; j++) {
		dp[0][j] = j;
	}
	// 自底向上求解
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			if (s[i - 1] == s[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			} else {
				dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1);
			}
		}
	}
	// 存储着整个s1 和 s2 的最小编辑距离
	return dp[m][n];
}
int min(int a, int b, int c) {
	return min(min(a, b), c);
}

// 章节1 经典动态规划问题
// Part2 子序列类型问题
// 示例2：经典动态规划：最长递增子序列
// 题目：给定一个无序整数数组，找到其中最长上升子序列的长度。
/*
 例子：输入[10, 9, 2, 5, 3, 7, 101, 18]
 输出：4
 解释：[2, 3, 7, 101]
*/
// 动态规划解法：
// dp[i]定义：表示以nums[i]这个数结尾的最长递增子序列的长度
int lenghtOFLIS(vector<int>& nums) {
	vector<int> dp(n, 1);
	for (int i = 0; i < nums.size(); i++) {
		for (int j = 0; j < i; j++) {
			if (nums[i] > nums[j]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	int res = 0;
	for (int i = 0; i < dp.size(); i++) {
		res = max(res, dp[i]);
	}
	return res;
}

// 示例3：信封嵌套问题
/*
 题目描述：
 给定一些标记了宽度和高度的信封，宽度和高度以整数对形式(w, h)出现，当另一个信封的宽度和高度
 都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
 请计算最多能有多少个信封能组成 一组 ”俄罗斯套娃“ 信封(即可以把一个信封放到另一个信封里面)
 说明：不允许旋转信封。
 
 例子：输入：envelopes = [[5,4], [6, 4], [6, 7], [2, 3]]
      输出：3
      解释：最多信封的个数为3，组合为：[2, 3] => [5, 4] => [6, 7]
*/
// 解法：先对宽度w进行升序排序，如果遇到w相同的情况，则按照高度h降序排序。之后把所有的h作为一个数组，
// 在这个数组上计算LIS的长度就是答案。
// 这个解法的关键在于，对应宽度w相同的数对，要对其高度h进行降序排序。因为两个宽度相同的信封不能相互包含的，
// 而逆序排序保证在w相同的数对中最多 只选取一个计入LIS。

static bool cmp(vector<int>& a, vector<int>& b) {
    if (a[0] == b[0]) {
        return a[1] > b[1];
    }
    return a[0] < b[0];
}

int maxEnvelopes(vector<vector<int>> & envelopes) {
	int n = envelopes.size();
	sort(envelopes.begin(), envelopes.end(), cmp);
	vector<int> height(n, 0);
	for (int i = 0; i < n; i++) {
		height[i] = envelopes[i][1];
	}
	return lenghtOFLIS(height);
}

// dp[i]定义：表示以nums[i]这个数结尾的最长递增子序列的长度
int lenghtOFLIS(vector<int>& nums) {
	vector<int> dp(n, 1);
	for (int i = 0; i < nums.size(); i++) {
		for (int j = 0; j < i; j++) {
			if (nums[i] > nums[j]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	int res = 0;
	for (int i = 0; i < dp.size(); i++) {
		res = max(res, dp[i]);
	}
	return res;
}

// Part2 子序列类型问题
// 示例4：动态规划套路：最大子数组和
// leet 53, 最大子序和
/*
  给定一个整数数组 nums，找到一个具有最大和的连续子数组(子数组最少包含一个元素)，返回其最大和。
  例子：输入[-2, 1, -3, 4, -1, 2, 1, -5, 4]
  输出：6
  解释：连续子数组[4, -1, 2, 1] 的和最大，为6
*/

// 思考：能否用滑动窗口，答案时不能的，因为数组中可以是负数，不知道什么时候去收缩左侧窗口，也就无法求出[最大子数组和]
// 这个问题需要动态规划技巧，但是dp数组的定义比较特殊

// 我们一般这样定义dp[i]： nums[0...i]中 最大的子数组和 为dp[i], 但是这样在已知dp[i-1]情况下，无法推断出dp[i]
// 因为子数组一定是连续的

// 比较巧妙的定义为： 以nums[i]为结尾的 【最大子数组和】 为dp[i]
int maxSubArray(vector<int>& nums) {
	int n = nums.size();
	if (n == 0) return 0;
	vector<int> dp(n, 0);
	// base cae
	// 第一个元素前面没有子数组
	dp[0] = nums[0];
	// 状态转移方程
	for (int i = 1; i < n; i++) {
		dp[i] = max(nums[i], dp[i - 1] + nums[i]);
	}
	// 得到nums的最大子数组
	int res = INT_MIN;
	for (int i = 0; i < n; i++) {
		res = max(res, dp[i]);
	}
	return res;
}
// 以上算法时间复杂度为O(N), 空间复杂度为O(N)
// 注意到dp[i] 仅和dp[i - 1]的状态相关，可以对其进行 状态压缩，将空间复杂度降低
int maxSubArray(vector<int>& nums) {
	int n = nums.size();
	if (n == 0) return 0;
	vector<int> dp(n, 0);
	int dp_0 = nums[0];
	int dp_1 = 0;
	int res = dp_0;
	for (int i = 1; i < n; i++) {
		// dp[i] = max(dp[i], dp[i - 1] + nums[i]);
		dp_1 = max(nums[i], dp_0 + nums[i]);
		dp_0 = dp_1;
		// 顺便计算最大的结果
		res = max(res, dp_1);
	}
	return res;
} 


// Part2 子序列类型问题
// 示例5：经典动态规划：最长公共子序列
// 秒杀3道动态规划题目
/*
  1143 最长公共子序列
  583 两个字符串的删除操作
  712 两个字符串的最小ASCII删除和
*/
// 最长公共子序列(Longest Common Subsequence 简称LSC) 是一道经典的动态规划题目
/*
  给你输入两个字符串 s1 和 s2, 请找出它们俩最长的公共子序列，并返回这个子序列的长度。
*/

// 前文 子序列解题模板中总结的一个规律：
/*
  对于两个字符串求子序列的问题，都是用两个指针i和j 分别在两个字符串上移动，大概率是动态规划的思路。
  最长公共子序列的问题，也可以遵循这个规律
*/
// 可以先写一个dp函数：
// 函数定义：计算s1[i...] 和 s2[j...] 最长公共子序列的长度
int dp(string s1, int i, string s2, int j);
// 根据这个定义，我们想要的答案时：
dp(s1, 0, s2, 0)

// 方法1：自顶向下的 dp函数 递归解法
vector<vector<int>> memo;
/* 主函数 */
int longestCommonSubsequence(string s1, string s2) {
	int m = s1.size();
	int n = s2.size();
	memo.resize(m);
	for( auto& row : memo) {
		row.resize(n, -1);
	}
	// 计算 ss1[0...] 和 s2[0...] 的lcs长度
	return dp(s1, 0, s2, 0);
}

// 定于：计算s1[i...] 和 s2[j...] 的最长公共子序列长度。
int dp(string s1, int i, string s2, int j) {
	// base case
	if (i == s1.size() || j == s2.size()) {
		return 0;
	}
	// 如果之前计算过，则这届返回备忘录中的答案
	if(memo[i][j] != -1) {
		return memo[i][j];
	}
	// 根据s1[i] 和 s2[j] 的情况做选择
	if (s1[i] == s2[j]) {
		// s1[i] 和 s2[j] 必然在lcs中
		memo[i][j] = 1 + dp(s1, i + 1, s2, j + 1);
	} else {
		// s1[i] 和 s2[j] 至少有一个不再lcs中
		// 穷举3中情况的结果，取其中最大的结果
		memo[i][j] = max(
				dp(s1, i + 1, s2, j), // 情况1：s1[i] 不在lcs中
				dp(s1, i, s2, j + 1)  // 情况2：s2[j] 不在 lcs中
				//, dp(s1, i + 1, s2, j + 1)  情况3：都不在lcs中 
				// 这里的情况3可以忽略，因为肯定是 小于等于情况1 和情况2
			);
	}
	return memo[i][j];
}

// 以上的 自顶向下方法挺好，但是我不太喜欢
// 还是喜欢 自底向上的迭代的动态规划思路
int longestCommonSubsequence(string s1, string s2) {
	int m = s1.size();
	int n = s2.size();
	if (m == 0 || n == 0)
		return 0;

	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	// 定义：s1[0,...i-1] 和 s2[0,...j-1] 的lcs的长度为dp[i][j]
	// 目标：s1[0,...m-1] 和 s2[0,...n-1] 的lcs的长度为dp[m][n]
	// base case: dp[0][...] = dp[...][0] = 0
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			// 现在 i 和 j 从1 开始，所以要减一
			if(s1[i - 1] == s2[j - 1]) {
				// s1[i - 1] 和 s2[j - 1] 必然在lcs中
				dp[i][j] = 1 + dp[i - 1][j - 1];
			} else {
				// s1[i-1] s2[j-1] 至少有一个不在 lcs中
				dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
			}
		}
	}
	rerurn dp[m][n];
}


// leet 583 字符串的删除操作
// 给定两个单词s1 和 s2, 找到使得s1和s2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
/*
  例子：
  输入：”sea“，”eat“
  输出：2
  解释：第一步将"sea" 变为”ea“, 第二步将”eat“ 变成”ea"
*/

// 解答：删除的最后结果，不就是它俩的最长公共子序列嘛
// 要计算删除的次数，就可以通过最长公共子序列的长度推导出来。
 int minDistince(string s1, string s2) {
 	int m = s1.size(), n = s2.size();
 	// 复用前文计算 lcs 长度的函数
 	int lcs = longestCommonSubsequence(s1, s2);
 	return m - lcs + n - lcs;
 }

// leet 712 最小ASCII 删除和
// 给定两个字符串s1，s2, 找到使两个字符串相等所需删除字符的ASCII值的最小和。
/*
  例子：
  输入：s1 = "sea", s2 = "eat"
  输出：231
  解释：在“sea” 中删除 “s” 并将 “s”的值（115） 加入总和
       在“eat”中 删除 “t” 将 “t”的值 116 加入总和
       结束时，两个字符串相等，115+116=231，就是符合条件的最小和。
*/
// 题解：可以按照计算最长公共子序列的函数的思路
// 稍微修改 base case 和状态转移部分，即可直接写出解法代码

// 方法：自顶向下的 dp函数 递归解法
vector<vector<int>> memo;
/* 主函数 */
int minimumDeleteSum(string s1, string s2) {
	int m = s1.size();
	int n = s2.size();
	memo.resize(m);
	for( auto& row : memo) {
		row.resize(n, -1);
	}
	
	return dp(s1, 0, s2, 0);
}

// 定义：将s1[i...] 和 s2[j...] 删除成相同的字符串
// 最小的ASCII 码之和 为dp(s1, i, s2, j)
int dp(string s1, int i, string s2, int j) {
	int res = 0;
	// base case
	if (i == s1.size()) {
		// 如果 s1 到头了，那么s2 剩下的都得删除
		for(; j < s2.size(); j++) {
			res += s2[j];
		}
		return res;
	}
	if (j == s2.size()) {
		// 如果 s2 到头了，那么s1 剩下的都得删除
		for(; i < s1.size(); i++) {
			res += s1[i];
		}
		return res;
	}
	if (memo[i][j] != -1) {
		return memo[i][j];
	}
	if (s1[i] == s2[j]) {
		// s1[i] 和 s2[j] 都在 lcs中，不用删除
		memo[i][j] = dp(s1, i + 1, s2, j + 1);
	} else {
		// s1[i] s2[j] 至少有一个不在lcs中，不用删除
		memo[i][j] = min(
				s1[i] + dp(s1, i + 1, s2, j),
				s2[j] + dp(s1, i, s2, j + 1)
			);
	}
	return memo[i][j];
}

// Part2 子序列类型问题
// 示例6：子序列解题模板：最长回文子序列
/*
  子序列问题 本身就相对子串、子数组更困难些，因为前者是不连续的序列，而后两者是连续的。
  子序列问题的套路，一般有两种模板，下面会讲。
*/
// 思路1：模板是一个一维的 dp数组：
int n = vec.size();
vector<int> dp(n, 0);
for (int i = 1; i < n; i++) {
	for (j = 0; j < i; j++) {
		dp[i] = 最值(dp[i], dp[j] + ...)
	}
}
// 前文介绍过的 最长递增子序列，在这个思路中 dp数组的定义为：
// 在子数组 array[0...i]中，以array[i] 结尾的目标子序列(最长递增子序列)的长度是dp[i]
// 为啥最长自增子序列需要这种思路呢？因为这样符合归纳法，可以找到状态转移的关系。

// 思路2：模板是一个二维的 dp数组：
int n = arr.size();
vector<vector<int>> dp(n, vector<int>(n, 0));

for (int i = 0; i < n; i++) {
	for (int j = 1; j < n; j++) {
		if (arr[i] == arr[j]) {
			dp[i][j] = dp[i][j] + ...
		} else {
			dp[i][j] = 最值(...)
		}
	}
}
这种思路相对运用多一些，尤其涉及两个字符串/数组的子序列。
本思路中dp数组 含义又分为 [只涉及一个字符串] 和 [涉及两个字符串]

1.
涉及两个字符串/数组时(比如 最长公共子序列)，dp数组的含义如下：
在子数组 arr1[0,...i] 和 子数组 arr2[0,...j]中，我们要求的子序列(最长公共子序列) 长度为 dp[i][j]

2.
只涉及一个字符串/数组时(比如，要讲的最长回文子序列)，dp数组的定义如下：
在子数组 array[i,...j] 中，我们要求的子序列(最长回文子序列)的长度为dp[i][j]

// 最长回文子序列
// 题目：给定一个字符串s，找到其中最长的回文子序列。
// 例子：输入 “bbbab” 输出 4，注：一个可能的最长回文子序列为 “bbbb”

// dp数组定义：在子串s[i,...,j] 中，最长混为子序列的长度为dp[i][j], 需要记住这个定义。
// 为啥这样定义二维数组，因为找状态转移需要归纳思维，说白了就是如何从已知的结果推出 未知的部分。
// 这样定义 容易发现状态转移关系。

// dp[i][j] 和 dp[i+1][j-1] dp[i][j-1] dp[i+1][j]相关
1.为了保证每次计算dp[i][j] 左、下、左下 三个方向的位置已经被计算出来，智能 斜着遍历 或者 反着遍历
// 选择 反着遍历
int longestPalindromesubseq(string s) {
	int n = s.size();
	// dp数组初始化为 0
	vector<vector<int>> dp(n, vector<int>(n, 0));
	// base case
	for (int i = 0; i < n; i++) {
		dp[i][i] = 1;
	}
	// 反着遍历 保证正确的状态转移
	for (int i = n - 1; i >= 0; i--) {
		for (int j = i + 1; j < n; j++) {
			// 状态转移方程
			if (s[i] == s[j]) {
				dp[i][j] = dp[i + 1][j - 1] + 2;
			} else {
				dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
			}
		}
	}
	return dp[0][n - 1];

}


// Part3 背包类型问题
// 示例1：经典动态规划：子集背包问题
1.题目1：分隔等和子集
给定一个只包含正整数的非空数组，是否可以将这个数组分隔成两个子集，使得两个子集的元素和相等。
例子1：
输入：[1, 5, 11, 5]
输出：true 
解释：数组可以分隔成 [1, 5, 5] 和 [11]

// 该题目可以转化成 背包问题：
给一个可装载重量为 sum/2 的背包和 N个物品，每个物品的重量为nums[i].
现在让你装物品，是否存在一种装法，能够恰好将背包装满？

// 按照背包的套路，给出如下定义：
dp[i][j] = x 表示，对于前i个物品，当前背包的重量为j时，若x 为 true, 则表示恰好可以装满备好。
若 x 为 false，则说明不能恰好将背包装满
// 代码如下：
bool canPartition(vector<int>& nums) {
	int sum = 0;
	for (int num : nums) 
		sum += num;
	// 和为奇数时，不可能划分成 两个和相等的集合
	if (sum % 2 != 0) return false;
	int n = sum / 2;
	sum = sum / 2;

	vector<vector<bool>> dp(n + 1, vector<bool>(sum + 1, false));
	// base case
	for (int i = 0; i <= n; i++) {
		dp[i][0] = true;
	}
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= sum; j++) {
			if (j - nums[i - 1] < 0) {
				// 背包容量不足，不能装入第i 个物品
				dp[i][j] = dp[i - 1][j];
			} else {
				// 装入或者不装入背包
				dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
			}
		}
	}
	return dp[n][sum];
} 

// 进行状态压缩
// 注意到dp[i][j] 都是通过 上一行dp[i - 1][...] 转移过来的，因此可以进行状态压缩
bool canPartition(vector<int>& nums) {
	int sum = 0, n = nums.size();
	for (int num : nums)
		sum += num;
	if (sum % 2 != 0) return false;
	sum = sum / 2;
	vector<bool> dp(sum + 1, false);
	// base case
	dp[0] = true;
	for (int i = 0; i < n; i++) {
		for (int j = sum; j >= 0; j--) {
			if (j - nums[i] >= 0)
				dp[j] = dp[j] || dp[j - nums[i - 1]];
 		}	
 	}
 	return dp[sum];
}
// 思路和上面相同，唯一需要注意的是：j应该从后往前反向遍历，因为每个物品(或者说数字) 只能用一次，以免之前的结果影响其它结果。
// 子集切割问题 就完全解决了，时间复杂度为O(n * sum), 空间复杂度为O(sum)

// Part3 背包类型问题
// 示例2：经典动态规划：完全背包问题
// leet 518 零钱兑换 II
// 题目：给定不同面额的硬币和一个 总金额，写出函数来计算可以凑成总金额的 硬币的组合数，假设每一种面额的硬币 有无限个。

/* 例子1：
   输入：amount = 5, coins = [1, 2, 5]
   输出：4
   解释：有4种方式 可以凑成总金额
*/
1.dp数组定义如下：
dp[i][j]的定义：若只使用前 i 个物品，当背包容量为j时，有dp[i][j]种方法 可装满背包。
换句话说
2.若只使用coins 中的前 i 个硬币的面值，若想凑出金额j, 有dp[i][j]种凑法
3.最终答案：dp[N][amount];
/* 代码如下 */
int change(int amount, vector<int> coins) {
	int n = coins.size();
	vector<vector<int>> dp(n + 1, vector<int>(amount + 1, 0));
	// base case
	for (int i = 0; i <= n; i++) {
		dp[i][0] = 1;
	}
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= amount; j++) {
			if (j - coins[i - 1] >= 0) {
				dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]];
			} else {
				dp[i][j] = dp[i - 1][j];
			}
		}
	}
	return dp[n][amount];
}

发现dp数组 的 转移只和 dp[i][...]和dp[i - 1][...]相关，可以进行状态压缩。
int change(int amount, vector<int> coins) {
	int n = coins.size();
	vector<int> dp(amount + 1, 0);
	// base case
	dp[0] = 1; 
	for (int i = 0; i < n; i ++) {
		for (int j = 1; j <= amount; j++) {
			if (j - coins[i] >= 0) {
				dp[j] = dp[j] + dp[j - coins[i]];
			}
		}
	}
	return dp[amount];
}
// 这个解法和之前的思路完全相同，将二维dp数组压缩为 一维，时间复杂度O(N * amount), 空间复杂度O(amount)
// 至此，零钱兑换问题，也通过背包问题解决了。

// Part3 背包类型问题
// 示例3：经典动态规划：0-1背包问题
1.题目：给你一个可装载重量为W的背包 和 N个物品，每个物品有重量和价值两个属性。
其中，第i个物品的重量为 wt[i] 价值为val[i] ,现在让你用这个背包装物品，最多能装的价值是多少？

2.分析。这个题目中的物品不可以分隔，要么装进包里，要么不装，不能说切成两块 装一半。
 这也许就是 0-1背包 这个名词的来历吧。

3.定义dp数组：
dp[i][w]的定义如下：对于前i个物品，当前背包的容量为w，这种情况下可以装的最大价值是 dp[i][w] = x
// 比如 dp[3][5] = 6, 表示：对于给定的一系列物品中，若只对前3个物品进行选择，当背包容量为5时，最多可以装下的价值为6.

4. base case
最终答案就是dp[N][N], base case为：dp[0][...] = dp[...][0] = 0
因为 没有物品或者背包没有空间的时候，能装的最大价值就是 0

5.细化上面的框架：
int dp[N+1][W+1]
dp[0][...] = 0;
dp[...][0] = 0;

for i in [1...N]:
	for w in [1...W]:
		dp[i][w] = max(
				把物品 i 装进背包，
				不把物品 i 装进背包 
			)
return dp[N][W]

/* 详细代码如下 */
int knapsack(int W, int N, vector<int>& wt, vector<int>& val) {
	// vector 全填0
	// base case 已初始化
	vector<vector<int>> dp(N + 1, vector<int>(W + 1, 0));
	for (int i = 1; i <= N; i++) {
		for (int w = 1; w <= W; w++) {
			if (W - wt[i - 1] < 0) {
				// 当前背包容量装不下，只能选择不装入背包
				dp[i][w] = dp[i - 1][w];
			} else {
				// 装入 或者 不装入背包
				dp[i][w] = max(dp[i - 1][w - wt[i - 1]] + val[i - 1], dp[i - 1][w]);
			}
		}
	}
	return dp[N][N];
}


// Part4 贪心类型问题
// 示例1：贪心算法之 区间调度问题
1.贪心算法可以认为是动态规划算法的一个特例，相比动态规划，使用贪心算法需要满足更多条件(贪心选择性质)，
  但是效率比动态规划要高。
2.什么是 贪心选择性：简单说就是：每一步都做出一个局部最优的选择，最终的结果就是全局最优的选择。

题目：解决一个很经典的贪心算法问题：Interval Scheduling（区间调度问题）。
给你很多形如[start, end]的闭区间，请你设计一个算法，算出这些区间中最多有几个互不相交的区间。

int intervalScheduling(vector<vector<int>>& ints) {}
思路： 
1.从区间集合 intvs中 选择一个区间x, 这个x 是当前所有区间中结束最早的(end最小)。
2.把所有与 x 区间相交的区间，从区间集合 intvs中删除。
3.重复步骤 1 和 2，直到intvs 为空为止。之前选择的那些 x 就是 最大不相交子集。

代码如下：
int intervalSchedule(vector<vector<int>>& intvs) {
	if (intvs.size() == 0) return 0;
	// 按 end 升序排序 
	sort(intvs.begin(), intvs.end(),cmpare);
	// 至少应该有一个区间 不相交
	int count = 1;
	// 排序后，第一个区间就是 x 
	int x_end = intvs[0][1];
	for (auto interval : intvs) {
		int start = interval[0];
		if (start >= x_end) {
			// 更新 x 
			count++;
			x_end = interval[1];
		}
	}
	return count;
}

static bool cmpare(vector<int>& a, vector<int>& b) {
    
    return a[1] < b[1];
}

题目2：// leet 435, 无重叠区间
给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
注意： 
1.可以认为区间的终点 总是大于它的起点
2.区间[1,2] 和 [2,3]的边界互相 接触，但并没有相互重叠
例子：
输入：[[1, 2], [2, 3], [3, 4], [1, 3]]
输出：1 
解释：移除[1, 3]后，剩下的区间没有重叠 

int eraseOverlapIntervals(vector<vector<int>>& intervals) {
	int n = intervals.size();
	return n - intervalSchedule(intervals);
}

题目3：用最少的箭 射爆气球
// leet 452
输入：[[10, 16], [2, 8], [1, 6], [7, 12]]
输出：2 
解释：对于该样例，我们可以在x = 6(射爆[2, 8], [1, 6]两个气球) 和 x = 11 (射爆另外两个气球)

// 只需要 稍微修改下算法就可以了。
int findMinArrowShots(vector<vector<int>> intvs) {
	// ...
	for (auto interval : intvs) {
		int start = interval[0];
		// 把 >= 改成 > 就行了
		if(start > x_end) {
			count++;
			x_end = interval[1];
		}
	}
	return count;
}



// Part4 贪心类型问题
// 示例2：经典贪心算法: 跳跃游戏
// 贪心算法可以理解为 一种特殊的动态规划问题，拥有一些更特殊的性质
// 可以进一步 降低动态规划的时间复杂度。

// 两道经典的贪心算法：跳跃游戏I 和 跳跃游戏II
// leet55 题目1：给定一个非负整数数组，你最初位于数组的第一个位置
// 数组中的每个元素代表 你在该位置可以跳跃的最大长度。
// 判断 你是否能够到达最后一个位置。
/*
 例子：输入：[2, 3, 1, 1, 4]
      输出：true
      解释：我们可以先跳1步，从位置0 到达位置 1,然后从位置1 跳3 步 达到最后一个位置。
*/

bool canJump(vector<int>& nums) {
	int n = nums.size();
	int farthest = 0;
	for (int i = 0; i < n - 1; i++) {
		// 不断计算能跳的最远距离
		farthest = max(farthest, i + nums[i]);
		// 可能碰到0 了，卡主了
		if(farthest <= i) return false;
	}
	return farthest >= n - 1;
}


// 跳跃游戏II
// leet 45
/*
  给定一个非负整数数组，你最初位于数组的第一个位置。
  数组中每个元素代表 你在 该位置可以跳跃的最大长度。
  你的目标是使用 最少的跳跃次数 到达最后一个位置
  例子：输入：[2, 3, 1, 1, 4]
      输出：2
      解释：我们可以先跳1步，从位置0 到达位置 1,然后从位置1 跳3 步 达到最后一个位置。
*/
// 现在问你，保证你一定能够跳到最后一格，请问你最少 要跳多少次，才能跳过去？

// 题解1：动态规划解法
// 采用自顶向下的递归动态规划，可以定义一个这样的dp函数：
定义：从索引p跳到最后一格，至少需要dp(nums, idx)步 
int dp(vector<int>& nums, int idx) {};

vector<int> memo;
// 主函数
int jump2(vector<int>& nums) {
	int n = nums.size();
	// 备忘录都初始化为 n, 相当于INT_MAX;
	// 因为从 0 跳到 n - 1 最多 n - 1步；
	memo = vector<int>(n, n);
	return dp(nums, 0);
}

int dp(vector<int>& nums, int idx) {
	int n = nums.size();
	// base case
	if (idx >= n - 1) {
		return 0;
	}
	// 子问题已经 计算过
	if (memo[idx] != n) {
		return memo[idx];
	}
	int steps = nums[p];
	// 你可以选择 跳 1步，2步...
	for(int i = 1; i <= steps; i++) {
		// 穷举每一个选择
		// 计算每一个子问题的结果
		int subproblem = dp(nums, idx + i);
		// 取其中 最小的 作为最终结果
		memo[idx] = min(memo[idx], subproblem + 1);
	}
	return memo[idx];
}
// 该算法的时间复杂度是：递归深度 * 每次递归需要的时间复杂度，即O(N^2)

// 解法2：贪心算法比动态规划多了一个性质：贪心选择性质
// 这个题目中，真的需要 [递归的] 计算出每一个子问题的结果，然后求最值吗？
// 直观的想一想，似乎不需要递归，只需要判断 哪一个选择 最具有 ‘潜力’ 即可。
// 下面看代码
int jump2(vector<int>& nums) {
	int n = nums.size();
	int farthest = 0, end = 0;
	int jumps = 0;
	for (int i = 0; i < n - 1; i++) {
		farthest = farthest(i + nums[i], farthest);
		if (i == end) {
			jumps++;
			end = farthest;
		}
	}
	return jumps;
}
// 本算法那时间复杂度 为：O(N), 空间复杂度为 O(1)


// Part5 其它经典问题：动态规划
// 示例1：经典动态规划：正则表达式
// 题目1：给定一个字符串(s) 和 一个字符模式(p) 实现支持'.' 和 '*' 的正则表达匹配
/*
  '.' 匹配任意单个字符
  '*' 匹配0个或多个前面的字符
  匹配应该是 覆盖 整个 字符串(s), 而不是部分字符串

  例子1：
  输入：
  s = "aa"
  p = "a*"
  输出: true

  例子2：
  输入：
  s = “aab”
  p = "c*a*b"
  输出：true
*/

// 先给出简单的代码
两个普通字符串进行比较，如何进行匹配？
bool isMatch(string text, string pattern) {
	if (text.size() != pattern.size()) {
		return false;
	}
	for (int j = 0; j < pattern.size(); j++) {
		if (pattern[j] != text[j])
			return false;
	}
	return true;
}

// 稍微改一下代码：
bool isMatch(string text, string pattern) {
	int i = 0;
	int j = 0;
	while (j < pattern.size()) {
		if (i >= text.size())
			return false;
		if (pattern[j++] != text[i++]) {
			return false;
		}
	}
	// 判断 pattern 和 text 是否一样长
	return j == text.size();
}
// 伪代码
def isMatch(text, pattern) -> bool:
	if pattern is empty: return (text is empty?)
	first_match = (text not empty) and pattern[0] == text[0]

	reutrn first_match and isMatch(text[1:], pattern[1:])

// 最终代码：
def isMatch(text, pattern) -> bool:
	if not pattern: 
		return not text
	first_match = bool(text) and pattern[0] in {text[0], '.'}
	if len(pattern) >= 2 and pattern[1] == '*':
		return isMatch(text, pattern[2:]) or \
	            first_match and isMatch(text[1:], pattern)
	    // 解释：如果发现有字符 和‘*’ 结合，
	    // 或者匹配该字符 0 次，然后调过该字符 和 ‘*’
	    // 或者 当 pattern[0] 和 text[0] 匹配后，移动 text
	else:
		return first_match and isMatch(text[1:], pattern[1:])

// 以上方法 有 重叠子问题 需要用 备忘录 或者 dp table 消除 重叠子问题，降低一下复杂度

// 带备忘录的递归
def isMatch(text, pattern) -> bool:
	memo = dict()
	def dp(i, j):
		if (i, j) in memo: return memo[(i, j)]
		if j == len(pattern): return i == len(text)

		first = i < len(text) and pattern[j] in {text[i], '.'}

		if j < len(pattern) - 2 and pattern[j + 1] == '*':
			ans = dp(i, j + 2) or first and dp(i + 1, j)
		else:
			ans = first and dp(i + 1, j + 1)
		return ans 

	return dp(0, 0)
// 总结：回顾整个解题过程，你应该能够 体会到算法设计的流程
// 从简单的类似问题入手，给基本的矿建主键组装新的逻辑，最终成为一个比较复杂、精巧的算法。

// Part5 其它经典问题：动态规划
// 示例2：经典动态规划：高楼扔鸡蛋
/*
 题目：你面前有一栋从1到N的共N层的楼，然后给你K个鸡蛋(K至少为1)
      现在确定这栋楼存在楼层 0 <= F <= N，
      在这层楼将鸡蛋扔下去，几点恰好没摔碎(高于F的楼层都会碎，低于F的楼层都不会碎)。
      现在问你，最坏情况下，你至少要扔几次鸡蛋，才能确定这个楼层F呢？
 PS: F可以为0， 比如说鸡蛋在1层都能摔碎，那么F = 0
*/
// 动态规划解法
# 当前状态为(K个鸡蛋，N层楼)
# 返回这个状态下的最优结果

def superEggDrop(K: int, N: int):
    
    memo = dict()
    def dp(K, N) -> int:
        # base case
        if K == 1: return N
        if N == 0: return 0
        # 避免重复计算
        if (K, N) in memo[(K, N)]:
            return memo[(K, N)]

        res = float('INF')
        # 穷举所有可能的选择
        for i in range(1, N + 1):
            res = min(res, 
                    max(dp(K, N - i),     # 碎
                        dp(K - 1, i - 1)  # 没碎
                        ) + 1              #在第 i 楼 扔了一次 
                )
        # 计入备忘录
        memo[(K, N)] = res
        return res 
    return dp(K, N)


int dp(int k, int n, vector<vector<int>>& memo, int res) {
    // base case
    if (k == 1) return N;
    if (n == 0) return 0;
    if (memo[k][n] != -1)
        return memo[k][n];
    for (int i = 1; i <= N; i++) {
        res = min(res, 
                 max(dp(k, n - i), dp(k - 1, i - 1)) + 1
                 );
    }
    memo[k][n] = res;
    return res;
}
int superEggDrop(int K, int N) {
    int res = INT_MAX;
    vector<vector<int>> memo(K, vector<int>(N, -1));
    return dp(K, N, memo, res);

}
// 以上算法的时间复杂度为O(KN^2)
// 动态规划的算法的时间复杂度就是子问题个数 * 函数本身的复杂度
// dp函数中有一个for循环 所以函数的时间复杂度为O(N)
// 子问题个数也就是不同状态组合的总数，显然是两个状态的乘积，也就是O(KN)
// 所以算法的总时间复杂度是O(KN^2),空间复杂度为子问题个数O(KN)



// Part5 其它经典问题：动态规划
// 示例3：经典动态规划：高楼扔鸡蛋(进阶)
// 二分搜索优化：利用状态转移方程的单调性
def superEggDrop(K: int, N: int) -> int:
    memo = dict()
    def dp(K, N):
        if k == 1: return N
        if N == 0: return 0
        if (K, N) in memo:
            return memo[(K, N)]
        //for i in range(1, N + 1):
        //    res = min(res, 
        //            max(dp(K, N - i),     # 碎
        //                dp(K - 1, i - 1)  # 没碎
        //                ) + 1              #在第 i 楼 扔了一次 
        //        )
        res = float('INF')
        # 利用二分搜索代替线性搜索
        lo, hi = 1, N 
        while lo <= hi:
            mid = (lo + hi) // 2
            broken = dp(K - 1, mid - 1) # 碎
            not_broken = dp(K, N - mid) # 没碎
            # res = min(max(碎，没碎) + 1)
            if broken > not_broken:
                hi = mid - 1
                res = min(res, broken + 1)
            else:
                lo = mid + 1
                res = min(res, not_broken + 1)

        memo[(K, N)] = res 
        return res 

    return dp(K, N)
// 以上算法的时间复杂度为O(K*N*logN)


// 解法2：重写状态转移
/*
  之前的状态转移定义：
  dp[k][n] = m
  表示 当前状态为k 个鸡蛋，面对n 层楼
  这个状态下最少扔鸡蛋次数为m 

  稍微改一下dp数组
  dp[k][m] = n
  当前有k个鸡蛋，可以尝试扔m次鸡蛋
  这个状态下，最坏情况下最多能确切测试一栋 n 层的楼
  比如说：dp[1][7] = 7 表示：
  现在有1个鸡蛋，允许你扔7次
  这个状态下最多给你 7层楼
  使得你可以确定楼层F 使得鸡蛋恰好摔不碎
*/
int superEggDrop(int K, int N) {
    // m最多不会超过 N 次(线性扫描)
    vector<vector<int>> dp(K + 1， vector<int>(N + 1, 0));
    // base case
    // dp[0][..] = 0
    // dp[..][0] = 0
    int m = 0;
    while(dp[K][m] < N) {
        m++;
        for (int k = 1; k <= K; k++) {
            dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1;
        }
    }
    return m;
}

// 以上while循环代码 等价于
for(int m = 1; dp[K][m] < N; m++) {
    for (int k = 1; k <= K; k++) {
        dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1;
    }
}


// Part5 其它经典问题：动态规划
// 示例4：经典动态规划：戳气球问题
// leet 312 题目如下：
/*
  有n个气球，编号为0 -> n-1, 每个气球上都标有一个数字，这些数字存在数组 nums中。
  现在要求你戳破所有的气球，每当你戳破一个气球i时，你可以获得 nums[left] * nums[i] * nums[right]个硬币
  这里的left和right 代表和i相邻的两个气球的序号。注意当你戳破气球i后，气球left和right就变成了相邻的气球。

  求所获得硬币的最大数量
*/

// 方法1：回溯思路
// 其实就是穷举戳气球的顺序，不同戳气球的顺序可能得到不同的分数。
// 这就是一个 全排列 问题嘛，代码思路如下：
int res = INT_MIN
/* 输入一组气球，返回戳破它们获得的最大分数 */
int maxCoins(vector<int> nums) {
    backtrack(nums, 0);
    return res;
}
/* 回溯算法的伪代码解法 */
void backtrack(vector<int>& nums, int score) {
    if (nums 为 空) {
        res = max(res, score);
        return;
    }
    for (int i = 0; i < nums.size(); i++) {
        int point = nums[i - 1] * nums[i] * nums[i + 1];
        int temp = nums[i];
        // 做选择
        在 nums中删除元素 nums[i]
        // 递归回溯
        backtrack(nums, score + point);
        // 撤销选择
        将temp 还原到 nums[i]
    }
}
// 解释：回溯算法 效率非常低，等同于全排列，时间复杂度是指数级

// 方法2：动态规划解法
// 这个和之前的动态规划系列的文章相比有什么特别之处呢？
// 原因在于：这个问题中我们每戳破一个气球 nums[i],得到的分数和该气球相邻的气球
// nums[i-1] 和 nums[i+1] 是有相关性的。
// 之前说过 动态规划应用的一个重要条件是：子问题必须独立
// 那么想用动态规划，必须巧妙的定义dp数组的含义，避免子问题产生相关性，才能推出合理的状态转移过程。

// 首先扩展下数组，可以认为nums[-1] = nums[n] = 1, 定义一个新数组
int maxCoins(vector<int> nums) {
    int n = nums.size();
    vector<int> points(n + 2);
    points[0] = points[n + 1] = 1;
    for (int i = 1; i <= n; i++) {
        points[i] = nums[i - 1];
    }
    // ...
}
// 现在的文婷 定义为：在一排气球points中，请你戳破气球0和气球n+1 之间的所有气球(不包括0和n+1),
// 使得最终两只气球0 和 n + 1连个气球，最终能够得到多少分？
// 定义dp数组的含义为：
/*
  dp[i][j] = x 表示，戳破气球i和气球j之间(开区间，不包含i和j)的所有气球，可以获得的最高分数为x
  根据定义 题目要求的结果就是：dp[0][n+1]
*/

// 遍历时，从下往上遍历，从左往右遍历写法
int maxCoins(vector<int> nums) {
    int n = nums.size();
    // 添加两侧的虚拟气球
    vector<int> points(n + 2);
    points[0] = points[n + 1] = 1;
    for (int i = 1; i <= n; i--) {
        points[i] = nums[i - 1];
    }
    // base case 已经都被初始化为0
    vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
    // 开始状态转移
    // i 应该从下往上
    for(int i = n; i >= 0; i--) {
        // j 应该从左到右
        for(int j = i + 1; j < n + 2; j++) {
            // 最后戳破的气球时哪个？
            for (int k = i + 1; k < j; k++) {
                // 择优做选择
                dp[i][j] = max(dp[i][j],
                            dp[i][k] + dp[k][j] + points[i] * points[i] * points[k]
                           );
            }
        }
    }
    return dp[0][n + 1];
}


// Part5 其它经典问题：动态规划
// 示例5：经典动态规划：博弈问题
// leet 877 题目如下：
/*
  亚里克斯和李用几堆石子做游戏，偶数堆石子排成一行，每堆都有正整数颗石子 piles[i]
  游戏以谁手中的石子最多来决出胜负，石子的总数是奇数，所以没有平局。
  亚里克斯和李轮流进行，亚里克斯先开始，每回合，玩家从行的开始或结束处取走整堆石头。
  这种情况一直持续到没有更多的石子堆为止，此时手中石子最多的玩家获胜。

  假设亚里克斯和李都发挥出最佳水平，当亚里克斯赢得比赛时 返回true, 当李赢得比赛时返回false。
示例：
    输入：[5, 3, 4, 5]
*/

// 解法1：
bool stoneGame(vector<int>& piles) {
    int length = piles.size();
    if (length <= 1) return true;
    //定义 dp[i][j],表示当剩下的石子堆为下标i到j时，当前玩家与另一个玩家的石子数量之差的最大值。
    vector<vector<int>> dp(length, vector<int>(left_height, 0));
    // base case
    // i > j， dp[i][j] = 0, 初始化已赋值
    for(int i = 0; i < length; i++) {
        dp[i][i] = piles[i];
    }
    for(int i = length - 2; i >= 0; i--) {
        for (int j = i + 1; j < length; j++) {
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
        }
    }
    return dp[0][length - 1] > 0;
}
// 解法2：降维
// dp[i][j]降维，因为计算第i行时，只与i和i+1相关
bool stoneGame2(vector<int>& piles) {
    int length = piles.size();
    if (length <= 1) return true;
    vector<int> dp(length, 0);
    for (int i = 0; i < length; i++) {
        dp[i] = piles[i];
    }
    for (int i = length - 2; i >= 0; i--) {
        for (int j = i + 1; j < length; j++) {
            dp[j] = max(piles[i] - dp[j], piles[j] - dp[j - 1]);
        }
    }
    return dp[length - 1] > 0;
}

// Part5 其它经典问题：动态规划
// 示例6：经典动态规划：四健问题
/*
  题目：假设你有一个特殊的键盘，包含下面的按键：
  key1: (A) 在屏幕上打印一个A
  key2：(ctrl-A): 选中整个屏幕
  key3：(ctrl-C): 复制选中区域到缓冲区
  key4：(ctrl-V): 将缓冲区内容输出到上次输入的结束位置，并显示在屏幕上。
  现在，你只可以按键N次，请问屏幕上最多可以显示几个 A呢？

  样例1：输入：N = 3
  输出：3
  解释：最多可以在屏幕上显示三个A， 通过如下顺序按键：
  A, A, A

  样例2：N = 7
  输出：9
  解释：最多可以在屏幕上显示 九个A,通过如下顺序按键。
  A，A，A，ctrl-a, ctrl-c, ctrl-v, ctrl-v
*/

// 思路：如何在N次敲击按钮后，得到最多的A？我们穷举呗，对于每次按键，我们可以穷举四种可能
// 很明显就是一个动态规划问题。
// 可以这样定义状态
/*
  第一个状态是剩余的按键次数，用n表示；
  第二个状态是当前屏幕上字符A 的数量，用a_num表示；
  第三个状态是剪切板张红字符A的数量；
  用copy表示。
  如此定义[状态]，就可以知道base case: 当剩余次数为0时，a_num就是我们想要的答案。
*/
可以把这几个选择 通过状态转移表示出来：
dp(n - 1, a_num + 1, copy)      #A 
解释：按下A键，屏幕上加一个字符，同时消耗 1 个操作数

dp(n - 1, a_num + copy, copy)   #ctrl-V 
解释：按下ctrl-v 粘贴，剪贴板中的字符加入屏幕，同时消耗 1 个操作数

dp(n - 2, a_num, a_num)         #ctrl-A, ctrl-C
解释：全选和复制必然是联合使用的，
剪切板中A的数量 变为屏幕上 A的数量，同时消耗 2 个操作数

// 代码如下：
def maxA(N: int) -> int:
    # 对于(n, a_num, copy) 这个状态 
    # 屏幕上能最终最多能有 dp(n, a_num, copy) 个 A
    def dp(n, a_num, copy):
        # base case
        if n <= 0: return a_num;
        return max(dp(n - 1, a_num + 1, copy),    #A
                   dp(n - 1, a_num + copy, copy), #ctrl-V 
                   dp(n - 2, a_num, a_num) 

            )
    # 可以按 N 次 按键，屏幕和剪切板里都还没有 A 
    return dp(N, 0, 0)

// 以下 消除一下 重叠子问题：
def maxA(N: int) -> int:
    # 备忘录
    memo = dict()
    def dp(n, a_num, copy):
        if n <= 0: return a_num;
        if (n, a_num, copy) in memo:
            return memo[(n, a_num, copy)]

        memo[(n, a_num, copy)] = max(dp(n - 1, a_num + 1, copy),    #A
                   dp(n - 1, a_num + copy, copy), #ctrl-V 
                   dp(n - 2, a_num, a_num) 
            )
        return memo[(n, a_num, copy)]

    return dp(N, 0, 0)
// 以上算法的时间复杂度比较高，不是很好。


第二种思路，比较优的解法：
分析：这次我们只定义一个 [状态]， 也就是剩余的敲击次数n。
这个算法是 基于这样的一个事实，最优按键序列一定只有两种情况：
1）要么一直按A: A,A,A,...,A(当N比较小时)。
2）要么是这么个形式：A,A,...,C-A,C-C,C-V,...C-V(当N比较大时)
明确之后，可以通过两种情况来设计算法：

int[] dp = new int[N + 1];
// 定义：dp[i] 表示 i 次操作后，最多能显示多少个A
for (int i = 0; i <= N; i++) {
    dp[i] = max(
            这次按A键，
            这次按ctrl-V键
        )
}

代码如下：
int maxA(int N) {
    vector<int> dp(N + 1， 0);
    dp[0] = 0;
    for(int i = 1; i <= N; i++) {
        // 按 A 键
        dp[i] = dp[i - 1] + 1;
        for (int j = 2; j < i; j++) {
            // 全选 & 复制 dp[j - 2], 连续粘贴 i - j 次
            // 屏幕上共 dp[j - 2] * (i - j + 1) 个A
            dp[i] = max(dp[i], dp[j - 2] * (i - j + 1);
        }
    }
    // N 次 按键之后 最多要有几个 A ?
    return dp[N];
}



// Part5 其它经典问题：动态规划
// 示例7：经典动态规划：KMP算法详解
// 题目描述：用pat表示模式串，长度为M，txt表示文本串，长度为N。
// KMP算法 是在 txt中查找子串 pat，如果存在，返回这个子串的起始索引，否则返回-1

// 解法1：暴力解法
int search(string pat, string txt) {
    int M = pat.size();
    int N = txt.size();
    for(int i = 0; i <= N - M; i++) {
        int j = 0;
        for (j = 0; j < M; j++) {
            if (pat[j] != txt[i + j])
                break;
        }
        // pat 全部匹配了
        if(j == M) return i;
    }
    return -1;
}

// 解法2：KMP算法，动态规划
public class KMP{
    private int[][] dp;
    private String pat;

    public KMP(String pat) {
        this.pat = pat;
        int M = pat.length();
        // dp[状态][字符] = 下一个状态
        dp = new int[M][256];
        // base case
        dp[0][pat.CharAt(0)] = 1;
        // 影子状态 X 初始化为0
        int X = 0;
        // 当前状态 j 从 1 开始
        for(int j = 1; j < M; j++) {
            for (int c = 0; c < 256; c++) {
                if (pat.CharAt(j) == c)
                    dp[j][c] = j + 1;
                else
                    dp[j][c] = dp[X][c];
            }
            // 更新影子状态
            X = dp[X][pat.CharAt(j)];
        }
    }

    public int search(String txt) {
        int M = pat.length();
        int N = txt.length();
        // pat的初始态为0
        for(int i = 0; i < N; i++) {
            // 计算pat 的下一个状态
            j = dp[j][txt.CharAt(i)];
            // 到达终止态，返回结果
            if(j == M) return i - M + 1;
        }
        // 没有达到终止态，匹配失效
        return -1;
    }
}


// Part5 其它经典问题：动态规划
// 示例8：经典动态规划：回文问题终极篇，最小代价构造回文串
// leetcode 1312, 让字符串称为回文串的最小插入次数。
// 这道题比较难，题目描述
/*
  给你一个字符串 s, 每一次操作你都可以在字符串的任意位置插入任意字符。
  请你返回让 s 称为回文串的 最少操作次数。
  [回文串] 是正读和反读 都相同的字符串。
  
  举例：s = "abcde", 算法返回 2,
  因为可以给 s 插入 2 个字符，变成回文串 "abeceba" 或者"aebcbea"
  s = "aba", 则算法返回0;
*/
解题：
dp[i][j]的定义如下：对字符串s[i...j]，最少需要进行dp[i][j]次插入才能变成回文串。
根据这个定义，我们要求整个s的最少插入次数，也就是想求dp[0][n-1]的大小；

// code 如下：
int minInsertions(string s) {
    int n = s.size();
    // 定义：对s[i...j], 最少需要插入dp[i][j] 次才能变成回文
    vector<vector<int>> dp(n, vector<int>(n, 0));
    // base case: i == j时，dp[i][j] = 0, 单个字符本身就是回文
    // dp数组 已经完全初始化为0，base case 已经初始化

    // 从下往上，从左往右遍历
    for(int i = n - 2; i >= 0; i--) {
        // 从左往右遍历
        for(int j = i + 1; j < n; j++) {
            // 根据s[i] 和 s[j]进行状态转移
            if(s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1;
            }
        }
    }
    return dp[0][n - 1];
}

// 进行状态压缩，到一维
int minInsertions(string s) {
    int n = s.size();
    std::vector<int> dp(n, 0);
    int temp = 0;
    for(int i = n - 2; i >= 0; i--) {
        // 记录dp[i+1][j-1];
        int pre = 0;
        for (int j = i + 1; j < n; j++) {
            temp = dp[j];
            if (s[i] = s[j]) {
                // dp[i][j] = dp[i+1][j-1];
                dp[j] = pre;
            } else {
                // dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1;
                dp[j] = min(dp[j], dp[j - 1]) + 1;
            }
            pre = temp;
        }
    }
    return dp[n - 1];-0iuu
}


