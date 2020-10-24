//动态规划
具体来说，动态规划的一般流程就是三步：暴力的递归解法 -> 带备忘录的递归解法 -> 迭代的动态规划解法。
就思考流程来说，就分为一下几步：找到状态和选择 -> 明确 dp 数组/函数的定义 -> 寻找状态之间的关系。
https://labuladong.gitbook.io/algo/dong-tai-gui-hua-xi-lie

首先，动态规划问题的一般形式就是求最值。动态规划其实是运筹学的一种最优化方法，只不过在计算机问题上应用比较多，比如说让你求最长递增子序列呀，最小编辑距离呀等等。
既然是要求最值，核心问题是什么呢？求解动态规划的核心问题是穷举。因为要求最值，肯定要把所有可行的答案穷举出来，然后在其中找最值呗。
动态规划这么简单，就是穷举就完事了？我看到的动态规划问题都很难啊！
首先，动态规划的穷举有点特别，因为这类问题存在「重叠子问题」，如果暴力穷举的话效率会极其低下，所以需要「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。
而且，动态规划问题一定会具备「最优子结构」，才能通过子问题的最值得到原问题的最值。
另外，虽然动态规划的核心思想就是穷举求最值，但是问题可以千变万化，穷举所有可行解其实并不是一件容易的事，只有列出正确的「状态转移方程」才能正确地穷举。
以上提到的重叠子问题、最优子结构、状态转移方程就是动态规划三要素。具体什么意思等会会举例详解，但是在实际的算法问题中，写出状态转移方程是最困难的，这也就是为什么很多朋友觉得动态规划问题困难的原因，我来提供我研究出来的一个思维框架，辅助你思考状态转移方程：
明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义。
按上面的套路走，最后的结果就可以套这个框架：

# 初始化 base case
dp[0][0][...] = base
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)


//经典例题
//322. 零钱兑换
//给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
示例 1:

输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1
//方法1，动态规划，动态规划-自上而下,采用备忘录，消除重叠计算
vector<int> mem;
int dp(vector<int>& coins, int amount) {
    if (amount == 0) return 0;
    if (amount < 0) return -1;
    if (mem[amount - 1]) return mem[amount - 1];
    int min = INT_MAX;
    for (auto coin : coins) {
        int res = dp(coins, amount - coin);
        if (res >= 0 && res < min)
            min = res + 1;
    }
    if (min == INT_MAX)
        mem[amount - 1] = -1;
    else
        mem[amount -1] = min;
    return mem[amount - 1];
}
int coinChange(vector<int>& coins, int amount) {
    if (amount < 1) return 0;
    mem.resize(amount);
    return dp(coins, amount);
}
//方法2：动态规划：自下而上，采用dp table记录
//自底向上
int coinChange(vector<int>& coins, int amount) {
    if (amount == 0) return 0;
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < coins.size(); j++) {
            if (i - coins[j] >= 0) {
                dp[i] = min(dp[i], 1 + dp[i - coins[j]]);
            }
        }
    }
    if (dp[amount] == amount + 1)
        return -1;
    return dp[amount];
}

//494. 目标和
给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。
返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
解题来源：https://labuladong.gitbook.io/algo/dong-tai-gui-hua-xi-lie/targetsum
//方法1，回溯法，超时
/* 回溯算法模板 */ //超时了
void backtrack(vector<int>&nums, int i, int rest) {
    // base case
    if (i == nums.size()) {
        if (rest == 0) {
            // 说明恰好凑出 target
            result++;
        }
        return;
    }
    // 给 nums[i] 选择 - 号
    rest += nums[i];
    // 穷举 nums[i + 1]
    backtrack(nums, i + 1, rest);
    // 撤销选择
    rest -= nums[i];

    // 给 nums[i] 选择 + 号
    rest -= nums[i];
    // 穷举 nums[i + 1]
    backtrack(nums, i + 1, rest);
    // 撤销选择
    rest += nums[i];
}

//方法2: 采用dp 备忘录解决重复问题
unordered_map<string, int> umap;
int findTargetSumWays3(vector<int>&nums, int target) {
    if (nums.size() == 0) return 0;
    return dp(nums, 0, target);
    return result;
}
int dp(vector<int>&nums, int i, long long rest) {
    // base case
    if (i == nums.size()) {
        if (rest == 0) return 1;
        return 0;
    }
    string key =to_string(i) + ":" + to_string(rest);
    if (umap.count(key))
        return umap[key];
    int result = dp(nums, i + 1, rest - nums[i]) + dp(nums, i + 1, rest + nums[i]);
    umap[key] = result;
    return result;
}

//方法3：/动态规划，转为子集合划分问题，而子集合划分问题就是典型的背包问题。
//如果我们把 nums 划分成两个子集 A 和 B，分别代表分配 + 的数和分配 - 的数，那么他们和 target 存在如下关系：
//sum（A）- sum(B) = target
//sum(A) = target + sum(B)
//sum(A) + sum(A) = target + sum(A) + sum(B)
//sum(A) = (target + sum(nums)) / 2
int findTargetSumWays(vector<int>&nums, int target) {
    int sum = 0;
    for (int n : nums) sum += n;
    // 这两种情况，不可能存在合法的子集划分
    if (sum < target || (sum + target) % 2 == 1) {
        return 0;
    }
    return subsets(nums, (sum + target) / 2);
}
int subsets(vector<int> &nums, int sum) {
    int n = nums.size();
    vector<vector<int>> dp(n + 1, vector<int>(sum + 1, 0));
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 1;
    }
    for (int i = 1; i <= n; i++) {
        for(int j = 0; j <= sum; j++) {
            if (j >= nums[i - 1]) {
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
            //cout << dp[i][j] << endl;
        }
    }
    return dp[n][sum];
}

//方法4：动态规划降维，终极版
int subsets(vector<int> &nums, int sum) {
    int n = nums.size();
    vector<int> dp(sum + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++) {
        for(int j = sum; j >= 0; j--) {
            if (j >= nums[i - 1]) {
                dp[j] = dp[j] + dp[j - nums[i - 1]];
            } else {
                dp[j] = dp[j];
            }
            //cout << dp[i][j] << endl;
        }
    }
    return dp[sum];
}

//5. 最长回文子串
//给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
/*示例 1：
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
*/
//方法1：双指针法
pair<int, int> expandAroundCenter(string &s, int left, int right) {
    while(left >= 0 && right < s.size()
          && s[left] == s[right]) {
        --left;
        ++right;
    }
    return {leff + 1, right - 1};
}

string longestPalindrome(string s) {
    int start = 0, end = 0;
    for (int i = 0; i < s.size(); i++) {
        auto [left1, right1] = expandAroundCenter(s, i, i);
        auto [left2, right2] = expandAroundCenter(s, i, i + 1);
        if (right1 - left1 > end - start) {
            start = left1;
            end = right1;
        }
        if (right2 - left2 > end - start) {
            start = left2;
            end = right2;
        }
    }
    return s.substr(start, end - start + 1);
}
//方法2：动态规划法
string longestPalindrome(string s) {
    int n = s.size();
    vector<vector<bool>> dp(n, vector<bool>(n));
    string ans;

    for (int len = 0; len < n; len++) {
        for (int i = 0; i + len < n; i++) {
            int j = i + len;
            if (0 == len) {
                dp[i][j] = true;
            } else if (1 == len) {
                dp[i][j] = (s[i] == s[j]);
            } else {
                dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
            }
            if (dp[i][j] && len + 1 > ans.size()) {
                ans = s.substr(i, len + 1);
            }
        }
    }
    return ans;
}
//509. 斐波那契数
/*斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
给定 N，计算 F(N)。
*/
//方法1：递归方法
int fib2(int N) {
    if (N == 1 || N == 2) return 1;
    return fib(N - 1) + fib(N - 2);
}
//方法2：带备忘录的递归解法
int fib3(int N) {
    if (N < 1) return 0;
    vector<int> mem(N + 1, 0);
    return helper(mem, N);
}
int helper(vector<int>& mem, int N) {
    //base case
    if (N == 1 || N == 2) return 1;
    //计算过的
    if (mem[N]) return mem[N];
    mem[N] = helper(mem, N - 1) + helper(mem, N - 2);
    return mem[N];
}
//方法3：dp数组的迭代解法
//我们可以把这个「备忘录」独立出来成为一张表，就叫做 DP table 吧，
//在这张表上完成「自底向上」的推算
int fib4(int N) {
    if (N < 1) return 0;
    if (N == 1 || N == 2) return 1;
    vector<int>dp(N + 1, 0);
    dp[1] = dp[2] = 1;
    for (int i = 3; i <= N; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[N];
}
//方法4：dp数组的迭代解法,去除dp数组，降低空间复杂度
//我们可以把这个「备忘录」独立出来成为一张表，就叫做 DP table 吧，
//在这张表上完成「自底向上」的推算
int fib(int N) {
    if (N < 1) return 0;
    if (N == 1 || N == 2) return 1;
    int pre = 1, cur = 1, sum = 0;
    for (int i = 3; i <= N; i++) {
        sum = pre + cur;
        pre = cur;
        cur = sum;
    }
    return sum;
}
//322. 零钱兑换
/*给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
示例 1:
输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1
*/
    
//方法一、搜索回溯 [超出时间限制]
int coinChange(int idx, vector<int>& coins, int amount) {
    if (amount == 0) return 0;
    if (idx < coins.size() && amount > 0) {
        int numCoins = amount / coins[idx];
        //cout << "numCoins: " << numCoins << endl;
        int minNum = INT_MAX;
        for (int i = 0; i <= numCoins; i++) {
            int cur = i * coins[idx];
            if (amount >= cur) {
                int res = coinChange(idx + 1, coins, amount - cur);
                if (res != -1) minNum = min(minNum, res + i);
            }
        }
        return minNum == INT_MAX ? -1 : minNum;
    }
    return -1;
}
int coinChange2(vector<int>& coins, int amount) {
    return coinChange(0, coins, amount);
}
//方法二、动态规划-自上而下 [通过]
vector<int> count;
int coinChange3(vector<int>& coins, int amount) {
    if (amount < 1) return 0;
    count.resize(amount);
    return dp(coins, amount);
}
int dp(vector<int>& coins, int rem) {
    if (rem < 0) return -1;
    if (rem == 0) return 0;
    if (count[rem - 1]) return count[rem - 1];
    int Min = INT_MAX;
    for (int coin : coins) {
        int res = dp(coins, rem - coin);
        if (res >= 0 && res < Min) {
            Min = res + 1;
        }
    }
    count[rem - 1] = Min == INT_MAX ? -1 : Min;
    return count[rem - 1];

}
//方法三、动态规划：自下而上 [通过]
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < coins.size(); ++j) {
            if (i >= coins[j]) {
                dp[i] = min(dp[i], dp[i - coins[j]] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}

//300. 最长上升子序列
给定一个无序的整数数组，找到其中最长上升子序列的长度。
示例:
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。

//方法1: 动态规划
//状态转移方程：dp[i]=max(dp[j])+1,其中0≤j<i且num[j]<num[i]
int lengthOfLIS2(vector<int>& nums) {
    //定义dp[i]为以 nums[i]结尾的最长上升子序列的长度
    int n = nums.size();
    if (n <= 1) return n;
    vector<int>dp(n, 1);
    for (int i = 0; i < n; i++) {
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

//方法2：二分查找
//纸牌游戏思路

int lengthOfLIS(vector<int>& nums) {
    vector<int> top(nums.size(), 0);
    //牌堆的数量
    int piles_num = 0;
    for (int i = 0; i != nums.size(); i++) {
        //要处理的poker
        int poker = nums[i];
        int left = 0, right = piles_num;
        //搜索左侧边界的二分查找
        while(left < right) {
            int mid = (left + right) / 2;
            if (top[mid] > poker) {
                right = mid;
            } else if (top[mid] < poker) {
                left = mid + 1;
            } else {
                right = mid;
            }
         //while 循环处理完
        }
        //如果找不到合适的牌堆，新建牌堆
        if(left == piles_num) piles_num++;
        top[left] = poker;
    }
    return piles_num;
}

//53. 最大子序和
/*给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
*/

//方法1：动态规划，
//以dp[i]表示已nums[i]结尾的最大和连续子数组，那么可以求出每个dp[i] = max(nums[i] + dp[i - 1], nums[i])
//所以dp[i]中取其大者即可
//时间复杂度O(N),空间复杂度O(N)
int maxSubArray2(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    vector<int> dp(n, 0);
    dp[0] = nums[0];
    for (int i = 1; i != n; i++) {
        dp[i] = max(nums[i], dp[i - 1] + nums[i]);
    }

    int res = INT_MIN;
    for (int i = 0; i != n; i++) {
        res = max(dp[i], res);
    }
    return res;
}

//方法2：动态规划
//因为迭代公式中，dp[i]仅与dp[i - 1]相关，所以可以用两个变量进行替换。
//dp0为初始值，更新dp1；然后将dp0 = dp1,
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    int dp0 = nums[0];
    int dp1 = 0
    int res = dp0;

    for (int i = 1; i != n; i++) {
        dp1 = max(nums[i], dp0 + nums[i]);
        dp0 = dp1;
        res = max(res, dp1);
    }
    return res;
}

//416. 分割等和子集
/*
给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
注意:
每个数组中的元素不会超过 100
数组的大小不会超过 200
示例 1:
输入: [1, 5, 11, 5]
输出: true
解释: 数组可以分割成 [1, 5, 5] 和 [11].
*/
//方法1：动态规划，本质上背包问题
//那么对于这个问题，我们可以先对集合求和，得出 sum，把问题转化为背包问题：
//给一个可装载重量为 sum / 2 的背包和 N 个物品，每个物品的重量为 nums[i]。现在让你装物品，是否存在一种装法，能够恰好将背包装满？
bool canPartition2(vector<int>& nums) {
    int sum = 0;
    for (int num : nums) {
        sum += num;
    }
    if (sum % 2 != 0) return false;
    sum = sum / 2;
    int n = nums.size();
    vector<vector<bool>> dp(n + 1, vector<bool>(sum + 1, false));
    for (int i = 0; i <= n; i++) {
        dp[i][0] = true;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= sum; j++) {
            if (j - nums[i - 1] < 0) {
                //背包容量不够，不能装入第i个物品
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
            }
        }
    }

    return dp[n][sum];
}
//方法2：动态规划，进行状态压缩
//再进一步，是否可以优化这个代码呢？注意到 dp[i][j] 都是通过上一行 dp[i-1][..] 转移过来的，之前的数据都不会再使用了。
//这就是状态压缩，其实这段代码和之前的解法思路完全相同，只在一行 dp 数组上操作，i 每进行一轮迭代，dp[j] 其实就相当于 dp[i-1][j]，所以只需要一维数组就够用了。
//唯一需要注意的是 j 应该从后往前反向遍历，因为每个物品（或者说数字）只能用一次，以免之前的结果影响其他的结果。
//至此，子集切割的问题就完全解决了，时间复杂度 O(n*sum)，空间复杂度 O(sum)。
bool canPartition(vector<int>& nums) {
    int sum = 0;
    for (int num : nums) {
        sum += num;
    }
    if (sum % 2 != 0) return false;
    sum = sum / 2;
    int n = nums.size();
    vector<bool> dp(sum + 1, false);
    //base case
    dp[0] = true;
    for (int i = 0; i < n; i++) {
        for (int j = sum; j >= 0; j--) {
            if (j - nums[i] >= 0) {
                dp[j] = dp[j] || dp[j - nums[i]];
            }
        }

    }
    return dp[sum];
}




