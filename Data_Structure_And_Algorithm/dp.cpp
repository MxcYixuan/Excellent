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
