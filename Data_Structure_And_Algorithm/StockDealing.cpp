// https://labuladong.gitbook.io/algo/dong-tai-gui-hua-xi-lie/1.5-qi-ta-jing-dian-wen-ti/tuan-mie-gu-piao-wen-ti

// 股票交易第一题：k = 1
// dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
// dp[i][1][1] = max(dp[i-1][1][1], dp[i-1][0][0] - prices[i])
//             = max(dp[i-1][1][1], -prices[i])
// 解释：k = 0 的 base case，所以 dp[i-1][0][0] = 0。
// 现在发现 k 都是 1，不会改变，即 k 对状态转移已经没有影响了。
// 可以进行进一步化简去掉所有 k：
// dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
// dp[i][1] = max(dp[i-1][1], -prices[i])
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
// 这个代码不是所以的case 都能通过，需要再看下
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

