// 第三章
// 必会算法技巧

// Part3.1 回溯算法(DFS算法)系列
// 题目：3.1.1 回溯算法解题套路框架
0.1 解决一个回溯算法，实际上就是一个决策树的遍历过程，只需要思考3个问题：
	1. 路径：也就是已经做出的选择
	2. 选择列表：也就是你当前可以做的选择
	3. 结束条件：也就是达到决策树底层，无法在做选择的条件。

0.2 回溯算法的框架：
result = []
def backtrack(路径，选择列表):
	if 满足结束条件:
		result.add(路径)
		return

	for 选择 in 选择列表:
		做选择
		backtrack(路径，选择列表)
		撤销选择
;

1.全排列问题
多叉树的遍历框架：
void traverse(TreeNode root) {
	for(TreeNode child : root.children)
		// 前序遍历需要的操作
		traverse(child);
		// 后续遍历需要的操作
}

1.1 前序遍历的代码 在进入某一个节点之前的那个时间点执行，
后续遍历代码在离开某个节点之后的那个时间点执行。

for 选择 in 选择列表:
	# 做选择
	将该选择从选择列表移出
	路径.add(选择)
	backtrack(路径，选择列表)
	# 撤销选择
	路径.remove(选择)
	将该选择再加入选择列表

我们只要在递归之前作出选择，在递归之后撤销刚才的选择。就能正确得到每个节点的选择列表和路径。

Java代码实现:
List<List<Integer>> res = new LinkedList<>();
/* 主函数，输入一组不重复的数字，返回它们的全排列 */
List<List<Integer>> permute(int[] nums) {
	// 记录 路径
	LinkedList<Integer> track = new LinkedList<>();
	backtrack(nums, track);
}

// 路径：记录在 track 中
// 选择列表：nums中不存在于 track 的那些元素
// 结束条件：nums中的元素全都在trace中出现
void backtrack(int[] nums, LinkedList<Integer> track) {
	// 触发结束条件
	if(track.size() == nums.length) {
		res.add(new LinkedList(track));
		return;
	}

	for(int i = 0; i < nums.length; i++) {
		// 排除不合法的选择
		if(track.contains(nums[i])) {
			continue;
		}
		// 做选择
		track.add(nums[i]);
		// 进入下一层决策树
		backtrack(nums, track);
		// 取消选择
		track.removeLast();
	}
}


1.2 C++解法，版本1
vector<vector<int>> result;
vector<int> path;

vector<vector<int>> premute(vector<int>& nums) {
	vector<bool> user(nums.size(), false);
	backtracking(nums, used);
	return result;
}

void backtracking(vector<int>& nums, vector<bool>& used) {
	if(path.size() == nums.size()) {
		result.push_back(path);
		return;
	}
	for(int i = 0; i < nums.size(); i++) {
		if(used[i] == true) 
			continue;
		used[i] = true;
		path.push_back(nums[i]);
		backtracking(nums, used);
		path.pop_back();
		used[i] = false;
	}
}

1.3 C++解法，版本2
vector<vector<int>> permute3(vector<int>& nums) {
	vector<vector<int>> res;
	vector<int> path;
	vector<bool> visited(nums.size(), false);
	back_tracking(res, path, nums, visited);
	return res;
}

void back_tracking(vector<vector<int>>& res, vector<int>& path, vector<int>& nums, vector<bool> visited) {
	if(path.size() == nums.size()) {
		res.push_back(path);
		return;
	}
	for(int i = 0; i < nums.size(); i++) {
		if(visited[i]) continue;
		visited[i] = true;
		path.push_back(nums[i]);
		back_tracking(res, path, nums, visited);
		path.pop_back();
		visited[i] = false;
	}
}

2. N 皇后问题
描述：给你一个N*N的棋盘，让你放置N个皇后，使得它们不能互相攻击。
注：皇后可以攻击同一行，同一列，左上左下右上右下的任意单位。

vector<vector<string>> res;

/* 输入棋盘变长 n, 返回所有合法的放置 */
vector<vector<string>> solveNQueens(int n) {
	// '.' 表示空，'Q' 表示皇后，初始化棋盘。
	vector<string> board(n, string(n, '.'));
	backtrack(board, 0);
	return res;
}

// 路径：board 中 小于 row 的那些行 都已经成功放置了皇后
// 选择列表：第 row 行 的所有列 都是放置皇后的选择
// 结束条件：row 超过 board 的最后一行
void backtrack(vector<string>& board, int row) {
	// 触发结束条件
	if(row == board.size()) {
		res.push_back(board);
		return;
	}

	int n = board[row].size();
	for(int col = 0; col < n; col++) {
		// 排除不合法的选择
		if(!isValid(board, row, col))
			continue;
		// 做选择
		board[row][col] = 'Q';
		backtrack(board, row + 1);
		// 撤销选择
		board[row][col] = '.';
	}
}

/* 是否可以在 board[row][col] 放置皇后？ */
bool isValid(vector<string>& board, int row, int col) {
	int n = board.size();
	// 检查 列 是否有皇后互相冲突
	for(int i = 0; i < n; i++) {
		if(board[i][col] == 'Q') {
			return false;
		}
	}
	// 检查右上方是否有有皇后互相冲突
	for(int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
		if(board[i][j] == 'Q')
			return false;
	}

	// 检查左上方 是否有皇后互相冲突
	for(int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
		if(board[i][j] == 'Q')
			return false;
	}
	return true;
}

3.0 总结：
写 backtrack函数时，需要维护走过的路径 和 当前可以做的 选择列表，当触发结束条件时，将路径记入结果集。


// Part3.1 回溯算法(DFS算法)系列
// 题目：3.1.2 回溯算法 团灭 排列/组合/子集问题
0. 题目为三道高频题，分别是求子集(subset), 求排列(permutation), 求组合(combination)

1.子集
问题：输入一个不包含重复数字的数组，要求算法输出这些数字的所有子集
比如：输入 nums = [1, 2, 3], 你的算法应该输出 8 个子集，包含空集和本身，顺序可以不同
[[],[1],[2],[3],[1,3],[2,3],[1,2],[1,2,3]]

分析：可以采用数学归纳法，发现：
subset([1, 2, 3]) - subset([1, 2]) = [3],[1,3],[2,3],[1,2,3]
而本身[1,2]的子集为：[[], [1], [2], [1, 2]]

所以可以用递归的方法去做;

vector<vector<int>> subsets(vector<int>& nums) {
	// base case， 返回一个空集
	if(nums.empty()) return {{}};
	// 把最后一个元素拿出来
	int n = nums.back();
	nums.pop_back();
	// 先递归算出 前面元素的所有子集
	vector<vector<int>> res = subsets(nums);
	int size = res.size();
	for(int i = 0; i < size; i++) {
		// 然后在之前的结果上追加
		res.push_back(res[i]);
		res.back().push_back(n);
	}
	return res;
} 
时间复杂度比较高：O(N*2^N)

1.2 回溯算法 
vector<vector<int>> res;
vector<vector<int>> subsets(vector<int>& nums) {
	// 记录走过的路径
	vector<int> track;
	backtrack(nums, 0, track);
	return res;
}

void backtrack(vector<int>& nums, int start, vector<int>& track) {
	res.push_back(track);

	// 注意 i 从 start 开始递增
	for(int i = start; i < nums.size(); i++) {
		// 做选择
		track.push_back(nums[i]);
		// 回溯
		backtrack(nums, i + 1, track);
		// 撤销选择
		track.pop_back();
	}
}

2.0 组合
题目：输入两个数字n, k, 算法输出[1..n]中 k 个数字的所有组合
直接套框架
vector<vector<int>> res;
vector<vector<int>> combine(int n, int k) {
	if(k <= 0 || n <= 0) return res;
	vector<int> track;
	backtrack(n, k, 1, track);
	return res;
}

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

3.0 排列，已讲

// Part3.1 回溯算法(DFS算法)系列
// 题目：3.1.3 解数独问题
题目描述：
编写一个程序，通过填充空格来解决数独问题。
一个数独的解法需遵循如下规则：

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
空白格用 '.' 表示。

1.0 解法1：
void solveSudoku(vector<vector<char>>& board) {
	backtrack(board, 0, 0);
}
bool backtrack(vector<vector<char>>& board, int row, int col) {
	int m = 9, n = 9;
	if(col == n) {
		return backtrack(board, row + 1, 0);
	}
	if(row == m) return true;
	for(int i = row; i < m; i++) {
		for (int j = col; j < n; j++) {
			if(board[i][j] != '.') {
				return backtrack(board, i, j + 1);
			}

			for(char ch = '1'; ch <= '9'; ch++) {
				if(!isValid(board, i, j, ch)) continue;
				board[i][j] = ch;
				if(backtrack(board, i, j + 1)) 
					return true;
				board[i][j] = '.';
			}
			return false;
		}
	}
}

bool isValid(vector<vector<char>>& board, int row, int col) {
	for(int i = 0; i < 9; i++) {
		if(board[row][i] == val) return false;
		if(board[i][col] == val) return false;

		if(board[(row/3)*3 + i / 3][(col/3)*3 + i % 3] == val) {
			return false;
		}
	}
	return true;
}


1.1 解法2：
void solveSudoku(vector<vector<char>>& board) {
	backtrack(board);
}
bool backtracking(vector<vector<char>>& board) {
	for(int i = 0; i < board.size(); i++) {
		for(int j = 0; j < board[0].size(); j++) {
			if(board[i][j] != '.')
				continue;
			for(char ch = '1'; ch <= '9'; ch++) {
				if(isValid(i, j, ch, board)) {
					board[i][j] = ch;
					if(backtracking(board)) return true;
					board[i][j] = '.';
				}
			}
			return false;
		}
	}
	return true;
}

bool isValid(int row, int col, char val, vector<vector<char>>& board) {
	for(int i = 0; i < 9; i++) {
		if(board[row][i] == val) return false;
		if(board[i][col] == val) return false;

		int statRow = (row / 3) * 3;
		int statCol = (col / 3) * 3;
		for(int i = statRow; i < statRow + 3; i++) {
			for(int j = statCol; j < statCol + 3; j++) {
				if(board[i][j] == val)
					return false;
			}
		}
	}
	return true;
}


// Part3.1 回溯算法(DFS算法)系列
// 题目：3.1.4 合法括号生成
0. 括号问题可以简单的分成两类：
   一类是：判断括号的合法性，可以借助 栈 数据结构
   一类是：合法括号的生成，一般要用到回溯的思想

// leetcode 22
题目：请你写一个算法，输入是一个正整数n, 输出是 n 对括号的所有合法组合。
vector<string> generateParaenthesis(int n);
0.1 两个性质；
    1.一个合法括号组合的左括号数量，一定等于右括号数量，这个显而易见。
    2.对于一个合法的括号字符串组合p，必然对于任何 0 <= i < len(p)都有：
    子串p[0...i] 中 左括号的数量都大于或等于 右括号的数量。

先套一下回溯算法的框架，伪代码如下：
void backtrack(int n, int i, string& track) {
	// i 代表当前的位置，共 2n 个位置
	// 穷举到最后一个位置了，得到一个长度为 2n 组合
	if(i == 2 * n) {
		print(track);
		return;
	}
	// 对于每个位置可以是左括号 或者 右括号两种选择
	for choice in ['(', ')'] {
		track.push(choice); // 做选择
		// 穷举下一个位置
		backtrack(n, i + 1, track);
		track.pop(choice); // 撤销选择
	}
}

正确的代码如下：
vector<string> generateParaenthesis(int n) {
	if(n == 0) return {};
	// 记录所有合法的括号组成
	vector<string> res;
	// 回溯过程中的路径
	string track;
	// 可用的左括号和右括号数量，初始化为n
	backtrack(n, n, track, res);
	return res;
}

// 可用的左括号数量为 left 个，可用的右括号数量为 right 个；
void backtrack(int left, int right, string& track, vector<string>& res) {
	// 左括号剩余得多，说明不合法；
	if(right < left) return ;
	// 数量小于0 肯定是不合法的
	if(left < 0 || right < 0)
		return;
	// 当所有括号都恰好用完时，得到一个合法的括号组合
	if(left == 0 && right == 0) {
		res.push_back(track);
		return;
	}
	// 尝试放一个左括号
	track.push_back('(');
	backtrack(left - 1, right, track, res);
	track.pop_back();

	// 尝试放一个右括号
	track.push_back(')');
	backtrack(left, right - 1, track, res);
	track.pop_back();
}


// Part3.2 BFS 算法系列
// 题目：3.2.1 BFS 算法解题套路框架
0. BFS算法的核心思想是：
   就是把一些问题抽象成图，从一个点开始，向四周开始扩散。
   一般来讲，BFS算法都是用 【队列】 数据结构，每次将一个节点周围的所有节点加入队列。

0.1 BFS 相对DFS 最主要的区别是：
	BFS 找到的路径一定是最短的，但代价就是空间复杂度比 DFS大很多。
0.2 算法框架：
	本质是，让你在一幅图中 找到从起点start到终点 target的最近距离，BFS 算法就是在干这个事

// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
	Queue<Node> q;     // 核心数据结构
	Set<Node> visited; // 避免走回头路

	q.offer(start);    // 将起点加入队列
	visited.add(start);
	int step = 0;

	while(q not empty) {
		int sz = q.size();
		/* 将当前队列中的所有节点向四周扩散 */
		for(int i = 0; i < sz; i++) {
			Node cur = q.poll();
			/* 重点：判断这里是否到达终点  */
			if(cur is target) {
				return step;
			}
			/* 将cur 的相邻节点加入队列 */
			for(Node x : cur.adj()) {
				if(x not in visited) {
					q.offer(x);
					visited.add(x);
				}
			}
		}
		// 划重点，更新步数在这里
		step++；
	} 
} 

1. 二叉树的最小高度
题目：给定一个二叉树，找出其最小深度

int minDepth(TreeNode* root) {
	if(root == nullptr) return 0;
	queue<TreeNode*> q;
	q.push(q);
	// root 本身就是一层，depth初始化为 1
	int depth = 1;

	while(!q.empty()) {
		int sz = q.size();
		/* 将当前队列中的所有节点 向四周扩散 */
		for(int i = 0; i < sz; i++) {
			TreeNode* cur = q.top();
			q.pop();
			/* 判断是否到达终点 */
			if(!cur->left && !cur->right) 
				return depth;
			/* 将 cur 的相邻节点 加入队列 */
			if(cur->left) {
				q.push(cur->left);
			}
			if(cur->right) {
				q.push(cur->right);
			}
		}
		depth++;
	}
	return depth;
}

2. 思考：
2.1 为什么 BFS 可以找到最短距离，DFS 不行吗？
   DFS也可以，但是需要把树中 所有树杈都探索万，才能找出最短路径
2.3 既然 BFS 这么好，为啥 DFS 还要存在？
	因为：BFS 可以找到最短距离，但是空间复杂度高，而 DFS 的空间爱你复杂度比较低
	DFS 无非就是递归堆栈，最坏情况下顶多 就是输的高度，也就是O(logN)

3.打开转盘锁
你有一个带有四个圆形拨轮的转盘锁。
每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。
每个拨轮可以自由旋转：例如把 '9' 变为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

示例1：
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。

int openLock(vector<string>& deadends, string target) {
    // 记录需要跳过的死亡密码
    set<string> deads;
    for(string s : deadends) {
        deads.insert(s);
    }
    // 记录已经穷举过的密码，防止走回头路
    set<string> visited;
    queue<string> q;
    // 从起点开始启动广度优先搜索
    int step = 0;
    q.push("0000");
    visited.insert("0000");
    while(!q.empty()) {
        int sz = q.size();
        /* 将当前队列中的所有节点 向周围扩散 */
        for(int i = 0; i < sz; i++) {
            string cur = q.front();
            q.pop();
            /* 判断是否到达终点 */
            if(deads.count(cur)) 
                continue;
            if(cur == target) {
                return step;
            }

            /* 将一个节点的未遍历相邻节点 加入队列 */
            for(int j = 0; j < 4; j++) {
                string up = plusOne(cur, j);
                if(!visited.count(up)) {
                    q.push(up);
                    visited.insert(up);
                }
                string down = minusOne(cur, j);
                if(!visited.count(down)) {
                    q.push(down);
                    visited.insert(down);
                }
            }
        }
        step++;
    }
    // 如果穷举完，都没找到目标密码，那就是找不到了
    return -1;
}
string plusOne(string s, int j) {
    string res = s;
    if(res[j] == '9') 
        res[j] = '0';
    else
        res[j] += 1;
    return res;
}

string minusOne(string s, int j) {
    string res = s;
    if(res[j] == '0') 
        res[j] = '9';
    else
        res[j] -= 1;
    return res;
}

4. 双向BFS优化
可以进一步提高算法的效率
传统BFS框架就是从起点开始 向四周扩散，遇到终点时停止；
而双向BFS则是从起点和终点同时开始扩散，档两边有交集时停止
int openLock(String[] deadends, String target) {
	Set<String> deads = new HashSet<>();
	for(String s : deadends)
		deads.add(s);
	// 用集合不用队列，可以快速判断元素是否存在
	Set<String> q1 = new HashSet<>();
	Set<String> q2 = new HashSet<>();

	Set<String> visited = new HashSet<>();

	int step = 0;
	q1.add("0000");
	q2.add(target);

	while(!q1.empty() && !q2.empty()) {
		// 哈希集合在遍历的过程中不能修改，用temp 存储扩散结果
		Set<String> temp = new HashSet<>();
		/* 将q1 中所有的节点向周围扩散 */
		for(String cur : q1) {
			if(deads.contains(cur))
				continue;
			if(q2.contains(cur)) {
				return step;
			}
			visited.add(cur);

			/* 将一个节点的未 遍历相邻节点 加入集合 */
			for(int j = 0; i < 4; j++) {
				String up = plusOne(cur, j);
				if(!visited.contains(up)){
					temp.add(up);
				}
				String down = minusOne(cur, j);
				if(!visited.contains(down)) {
					temp.add(down);
				}
			}
		}
		step++;
		// temp 相当于 q1
		// 这里交换 q1 q2, 下一轮while 就是扩散 q2
		q1 = q2;
		q2 = temp;
	}
	return -1;
}


// Part3.2 BFS 算法系列
// 题目：3.2.2 BFS 暴力搜索算法
// leetcode 773. 滑动谜题
题目描述：
在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示.
一次移动定义为选择 0 与一个相邻的数字（上下左右）进行交换.
最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。
给出一个谜板的初始状态，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。

示例1：
输入：board = [[1,2,3],[4,0,5]]
输出：1
解释：交换 0 和 5 ，1 步完成

int slidingPuzzle(vector<vector<int>>& board) {
	int m = 2, n = 3;
	string start = "";
	string target = "123450";

	// 2 * 3 数组转化为字符串
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			start.push_back(board[i][j] + '0');
		}
	}
	// 记录一堆字符串的相邻索引
	vector<vector<int>> neighbor = {
		{1, 3},
		{0, 4, 2},
		{1, 5},
		{0, 4},
		{3, 1, 5},
		{4, 2}
	};

	/* BFS 框架开始 */
	queue<string> q;
	unordered_set<string> visited;
	q.push(start);
	visited.insert(start);

	int step = 0;
	while(!q.empty()) {
		int sz = q.size();
		for(int i = 0; i < sz; i++) {
			string cur = q.front(); q.pop();
			// 判断是否达到目标局面
			if(target == cur) {
				return step;
			}
			// 找到数字 0 对应的索引
			int idx = 0;
			for(; cur[idx] != '0'; idx++) {
				string new_board = cur;
				swap(new_board[adj], new_board[idx]);
				// 防止走回头路
				if(!visited.count(new_board)) {
					q.push(new_board);
					visited.insert(new_board);
				}
			}
		}
		step++;
	}
	return -1;
	/* BFS 算法框架结束 */
}


// Part3.3 其它算法篇
// 题目：3.3.1 小而美的算法技巧：前缀和数组
题目：给定一个整数数组和一个整数k，你需要找到该数组中和为k的连续的子数组的个数。
示例1：
输入：nums = [1, 1, 1], k = 2
输出：2， [1, 1] 与 [1, 1]为两种不同的情况

1.前缀和实现 
思路：对于一个给定的数组nums, 我们额外开辟一个前缀和数组 进行预处理
int n = nums.length;
// 前缀和数组
int[] preSum = new int[n + 1];
preSum[0] = 0;
for(int i = 0; i < n; i++) {
	preSum[i + 1] = preSum[i] + nums[i];
}

1.1 借助前缀和 解法： 
int subarraySum(int[] nums, int k) {
	int n = nums.size();
	// 前缀和构造
	int[] sum = new int[n + 1];
	sum[0] = 0;
	for(int i = 0; i < n; i++) {
		sum[i + 1] = sum[i] + nums[i];
	}

	int ans = 0;
	// 穷举所有子数组
	for(int i = 1; i < n; i++) {
		for(int j = 0; j < i; j++) {
			// sum of nums[j..i-1]
			if(sum[i] - sum[j] == k) {
				ans++;
			}
		}
	}
	return ans;
}

时间复杂度为O(N^2), 空间复杂度为O(N)

1.2 优化解法：
int subarraySum(int[] nums, int k) {
	int n = nums.size();
	// map: 前缀和 -> 该前缀和出现的次数
	HashMap<Integer, Integer> preSum = new HashMap<>();
	// base case
	preSum.put(0, 1);

	int ans = 0, sum0_i = 0;
	for(int i = 0; i < n; i++) {
		sum0_i += nums[i];
		// 这是我们想要找的 前缀和 nums[0..j]
		int sum0_j = sum0_i - k;
		// 如果前面有这个前缀和，则直接更新答案
		if(preSum.containsKey(sum0_j)) {
			ans += preSum.get(sum0_j);
		}
		// 把前缀和 nums[0..i] 加入并记录出现次数
		preSum.put(sum0_i, preSum.getOrDefault(sum0_i, 0) + 1);
	}
	return ans;
}

1.3 总结：让你统计班上同学考试成绩在不同分数段的百分比，也可以利用前缀和 技巧
int[] scores; // 存储着所有同学的分数
// 试卷满分 150分
int[] count = new int[150 + 1];
// 记录每个分数 有几个同学
for(int score : scores) {
	count[score]++;
}
// 构造前缀和
for(int i = 1; i > count.length; i++) {
	count[i] = count[i] + count[i - 1];
}

// Part3.3 其它算法篇
// 题目：3.3.2 小而美的算法技巧：查分数组
// leetcode 1109
应用：
前缀和主要试用的场景是：原始数组不会被修改的情况下，频繁查询某个区间的累加和。
0.1 前缀和 核心代码
class PrefixSum {
	// 前缀和数组
	private int[] prefix;

	/* 输入一个数组，构造前缀和 */
	public PrefixSum(int[] nums) {
		prefix = new int[nums.length + 1];
		// 计算 nums 的累加和
		for(int i - 1; i < prefix.length; i++) {
			prefix[i] = prefix[i - 1] + nums[i - 1];
		}
	}
	/* 查询闭区间 [i, j]的累加和 */
	public int query(int i, int j) {
		return prefix[j + 1] - prefix[i];
	}
};

0.2 差分数组代码逻辑：
int[] res = new int[diff.length];
// 根据差分数组 构造结果数组
res[0] = diff[0];
for(int i = 1; i < diff.length; i++) {
	res[i] = res[i - 1] + diff[i];
}

// 把差分数组 抽象成一个类
class Difference {
	// 差分数组
	private int[] diff;

	public Difference(int[] nums) {
		assert nums.length > 0;
		diff = new int[nums.length];
		// 构造差分数组
		diff[0] = nums[0];
		for(int i = 1; i < nums.length; i++) {
			diff[i] = nums[i] - nums[i - 1];
		}

		/* 给闭区间 [i, j]增加 val (可以是负数) */
		public void increment(int i, int j, int val) {
			diff[i] += val;
			if(j + 1 < diff.length) {
				diff[j + 1] -= val;
			}
		}

		public int[] result() {
			int[] res = new int[diff.length];
			// 根据差分数组 构造结果数组
			res[0] = diff[0];
			for(int i = 1; i < diff.length; i++) {
				res[i] = res[i - 1] + diff[i];
			}
			return res;
		}
	}
};

1.算法实战
航班预订统计
这里有 n 个航班，它们分别从 1 到 n 进行编号。

有一份航班预订表 bookings ，表中第 i 条预订记录 bookings[i] = [firsti, lasti, seatsi] 
意味着在从 firsti 到 lasti （包含 firsti 和 lasti ）的 每个航班 上预订了 seatsi 个座位。
请你返回一个长度为 n 的数组 answer，其中 answer[i] 是航班 i 上预订的座位总数。

int[] corpFlightBookings(int[][] bookings, int n) {
	int[] nums = new int[n];
	// 构造差分解法
	Difference df = new Difference(nums);

	for(int[] booking : bookings) {
		// 注意 转成数组索引 要减一
		int i = booking[0] - 1;
		int j = booking[1] - 1;
		int val = booking[2];

		df.increment(i, j, val);
	}
	// 返回最终的结果数组
	return df.result();
}


// Part3.3 其它算法篇
// 题目：3.3.3 小快速排序亲兄弟：快速选择算法
题目描述：
给你输入一个无序数组nums和一个正整数 k, 让你计算nums中第 k 大的数
// leetcode 215 数组中的第K个最大元素
0.1 在未排序的数组中 找到 第 k 个最大的元素。请注意，你需要找的是数组排序后的第k个最大元素。
    而不是第 k 个元素。 
示例1：
输入：[3, 2, 1, 5, 6, 4] 和 k = 2 
输出：5 

1.解法1，二叉堆解法
int findKthLargest(vector<int>& nums, int k) {
	// 小顶堆，堆顶是最小元素
	priority_queue<int, vector<int>, greater<int>> pq;
	for(auto n : nums) {
		pq.push(n);
		// 堆中元素 多于 k 个时，删除堆顶元素
		if(pq.size() > k) {
			pq.pop();
		}
	}
	return pq.top();
}

时间复杂度为:O(NlogK)

2.解法2，快速选择排序
先写一下快速排序
/* 快速排序主函数 */
void sort(vector<int>& nums) {
	// 一般要在这 用洗牌算法将 nums数组打乱
	// 以保证较高的效率，我们暂时省略这个细节
	sort(nums, 0, nums.size() - 1);
}
/* 快速排序核心逻辑 */
void sort(vector<int>& nums, int low, int high) {
	if(low >= high) return;

	// 通过交换元素，构建分界点索引 p;
	int p = partition(nums, low, high);
	// 现在 nums[low...p-1] 都小于nums[p]
	// 且   nums[p+1...high] 都大于nums[p]
	sort(nums, low, p - 1);
	sort(nums, p + 1, high);
}

int partition(vector<int>& nums, int low, int high) {
	if(low == high) return low;

	// 将 nums[low] 作为 默认分界点 pivot
	int pivot = nums[low];
	// j = high + 1 因为 while中先执行 --
	while(true) {
		// 保证nums[low..i] 都小于 pivot
		while(nums[++i] < pivot) {
			if(i == high) break;
		}
		// 保证nums[j..high] 都大于 pivot
		while(nums[--j] > pivot) {
			if(j == low) break;
		}
		if(i >= j) break;
		// 如果走到这里，一定有
		// nums[i] > pivot && nums[j] < pivot
		// 所以需要交换 nums[i] 和 nums[j]
		// 保证 nums[low...i] < pivot < nums[j...high]
		swap(nums, i, j);
	}
	// 将 pivot 值交换到正确的位置
	swap(nums, j, low);
	// 现在 nums[low...j-1] < nums[j] < nums[j+1..high]
	return j;
}

// 交换数组中的两个元素
void swap(int[], int i, int j) {
	int temp = nums[i];
	nums[i] = nums[j];
	nums[j] = temp;
}
注：partition函数的细节比较多，上述代码才考了<算法4> 是最漂亮简洁的一种。

题目要 求的是[第K个最大元素]，这个元素其实就是nums升序排序后[索引]
为 len(nums) - k;
2.1 
int findKthLargest(vector<int>& nums, int k) {
	int low = 0, high = nums.size() - 1;
	// 索引转化
	k = nums.size() - k;
	while(low <= high) {
		// 在nums[low.high] 选择一个分界点
		int p = partition(nums, low, high);
		if(p < k) {
			// 第 k 大元素 在nums[p+1...high]
			low = p + 1;
		} else if (p > k) {
			// 第 k 大元素 在nums[low... p-1]
			high = p - 1;
		} else {
			// 找到第 k 大 元素
			return nums[p];
		}
	}
	return -1;
}

2.2 为了尽可能防止极端情况发生，我们需要在算法开始时，对nums数组来一次随机打乱。
int findKthLargest(int[] nums, int k) {
	// 首先随机打乱数组
	shuffle(nums);
	int low = 0, high = nums.length - 1;
	k = nums.length - k;
	while(low <= high) {
		// ...
	}
	return -1;
}

// 对数组元素 进行随机打乱
void shuffle(int[] nums) {
	int n = nums.length;
	Random rand = new Random();
	for(int i = 0; i < n; i++) {
		// 从 i 到最后随机选择一个元素
		int r = i + rand.nextInt(n - i);
		swap(nums, i, r);
	}
}
算法时间复杂度为O(N)


// Part3.3 其它算法篇
// 题目：3.3.4 分治算法详解：表达式的不同优先级
// leetcode 241： 分治算法思想
0.1 最典型的分治算法就是归并排序了，核心逻辑如下：
void sort(int[] nums, int low, int high) {
	int mid = (low + high) / 2;
	/************分*************/
	// 对数组的两部分进行排序
	sort(nums, low, mid);
	sort(nums, mid + 1, high);
	/************治************/
	// 合并两个排好序的子数组
	merge(nums, low, mid, high);
}


1.题目描述：为运算表达式设计优先级
给定一个含有数字和运算符的字符串，为表达式括号，改变其运算优先级以求出不同的结果。
你需要给出所有可能的组合的结果。有效的运算符号包含 + 、-以及 *.

示例1：
输入: "2-1-1"
输出: [0, 2]
解释: 
((2-1)-1) = 0 
(2-(1-1)) = 2

示例2：
输入: "2*3-4*5"
输出: [-34, -14, -10, -10, 10]
解释: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10

核心代码逻辑：
List<Integer> diffWaysToCompute("(1 + 2 * 3) - (4 * 5)") {
	List<Integer> res = new LinkedList<>();
	/**********分*************/
	List<Integer> left = diffWaysToCompute("1 + 2 * 3");
	List<Integer> right = diffWaysToCompute("4 * 5");
	/***********治************/
	for(int a : left) {
		for(int b : right) {
			res.add(a - b);
		}
	}
	return res;
}

1.1 整体代码如下
List<Integer> diffWaysToCompute(String input) {
	List<Integer> res = new LinkedList<>();
	for(int i = 0; i < input.length(); i++) {
		char c = input.charAt(i);
		// 扫描算式 input 中的运算符
		if(c == '-' || c == '*' || c == '+') {
			/***********分**********/
			// 以运算符为中心，分割成两个字符串，分别递归的计算
			List<Integer> left = diffWaysToCompute(input.substring(0, i));
			List<Integer> right = diffWaysToCompute(input.substring(i+1));
			/**********治**********/
			// 通过子问题的结果，合并原问题的结果
			for(int a : left) {
				for(int b : right) {
					if(c == '+')
						res.add(a + b);
					else if(c == '-')
						res.add(a - b);
					else if(c == '*')
						res.add(a * b);
				}
			}
		}
	}
	// base case
	// 如果 res 为空，说明算式 是一个数字，没有运算符
	if(res.isEmpty()) {
		res.add(Integer.parseInt(input));
	}
	return res;
}

注：这就是典型的分治思路，先分后治，先按照运算符将原问题拆解成多个子问题，
然后通过子问题的结果来合成原问题的结果。
当算式中不存在运算符的时候，就不会触发if语句，也就不会给res中添加任何元素。

1.2 可以做个剪枝优化：使用备忘录
HashMap<String, List<Integer>> memo = new HashMap<>();
List<Integer> diffWaysToCompute(String input) {
	// 避免重复计算
	if(memo.containsKey(input)) {
		return memo.get(input);
	}
	/*******其它都不变***********/
	List<Integer> res = new LinkedList<>();
	for(int i = 0; i < input.length(); i++) {
		// ...
	}
	if(res.isEmpty()) {
		res.add(Integer.parseInt(input));
	}
	/***************/

	// 将结果添加进 备忘录
	memo.put(input, res);
	return res;
}

// Part3.4 数学运算技巧
// 题目：3.4.1 常用的位运算符
0 几个有趣的位操作
0.1 利用或操作 "|" 和 空格 将英文字符转换为小写
('a' | ' ') = 'a'
('A' | ' ') = 'a'

0.2 利用与操作 "&" 和 下划线 将英文字符转换为大写
('b' & '_') = 'B'
('B' & '_') = 'B'

0.3 利用异或操作 '^' 和 空格进行应为字符 大小写互换
('d' ^ ' ') = 'D'
('D' ^ ' ') = 'd'

以上操作能够产生 奇特效果的原因 在于 ASCII编码，字符其实就是数字，恰巧这些字符对应的数字通过位运算 就能得到正确的结果。

0.4 判断两个数是否异号
int x = -1, y = ;
bool f = ((x ^ y) < 0); // true

int x = 3, y = 2;
bool f = ((x ^ y) < 0); // false

0.5 不用临时遍历 交换两个数
int a = 1, b = 2;
a ^= b;
b ^= a;
a ^= b;
// 现在 a = 2, b = 1;

0.6 加一
int n = 1;
n = -~n;
// 现在 n = 2;

0.7 减一
int n = 2;
n = ~-n;
// 现在 n = 1;

1.算法常用操作
  n & (n - 1) 这个操是算法中常见的，作用是消除数字 n 的二进制表示中的最后一个1
  其核心逻辑就是：n - 1 一定可以消除最后一个1，同时把其后的 0 都变成1，
                这样再和 n 做一次 & 运算，就可以仅仅把最后一个1 变成 0 了 

1.1 计算汉明权重(Hamming Weight)
// leetcode 191
编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

// 方法1： n & (n - 1) 消除最后一个1
int HammingWeight(uint32_t n) {
	int res = 0;
	while(n != 0) {
		n = n & (n - 1);
		res++;
	}
	return res;
}

// 方法2：通过位遍历，逐个检测
int hammingWeight(uint32_t n) {
    int ret = 0;
    for(int i = 0; i < 32; i++) {
        if(n & (1 << i)) {
            ret++;
        }
    }
    return ret;
}

1.2 判断一个数 是不是 2 的指数
一个数如果是 2 的指数，那么它的二进制表示 一定只含有一个1：
2^0 = 1 = 0b0001
2^1 = 2 = 0b0010
2^2 = 4 = 0b0100

bool isPowerOfTwo(int n) {
	if(n <= 0) return false;
	return (n & (n - 1)) == 0;
}

1.3 查找只出现一次的元素
// leetcode 146
题目：给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
说明：你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？
示例1：
输入：[2, 2, 1]
输出：1

异或运算的性质：一个数和它本身做异或运算结果为0，即 a ^ a = 0; 
              一个数和 0 做异或运算的结果为它本身，a ^ 0 = a;
			  且，异或满足交换律
int singleNumber(vector<int>& nums) {
	int res = 0;
	for(int n : nums) {
		res ^= n;
	}
	return res;
}          

// Part3.4 数学运算技巧
// 题目：3.4.2 阶乘相关的算法题

1.题目1：阶乘后的零 // leetcode 172
输入一个整数n, 返回n! 结果尾数中零的个数
比如 n = 5, 阶乘为5! = 120，末尾有一个0;

分析：两数相乘结果末尾有0，一定是因为两个数中有因子 2 和 5，因为 10 = 2 * 5
现在问题转化为：n！最多可以分解出多少个因子5？

int trailingZeroes(int n) {
	int res = 0;
	long divisor = 5;
	while(divisor <= n) {
		res += n / divisor;
		divisor *= 5;
	}
	return res;
}

考虑到divisor变量使用long 型，因为假如 n 比较大，考虑while循环的结束条件，divisor可能会整型溢出。
代码可以更简单些
int trailingZeroes(int n) {
	int res = 0;
	for(int d = n; d / 5 > 0; d = d / 5) {
		res += d / 5;
	}
}


2.题目2：给你一个非负整数K，问你有多少个n, 使得n!结果末尾有K个 0；
二分查找搜索左侧边界 和 右侧边界
// 逻辑不变，数据类型全部改成 long
long trailingZeroes(long n) {
	long res = 0;
	for(long d = n; d / 5; d = d / 5) {
		res += d / 5;
	}
	return res;
}

// 二分查找框架
/* 主函数 */
int preimageSizeFZF(int K) {
	// 左边界和右边界之差 + 1 就是答案
	return right_bound(K) - left_bound(K) + 1;
}
/* 搜索trailingZeroes(n) == K 的左侧边界 */
long left_bound(int target) {
	long low = 0, high = LONG_MAX;
	while(low < high) {
		long mid = low + (high - low) / 2;
		if(trailingZeroes(mid) < target) {
			low = mid + 1;
		} else if(trailingZeroes(mid) > target) {
			high = mid;
		} else {
			high = mid;
		}
	}
	return low;
}

/* 搜索trailingZeroes(n) == K 的右侧边界 */
long right_bound(int target) {
	long low = 0, high = LONG_MAX;
	while(low < high) {
		long mid = low + (high - low) / 2;
		if(trailingZeroes(mid) < target) {
			low = mid + 1;
		} else if (trailingZeroes(mid) > target) {
			high = mid;
		} else {
			low = mid + 1;
		}
	}
	return low - 1;
}


// Part3.4 数学运算技巧
// 题目：3.4.3 如何高效寻找素数
素数的定义：如果一个数只能被1和它本身整除，那么这个数就是素数

0.1 解法1，复杂度很高O(N^2)
int countPrimes(int n) {
	int count = 0;
	for(int i = 2; i < n; i++) {
		if(isPrim(i)) count++;
	}
	return count;
}
// 判断整数 n 是否是素数
boolean isPrim(int n) {
	for(int i = 2; i < n; i++) {
		if(n % i == 0)
			return false;
	}
	return true;
}

0.2 解法2，只需要循环到sqrt(n) 就可以了，换句话说，如果在[2, sqrt(n)] 这个区间之内没有发现可整除因子
    就可以直接断定 n 是素数了。
int countPrimes(int n) {
	int count = 0;
	
	boolean[] isPrim = new boolean[n];
	// 将数组初始化为 true
	Array.fill(isPrim, true);
	for(int i = 2; i < n; i++) {
		if(isPrim[i] {
			// i 的倍数 不可能是素数了
			for(int j = 2 * i; j < n; j+= i) {
				isPrim[j] = false;
			}
		}
	}
	int count = 0;
	for(int i = 2; i < n; i++) {
		if(isPrim[i])
			count++;
	}
	return count;

}

0.3 完整的最终代码
int countPrimes(int n) {
	boolean[] isPrim = new boolean[n];
	Array.fill(isPrim, true);
	for(int i = 2; i * i < n; i++) {
		if(isPrim[i]) {
			for(int j = i * i; j < n; j += i) {
				isPrim[j] = false;
			}
		}
	}

	int count = 0;
	for(int i = 2; i < n; i++) {
		if(isPrim[i]) count++;
	}
	return count;
}


// Part3.4 数学运算技巧
// 题目：3.4.4 如何高效进行模幂运算
// leetcode 372
int superPow(int a, vector<int>& b);

要求你的算法返回幂运算 a^b 的计算结果 与 1337 取模后的结果。
0.1 
推导：superPow(a, [1, 5, 6, 4])
  => superPow(a, [1, 5, 6])；

// 计算 a 的 k 次方的结果, 后文会实现;
int myPow(int a, int k);

int superPow(int a, vector<int>& b) {
	// 递归的base case
	if(b.empty()) return 1;
	// 取出最后一个数
	int last = b.back();
	b.pop_back();
	// 将原问题化简，缩小规模递归求解
	int part1 = myPow(a, last);
	int part2 = myPow(superPow(a, b), 10);
	// 合并结果
	return part1 * part2;
}

0.2 如何处理mod, 避免结果太大而导致的整型溢出
一个模运算的技巧：
(a*b)%k = (a%k)(b%k)%k 

int base = 1337;
// 计算 a 的 k 次方 然后与 base 求模的结果
int myPow(int a, int k) {
	// 对因子求模
	a %= base;
	int res = 1;
	for(int i = 0; i < k; i++) {
		// 这里有乘法，是潜在的溢出点
		res *= a;
		// 对乘法结果求模
		res %= base;
	}
	return res;
}

int superPow(int a, vector<int>& b) {
	if(b.empty()) return 1;
	int last = b.back();
	b.pop_back();
	int part1 = myPow(a, last);
	int part2 = myPow(superPow(a, b), 10);
	// 每次乘法都要求模
	return (part1 * part2) % base;
}
先对因子 a 求模，然后每次都对乘法结果res 求模；
以上 可以通过lt上所有测试，时间负载度为O(N),

0.3 进阶版，快速幂算法
int base = 1337;
int mypow(int a, int k) {
	if(k == 0) return 1;
	a %= base;
	if(k % 2 == 1) {
		// k 是奇数
		return (a * mypow(a, k - 1)) % base;
	} else {
		// k 是偶数
		int sub = mypow(a, k / 2);
		return (sub * sub) % base;
	}
}

// Part3.4 数学运算技巧
// 题目：3.4.5 如何快速寻找缺失的元素
题目：给定一个包含0，1，2，..., n中 n个数的序列，找出0...n中没有出现的序列中的那个数。
示例1：
输入：[3, 0, 1]
输出：2 

0.1 方法1：位运算
    一个数和它本身做异或运算结果为0，一个数和0做异或运算，还是它本身。
    2 ^ 3 ^ 2 = 3 ^ (2 ^ 2) = 3 ^ 0
先把索引补一位，然后让每个元素和自己相等的索引相对应
int missingNumber(vector<int>& nums) {
	int n = nums.size();
	int res = 0;
	// 先和新补的索引做异或
	res ^= n;
	// 和其它的元素、索引做异或
	for(int i = 0; i < n; i++) {
		res ^= i ^ nums[i];
	}
	return res;
}
时间复杂度为O(N)

0.2 利用等差数列求和公式
int missingNumber(vector<int>& nums) {
	int n = nums.size();
	int expect = (0 + n) * (n + 1) / 2;
	int sum = 0;
	for(int x : nums) {
		sum += x;
	}
	return expect - sum;
}

0.3 考虑 expect 整型溢出 
int missingNumber(vector<int>& nums) {
	int n = nums.size();
	int res = 0;
	// 新补的索引
	res += n - 0;
	// 生效的索引和元素的差和 加起来
	for(int i = 0; i < n; i++) {
		res += i - nums[i];
	}
	return res;
}


// Part3.4 数学运算技巧
// 题目：3.4.6 高效寻找缺失和重复的数字
// leetcode 645
题目：
集合 s 包含从 1 到 n 的整数。
不幸的是，因为数据错误，导致集合里面某一个数字复制了成了集合里面的另外一个数字的值，导致集合 丢失了一个数字 并且 有一个数字重复 。
给定一个数组 nums 代表了集合 S 发生错误后的结果。
请你找出重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。

0. 分析：有一个元素重复了，同时导致一个元素缺失了，产生的现象是：
	会导致有两个元素对应到了同一个索引，而且会有一个索引没有元素对应过去。
可以通过 将每个索引对用的元素变成负数，以表示这个索引 被 对应过一次了。

vector<int> findErrorNums(vector<int>& nums) {
	int n = nums.size();
	int dup = -1;
	for(int i = 0; i < n; i++) {
		// 索引应该从 0 开始
		int index = abs(nums[i]) - 1;
		if(nums[index] < 0) {
			dup = abs(nums[i]);
		} else {
			nums[index] *= -1;
		}
	}
	int missing = -1;
	for(int i = 0; i < n; i++) {
		if(nums[i] > 0) {
			// 将索引转换成元素
			missing = i + 1;
		}
	}
	return {dup, missing};
}


// Part3.4 数学运算技巧
// 题目：3.4.7 随机算法之水塘抽样算法
// leetcode 382, 398
关于水塘抽样算法，本质上是一种随机概率算法，解法应该说 会者不难，难者不会
goodle题目：给你一个未知长度的链表，请你设计一个算法，只能遍历一次，随机地返回链表中的一个节点。

0 结论：当你遇到第 i 个元素时，应该有 1/i 的概率选择该元素，1 - 1/i 的概率保持原有的选择  // 重要
/* 返回链表中一个随机节点的值 */
int getRandom(ListNode head) {
	Random r = new Random();
	int i = 0, res = 0;
	ListNode p = head;
	while(p != null) {
		// 生成一个[0, i) 之间的整数
		// 这个整数 等于 0 的概率是 1/i 
		if(r.nextInt(++i) == 0) {
			res = p.val;
		}
		p = p.next;
	}
	return res;
}

1. 如果要随机选择 k 个数，只要在第 i 个元素处 已 k / i 的概率选择该元素，以 1- k/i 的概率保持原有选择即可。
/* 返回链表中 k 个随机节点的值 */
int[] getRandom(ListNode head, int k) {
	Random r = new Random();
	int[] res = new int[k];
	ListNode p = head;

	// 前 k 个元素先默认选上
	for(int j = 0; j < k && p != null; j++) {
		res[j] = p.val;
		p = p.next;
	}
	int i = k;
	while(p != null) {
		// 生成一个[0, i)之间的整数
		int j = r.nextInt(++i);
		// 这个整数小于 k 的概率就是 k/i
		if(j < k) {
			res[j] = p.val;
		}
		p = p.next;
	}
	return res;
}

// Part3.4 数学运算技巧
// 题目：3.4.8 如何高效的 对有序数组/链表去重？
见 快慢指针


// Part4 高频面试系列
// 题目：4.1 关于吃葡萄的算法题
题目：有三种葡萄，每种分别有a, b, c颗，现在有三个人，第一个人只吃第一种和第二种葡萄，
     第二个人只吃第二种和第三种葡萄，第三个人只吃第一种和第三种葡萄。
现在给你输入a, b, c三个值，请你适当安排，让三个人吃完所有的葡萄，算法返回 吃的最多的人最少要吃多少颗葡萄。

long solution(long a, long b, long c) {
	vector<long> nums = {a, b, c};
	sort(nums.begin(), nums.end());
	long sum = a + b + c;

	// 能够构成三角形，可完全平分
	if(nums[0] + nums[1] > nums[2]) {
		return (sum + 2) / 3;
	}
	// 不能构成三角形，平分最长边的情况
	if(2 * (nums[0] + nums[1]) < nums[2]) {
		return (nums[2] + 1) / 2;
	}
	// 不能构成三角形，但依然可以完全平分的情况
	return (sum + 2) / 3;
}


// Part4 高频面试系列
// 题目：4.2 烧饼排序算法
题目：给你一个整数数组 arr ，请使用 煎饼翻转 完成对数组的排序。
一次煎饼翻转的执行过程如下：
	1.选择一个整数 k ，1 <= k <= arr.length
	2.反转子数组 arr[0...k-1]（下标从 0 开始）
例如，arr = [3,2,1,4] ，选择 k = 3 进行一次煎饼翻转，反转子数组 [3,2,1] ，得到 arr = [1,2,3,4] 。
以数组形式返回能使 arr 有序的煎饼翻转操作所对应的 k 值序列。
任何将数组排序且翻转次数在 10 * arr.length 范围内的有效答案都将被判断为正确。

示例 1：
输入：[3,2,4,1]
输出：[4,2,4,3]
解释：
我们执行 4 次煎饼翻转，k 值分别为 4，2，4，和 3。
初始状态 arr = [3, 2, 4, 1]
第一次翻转后（k = 4）：arr = [1, 4, 2, 3]
第二次翻转后（k = 2）：arr = [4, 1, 2, 3]
第三次翻转后（k = 4）：arr = [3, 2, 1, 4]
第四次翻转后（k = 3）：arr = [1, 2, 3, 4]，此时已完成排序。 

代码实现：递归求解

vector<int> res;
vector<int> pancakeSort(vector<int>& arr) {
	sort(arr, arr.size());
	return res;
}

void reverse(vector<int>& arr, int start, int end) {
	while(start < end) {
		int tmp = arr[start];
		arr[start] = arr[end];
		arr[end] = tmp;
		start++, end--;
	}
}


void sort(vector<int>& arr, int n) {
	// base case
	if(n == 1) return;
	// 寻找最大索引，和最大值
	int maxArr = 0;
	int maxArrIndex = 0;
	for(int i = 0; i < n; i++) {
		if(arr[i] > maxArr) {
			maxArr = arr[i];
			maxArrIndex = i;
		}
	}
	// 第一次翻转，将最大饼翻到最上面
	reverse(arr, 0, maxArrIndex);
	res.push_back(maxArrIndex + 1);
	// 第二次翻转，将最大饼翻到最下面
	reverse(arr, 0, n - 1);
	res.push_back(n);
	//递归调用
	sort(arr, n - 1);
}


// Part4 高频面试系列
// 题目：4.3 字符串乘法
题目：给定两个以字符串形式表示的非负整数 num1 和 num2，
返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

示例 1:
输入: num1 = "2", num2 = "3"
输出: "6"

示例 2:
输入: num1 = "123", num2 = "456"
输出: "56088"

string mulitply(string num1, string num2) {
	int m = num1.size();
	int n = num2.size();
	// 结果最多 m + n 位
	vector<int> res(m + n, 0);
	// 从个位数开始 逐位相乘
	for(int i = n - 1; i >= 0; i--) {
		for(int j = m - 1; j >= 0; j--) {
			int mul = (num2[i] - '0') * (num1[j] - '0');
			// 乘积在 res 对应的索引位置
			int p1 = i + j, p2 = i + j + 1;
			// 叠加到res上
			int sum = mul + res[p2];
			res[p2] = sum % 10;   // 取余在低位数
			res[p1] += sum / 10;  // 取整在高位数，且和原高位数累加
		}
	}
	// 结果前缀可能存在0， 未使用，排除掉
	int i = 0;
	while(i < res.size() && res[i] == 0) {
		i++;
	}
	// 将数字 转换成 字符
	string str;
	for(; i < res.size(); i++) {
		str.push_back(str[i] + '0');
	}
	return str.size() == 0 ? "0" : str;
}


// Part4 高频面试系列
// 题目：4.4 如何拆解复杂问题：实现一个计算器
0.1 字符串转整数
string s = "458";
int n = 0;
for(int i = 0; i < s.size(); i++) {
	char c = s[i];
	n = 10 * n + (c - '0'); 
}

0.2 处理加减法
如果输入的这个算式 只包含加减法，而且不存在空格，如何计算结果
分析：放入栈中
int calculate(string s) {
	stack<int> stk;
	// 记录算式中的数字
	int num = 0;
	// 记录 num 前的符合，初始化为 +
	char sign = '+';
	for(int i = 0; i < s.size(); i++) {
		char c = s[i];
		// 如果是数字，连续读取到 num
		if(isdigit(c)) {
			num = 10 * num + (c - '0');
		}
		// 如果不是数字，就是遇到了下一个符号，
		// 之前的数字和符号就要存进栈中
		if(!isdigit(c) || i == s.size() - 1) {
			switch (sign) {
				case '+':
					stk.push(num); break;
				case '-':
					stk.push(-num); break;
			}
			// 更新符号为当前符号，数字清零
			sign = c;
			num = 0;
		}
	}
	// 将栈中所有结果求和 就是答案
	int res = 0;
	while(!stk.empty()) {
		res += stk.top();
		stk.pop();
	}
	return res;
} 

0.3 处理乘除法
for(int i = 0; i < s.size(); i++) {
	char c = s[i];
	if(isdigit(c)) {
		num = 10 * num + (c - '0');
	}
	if((!isdigit(c) && c != ' ')|| i == s.size() - 1) {
		switch (sign) {
			int pre;
			case '+':
				stk.push(num); break;
			case '-':
				stk.push(-num); break;
			// 只要拿出前一个数字 做对应的运算即可
			case '*':
				pre = stk.top();
				stk.pop();
				stk.push(pre * num);
				break;
			case '/':
				pre = stk.top();
				stk.pop();
				stk.push(pre / num);
		}
		// 更新符号为当前符号，数字清零
		sign = c;
		num = 0;
	}
}

0.4 处理括号
将之前的版本，翻译成python版本
def calculate(s: str) -> int:
	
	def helper(s: List) -> int:
		stack = []
		sign = '+'
		num = 0

		while len(s) > 0:
			c = s.pop(0)
			if c.isdigit():
				num = num * 10 + int(c)

			if(not c.isdigit() and c != ' ') or len(s) == 0:
				if sign == '+':
					stack.append(num)
				if sign == '-':
					stack.append(-num)
				if sign == '*':
					stack[-1] = stack[-1] * num
				if sign == '/':
					// python 除法 向 0 取整的写法
					stack[-1] = int(stack[-1] / float(num))
				num = 0
				sign = c 

		return sum(stack)
	# 需要吧字符串转成列表 方便操作
		return helper(list(s))

因为括号具有递归性质。换句话说，括号包含的算式，我们直接视为一个数字就行了。
递归的开始条件和结束条件是什么？
 遇到（ 开始递归，遇到) 结束递归：

def calculate(s: str) -> int:

	def helper(s: List) -> int:
		stack = []
		sign = '+'
		num = 0

		while len(s) > 0:
			c = s.pop(0)
			if c.isdigit():
				num = 10 * num + int(c)
			// 遇到左括号 开始递归计算num
			if c == '(':
				num = helper(s)

			if (not c.isdigit() and c != ' ') or len(s) == 0:
				if sign == '+': ...
				elif sign == '-':
				elif sign == '*':
				elif sign == '/':
				num = 0
				sign = c 
			// 遇到右括号 返回递归结果
			if c == ')': break
		return sum(stack)

	return helper(list(s))


// Part4 高频面试系列
// 题目：4.5 如何高效解决接雨水问题
题目：给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
示例1：
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水;

0 暴力解法->备忘录解法->双指针解法
1.暴力解法
int trap(vector<int>& height) {
	int n = height.size();
	int res = 0;
	for(int i = 1; i < n - 1; i++) {
		int l_max = 0, r_max = 0;
		// 找到右边最高的柱子
		for(int j = i; j < n; j++) {
			r_max = max(r_max, height[j]);
		}
		// 找左边最高的柱子
		for(int j = i; j >= 0; j--) {
			l_max = max(l_max, height[j]);
		}
		// 如果自己就是最高的话
		// l_max == r_max == height[i];
		res += min(l_max, r_max) - height[i];
	}
	return res;
}

该方法时间复杂度为O(N^2), 空间复杂度为O(1)

2.备忘录优化
先开两个数组 r_max 和 l_max 充当备忘录，l_max[i] 表示位置i 左边最高的柱子高度，
r_max[i] 表示位置i 右边最高的柱子高度。
int trap(vector<int>& height) {
	if(height.empty()) return 0;

	int n = height.size();
	int res = 0;
	// 数组充当备忘录
	vector<int> l_max(n), r_max(n);
	// 初始化 base case
	l_max[0] = height[0];
	r_max[n - 1] = height[n - 1];
	// 从左向右计算r_max
	for(int i = 1; i < n; i++) {
		l_max[i] = max(height[i], l_max[i - 1]);
	}
	// 从右向左计算 r_max
	for(int i = n - 2; i >= 0; i--) {
		r_max[i] = max(height[i], r_max[i + 1]);
	}
	// 计算答案
	for(int i = 1; i < n - 1; i++) {
		res += min(l_max[i], r_max[i]) - height[i];
	}
	return res;
}
改方法时间复杂度为O(N), 但是空间复杂度为O(N)

3.双指针解法
使用双指针边走边算，节省下空间复杂度

int trap(vector<int>& height) {
	if(height.empty()) return 0;
	int n = height.size();
	int left = 0, right = n - 1;
	int res = 0;

	int l_max = height[0];
	int r_max = height[n - 1];

	while(left <= right) {
		l_max = max(l_max, height[left]);
		r_max = max(r_max, height[right]);

		// res += min(l_max, r_max) - height[i];
		if(l_max < r_max) {
			res += l_max - height[left];
			left++;
		} else {
			res += r_max - height[right];
			right--;
		}
	}
	return res;
}
该方法时间复杂度为O(N), 空间复杂度为O(1)


// Part4 高频面试系列
// 题目：4.6 如何寻找最长回文子串
题目：给定一个字符串s, 找到 s 中最长的回文子串，你可以假设 s 的最大长度为1000.
示例1：
输入：”babad“
输出：”bab“
注意：”aba“也是一个有效答案

首先实现一个函数，来寻找最长回文串，这个函数有一点技巧的；
string palindrome(string& s, int left, int right) {
	// 防止索引越界
	while(left >= 0 && right < s.size() && s[left] == s[right]) {
		// 向两边展开
		left--;
		right++;
	}
	// 返回以 s[left] 和 s[right] 为中心的最长回文串
	return s.substr(left + 1, right - left - 1);
}

完整代码：
string longestPalindrome(string s) {
	string res;
	for(int i = 0; i < s.size(); i++) {
		// 以 s[i] 为中心的最长回文子串
		string s1 = palindrome(s, i, i);
		// 以 s[i] 和 s[i + 1] 为中心的最长回文子串
		string s2 = palindrome(s, i, i + 1);
		res = res.size() > s1.size() ? res : s1;
		res = res.size() > s2.size() ? res : s2;
	}
	return res;
}
改方法 时间复杂度为O(N^2), 空间复杂度为O(1)


// Part4 高频面试系列
// 题目：4.7 如何判断括号的合法性
注意：每个右括号")"的左边 必须有一个左括号 "(" 和它匹配；
分析：采用栈的方式：遇到左括号就入栈，遇到右括号就去栈中寻找最近的左括号，看是否匹配

bool isValid(string str) {
	stack<char> left;
	for(char c : str) {
		if(c == '(' || c == '{' || c == '[') {
			left.push(c);
		} else {
			if(!left.empty() && leftOf(c) == left.top()) {
				left.pop();
			} else {
				// 和最近的左括号不匹配
				return false;
			} 
		}
	}
	return left.empty();
}

char leftOf(char c) {
	if(c == '}') return '{';
	if(c == ')') return '(';
	return '[';
}


// Part4 高频面试系列
// 题目：4.8 判断是否是完美矩阵
// leetcode 391
题目：输入一个数组rectangles, 里面装有若干四元组(x1, y1, x2, y2), 
每个四元组就是记录一个矩形的左下角和右上角顶点坐标。
也就是说，输入的rectangles数组 实际上就是很多小矩形，题目要求我们输出一个布尔值，
判断这些小矩形能否构成一个 完美矩形

分析：想判断最终形成的图形是否是完美矩形，需要从[面积]和[顶点]两个监督来处理。
0.1 初步代码
def isRectangleCover(rectangles: List[List[int]]) -> bool:
	X1, Y1 = float('inf'), float('inf')
	X2, Y2 = -float('inf'), -float('inf')
	// 记录所有小矩形的面积之和
	actual_area = 0
	for x1, y1, x2, y2 in rectangles:
		// 计算完美矩阵的理论坐标
		X1, Y1 = min(X1, x1), min(Y1, y1);
		X2, Y2 = min(X2, x2), min(Y2, y2);
		// 累加所有小矩阵面积之和
		actual_area += (x2 - x2) * (y2 - y1)

	// 计算完美矩阵的理论面积
	expected_area = (X2 - X1) * (Y2 - Y1)
	// 面积应该相同
	if actual_area != expected_area:
		return False

	return True 

0.2 如果面积相同，是否可以证明最终形成的图形是完美矩阵，一定不存在空缺或者重叠吗？
	即使面积相同，并不能完全保证不存在空缺或者重叠，我们需要从【顶点】的维度来辅助判断
	显然，完美矩阵一定只有4个顶点。

def isRectangleCover(rectangles: List[List[int]]) -> bool:
	X1, Y1 = float('inf'), float('inf')
	X2, Y2 = -float('inf'), -float('inf')

	actual_area = 0
	// 哈希集合，记录最终图形的顶点
	points = set()
	for x1, y1, x2, y2 in rectangles:
		X1, Y1 = min(X1, x1), min(Y1, y1)
		X2, Y2 = min(X2, x2), min(Y2, y2)

		actual_area += (x2 - x1) * (y2 - y1)
		// 先计算出 小矩形每个点的坐标
		p1, p2 = (x1, y1), (x2, y2)
		p3, p4 = (x2, y1), (x2, y2)
		// 对于每个点，如果存在集合中，删除它：
		// 如果不存在集合中，添加它
		// 在集合中剩下的点 都是出现奇数次的点
		for p in [p1, p2, p3, p4]:
			if p in points:
				points.remove(p)
			else:
				points.add(p)

	expected_area = (X2 - X1) * (Y2 - Y1)
	if actual_area != expected_area:
		return False 
	return True;

0.3 不仅要保证len(points) == 4, 而且要保证points 中最终剩下的点坐标就是完美矩阵的4个理论坐标

def isRectangleCover(rectangles: List[List[int]]) -> bool:
	X1, Y1 = float('inf'), float('inf')
	X2, Y2 = -float('inf'), -float('inf')

	actual_area = 0
	// 哈希集合，记录最终图形的顶点
	points = set()
	for x1, y1, x2, y2 in rectangles:
		X1, Y1 = min(X1, x1), min(Y1, y1)
		X2, Y2 = min(X2, x2), min(Y2, y2)

		actual_area += (x2 - x1) * (y2 - y1)
		// 先计算出 小矩形每个点的坐标
		p1, p2 = (x1, y1), (x2, y2)
		p3, p4 = (x2, y1), (x2, y2)
		// 对于每个点，如果存在集合中，删除它：
		// 如果不存在集合中，添加它
		// 在集合中剩下的点 都是出现奇数次的点
		for p in [p1, p2, p3, p4]:
			if p in points:
				points.remove(p)
			else:
				points.add(p)

	expected_area = (X2 - X1) * (Y2 - Y1)
	if actual_area != expected_area:
		return False 
	// 判断最终剩下的顶点个数 是否为4
	if len(points) != 4:
		return False
	// 判断留下的 4 个顶点是否是完美矩阵的顶点
	if(X1, Y1) not in points: return False
	if(X1, Y2) not in points: return False 
	if(X2, Y1) not in points: return False 
	if(X2, Y2) not in points: return False

	// 面积和顶点都对应，说明矩形符合题意
	return True 


// Part4 高频面试系列
// 题目：4.9 如何算法调度考生的座位？
// leetcode 855
题目：在考场里，一排有 N 个座位，分别编号为 0, 1, 2, ..., N-1 。
当学生进入考场后，他必须坐在能够使他与离他最近的人之间的距离达到最大化的座位上。如果有多个这样的座位，他会坐在编号最小的座位上。
(另外，如果考场里没有人，那么学生就坐在 0 号座位上。)

返回 ExamRoom(int N) 类，它有两个公开的函数：其中，函数 ExamRoom.seat() 会返回一个 int （整型数据），代表学生坐的位置；
函数 ExamRoom.leave(int p) 代表坐在座位 p 上的学生现在离开了考场。
每次调用 ExamRoom.leave(p) 时都保证有学生坐在座位 p 上;

class ExamRoom {
	// 构造函数，传入作为总数 N
	public ExamRoom(int N);
	// 来了一名考生，返回你给他分配的座位
	public int seat();
	// 坐在 p 位置的考生离开了
	// 可以认为 p 位置一定坐有考生
	public void leave(int p);
};

0 分析，如果将每两个相邻的考生看做线段的两端点，新安排考生就是找最长的线段，然后让该考生在中间把这个线段[二分]，
  中点就是给他分配的作为。leave(p) 其实就是去除端点p, 使得相邻两个线段合并为一个。

采用数据结构中的平衡二叉搜索树，取最值，也可以修改、删除任意一个值，而且时间复杂度都是O(logN).
暂时先不考虑，如果有多个可选座位时，需要选择索引最小的作为。

class ExamRoom {
	// 将端点p 映射到以 p 为左端点的线段
	private Map<Integer, int[]> startMap;
	// 将端点 p 映射到 以 p 为右端点的线段
	private Map<Integer, int[]> endMap;
	// 根据线段长度从小到大存放所有线段
	private TreeSet<int[]> pq;
	private int N;

	// 构造函数，传入作为总数 N
	public ExamRoom(int N) {
		this.N = N;
		startMap = new HashMap<>();
		endMap = new HashMap<>();
		pq = new TreeSet<>((a, b) -> {
			// 算出两个线段的长度
			int distA = distance(a);
			int distB = distance(b);
			// 长度更长的更大，排后面
			return distA - distB;
		});
		// 在有序集合中 先放一个虚拟线段
		addInterval(new int[] {-1, N});
	}

	/* 去除一个线段 */
	private void removeInterval(int[] intv) {
		pq.remove(intv);
		startMap.remove(intv[0]);
		endMap.remove(intv[1]);
	}

	/* 增加一个线段 */
	private void addInterval(int[] intv) {
		pq.add(intv);
		startMap.put(intv[0], intv);
		endMap.put(intv[1], intv);
	}

	/* 计算一个线段的长度 */
	private int distance(int[] intv) {
		return intv[1] - intv[0] - 1;
	}

	// 来了一名考生，返回你给他分配的座位
	public int seat() {
		// 从有序集合 拿出最长的线段
		int[] longest = pq.last();
		int x = longest[0];
		int y = longest[1];
		int seat;
		if(x == -1) {
			// 情况1
			seat = 0;
		} else if (y == N) {
			// 情况2
			seat = N - 1;
		} else { // 情况3
			seat = (y - x) / 2 + x;
		}
		// 将最长的线段 分成两段
		int[] left = new int[] {x, seat};
		int[] right = new int[] {seat, y};
		removeInterval(longest);
		addInterval(left);
		addInterval(right);
		return seat;
	}
	// 坐在 p 位置的考生离开了
	// 可以认为 p 位置一定坐有考生
	public void leave(int p) {
		// 将 p 左右的线段找出来
		int[] right = startMap.get(p);
		int[] left = endMap.get(p);
		// 合并两个线段 成为一个线段
		int[] merged = new int[] {left[0], right[1]};
		removeInterval(left);
		removeInterval(right);
		addInterval(merged);
	}
};


1. 进阶，刚才忽略的问题是，多个选择时，需要选择索引最小的那个座位，修改代码如下：
修改下有序数据结构的排序方式，
pq = new TreeSet<>((a, b) -> {
	int distA = distance(a);
	int distB = distance(b);
	// 如果长度相同，就比较索引
	if(distA == distB) {
		return b[0] - a[0];
	}
	return distA - distB;
});

distance的计算方式，也需要修改下，不能简单地让它计算一个线段两个端点间的长度，
而是让它计算该线段中点到端点的长度。
private int distance(int[] intv) {
	int x = intv[0];
	int y = intv[1];
	if(x == -1) return y;
	if(y == N) return N - 1 - x;
	// 中点到端点的长度
	return (y - x) / 2;
}

// Part4 高频面试系列
// 题目：4.10 二分查找的妙用，判定子序列
题目，如何判定字符串 s 是否是字符串t 的子序列(可以假定 s 长度比较小，且 t 的长度非常大)。 
例子1：
s = "abc", t="ahbgdc", return true 

0.1 简单的解法
bool isSubsequence(string s, string t) {
	int i = 0, j = 0;
	while(i < s.size() && j < t.size()) {
		if(s[i] == t[j]) i++;
		j++;
	}
	return i == s.size();
}

如果给你一系列字符串s1, s2, ...和字符串t, 你需要判定每个串 s 是否是 t 的子序列
(可以假定 s 相对较短，t 很长)。
0.2 二分思路，主要是对 t 进行预处理，用一个字典 index 将每个字符出现的索引位置，按照顺序存储下来
(对于ASCII字符，可以用大小为256的数组充当字典)

int m = s.length(), n = t.length();
ArrayList<Integer>[] index = new ArrayList[256];
// 先记下 t 中每个字符出现的位置
for(int i = 0; i < n; i++) {
	char c = t.charAt(i);
	if(index[c] == null) {
		index[c] = new ArrayList<>();
	}
	index[c].add(i);
}

0.3 再谈二分查找，对于搜索左侧边界的二分查找，有一个特殊性质：
当val 不存在时，得到的索引恰好是比val大的最小元素索引；
// 查找左侧边界的二分查找
int left_bound(ArrayList<Integer>arr, int target) {
	int low = 0, high = arr.size();
	while(low < high) {
		int mid = low + (high - low) / 2;
		if(target > arr.get(mid)) {
			low = mid + 1;
		} else {
			high = mid;
		}
	}
	return low;
}

0.4 代码实现，这里以处理单个字符串 s 为例，对于多个字符串 s，把预处理部分单独抽出来即可。

boolean isSubsequence(String s, String t) {
	int m = s.length(), n= t.length();
	// 对 t 进行预处理
	ArrayList<Integer> index = new ArrayList[256];
	for(int i = 0; i < n; i++) {
		char c = t.charAt(i);
		if(index[c] == null) {
			index[c] = new ArrayList<>();
		}
		index[c].add(i);
	}
	// 串 t 上的指针
	int j = 0;
	// 借助index 查找 s[i]
	for(int i = 0; i < m; i++) {
		char c = s.charAt(i);
		// 整个 t 压根没有字符 c
		if(index[c] == null) return false;
		int pos = left_bound(index[c], j);
		// 二分搜索区间中 没有找到字符 c
		if(pos == index[c].size()) return false;
		// 向前移动指针 j
		j = index[c].get(pos) + 1;
	}
	return true;
}

结论：记住二分查找，算法的效率是可以大幅提升的：
预处理时需要O(N) 时间，每次匹配子序列的时间是O(MlogN),比之前每次匹配都要O(N) 的时间要高效的多。
当然，如果只需要判断一个s是否是t的子序列，是不需要二分查找的，一开始O(N)解法是最好的。
因为虽然二分查找解法处理每个 s 只需要 O(MlogN), 但是还需要O(N) 时间构造index 字典预处理，所以处理单个 s 时没有必要。

