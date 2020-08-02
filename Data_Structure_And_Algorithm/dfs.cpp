//133. 克隆图
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

class Node {
    public int val;
    public List<Node> neighbors;
}

//采用stack的迭代法
Node* cloneGraph(Node* node) {
    if(!node) return nullptr;
    stack<Node*> s;
    s.push(node);
    unordered_map<Node*, Node*> u_map;
    u_map[node] = new Node(node->val);

    while(!s.empty()) {
        Node* tmp = s.top();
        s.pop();
        Node* cloneNode = u_map[tmp];
        for (auto &neighbor: tmp->neighbors) {
            if(!u_map.count(neighbor)) {
                Node* t = new Node(neighbor->val);
                u_map[neighbor] = t;
                s.push(neighbor);
            }
            cloneNode->neighbors.push_back(u_map[neighbor]);
        }
    }
    return u_map[node];

}
//递归法
unordered_map<Node*, Node*> u_map;
Node* cloneGraph2(Node* node) {
    if(!node) return nullptr;
    if (u_map.count(node)) {
        return u_map[node];
    }
    Node* cloneNode = new Node(node->val);
    u_map[node] = cloneNode;
    for (auto neighbor : node->neighbors) {
        cloneNode->neighbors.push_back(cloneGraph(neighbor));
    }
    return cloneNode;
}

//200. 岛屿数量
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。

示例 1:

输入:
[
['1','1','1','1','0'],
['1','1','0','1','0'],
['1','1','0','0','0'],
['0','0','0','0','0']
]
输出: 1
示例 2:

输入:
[
['1','1','0','0','0'],
['1','1','0','0','0'],
['0','0','1','0','0'],
['0','0','0','1','1']
]
输出: 3
解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。
void dfs(vector<vector<char>>& grid, int i, int j) {
    int m = grid.size();
    int n = grid[0].size();
    grid[i][j] = '0';
    if (i - 1 >= 0 && grid[i - 1][j] == '1') dfs(grid, i - 1, j);
    if (i + 1 < m && grid[i + 1][j] == '1') dfs(grid, i + 1, j);
    if (j - 1 >= 0 && grid[i][j - 1] == '1') dfs(grid, i, j - 1);
    if (j + 1 < n && grid[i][j + 1] == '1') dfs(grid, i, j + 1);

}

int numIslands(vector<vector<char>>& grid) {
    int num_lands = 0;
    int m = grid.size();
    if(!m) return 0;
    int n = grid[0].size();
    if (!n) return 0;


    for (int i = 0; i != m; i++) {
        for (int j = 0; j != n; j++) {
            if (grid[i][j] == '1') {
                ++num_lands;
                dfs(grid, i, j);
            }
        }
    }
    return num_lands;
}

