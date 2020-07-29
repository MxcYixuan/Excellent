//144. 二叉树的前序遍历
vector<int> res;
    vector<int> preorderTraversal(TreeNode* root) {
        if(root != nullptr) {
            res.push_back(root->val);
            preorderTraversal(root->left);
            preorderTraversal(root->right);
        }
        return res;
}
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode*> s;
    TreeNode* curr = root;
    while(curr || !s.empty()) {
        while(curr) {
            s.push(curr);
            res.push_back(curr->val);
            curr = curr->left;
        }
        curr = s.top();
        s.pop();
        curr = curr->right;
    }
    return res;
}
//145. 二叉树的后序遍历
//给定一个二叉树，返回它的 后序 遍历。
vector<int> postorderTraversal(TreeNode* root) {
    vector<int> res;
    if(root == nullptr) return res;
    stack<TreeNode*> s;
    s.push(root);
    while(s.size()){
        root = s.top();
        s.pop();
        if(root->left) s.push(root->left);
        if(root->right) s.push(root->right);
        res.push_back(root->val);
    }
    reverse(res.begin(),res.end());
    return res;
}


//面试题 04.03. 特定深度节点链表
//给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表
//（比如，若一棵树的深度为 D，则会创建出 D 个链表）。返回一个包含所有深度的链表的数组

//[1,2,3,4,5,null,7,8]
/*      1
       /  \ 
      2    3
     / \    \ 
    4   5    7
   /
  8
*/ 
//输出：[[1],[2,3],[4,5,7],[8]]
vector<ListNode*> listOfDepth(TreeNode* tree) {
        vector<ListNode*> res;
        if(!tree) return res;
        queue<TreeNode*> que;
        que.push(tree);
        while(!que.empty()) {
            int size = que.size();
            ListNode* head = nullptr;
            ListNode* p = nullptr;
            TreeNode* tmp = nullptr; 
            for (int i = 0; i != size; i++) {
                tmp = que.front();
                que.pop();
                if (i == 0) {
                    head = new ListNode(tmp->val);
                    p = head;  
                } else {
                    ListNode* li = new ListNode(tmp->val);
                    p->next = li;
                    p = p->next;
                }
                if (tmp->left) que.push(tmp->left);
                if (tmp->right) que.push(tmp->right);     
            }
            res.push_back(head);
        }
        return res;
    }

//1367. 二叉树中的列表
/*给你一棵以 root 为根的二叉树和一个 head 为第一个节点的链表。
如果在二叉树中，存在一条一直向下的路径，且每个点的数值恰好一一对应以 head 为首的链表中每个节点的值，
那么请你返回 True ，否则返回 False 。
一直向下的路径的意思是：从树中某个节点开始，一直连续向下的路径。
*/
bool isSubPath(ListNode* head, TreeNode* root) {
        //DFS遍历数
        if (!root) return false;
        bool result1 = dfs(head, root);
        return result1 || isSubPath(head, root->left) 
                       || isSubPath(head, root->right);
    }

    bool dfs(ListNode* head, TreeNode* root) {
        if (!head) return true;
        if (!root) return false;
        if (head->val != root->val) return false;
        return dfs(head->next, root->left) 
            || dfs(head->next, root->right);
    }

//669. 修剪二叉搜索树
/*给定一个二叉搜索树，同时给定最小边界L 和最大边界 R。通过修剪二叉搜索树，使得所有节点的值在[L, R]中 (R>=L) 。
你可能需要改变树的根节点，所以结果应当返回修剪好的二叉搜索树的新的根节点。
*/
TreeNode* trimBST(TreeNode* root, int L, int R) {
        if (!root) return nullptr;
        if (root->val < L) return trimBST(root->right, L, R);
        if (root->val > R) return trimBST(root->left, L, R);
        root->left = trimBST(root->left, L, R);
        root->right = trimBST(root->right, L, R);
        return root;

    }

//112. 路径总和
//给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
//说明: 叶子节点是指没有子节点的节点。
bool hasPathSum(TreeNode* root, int sum) {
    if (!root) return false;
    if (!root->left && !root->right)
        return sum == root->val;
    return hasPathSum(root->left, sum - root->val) ||
           hasPathSum(root->right, sum - root->val);
}

//113. 路径总和 II
//给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
//说明: 叶子节点是指没有子节点的节点。

void dfs(TreeNode* root, vector<int>& tmp, int sum, vector<vector<int>>&res) {
    if (root == nullptr) return;
    tmp.push_back(root->val);
    if (root->val == sum && root->left == nullptr && root->right == nullptr)
        res.push_back(tmp);
    if(root->left)
        dfs(root->left, tmp, sum - root->val, res);
    if (root->right)
        dfs(root->right, tmp, sum - root->val, res);
    tmp.pop_back();
}
vector<vector<int>> pathSum(TreeNode* root, int sum) {
    vector<int> tmp;
    vector<vector<int>> res;
    dfs(root, tmp, sum, res);
    return res;
}

//116. 填充每个节点的下一个右侧节点指针
/*给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}*/
//填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
//初始状态下，所有 next 指针都被设置为 NULL。
//方法1：层次遍历迭代法
Node* connect(Node* root) {
    if (!root) return nullptr;
    queue<Node*> que;
    que.push(root);
    while(!que.empty()) {
        int size = que.size();
        for (int i = 0; i != size; i++) {
            Node* node = que.front();
            que.pop();
            if(node->left) que.push(node->left);
            if(node->right) que.push(node->right);
            if(i < size - 1) node->next = que.front();
        }
    }
    return root;
}
//方法2：递归
Node* connect(Node* root) {
    if (!root || !root->left)
        return root;
    root->left->next = root->right;
    if (root->next)
        root->right->next = root->next->left;
    root->left = connect(root->left);
    root->right = connect(root->right);
    return root;

}


//给定一个非空二叉树，返回其最大路径和。
//本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
示例 1:
输入: [1,2,3]

       1
      / \
     2   3
输出: 6
示例 2:

输入: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出: 42

       
int maxPathSum(TreeNode* root) {
    int val = INT_MIN;
    maxPathSum2(root, val);
    return val;
}
int maxPathSum2(TreeNode* root, int &val) {
    if (root == nullptr) return 0;
    int left = max(0, maxPathSum2(root->left, val));
    int right = max(0, maxPathSum2(root->right, val));
    val = max(val, (left + right + root->val));
    return max(left, right) + root->val;
}

int maxPathSum(TreeNode* root, int &val) {
    if (root == NULL) return 0;
    int left = maxPathSum(root->left, val);
    int right = maxPathSum(root->right, val);
    int condition1 = root->val + max(0, left) + max(0, right);
    int condition2 = root->val + max(0, max(left, right));
    val = max(val, max(condition1, condition2));
    // 返回经过root的单边最大分支给上游
    return condition2;
}

//129. 求根到叶子节点数字之和
/*给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
例如，从根到叶子节点路径 1->2->3 代表数字 123。
计算从根到叶子节点生成的所有数字之和。
说明: 叶子节点是指没有子节点的节点
示例 1:

输入: [1,2,3]
    1
   / \
  2   3
输出: 25
解释:
从根到叶子节点路径 1->2 代表数字 12.
从根到叶子节点路径 1->3 代表数字 13.
因此，数字总和 = 12 + 13 = 25.
*/

int sumNumbers(TreeNode* root) {
    //方法1：广度优先遍历
    if(!root) return 0;
    int ans = 0;
    queue<TreeNode*> que;
    que.push(root);
    while(!que.empty()) {
        int size = que.size();
        for (int i = 0; i != size; i++) {
            TreeNode* node = que.front();
            que.pop();
            if (!node->left && !node->right) ans += node->val;
            if (node->left) {
                node->left->val = 10 * node->val + node->left->val;
                que.push(node->left);
            }
            if (node->right) {
                node->right->val = 10 * node->val + node->right->val;
                que.push(node->right);
            }
        }
    }
    return ans;
}


//方法2：递归
int dfs(TreeNode* root, int sum) {
    if(!root) return 0;
    if (!root->left && !root->right)
        return 10 * sum + root->val;
    return dfs(root->left, 10 * sum + root->val) +
           dfs(root->right, 10 * sum + root->val);
}

int sumNumbers(TreeNode* root) {
    return dfs(root, 0);
}

