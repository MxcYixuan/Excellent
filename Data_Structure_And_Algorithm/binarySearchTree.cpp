// 给定二叉搜索树的后续遍历数组，重建二叉搜索树
// vec = {2, 4, 3, 6, 8, 7, 5}
TreeNode* rebuildTreeFromPostOrder(vector<int> &vec, int left, int right) {
    if (left > right) return nullptr;

    TreeNode* root = new TreeNode(vec[right]);
    if (left == right) return root;

    int id = 0;
    for (int i = left; i != right; i++) {
        if (vec[i] >= vec[right]) {
            id = i;
            break;
        }
    }
    root->left = rebuildTreeFromPostOrder(vec, left, id  - 1);
    root->right = rebuildTreeFromPostOrder(vec, id, right - 1);
    return root;

}

// 给定二叉搜索树的后续遍历数组，重建二叉搜索树
// vec = {2, 4, 3, 6, 8, 7, 5}
// 采用二分查找
TreeNode* rebuildTreeFromPostOrder2(vector<int> &vec, int left, int right) {
    if (left > right) return nullptr;

    TreeNode* root = new TreeNode(vec[right]);
    if (left == right) return root;

    int id = 0;
    int l = left;
    int r = right - 1;

    while (l <= r) {
        int mid = l + ((r - l) >> 1);
        if (vec[mid] < vec[r]) {
            l = mid + 1;
        } else {
            r = mid - 1;
            id = mid;
        }

    }
    root->left = rebuildTreeFromPostOrder(vec, left, id  - 1);
    root->right = rebuildTreeFromPostOrder(vec, id, right - 1);
    return root;
}

//124. 二叉树中的最大路径和
/*给定一个非空二叉树，返回其最大路径和。
本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
*/
int maxPathSum(TreeNode* root) {
    int val = INT_MIN;
    maxVal(root, val);
    return val;
}
int maxVal(TreeNode* root, int &val) {
    if (root == nullptr) return 0;
    int left = max(0, maxVal(root->left, val));
    int right = max(0, maxVal(root->right, val));
    val = max(val, (left + right + root->val));
    return max(left, right) + root->val;
}

//96. 不同的二叉搜索树
/*给定一个整数 n，生成所有由 1 ... n 为节点所组成的 二叉搜索树.*/
vector<TreeNode*> generateTrees(int n) {
    if(n == 0) return {};
    return helper(1, n);
}
vector<TreeNode*> helper(int start, int end) {
    if (start > end) return {nullptr};
    vector<TreeNode*> res;
    for (int i = start; i <= end; i++) {
        vector<TreeNode*> left_trees = helper(start, i - 1);
        vector<TreeNode*> right_trees = helper(i + 1, end);
        for (auto l : left_trees) {
            for (auto r : right_trees) {
                TreeNode* node = new TreeNode(i);
                node->left = l;
                node->right = r;
                res.push_back(node);
            }
        }

    }
    return res;
}

//98. 验证二叉搜索树
//方法1：中序遍历放入栈中，再循环迭代
vector<int> t;
bool isValidBST3(TreeNode* root) {
    if(root == NULL) return true;
    //vector<int> t;
    dfs(root);
    for(int i =0;i < t.size() - 1;i++){
        if(t[i + 1] <= t[i]) return false;
    }
    return true;

}
void dfs(TreeNode* root){
    if(root -> left) dfs(root->left);
    t.push_back(root -> val);
    if(root -> right) dfs(root->right);
}
//方法2：递归调用
bool helper(TreeNode* root, long long lower, long long upper) {
    if (root == nullptr) return true;
    if (root->val <= lower || root->val >= upper)
        return false;
    return helper(root->left, lower, root->val) && helper(root->right, root->val, upper);
}
bool isValidBST2(TreeNode* root) {

    return helper(root, LONG_MIN, LONG_MAX);
}
//方法3：中序判断，使用栈
bool isValidBST(TreeNode* root) {
    stack<TreeNode*> s;
    long long value = (long long)INT_MIN - 1;
    while(!s.empty() || root != nullptr) {
        while(root != nullptr) {
            s.push(root);
            root = root->left;
        }
        root = s.top();
        s.pop();
        if (root->val <= value) return false;
        value = root->val;
        root = root->right;
    }
    return true;
}


