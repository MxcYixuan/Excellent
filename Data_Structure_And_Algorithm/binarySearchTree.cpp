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

//102. 二叉树的层序遍历
//给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if(!root) return res;
    queue<TreeNode*> que;
    que.push(root);
    while(!que.empty()) {
        int len = que.size();
        vector<int> tmp;
        for (int i = 0; i != len; i++) {
            TreeNode* node = que.front();
            que.pop();
            tmp.push_back(node->val);
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        res.push_back(tmp);
    }
    return res;
}
//103. 二叉树的锯齿形层次遍历
//给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    //BFS
    vector<vector<int>>res;
    if(!root)return res;
    queue<TreeNode*>q;
    q.push(root);
    int x=0;  //判断当前行奇偶
    while(!q.empty()){
        int l=q.size();
        vector<int>tmp;
        for(int i=0;i<l;i++){
            TreeNode* t=q.front();
            q.pop();
            tmp.push_back(t->val);
            if(t->left)q.push(t->left);
            if(t->right)q.push(t->right);
        }
        x++;
        if(x%2==0)reverse(tmp.begin(),tmp.end());
        res.push_back(tmp);
    }
    return res;
}

//105. 从前序与中序遍历序列构造二叉树
//根据一棵树的前序遍历与中序遍历构造二叉树。
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int inSize = inorder.size();
    int preSize = preorder.size();
    if (inSize != preSize) return nullptr;

    unordered_map<int, int> umap;
    for (int i = 0; i != inorder.size(); i++) {
        umap[inorder[i]] = i;
    }
    TreeNode* root = buildTree(preorder, inorder, umap, 0, preSize-1, 0, inSize-1);
    return root;
}

TreeNode* buildTree(vector<int> &preorder, vector<int> &inorder, unordered_map<int, int>&umap,
                    int preLeft, int preRight, int inLeft, int inRight) {
    if (preLeft > preRight || inLeft > inRight)
        return nullptr;
    int val = preorder[preLeft];
    int index = umap[val];
    TreeNode* node = new TreeNode(val);
    node->left = buildTree(preorder, inorder, umap, preLeft+1, preLeft+index-inLeft, inLeft, index - 1);
    node->right = buildTree(preorder, inorder, umap, preLeft+index-inLeft+1,preRight, index+1, inRight);
    return node;
}

//106. 从中序与后序遍历序列构造二叉树
//根据一棵树的中序遍历与后序遍历构造二叉树。
//你可以假设树中没有重复的元素。
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {

    if (inorder.size() == 0 || postorder.size() == 0) return nullptr;
    unordered_map<int, int> umap;
    for (int i = 0; i != inorder.size(); i++) {
        umap[inorder[i]] = i;
    }
    TreeNode* root = buildTree(inorder, postorder, umap, 0, inorder.size()-1, 0, inorder.size()-1);
    return root;
}

TreeNode* buildTree(vector<int>&inorder, vector<int>&postorder, unordered_map<int, int>&umap,
                    int postLeft, int postRight,
                    int inLeft, int inRight) {
    if (inLeft > inRight || postLeft > postRight)
        return nullptr;
    int val = postorder[postRight];
    int index = umap[val];
    TreeNode* node = new TreeNode(val);
    node->left = buildTree(inorder, postorder, umap, postLeft, postLeft+index-inLeft-1, inLeft, index-1);
    node->right = buildTree(inorder, postorder, umap, postLeft+index-inLeft,postRight-1, index+1,inRight);
    return node;
}
//108. 将有序数组转换为二叉搜索树
//将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
//本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
TreeNode* dfs(vector<int>& nums, int left, int right) {
    if (left > right)
        return nullptr;
    int mid = left + (right - left) / 2;
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = dfs(nums, left, mid - 1);
    root->right = dfs(nums, mid + 1, right);
    return root;
}

TreeNode* sortedArrayToBST(vector<int>& nums) {
    return dfs(nums, 0, nums.size() - 1);
}

//110. 平衡二叉树
//给定一个二叉树，判断它是否是高度平衡的二叉树。
//本题中，一棵高度平衡二叉树定义为：
//一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
int height(TreeNode* root) {
    if(!root) return 0;
    return 1 + max(height(root->left), height(root->right));
}

bool isBalanced(TreeNode* root) {
    if (!root) return true;
    int left = height(root->left);
    int right = height(root->right);
    if (abs(left - right) >= 2)
        return false;
    return isBalanced(root->left) && isBalanced(root->right);
}

//111. 二叉树的最小深度
//给定一个二叉树，找出其最小深度。
//最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
//说明: 叶子节点是指没有子节点的节点。
int minDepth(TreeNode* root) {
    if(!root) return 0;
    if(!root->left && !root->right) return 1;
    if (!root->left && root->right) return minDepth(root->right) +1;
    if (root->left && !root->right) return minDepth(root->left) +1;
    return min(minDepth(root->left), minDepth(root->right)) + 1;
}


