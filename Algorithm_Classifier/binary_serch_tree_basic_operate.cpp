// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

// 二叉树的前序遍历
void preorder_tree(TreeNode* root) {
    if (!root) return;
    cout << root->val << '\t';
    preorder_tree(root->left);
    preorder_tree(root->right);
}

// 二叉树的中序遍历
void inorder_tree(TreeNode* root) {
    if (!root) return;

    inorder_tree(root->left);
    cout << root->val << '\t';
    inorder_tree(root->right);
}

// 二叉树的后序遍历
void postorder_tree(TreeNode* root) {
    if (!root) return;
    postorder_tree(root->left);
    postorder_tree(root->right);
    cout << root->val << '\t';
}
