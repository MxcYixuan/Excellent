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
