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


