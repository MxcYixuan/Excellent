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
