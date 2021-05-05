// labuladong 第二章
// 数据结构的基本操作
// 对于任何数据结构，其基本操作，无非是 遍历+访问，再具体点：
增删查改

1.数组遍历框架
void traverse(vector<int> arr) {
	for (int i = 0; i < arr.size(); i++) {
		// 迭代访问 arr[i]
	}
}

2.链表遍历框架，定性的线性迭代结构：
/* 基本的单链表节点 */
class ListNode {
	int val;
	ListNode* next;
};

void traverse(ListNode* head) {
	for (ListNode* p = head; p != nullptr; p = p->next) {
		// 迭代访问 p.val
	}
}

void traverse(ListNode* head) {
	// 递归访问 head.val
	traverse(head->next);
}

3.二叉树的遍历框架
/* 基本的二叉树节点 */
class TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {

	}
};

4.N叉树的遍历框架；
4.1 Java版本
/* 基本的N 叉树节点 */
class Node {
	int val;
	Node[] children;
};

void traverse(Node root) {
	for (Node child : root.children) {
		traverse(child)
	}
}
N叉树的遍历，又可以扩展为图的遍历，因为图就是好几个N叉树的结合。
4.2 C++版本

class Node{
	int val;
	vector<Node*> children;
	Node(int _val) {
		val = _val;
	}
	Node(int _val, vector<Node*>_children) {
		val = _val;
		children = _children;
	}
};
void traverse(Node* root) {
	for(Node* child : root->children) {
		traverse(child);
	}
}

// Part2.2 训练递归思维，链表题目
// 题目：2.2.1 递归翻转单链表
ListNode* reverse(ListNode* head) {
	if(!head->next) return head;
	ListNode* last = reverse(head->next);
	head->next->next = head;
	head->next = nullptr;
	return last;
}


ƒ
// 描述：给你一个链表，每k个节点一组进行反转，请你返回翻转后的链表
// k是一个正整数，它的值小于或等于 链表的长度。
// 如果节点总数 不是k 的整数倍，纳闷请将最后剩余的节点 保持原有顺序
/*
  例子1：给定链表：1->2->3->4->5
  当k=2时，应当返回 2->1->4->3->5
  当k=3时，应当返回 3->2->1->4->5
*/

2.2.1中我们用递归翻转链表，这次我们用迭代的方式去解决。
首先实现
1.step
// 翻转以 a 为头结点的链表
ListNode* reverse(ListNode* a) {
	ListNode* pre, curr, next;
	pre = nullptr;
	curr = a;
	next = a;
	while(curr != null) {
		next = curr->next;
		// 逐个节点反转
		curr->next = pre;
		// 更新指针位置
		pre = cur;
		cur = next;
	}
	// 返回反转后的头节点
	return pre;
}

2.step // 反转区间[a, b)的元素，注意是左闭右开
ListNode* reverse(ListNode* a, ListNode* b) {
	ListNode* pre, curr, next;
	pre = nullptr;
	curr = a;
	next = a;
	// while 终止的条件改一下就好了
	while(curr != b) {
		next = curr->next;
		curr->next = pre;
		pre = curr;
		curr = next;
	}
	// 返回反转后的头结点
	return pre;
}

3.现在通过迭代，实现了反转部分链表的功能，以下实现reverseKGroup()函数
ListNode* reverseKGroup(ListNode* head, int k) {
	if (head == nullptr) return nullptr;
	// 区间[a, b) 包含 k 元素
	ListNode* a, b;
	a = b = head;
	for (int i = 0; i < k; i++) {
		// 不足k个，不需要反转，这是 base case
		if（b == nullptr） return head;
		b = b->next;
	}
	// 反转 前k个元素
	ListNode* newHead = reverse(a, b);
	// 递归反转 后续链表 并连接起来
	a->next = reverseKGroup(b, k);
	return newHead;
}


// Part2.2 训练递归思维，链表题目
// 题目：2.2.3 如何高效判断 是否回文链表？
寻找回文串的核心思想是从中心向
string palindrome(string& s, int l, int r) {
	// 防止索引越界
	while(l >= 0 && r <= s.size() && s[l] == s[r]) {
		// 向两边展开
		l--; r++;
	}
	// 返回以 s[l] 和 s[r] 为中心的最长回文串
	return s.substr(l + 1, r - l - 1);
}
注：回文串长度可能为奇数，也可能为偶数，长度为奇数时，只存在一个中心点，
长度为偶数时，就存在两个中心点，所以l和r 就是要处理这两种情况。

以往例子：判断一个字符串是否是回文串？
不需要考虑奇偶情况，只需要双指针，从两端向中间逼近即可。
bool isPalindrome(string& s) {
	int left = 0;
	int right = s.size() - 1;
	while(left < right) {
		if (s[left] != s[right]) {
			return false;
		}
		left++;
		right--;
	}
	return true; 
}

// 以下的重点是，题目：输入一个单链表的头节点，判断这个链表中的数字是不是回文？
/*
  例子：
  输入：1->2->null         输出：false

  输入：1->2->2->1->null   输出：true
*/
分析：这道题的难点在于：单链表无法倒着遍历，也就无法使用双指针技巧。
最简单的方法是：把原始链表翻转，存入一条新的链表，然后比较这两条链表是否相同

1.以下是巧妙解法：
借助二叉树后续遍历的思路，不需要显示的反转原始链表，也可以倒序遍历链表。
链表其实也可以有：前序遍历和后续遍历：

void traverse(ListNode* head) {
	// 前序遍历代码
	traverse(head->next);
	// 后续遍历代码
}

//例如
/* 倒序打印单链表中的元素值 */
void traverse(ListNode* head) {
	if (head == nullptr) return;
	traverse(head->next);
	// 后序遍历代码
	print(head->val);
}
因此，可以稍作修改，模仿双指针实现 回文判断的功能：

ListNode* left;
bool isPalindrome(ListNode* head) {
	left = head;
	return traverse(head);
}

bool traverse(ListNode* right) {
	if (right == nullptr) return true;
	bool res = traverse(right->next);
	// 后序遍历代码
	res = res && (right->val == left->val);
	left = left->next;
	return res;
}
分析：无论是造一条反转链表，还是利用后序遍历，算法的时间和空间复杂度都是O(N)
下面考虑将 空间复杂度降到O(1)

主要思路：通过快慢指针，来找到链表的中点。

bool isPalindrome(ListNode* head) {
	ListNode* slow;
	ListNode* fast;
	slow = fast = head
	while(fast != NULL && fast->next != NULL) {
		slow = slow->next;
		fast = fast->next->next;
	}
	// slow 指针现在指向链表中点
	// 如果fast 指针没有指向NULL, 说明链表长度为奇数，slow还要再前进一步：
	if(fast != NULL) 
		slow = slow->next;
	// 现在开始比较 回文串了
	ListNode* left = head;
	ListNode* right = traverse(slow);
	while(right != NULL) {
		if(left->val != right->val) 
			return false;
		left = left->next;
		right = right->next;
	}
	return true;
}
ListNode* reverse(ListNode* head) {
	ListNode* pre = NULL;
	ListNode* curr = head;
	while(curr != NULL) {
		ListNode* next = curr->next;
		curr->next = pre;
		pre = curr;
		curr = next;
	}
	return pre;
}
算法的时间复杂度为O(N), 空间复杂度为O(1)
总结一下：
首先，
1.寻找回文串是从中间向两端扩展
2.判断回文串 是从两端向中间收缩
3.对于单链表，无法直接倒序遍历，可以造一条新的反转链表，可以利用链表的后序遍历
  也可以用栈结构倒序处理单链表
4.具体到回文链表的判断问题，由于回文的特殊性，可以不完全反转链表，而是仅仅反转部分链表，
  降空间复杂度降到O(1),不过需要注意链表长度的奇偶。


// Part2.3 训练递归思维，链表题目
// 题目：2.3.1 二叉树递归框架I
/*
  题目：leetcode 226, 翻转二叉树
       leetcode 114, 将二叉树展开为链表
       leetcode 116, 填充二叉树节点的右侧指针
*/
二叉树的结构：
思维：快速排序是二叉树的前序遍历，归并排序是二叉树的后序遍历。
1.快速排序框架
void sort(vector<int>& nums, int low, int high) {
	/* 前序遍历位置 */
	// 通过交换元素，构建分界点p
	int p = partition(nums, low, high);

	sort(nums, low, p - 1);
	sort(nums, p + 1, high);
}

2.归并排序框架
void sort(vector<int>& nums, int low, int high) {
	int mid = (low + high) / 2;
	sort(nums, low, mid);
	sort(nums, mid + 1, high);
	/* 后序遍历位置 */
	// 合并两个排好序的子数组
	merge(nums, low, mid, high);
}

3.练习1， 计算一颗二叉树共有几个节点
// 定义：count(root), 返回以root为根的树有多少节点
int count(TreeNode* root) {
	// base case
	if (root == NULL) return 0;
	return 1 + count(root->left) + count(root->right);
}

4.翻转二叉树
// 将整棵树的节点翻转
TreeNode* invertTree(TreeNode* root) {
	// base case
	if (root == NULL) {
		return NULL;
	}
	/* 前序遍历位置 */
	TreeNode* tmp = root->left;
	root->left = root->right;
	root->right = tmp;
	// 让左右子节点 继续翻转它们的子节点
	invertTree(root->left);
	invertTree(root->right);

	return root;
}

5.填充二叉树节点的右侧指针
//给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点，二叉树定义如下：
struct Node {
	int val;
	Node* left;
	Node* right;
	Node* next;
};
// 填充它的每个next指针，让这个指针指向下一个右侧节点，如果找不到下一个右侧节点，则将next指针设置为NULL.


// 这道题的难点在于，如何把题目的要求细化成每个节点需要做的事情，只依赖一个节点，做不到，那么久安排两个节点。
void connectTwoNode(Node* node1, Node* node2) {
	if (node1 == NULL || node2 == NULL) {
		return;
	}
	/* 前序遍历位置 */
	// 将传入的两个节点 连接
	node1->next = node2;

	// 连接相同父节点的两个子节点
	connectTwoNode(node1->left, node1->right);
	connectTwoNode(node2->left, node2->right);

	// 连接跨越父节点的两个子节点
	connectTwoNode(node1->right, node2->left);
}

6.将二叉树展开为链表
// leetcode 114
/*
  给定一个二叉树，原地将它展开为一个单链表，例如，给定二叉树：
          1
         / \
        2   5
       /  \   \
      3   4    6
   
   将其展开为：
            1
              \
               2
                \
                 3
                  \
                   4
                    \
                     5
                      \
                       6
*/ 

6.1思路：
1.将root 的左子树 和 右子树拉平
2.将root 的右子树接到 左子树下放，然后将整个左子树作为右子树

// 定义： 将以root 为根的树 拉平为链表
void flatten(TreeNode* root) {
	// base case
	if (root == NULL) return ;
	flatten(root->left);
	flatten(root->right);

	/* 后序遍历位置 */
	// 1. 左右子树已经被拉平成一条链表
	TreeNode* left = root->left;
	TreeNode* right = root->right;

	// 2. 将左子树作为右子树
	root->left = NULL;
	root->right = left;

	// 将 原先的右子树接到 当前右子树的末端
	TreeNode* p = root;
	while(p->right != NULL) {
		p = p->right;
	}
	p->right = right;
}


// Part2.3 训练递归思维，链表题目
// 题目：2.3.2 二叉树递归框架II
// LeetCode 654 最大二叉树
// leetcode 105 从前序与中序遍历序列 构造二叉树
// leetcode 106 从中序与后序遍历序列 构造二叉树

1.构造最大二叉树
/*
  描述：给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
   1.二叉树的根 是数组中的最大元素。
   2.左子树是通过数组中最大值左边部分 构造出的最大二叉树
   3.右子树是通过数组中最大值右边部分 构造出的最大二叉树
  通过给定的数组构建最大二叉树，并且输出这个树的根节点
*/
代码如下：
/* 主函数 */
TreeNode* constructMaximunBinaryTree(vector<int>& nums) {
	return build(nums, 0, nums.size() - 1);
}
/* 将nums[low...high] 构造成符合条件的树，返回根节点*/
TreeNode* build(vector<int>& nums, int low, int high) {
	// base  case
	if (low > high) {
		return NULL;
	}
	// 找到数组中的最大值和对应的索引
	int index = -1, maxVal = INT_MIN;
	for (int i = low; i <= high; i++) {
		if (maxVal < nums[i]) {
			index = i;
			maxVal = nums[i];
		}
	}
	TreeNode* root = new TreeNode(maxVal);
	// 递归调用构造左右子树
	root->left = build(nums, low, index - 1);
	root->right = build(nums, index + 1, high);

	return root; 
}

2.从前序与中序遍历序列中，构造二叉树
/*
  根据一棵树的前序遍历与中序遍历构造二叉树
  注意：假定树中没有重复元素。
  例如：
  前序遍历 preorder = [3, 9, 20, 15, 7]
  中序遍历  inorder = [9, 3, 15, 20, 7]
  返回如下的二叉树：
         3
        / \
       9   20
          /  \
         15   7
*/

/* 主函数 */
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
	return build(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}
/*
  若前序遍历数组为：preorder[preStart, preEnd]
  后续遍历数字为 ： inorder[inStart, inEnd]
  构造二叉树 返回该二叉树的根节点
*/
TreeNode* build(vector<int>& preorder, int preStart, int preEnd, vector<int>& inorder, int inStart, int inEnd) {
	// base case
	if(preStart > preEnd) {
		return NULL;
	}

	// root 节点 对应的值 就是前序遍历数组中的第一个
	int rootVal = preorder[preStart];
	// rootVal 在中序遍历数组中的索引
	int index = 0;
	for (int i = inStart; i <= inEnd; i++) {
		if (inorder[i] == rootVal) {
			index = i;
			break;
		}
	}
	TreeNode* root = new TreeNode(rootVal);
	// 递归的构造左右子树
	root->left = build(preorder, preStart + 1, index - inStart - preStart, inorder, inStart, index - 1);
	root->right = build(preorder,preStart + index - inStart + 1, preEnd, inorder, index + 1, inEnd);

	return root; 
}


3.通过后序和中序遍历结果 构造二叉树
/*
 根据一颗数的中序 与 后序遍历 构造二叉树
 注意：可以假定树中 没有重复元素。。
 例如：
 中序遍历 inorder = [9, 3, 15, 20, 7]
 后序遍历 postorder = 9, 15, 7, 20, 3]
 返回如下的二叉树：
         3
        / \
       9   20
          /  \
         15   7
*/ 
代码和上文类似
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
	return build(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
}
TreeNode* build(vector<int>& inorder, int inStart, int inEnd, vector<int>& postorder, int postStart, int postEnd) {

	if (inStart > inEnd) {
		return NULL;
	}
	// root 节点对应的值 就是后序遍历数组的最后一个元素
	int rootVal = postorder[postEnd];
	// rootVal 在中序遍历数组中的索引
	int index = 0;
	for (int i = inStart; i <= inEnd; i++) {
		if (inorder[i] == rootVal) {
			index = i;
			break;
		}
	}
	TreeNode* root = new TreeNode(rootVal);
	// 递归构造左右子树
	int leftSize = index - inStart;
	root->left = build(inorder, inStart, index - 1, postorder, postStart, index - 1 - inStart + postStart);
	root->right = build(inorder, index + 1, inEnd, postorder, postStart + index - inStart, postEnd - 1);

	return root;

}



// Part2.3 训练递归思维，链表题目
// 题目：2.3.3 二叉树递归框架III
// 给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。
// 两棵树重复是指它们具有相同的结构以及相同的结点值。
/*
  例子：
    	1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
下面是两个重复的子树：
 	  2
     /
    4
和  4
*/

//思路：问题的难点在于如何在遍历的时候对之前遍历过的子树进行描述和保存。
//这里就需要使用之前使用过的二叉树序列化的手法，将遍历到的二叉树进行序列化表达，
//我们知道序列化的二叉树可以唯一的表示一棵二叉树，并可以用来反序列化。
// 我们只需在遍历的过程中将每次的序列化结果保存到一个HashMap中，并对其进行计数，如果重复出现了
//那么将当前的节点添加到res中即可
vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
	vector<TreeNode*> res;
	unordered_map<string, int> umap;
	dfs(root, res, umap);
	return res;
}

string dfs(TreeNode* root, vector<TreeNode*>& res, unordered_map<string, int>&umap) {
	if (root == NULL) {
		return "";
	}
	// 二叉树先序序列化
	string left = dfs(root->left, res, umap);
	string right = dfs(root->right, res, umap);
	string str = to_string(root->val) + "," + left + "," + right;
	if (umap[str] == 1) {
		res.push_back(root);
	}
	umap[str]++;
	return str;
}



vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
	vector<TreeNode*> res;
	unordered_map<string, int> umap;
	dfs(root, res, umap);
	return res;
}

string dfs(TreeNode* root, vector<TreeNode*>& res, unordered_map<string, int>& umap) {
	if (root == NULL) return "";
	string left = dfs(root->left);
	string right = dfs(root->right);

	string result = to_string(root->val) + "," + left + "," + right;
	if (umap[result] == 1) {
		res.push_back(root);
	}
	umap[result]++;
	return result;
}


// Part2.3 训练递归思维，链表题目
// 题目：2.3.4 二叉搜索树I(Binary Search Tree)
0.二叉搜索树并不复杂，但构建起了数据结构领域的半壁江山，
 直接基于BST的数据结构有AVL树，红黑树等，拥有了自平衡性。
 可以提供logN级别的增删查改效率；B+树，线段树结构都是基于BST思想设计的。
0.1 BST性质：中序遍历是有序的(升序)

void traverse(TreeNode* root) {
	if (root == NULL) return;
	traverse(root->left);
	// 中序遍历代码位置
	print(root->val);
	traverse(root->right);
}

1. 题目1：寻找第k小的元素
// leetcode 230 二叉搜索树中第K小的元素。
/*
  给定一个二叉搜索树，编写一个函数kthSmallest 来查找其中第k个元素，
  你可以假设K总是有效的
*/
1.1 思路1，中序遍历，然后选择第k个

int kthSmallest(TreeNode* root, int k) {
	// 利用 BST的中序特性遍历
	// 记录结果
	int res = 0;
	// 记录当前元素的排名
	int rank = 0;
	traverse(root, k, res, rank);
	return res;
}

//
void traverse(TreeNode* root, int k, int& res, int& rank) {
	if (root == NULL) {
		return;
	}
	traverse(root->left, k, res, rank);
	/* 中序遍历代码位置 */
	rank++;
	if (k == rank) {
		// 找到第k小的元素
		res = root->val;
		return;
	}
	/********************/
	traverse(root->right, k, res, rank);
}

2.BST累计树
/*
  leetcode 538 把二叉搜索树 转换为累加树
  给出二叉搜索树的根节点，该树的节点值各不相同，请你将其转换为累加树，使每个节点node的新值
  等于原树中 大于或等于 node.val的值之和。
*/
TreeNode* covertBST(TreeNode* root) {
	traverse(root);
	return root;
}
// 记录 累加和
int sum = 0;
void traverse(TreeNode* root) {
	if (root == NULL) {
		return;
	}
	traverse(root->right);
	// 维护累加和
	sum += root->val;
	// 将 BST 转为累加树
	root->val = sum;
	traverse(root->left);
}


// Part2.3 训练递归思维，链表题目
// 题目：2.3.5 二叉搜索树II(Binary Search Tree)
1.题目:主要是BST的基础操作：判断BST的合法性、增、删、查。(其中删和判断合法性略微复杂)
0.有坑的做法：
bool isValidBST(TreeNode* root) {
	if (root == NULL) return true;
	if (root->left != NULL && root->val <= root->left->val) {
		return false;
	} 
	if (root->right != NULL && root->val >= root->right->val) {
		return false;
	}
	return isValidBST(root->left) && isValidBST(root->right);
}
// 这个算法的错误在于，BST的每个节点应该要小于 右边子树的所有节点。
// 而下面的情况显然不是BST，但是这个算法会把它判定为合法BST:
/*
                 10
                /  \
               5    15
                   /  \
                  6    20

   显然按照上述算法，满足条件，但并不是BST
*/
1.以下是正确的代码
bool isValidBST(TreeNode* root) {
	return isValidBST(root, NULL, NULL);
}
/* 限定以root 为根的子树节点必须满足 max.val > root.val > min.val */
bool isValidBST(TreeNode* root, TreeNode* min, TreeNode* max) {
	// base case
	if (root == NULL) return true;

	// 若root->val 不符合max 和 min的限制，说明不是合法BST
	if (min != NULL && root->val <= min->val);
		return false;
	if (max != NULL && root->val >= max->val);
		return false;
	// 限定左子树的最大值是root->val, 右子树的最小值是root->val
	return isValidBST(root->left, min, root) && 
	       isValidBST(root->right, root, max); 
}
注：通过使用辅助函数，增加函数参数列表，在参数中携带额外信息，
将这种约束传递给子树的所有节点，这也是二叉树算法的一个小技巧吧。

2.在BST中搜索一个数
如果是 在二叉树中寻找元素，可以这样写代码
bool isInBST(TreeNode* root, int target) {
	if (root == NULL) 
		return false;
	if (root->val == target)
		return true;
	// 当前节点没找到就递归地去左右子树寻找
	return isInBST(root->left, target) || isInBST(root->right, target);
}
这样写并美誉问题，但是适用于所有的普通二叉树，BST性质并没有用到。
写法2：
bool isInBST(TreeNode* root, int target) {
	if (root == NULL) return false;
	if (root->val == target) 
		return true;
	if (root->val < target) {
		return isInBST(root->right, target);
	}
	if (root->val > target) {
		return isInBST(root->left, target);
	}
}

因此，可以对原始框架进行改造，抽象出一套针对BST的遍历框架
void BST(TreeNode* root, int target) {
	if (root->val == target) {
		// 找到目标，做点什么
	}
	if (root->val < target) {
		BST(root->right, target);
	}
	if (root->val > target) {
		BST(root->left, target);
	}
}

3.在BST中插入一个数
TreeNode* insertIntoBST(TreeNode* root, int val) {
	// 找到空位置 插入新节点
	if (root == NULL) {
		return new TreeNode(val);
	}
	// if(root->val == val) 
	//  BST 中一般不会插入已存在的元素
	if(root->val < val) {
		root->right = insertIntoBST(root->right, val);
	} 
	if(root->val > val) {
		root->left = insertIntoBST(root->left, val);
	}
	return root;
}

4.在BST中删除一个数
先写框架
TreeNode* deleteNode(TreeNode* root, int key) {
	if (root->val == key) {
		// 找到了 进行删除
	} else if (root->val > key) {
		// 去左子树找
		root->left = deleteNode(root->left, key);
	} else if (root->val < key) {
		// 去右子树找
		root->right = deleteNode(root->right, key);
	}
	return root;
}
如何删除，要分三种情况

4.1 情况1：删除的节点A，恰好是末端节点，两个子节点都为空，那么直接去掉即可
if(root->left == NULL && root->right == NULL)
	return NULL;

4.2 情况2：A只有一个非空子节点，那么它要让这个孩子接替自己的位置。
// 排查情况1后，
if(root->left == NULL) return root->right;
if(root->right == NULL) return root->left;

4.3 情况3：A有两个子节点，为了不破坏BST性质，A必须找到左子树中的最大的那个节点，或者右子树的最小那个节点来接替自己
if(root->left != NULL && root->right != NULL) {
	// 找到右子树的最小节点
	TreeNode* minNode = getMin(root->right);
	// 把root 改为minNode
	root->val = minNode->val;
	// 转而去删除 minNode
	root->right = deleteNode(root->right, minNode->val);
}

那么整体的简化代码如下：
TreeNode* deleteNode(TreeNode* root, int key) {
	if (root == NULL) 
		return NULL;
	if (root->val == key) {
		// 这两个if 把情况1 和 情况2 都正确处理了。
		if (root->left == NULL) return root->right;
		if (root->right == NULL) return root->left;
		// 处理情况3：
		TreeNode* minNode = getMin(root->right);
		root->val = minNode->val;
		root->right = deleteNode(root->right, minNode->val);
	} else if (root->val > key) {
		root->left = deleteNode(root->left, key);
	} else if (root->val < key) {
		root->right = deleteNode(root->right, key);
	}
	return root;
}

TreeNode* getMin(TreeNode* node) {
	// BST 最左边的就是最小的
	while(node->left != NULL) {
		node = node->left;
	}
	return node;
}





// Part2.3 训练递归思维，链表题目
// 题目：2.3.6 二叉树的序列化
leetcode 297: 给你输入一颗二叉树的根节点root, 要求你实现如下一个类
class Codec {
	// 把一个二叉树序列化成字符串
	string serialize(TreeNode*) {

	}
	// 把字符串反序列化成二叉树
	TreeNode* deserialize(string data) {

	} 
};

1.解法1：前序遍历解法
class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        ostringstream out;
        serialize(root, out);
        return out.str();
    }
    void serialize(TreeNode* root, ostringstream& out) {
        if (root == NULL) {
            out << "#" << " ";
            return;
        } 
        /********前序遍历位置*********/
        out << root->val << " ";
        /**************************/
        serialize(root->left, out);
        serialize(root->right, out);
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        istringstream is(data);
        return dfs(is);
    }
    TreeNode* dfs(istringstream& s) {
        string t;
        s >> t;
        if (t == "#") 
            return NULL;
        /*********前序遍历位置************/
        // 最左侧就是根节点
        TreeNode* node = new TreeNode(stoi(t));
        /******************************/
        node->left = dfs(s);
        node->right = dfs(s);
        return node;
    }
};

2.解法2：后续遍历解法
后序遍历可以，但是没必要，前序就可以了，后序处理起来，稍微复杂些。

3.解法3：层次遍历
class Codec {
public:
	// Encodes a tree to as single string.
	string serialize(TreeNode* root) {
		ostringstream out;
		queue<TreeNode*> que;
		que.push(root);
		while(!que.empty()) {
			TreeNode* tmp = que.front();
			que.pop();
			if (!tmp) {
				out << "#" << " ";
			} else {
				out << tmp->val << " ";
				que.push(tmp->left);
				que.push(tmp->right);
			}
		}
		return out.str();
	}
	// Decodes your encodes data to tree;
	TreeNode* deserialize(string data) {
		istringstream input(data);
		string val;
		vector<TreeNode*> vec;
		while(input >> val) {
			if (val == "#") {
				vec.push_back(NULL);
			} else {
				vec.push_back(new TreeNode(stoi(val)));
			}
		}
		int j = 1;                                          // i每往后移动一位，j移动两位，j始终是当前i的左子下标
        for (int i = 0; j < vec.size(); ++i) {              // 肯定是j先到达边界，所以这里判断j < vec.size()
            if (vec[i] == NULL) continue;                   // vec[i]为null时跳过。
            if (j < vec.size()) vec[i]->left = vec[j++];    // 当前j位置为i的左子树
            if (j < vec.size()) vec[i]->right = vec[j++];   // 当前j位置为i的右子树
        }
        return vec[0];
	}
};


// Part2.3 训练递归思维，链表题目
// 题目：2.3.7 扁平化嵌套列表迭代器
/*
 题目描述：leetcode 341
 给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。
 列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。 

 示例1：
 输入：[[1, 1], 2, [1, 1]]
 输出：[1, 1, 2, 1, 1]
 解释：通过重复调用next 直到 hasNext 返回false，next 返回的元素的顺序应该是：[1, 1, 2, 1, 1].
*/
 /**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */

 class NestedIterator {
 private:
 	stack<NestedInteger> st;
 public:
 	NestedIterator(vector<NestedInteger>& nestedList) {
 		for(auto iter = nestedList.rbegin(); iter ! =nestedList.rend(); iter++) {
 			st.push(*iter);
 		}
 	}

 	int next() {
 		auto res = st.top();
 		st.pop();
 		return res.getInteger();
 	}

 	bool hasNext() {
 		while(!st.empty()) {
 			auto cur = st.top();
 			if(cur.isInteger()) 
 				return true;
 			st.pop();
 			auto curList = cur.getList();
 			for(auto iter = curList.rbegin(); iter != curList.rend(); iter++) {
 				st.push(*iter);
 			}
 		}
 		return false;
 	}

 };


// Part2.3 训练递归思维，链表题目
// 题目：2.3.8 用git来讲讲二叉树 最近公共祖先
// leetcode 236 二叉树的最近公共祖先
0.题目：给定一个二叉树，找到该树中两个指定节点的最近公共祖先。
最近公共祖先的定义为：“对于有根树T的两个节点p、q，最近公共祖先表示一个节点x，
满足x是p、q的祖先且x的深度尽可能大” （一个节点也可以是它自己的祖先）
/*
示例1：        3
             /  \
            5    1
           / \  / \
          6   2 0  8
             / \
            7   4
输入：root = [3, 5, 1, 6, 2, 0, 8, null, null, 7, 4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3. 
*/

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (root == NULL) return NULL;
	if (root == p || root == q) return root;
	TreeNode* left = lowestCommonAncestor(root->left, p, q);
	TreeNode* right = lowestCommonAncestor(root->right, p, q);

	if (left != NULL && right != NULL) {
		return root;
	}
	if (left == NULL && right == NULL) {
		return NULL;
	}
	return (left == NULL) ? right : left;
}



// Part2.3 训练递归思维，链表题目
// 题目：2.3.9 完全二叉树的节点数，你真的会算吗？
1.一颗普通二叉树的节点个数
int countNodes(TreeNode* root) {
	if(root == NULL) 
		return 0;
	return 1 + countNodes(root->left) + countNodes(root->right);
}
时间复杂度为O(N)

2.一颗满二叉树的节点数，节点总数和树的高度呈指数关系，时间复杂度为O(logN)
int countNodes(TreeNode* root) {
	int height = 0;
	while(root != NULL) {
		root = root->left;
		height++;
	}
	// 节点总数为 2^height - 1;
	return pow(2, height) - 1;
}

3.一颗完全二叉树的节点数
int countNodes(TreeNode* root) {
	TreeNode* left = root;
	TreeNode* right = root;
	int left_height = 0, right_height = 0;
	while(left != NULL) {
		left = left->left;
		left_height++;
	}
	while(right != NULL) {
		right = right->right;
		right_height++;
	}
	if (left_height == right_height) {
		return pow(2, left_height) - 1;
	}
	// 如果左右高度不同，则按照普通二叉树的逻辑计算
	return 1 + countNodes(root->left) + countNodes(root->right);
}



// Part2.4 手把手设计数据结构
// 题目：2.4.1 Union-Find 算法详解
0.Union-Find 并查集算法详解
主要解决图论中的 [动态连通性]问题
0.1 问题介绍：动态连通性 其实可以抽象成给一幅图连线，现在我们的Union-Find算法主要需要实现这两个API
class UF {
	/* 将p 和 q 连接 */
	public void union(int p, int q);
	/* 判断 p 和 q 是否连通 */
	public bool connected(int p, int q);
	/* 返回图中有多少个连通分量 */
	public int count();
};
// 这里说的连通 是一种等价关系，也就是有以下三个性质。
// 1、自反性：节点 p 和 节点 q 是连通的。
// 2、对称性：如果节点p和q连通，那么q和p也是连通的。
// 3、传递性：如果节点 p 和 q 连通，q 和 r 连通，那么 p 和 r 也连通

0.2 基本思路：可以用森林(若干颗树)来表示 图 的动态连通性，用数组来具体实现这个森林。
0.3 怎么用森林来表示连通性呢？我们设定树的每个节点 有一个指针指向其父节点，如果是根节点的话，这个指针指向自己。

以 10 个节点的图 举例，一开始的时候 并没有相互连通，就是这样的：

class UF {
	// 记录连通分量
	private int count;
	// 节点 x 的根节点是 parent[x]
	private int[] parent;

	/* 构造函数，n为图的节点总数 */
	public UF(int n) {
		// 一开始互不连通
		this.count = n;
		// 父节点指针 初始指向自己
		parent = new int[n];
		for (int i = 0; i < n; i++) {
			parent[i] = i;
		}
	};
	/* 其它函数 */

};

如果某两个节点被连通，则让其中的(任意)一个节点的根节点 接到 另一个节点的根节点上；
public void union(int p, int q) {
	int rootP = find(p);
	int rootQ = find(q);
	if (rootP == rootQ) {
		return;
	}
	// 将两颗树 合并为一棵树
	parent[rootP] = rootQ;
	// parent[rootQ] = rootP 也一样
	count--; // 两个分量 合二为一
}

/* 返回某个节点 x 的根节点 */
public int find(int x) {
	// 根节点的 parent[x] == x
	while(parent[x] != x) {
		x = parent[x];
	}
	return x;
}

/* 返回当前的连通 分量个数 */
public int count() {
	return count;
} 

// 如果 节点 p 和 q 连通的话，他们一定有相同的根节点
public bool connected(int p, int q) {
	int rootP = find(p);
	int rootQ = find(q);
	return rootP == rootQ;
}

基本的算法已经晚了，分析下算法的复杂度？
极端情况下，find, union, connected 的时间复杂度都是O(N)
这个复杂度很不理想，毕竟图论解决的都是诸如 社交网络这样数据规模巨大的问题。
对于 union 和 connected 的调用非常频繁，每次调用需要线性时间 完全不能忍受的。

1.下面是如何避免 树的不平衡问题？
  平衡性优化
分析：在union过程中，一开始就是简单粗暴的 把 p 所在的树接到 q 所在的树的根节点下面，
那么这里可能就会出现[头重脚轻]的不平衡情况。
我们希望，小一些的树 接到 大一些的树下面，这样就能避免头重脚轻，更平衡一些，
解决方法是：额外使用一个size数组，记录每棵树包含的节点数，我们不妨称为[重量]；

class UF {
	private int count;
	private int[] parent;

	// 新增一个数组 记录树的重量
	private int[] size;

	public UF(int n) {
		this.count = n;
		parent = new int[n];
		// 最初每棵树 只有一个节点
		// 重量应该初始化为 1
		size = new int[n];
		for (int i = 0; i < n; i++) {
			parent[i] = i;
			size[i] = 1;
		}
	}
	/* 其它函数 */
};

下面修改下 union方法
public void union(int p, int q) {
	int rootP = find(p);
	int rootQ = find(q);
	if (rootP == rootQ) {
		return;
	}
	// 小树接到 大树 下面，较平衡
	if(size[rootP] > size[rootQ]) {
		parent[rootQ] = rootP;
		size[rootP] += size[rootQ];
	} else {
		parent[rootP] = rootQ;
		size[rootQ] += size[rootP];
	}
	count--;
}
1.1 分析：通过比较树的重量，就可以保证树的生长相对平衡，树的高度大致在logN这个数量级上，极大提升执行效率。
1.2 此时，find, union, connected的时间复杂度都下降到O(logN)，即便数据规模上亿，所需时间也非常少。

2.路径压缩
// 这不的优化特别简单，所以非常巧妙
// 我们能不能进一步压缩每棵树的高度，使树高始终保持为常数？
// 这样 find 就能以 O(1)的时间 找到某一节点的根节点，相应的 connected 和union 复杂度都下降到 O(1)
只需要在find函数中 加一行代码
private int find(int x) {
	while(parent[x] != x) {
		// 进行路径压缩，重要
		parent[x] = parent[parent[x]];
		x = parent[x];
	}
	return x;
}

完整代码如下：
class UF {
	// 连通分量个数
	private int count;
	// 存储一棵树
	private int[] parent;
	// 记录树的重量
	private int[] size;

	public UF(int n) {
		this.count = n;
		parent = new int[n];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			parent[i] = i;
			size[i] = 1;
		}
	}

	public void union(int p, int q) {
		int rootP = find(p);
		int rootQ = find(q);
		if(rootP == rootQ) {
			return;
		}
		// 小树接到 大树下面，较平衡
		if(size[rootP] > size[rootQ]) {
			parent[rootQ] = rootP;
			size[rootP] += size[rootQ];
 		} else {
 			parent[rootP] = rootQ;
 			size[rootQ] += size[rootP];
 		}
 		count--;
	}

	public bool connected(int p, int q) {
		int rootP = find(p);
		int rootQ = find(q);
		return rootQ == rootP;
	}

	public int find(int x) {
		while(parent[x] != x) {
			parent[x] = parent[parent[x]];
			x = parent[x]
		}
		return x;
	}
};

分析：构造函数初始化数据结构 需要O(N)的时间和空间复杂度；
连通两个节点union、判断两个节点的连通性connected、计算连通分量count, 所需要的时间复杂度均为O(1)


// Part2.4 手把手设计数据结构
// 题目：2.4.2 Union-Find 并查集算法应用
1. DFS的替代方案
 能够使用 DFS 深度优先算法解决的问题，也可以用Union-Find 算法解决。
1.1 题目 leetcode 130， 
被围绕的区域：给你一个M*N的二维矩阵，其中包含字符 X 和 O，让你找到矩阵中完全 被X 围住的O, 并且把它们替换成X。
可以用Union-Find算法解决，虽然实现复杂一些，甚至效率也略低，但是通用思想可以学一学

// 可以把那些不需要被替换的O 看成一个拥有独门绝技的门派，它们有一个共同祖师爷叫 dummy，这些O 和 dummy互相连通，
// 而那些 需要被替换的O 与 dummy不连通
代码如下：
void slove(char[][] board) {
	if(board.length == 0) return;

	int m = board.length;
	int n = board[0].length;

	// 给dummy 留一个 额外位置
	UF uf new UF(m * n + 1);
	int dummy = m * n;
	// 将首列和末列的O 与dummy连通
	for(int i = 0; i < m; i++) {
		if(board[i][0] == 'O') {
			uf.union(i * n, dummy);
		}
		if(board[i][n - 1] == 'O') {
			uf.union(i * n + n - 1, dummy);
		}
	}
	// 将 首行 和 末行的O 与 dummy连通
	for(int j = 0; j < n; j++) {
		if(board[0][j] == 'O') {
			uf.union(j, dummy);
		}
		if(board[m - 1][j] == 'O') {
			uf.union(n * (m - 1) + j, dummy);
		}
	}
	// 方向数组 d 是 上下左右搜索的常用手法
	int[][] d = new int[][] {{1, 0}, {0, 1}, {0, -1}, {-1, 0}};
	for (int i = 1; i < m - 1; i++) {
		for(int j = 1; j < n - 1; j++) {
			if(board[i][j] == 'O') {
				// 将此 O 与 上下左右的 O 连通
				for(int k = 0; k < 4; k++) {
					int x = i + d[k][0];
					int y = j + d[k][1];
					if(board[x][y] == 'O') {
						uf.union(x * n + y, i * n + j);
					}
				}
			}
		}
	}
	// 所有不和 dump 连通的 O, 都要被替换
	for(int i = 1; i < m - 1; i++) {
		for (int j = 1; j < n - 1; j++) {
			if(!uf.connected(dummy, i * n + j))
				board[i][j] = 'X';
		}
	}
}  
上述方法的主要思路：适时增加虚拟节点，想办法让元素[分门别类]，建立动态连通关系


2. 判定合法算式
 这个题目 Union-Find 算法应用起来，就十分优美了
题目描述：给你一个数组 equations, 装着若干字符串表示的算式。每个算式equations[i]
         长度都是4，而且只有这两种情况：a==b 或者 a!=b,其中 a,b可以是任意小写字母。
         写一个算法，如果equations中 所有算式都不会互相冲突，返回true, 否则返回false.
2.1 采用Union-Find算法的核心思想：
	将equations中的算式根据 == 和 != 分成两部分，先处理 == 算式，使得他们通过相等关系各自勾结成门派；
	然后处理!= 算式，检查不等关系是否破坏了相等的连通性。
代码如下：
boolean equationsPossible(String[] equations) {
	// 26 个 英文字母
	UF uf = new UF(26);
	// 先让相等的字母 形成连通分量
	for(String eq : equations) {
		if(eq.charAt(1) == '=') {
			char x = eq.charAt(0);
			char y = eq.charAt(3);
			uf.union(x - 'a', y - 'a');
		}
	}
	// 检查不等关系是否 打破相等关系的连通性
	for(String eq : equations) {
		if(eq.charAt(1) == '!') {
			char x = eq.charAt(0);
			char y = eq.charAt(3);
			// 如果相等关系成立，就是逻辑冲突
			if(uf.connected(x - 'a', y - 'a')) {
				return false;
			}
		}
	}
	return true;
} 

3.总结一下：
3.1 使用Union-Find算法，主要是如何把原问题转化成图的动态连通性问题，对于算法合法性问题，可以直接利用等价关系，
    对于棋盘包围问题，则是利用一个虚拟节点，营造出动态连通特性。
3.2 另外，将二维数组映射到一维数组，利用方向数组 d 来简化代码量，都是在写算法时 常用的一些小技巧。


// Part2.4 手把手设计数据结构
// 题目：2.4.3 LRU算法(Least Recently Used) 
// leetcode 146 
// 题目：
0. 运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。
当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
 

进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？

// 以Java代码为例，逐步去实现，主要是哈希双向链表
1.1 实现一个双链表节点
class Node {
	public int key, val;
	public Node next, prev;
	public Node(int k, int v) {
		this.key = key;
		this.val = val;
	}
}  

1.2 依靠Node类型，构建一个双链表，实现几个LRU算法必须的API：
class DoubleList {
	// 头尾虚节点
	private Node head, tail;
	// 李阿巴巴元素数
	private int size;

	public DoubleList() {
		// 初始化双向链表数据
		head = new Node(0, 0);
		tail = new Node(0, 0);
		head.next = tail;
		tail.prev = head;
		size = 0;
	}

	// 在链表尾部添加节点 x, 时间O(1)
	public void addLast(Node x) {
		x.prev = tail.prev;
		x.next = tail;
		tail.prev.next = x;
		tail.prev = x;
		size++；
	}

	// 删除链表中的 x 节点(x 一定存在)
	// 由于是双链表且给的目标 Node 节点，时间O(1)
	public void remove(Node x) {
		x.prev.next = x.next;
		x.next.prev = x.prev;
		size--;
	}

	// 删除链表中第一个节点，并返回该节点，时间O(1)
	public Node removeFirst() {
		if(head.next == tail) {
			return null;
		}
		Node first = head.next;
		remove(first);
		return first;
	}

	// 返回链表的长度，O(1)
	public int size() {
		return size;
	}
};

1.3 现在实现LRU算法，把它和哈希表结合起来，先搭出框架
class LRUCache {
	// key -> Node(key, val)
	private HashMap<Integer, Node> map;
	// Node(k1, v1) <-> Node(k2, v2)....
	private DoubleList cache;
	// 最大容量
	private int cap;

	public LRUCache(int capacity) {
		this.cap = capacity;
		map = new HashMap<>();
		cache = new DoubleList();
	} 
};
分析：先不慌去实现 get 和 put方法，由于我们同时摇维护一个双链表cache和一个哈希表map，
     很容易遗留掉一些操作。
解决方法是：在两种数据结构之上，提供一层抽象API

class LRUCache {
	// key -> Node(key, val)
	private HashMap<Integer, Node> map;
	// Node(k1, v1) <-> Node(k2, v2)....
	private DoubleList cache;
	// 最大容量
	private int cap;

	public LRUCache(int capacity) {
		this.cap = capacity;
		map = new HashMap<>();
		cache = new DoubleList();
	} 

	// 以下为抽象API
	/* 将某个key 提升为最近使用的 */
	private void makeRecently(int key) {
		Node x = map.get(key);
		// 先从链表中删除 这个节点
		cache.remove(x);
		// 重新插入到队尾
		cache.addLast(x);
	}

	/* 添加最近使用的元素 */
	private void addRecently(int key, int val) {
		Node x = new Node(key, val);
		// 链表尾部 就是最近使用的元素
		cache.addLast(x);
		// 别忘了 在map中 添加 key 的映射
		map.put(key, x);
	}

	/* 删除某一个key */
	private void deleteKey(int key) {
		Node x = map.get(key);
		// 从链表中删除
		cache.remove(x);
		// 从map中 删除
		map.remove(key);
	}

	/* 删除最久未使用的元素 */
	private void removeLeastRecently() {
		// 链表头部的第一个元素 就是最久未使用的
		Node deleteNode = cache.removeFirst();
		// 同时 别忘了 从map 中 删除它的 key
		int deleteKey = deleteNode.key;
		map.remove(deleteKey);
	}

	// 以下是get 和 put方法
	public int get(key) {
		if(!map.containsKey(key)) {
			return -1;
		}
		// 将数据提升为 最近使用的
		makeRecently(key);
		reutrn map.get(key).val;
	}

	public void put(int key, int val) {
		if(map.containsKey(key)) {
			// 删除旧的数据
			deleteKey(key);
			// 新插入的数据 为最近使用的数据
			addRecently(key, val);
			return;
		}
		if(cap == cache.size()) {
			// 删除最久未使用的 元素
			removeLeastRecently();
		}

		// 添加为最近使用的元素
		addRecently(key, val);
	}
};
// 至此，LRU算法的原理 实现了
1.4 以下是使用Java 内置类型LinkedHashMap 来实现LRU算法，逻辑和之前完全一致
class LRUCache {
	int cap;
	LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();

	public LRUCache(int capacity) {
		this.cap = capacity;
	}

	private void makeRecently(int key) {
		int val = cache.get(key);
		// 删除 key，重新插入到队尾
		cache.remove(key);
		cache.put(key, val);
	}

	public int get(int key) {
		if(!cache.containsKey(key)) {
			return -1;
		}
		// 将 key 变为 最近使用
		makeRecently(key);
		return cache.get(key);
	}

	public void put(int key, int val) {
		if(cache.containsKey(key)) {
			// 修改 key 的值
			cache.put(key, val);
			makeRecently(key);
			return;
		}

		if(cache.size() >= this.cap) {
			// 链表头部 就是最久未使用的key
			int oldestKey cache.keySet().iterator.next();
			cache.remove(oldestKey);
		}
		// 将新的 key 添加到 链表尾部
	} 
};

1.5 重要，c++版本实现，简单精妙，一定要会
// 一定要会，一定要会，一定要会
class LRUCache {
public:
	int max;
	unordered_map<int, int> hashmap;
	list<int> keylist;
	LRUCache(int capacity) {
		max = capacity;
	}

	void adjustKey(int key) {
		auto it = find(keylist.begin(). keylist.end(), key);
		keylist.erase(it);
		keylist.push_back(key);	
	}

	int get(int key) {
		if (hashmap.find(key) == hashmap.end()) {
			return -1;
		}
		adjustKey(key);
		return hashmap[key];
	}

	void put(int key, int value) {
		if(hashmap.find(key) != hashmap.end()) {
			hashmap[key] = value;
			adjustKey(key);
		} else {
			hashmap[key] == value;
			keylist.push_back(key);
			if(keylist.size() > max) {
				hashmap.erase(keylist.front());
				keylist.pop_front();
			}
		}
	}

};


// Part2.4 手把手设计数据结构
// 题目：2.4.4 LFU算法  leetcode 460 
   注：LFU算法 相当于 淘汰访问频次最低的数据，如果访问频次最低的数据有多条，需要淘汰最旧的数据。
题目要求 写一个类，接受一个capacity参数，实现get和put方法
class LFUCache {
	// 构造容量为 capacity 的缓存
	public LFUCache(int capacity) {}
	// 在缓存中 查询 key
	public int get(int key) {}
	// 将key 和 val 存入 缓存
	public void put(int key, int val) {}
};

0.1 先写出LFU算法的基本数据结构
class LFUCache {
	// 从key 到 val 的映射，后文称为 KV 表
	HashMap<Integer, Integer> keyToVal;
	// key 到 freq 的映射，后文称为 KF 表
	HashMap<Integer, Integer> keyToFreq;
	// freq 到 key 列表的映射，后文称为 FK 表
	HashMap<Integer, LinkedHashSet<Integer>> freqToKeys;
	// 记录最小的 频次
	int minFreq;
	// 记录 LFU 缓存的最大容量
	int cap;

	public LFUCache(int capacity) {
		keyToVal  = new HashMap<>();
		keyToFreq = new HashMap<>();
		freqToKeys = new HashMap<>();
		this.cap = capacity;
		this.minFreq = 0;
	}

	public int get(int key) {
		if(!keyToVal.containsKey(key)) {
			return -1;
		}
		// 增加 key 对应的 freq
		increaseFreq(key);
		return keyToVal.get(key);
	}

	public void put(int key, int val) {
		if(this.cap <= 0) return;

		/* 若 key 已存在，修改对应的 val 即可 */
		if(keyToVal.containsKey(key)) {
			keyToVal.put(key, val);
			// key对应的 freq 加1
			increaseFreq(key);
			return;
		}

		/* key 不存在，需要插入 */
		/* 容量已满的话，需要淘汰一个 freq 最小的 key */
		if(this.cap <= keyToVal.size()) {
			removeMinFreqKey();
		}

		/* 插入 key 和 val, 对应的 freq 为 1 */
		// 插入 KV 表
		keyToVal.put(key, val);
		// 插入 KF 表
		keyToFreq.put(key, 1);
		// 插入 FK 表
		freqToKeys.putIfAbsent(1, new LinkedHashSet<>());
		freqToKeys.get(1).add(key);
		// 插入新 key 后最小的 freq 肯定是 1
		this.minFreq = 1;
	}

	// 以下是核心代码
	private void removeMinFreqKey() {
		// freq 最小的 key 列表
		LinkedHashSet<Integer> keyList = freqToKeys.get(this.minFreq);
		// 其中最先被插入的那个 key 就是被淘汰的 key
		int deleteKey = keyList.iterator().next();
		/* 更新 FK 表 */
		keyList.remove(deleteKey);
		if(keyList.isEmpty()) {
			freqToKeys.remove(this.minFreq);
		}
		/* 更新 KV 表 */
		keyToVal.remove(deleteKey);
		/* 更新 KF 表 */
		keyToFreq.remove(deleteKey);
	}

	private void increaseFreq(int key) {
		int freq = keyToFreq.get(key);
		/* 更新 KF 表 */
		keyToFreq.put(key, freq + 1);
		/* 更新 FK 表 */
		// 将 key 从 freq 对应的列表中删除
		freqToKeys.get(freq).remove(key);
		// 将 key 加入 freq + 1 对应的列表中
		freqToKeys.putIfAbsent(freq + 1, new LinkedHashSet<>());
		freqToKeys.get(freq + 1).add(key);

		// 如果 freq 对应的列表空了，移除这个 freq
		if(freqToKeys.get(freq).isEmpty()) {
			freqToKeys.remove(freq);
			// 如果 这个 freq 恰好 是 minFreq, 更新 minFreq
			if(freq == this.minFreq) {
				this.minFreq++;
			}
		}
	}

};



// Part2.4 手把手设计数据结构
// 题目：2.4.5 数据流的中位数 //leetcode 295
// 中位数 是有序列表中间的数，如果长度是偶数，中位数则是中间两个数的平均值。
// 设计一个支持以下两种操作的数据结构：
    void addNum(int num) -从数据流中添加一个整数到数据结构中;
    double findMedian()  -返回目前所有元素的中位数;


1.0 Java版本做法
设计这样一个类：
class MedianFinder {
	// 添加一个数字
	public void addNum(int num) {}
	// 计算当前添加的所有数字的中位数
	public double findMedian() {}
};


class MedianFinder {
	private PriorityQueue<Integer> large;
	private PriorityQueue<Integer> small;

	public MedianFinder() {
		// 小顶堆
		large = new PriorityQueue<>();
		// 大顶堆
		small = new PriorityQueue<>((a, b) -> {
			return b - a;
		});
	}

	public double findMedian() {
		// 如果元素不一样多，多的那个元素的堆顶元素就是中位数
		if(large.size() < small.size()) {
			return small.peek();
		} else if (large.size() > small.size()) {
			return large.peek();
		}
		// 如果元素一样多，两个堆 堆顶元素的平均数是中位数
		return (large.peek() + small.peek()) / 2.0;
	}

	public void addNum(int num) {
		if(small.size() >= large.size()) {
			small.offer(num);
			large.offer(small.poll());
		} else {
			large.offer(num);
			small.offer(large.poll());
		}
	}
};

2.1 C++ 版本，简单排序
class MedianFinder {
    vector<double> store;

public:
    // Adds a number into the data structure.
    void addNum(int num) {
        store.push_back(num);
    }

    // Returns the median of current data stream
    double findMedian() {
        sort(store.begin(), store.end());

        int n = store.size();
        return (n & 1 ? store[n / 2] : (store[n / 2 - 1] + store[n / 2]) * 0.5);
    }
};

2.2 c++版本，插入排序
class MedianFinder {
	vector<int> store;
public:
	void addNum(int num) {
		if (store.empty()) {
			store.push_back(num);
		} else {
			store.insert(lower_bound(store.begin(), store.end(), num), num);
			// 解释下
			// lower_bound,指返回第一个大于或等于num的迭代器，并返回这个元素对应的迭代器
			// insert, 表示在指定的位置(对应的迭代器)之前，插入元素num，并返回指向这个元素的迭代器。
		}
	}

	double findMedian() {
        sort(store.begin(), store.end());

        int n = store.size();
        return (n & 1 ? store[n / 2] : (store[n / 2 - 1] + store[n / 2]) * 0.5);
    }
};

2.3 c++版本，两个堆实现，大顶堆和小顶堆
/*
 优先队列声明：
 priority_queue<type, container, function>
 注：其中第一个参数不可以忽略，后面两个忽略
 type: 数据类型
 container: 实现优先队列的底层容器，要求必须是以数组形式实现的容器。
 function: 元素之间的比较方式
*/
priority_queue<int> q; //定义一个优先队列，按照元素从大到小的顺序出队
// 等同于
priority_queue<int, vector<int>, less<int> >q;
// 另外一种按元素 从 小到大顺序出队
priority_queue<int, vector<int>, greater<int> >q;

class MedianFinder {
	priority_queue<int> small; // 大顶堆
	priority_queue<int, vector<int>, greater<int>> large; // 小顶堆
public:
	void addNum(int num) {
		if(small.size() >= large.size()) {
			small.push(num);
			large.push(small.top());
			small.pop();
		} else {
			large.push(num);
			small.push(large.top());
			large.pop();
		}
	}
	double findMedian() {
		if(large.size() < small.size()) {
			return small.top();
		} else if (large.size() > small.size()) {
			return large.top();
		}

		return (large.top() + small.top()) * 0.5;
	}
};


// Part2.4 手把手设计数据结构
// 题目：2.4.6 设计朋友圈时间线功能 

class Twitter {
	private static int timestamp = 0;
	private static class Tweet {
		private int id;
		private int time;
		private Tweet next;

		// 需要传入推文内容(id) 和 发文时间
		public Tweet(int id, int time) {
			this.id = id;
			this.time = time;
			this.next = null;
		}
	};
	private static class User {
		private int id;
		public Set<Integer> followed;
		// 用户发表的推文 链表头结点
		public Tweet head;

		public User (int userId) {
			followed = new HashSet<>();
			this.id = userId;
			this.head = null;
			// 关注一下自己
			follow(id);
		}
		public void follow(int userId) {
			followed.add(userId);
		}

		public void unfollow(int userId) {
			if(userId != this.id) {
				followed.remove(userId);
			}
		}

		public void post(int tweetId) {
			Tweet  twt = new Tweet(tweet, timestamp);
			timestamp++;
			// 将新建的推文插入到链表头
			// 越靠前的推文 time 值越大
			twt.next = head;
			head = twt;
		}
	};

	// 我们需要一个映射将 userId 和 User 对象对应起来
	private HashMap<Integer, User> userMap = new HashMap<>();


    /** Initialize your data structure here. */
    public Twitter() {

    }
    
    /** Compose a new tweet. */
    /* user 发表一条 tweet 动态 */
    public void postTweet(int userId, int tweetId) {
    	// 若 userId 不存在，则新建
    	if(!userMap.containsKey(userId)) {
    		userMap.put(userId, new User(userId));
    	}
    	User u = userMap.get(userId);
    	u.post(tweetId);
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
    	List<Integer> res = new ArrayList<>();
    	if(!userMap.containsKey(userId)) return res;
    	// 关注列表的 用户id
    	Set<Integer> users = userMap.get(userId).followed;
    	// 自动通过 time 属性 从大到小排序，容量为 users 的大小
    	PriorityQueue<Tweet> pq = new PriorityQueue<>(users.size(), (a, b)->(b.time - a.time));

    	// 首先将所有的链表 头节点 插入优先级队列
    	for(int id : users) {
    		Tweet twt = userMap.get(id).head;
    		if(twt == null) continue;
    		pq.add(twt);
    	}

    	while(!pq.isEmpty()) {
    		// 最多返回10条就够了
    		if(res.size() == 10) break;
    		// 弹出 time值 最大的(最近发表的)
    		Tweet twt = pq.poll();
    		res.add(twt.id);
    		// 将下一篇 Tweet 插入进行排序
    		if(twt.next != null) {
    			pq.add(twt.next);
    		}
    	}
    	return res;

    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    /* follower 关注 followee */
    public void follow(int followerId, int followeeId) {
    	// 若 follower 不存在，则新建
    	if(!userMap.containsKey(followerId)) {
    		User u = new User(followerId);
    		userMap.put(followerId, u);
    	}
    	// 若 followee 不存在，则新建
    	if(!userMap.containsKey(followeeId)) {
    		User u = new User(followeeId);
    		userMap.put(followeeId, u);
    	}
    	userMap.get(followerId).follow(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    /* follower 取关 followee, 如果ID 不存在，则什么都不做 */
    public void unfollow(int followerId, int followeeId) {
    	if(userMap.containsKey(followerId)) {
    		User flwer = userMap.get(followerId);
    		flwer.unfollow(followeeId);
    	}
    }

};

// Part2.4 手把手设计数据结构
// 题目：2.4.7 单调栈解决三道算法题
0.1 所谓单调栈，只是利用了一些巧妙的逻辑，使得每次新元素入栈后，站内的元素都保持有序(单调递增或单调递减)
1.题目1 leetcode 496,  下一个更大元素
  给你一个数组，返回一个等长的数组，对应的索引存储着下一个更大元素，如果没有更大的元素，就存1.
  函数签名：
vector<int> nextGreaterElement(vector<int>& nums);
比如：输入一个数组 nums = [2, 1, 2, 4, 3]
        返回数组 res = [4, 2, 4, -1, -1]

1.1 分析与思考：把数组的元素想象成并列站立的人，元素大小想象成人的身高，这些人面对你站成一列，如何求元素[2]的Next Greater Number呢？
vector<int> nextGreaterElement(vector<int>& nums) {
	vector<int> res(nums.size()); //存放答案的数组
	stack<int> s;
	// 倒着往栈里放
	for(int i = nums.size() - 1; i >= 0; i--) {
		// 判断个子高矮
		while(!s.empty() && nums[i] >= s.top()) {
			s.pop();
		}
		// nums[i] 身后的 next greater number
		res[i] = s.empty() ? -1 : s.top();
		// 
		s.push(nums[i]);
	}
	return res;
}


vector<int> nextGreaterElement2(vector<int>& nums) {
	vector<int> res(nums.size());
	stack<int> sta;
	for(int i = nums.size() - 1; i >= 0; i--) {

		while(!sta.empty() && nums[i] >= sta.top()) {
			sta.pop();
		}
		if(sta.empty()) {
			res[i] = -1;
		} else {
			res[i] = sta.top();
		}
		sta.push(nums[i]);

	}
}

1.2 这就是单调队列 解决问题的模板。for循环要从后往前扫描，因为我们借助的是栈的结构，倒着入栈。
    while是把 两个[个子高]元素之间的元素排除，因为他们存在没有意义，前面挡着个更高元素，所以
    他们不可能被作为后续进来的元素的 Next Greater Number。

时间复杂度，尽管for和while循环，但是复杂度为O(N);

2.0 题目 leetcode 一月有多少天？
    给你一个数组T，这个数组存放的是近几天的天气温度，你返回一个等长的数组，
    计算：对于每一天，你还要至少等多少天才能等到一个更暖和的气温；如果等不到那一天，填0.
函数签名：
vector<int> dailyTemperatures(vector<int>& T);
比如 T = [73, 74, 75, 71, 69, 76]
返回     [1,   1,  3, 2,   1,  0]
vector<int> dailyTemperatures(vector<int>& T) {
	vector<int> res(T.size());
	// 这里放元素索引，而不是元素
	stack<int> s;
	/* 单调栈模板 */
	for(int i = T.size() - 1; i >= 0; i--) {
		while(!s.empty() && T[s.top() <= T[i]]) {
			s.pop();
		}
		// 得到索引间距
		res[i] = s.empty() ? 0 : (s.top() - i);
		// 将索引入栈，而不是元素
		s。push(i);
	}
	return res;
}

3.0 如何处理环形数组
// leetcode 503
// 比如 输入[2, 1, 2, 4, 3]
//     输出[4, 2, 4, -1, 4]
vector<int> nextGreaterElements(vector<int>& nums) {
	int n = nums.size();
	vector<int> res(n);
	stack<int> s;
	// 假装这个数组长度翻倍了
	for(int i = 2 * n - 1; i >= 0; i--) {
		// 索引要求模，其它和模板一样
		while(!s.empty() && s.top() <= nums[i % n]) {
			s.pop();
		}
		res[i % n] == s.empty() ? -1 : s.top();
		s.push(nums[i % n]);
	}
	return res;
}


// Part2.4 手把手设计数据结构
// 题目：2.4.8 单调队列系列，数据结构解决滑动窗口问题
0.1 单调队列 就是一个队列，巧妙的是，队列中的元素，全都是单调递增(或递减)的。 
// leetcode 239
1.优先队列方法
分析：初始时，我们将数组nums的前k个元素，放入优先队列中，每当向右移动窗口时，我们
就可以把一个新的元素放入优先队列中，此时堆顶元素就是堆中所有元素的最大值。
然而这个最大值可能并不在滑动窗口中，在这种情况下，这个数在数组nums中的位置 出现在滑动窗口左边界左侧。
为了方便 判断堆顶元素与滑动窗口的位置关系，可以在优先队列中存储二元组(num, index), 表示元素num在数组中的下标 index;

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> ans;
        priority_queue<pair<int, int>> q;
        for(int i = 0; i < k; i++) {
            q.push(pair(nums[i], i));
            //q.emplace(nums[i], i);
        }
        ans.push_back(q.top().first);
        for(int i = k; i < n; i++) {
            q.push(pair(nums[i], i)); 
            //q.emplace(nums[i], i);

            while(q.top().second <= i - k) {
                q.pop();
            }
            ans.push_back(q.top().first);
        }
        return ans;
    }


2.双端队列
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	int n = nums.size();
	deque<int> q;
	for(int i = 0; i < k; ++i) {
		while(!q.empty() && nums[i] >= nums[q.back()]) {
			q.pop_back();
		}
		q.push_back(i);
	}

	vector<int> ans = {nums[q.front()]};
	for(int i = k; i < n; ++i) {
		while(!q.empty() && nums[i] >= nums[q.back()]) {
			q.pop_back();
		}
		q.push_back(i);
		while(q.front() <= i - k) {
			q.pop_front();
		}
		ans.push_back(nums[q.front()]);
	}
	return ans;
}


// Part2.4 手把手设计数据结构
// 题目：2.4.9 二叉堆，实现优先级队列
0.1 二叉堆，主要操作为两个，sink(下沉)和swim(上浮)，用以维护二叉堆的性质。
    其应用主要两个：一种是堆排序，另外一种是很有用的数据结构 [优先级队列]
0.2 二叉堆与二叉树什么关系？
    二叉堆其实就是一种特殊的二叉树(完全二叉树)，只不过存储在数组里。
    一般的链表二叉树，我们操作节点的指针，而在数组里，我们把数组索引作为指针。

// 父节点的索引
int parent(int root) {
	return root / 2;
}
// 左孩子的索引
int left(int root) {
	return root * 2;
}

// 右孩子的索引
int right(int root) {
	return root * 2 + 1;
}

0.3 二叉堆还分为最大堆和最小堆，
    最大堆的性质是：每个节点都大于等于它的两个子节点
    最小堆的性质是：每个节点都小于等于它的子节点

1.下面市县一个简化的优先队列。采用了Java的泛型，Key可以是一种可比较大小的数据类型
   你可以认为它是int, char等。

public class MaxPQ <Key extends Comparable<Key>> {
	// 存储元素的数组
	private Key[] pq;
	// 当前 Priority Queue 中的元素个数
	private int N = 0;

	public MaxPQ(int cap) {
		// 索引 0 不用， 所以多分配一个空间
		pq = (Key[] new Comparable[cap + 1]);
	}
	/* 返回当前队列中的最大元素 */
	public key max() {
		return pq[1];
	}
	/* 交换数组的两个元素 */
	private void exch(int i, int j) {
		Key tmp = pq[i];
		pq[i] = pq[j];
		pq[j] = tmp;
	}
	/* pq[i] 是否比pq[j] 小？ */
	private boolean less(int i, int j) {
		return pq[i].compareTo(pq[j]) < 0;
	}

	/* 实现swim 上浮方法 */
	private void swim(int k) {
		// 如果浮到堆顶，就不能再上浮了
		while(k > 1 && less(parent[k], k)) {
			// 如果第 k 个元素 比上层大
			// 将 k 换上去
			exch(parent[k], k);
			k = parent(k);
		}
	}
	/* 实现下沉的方法 */
	private void sink(int k) {
		// 如果沉到堆底，就沉不下去了
		while(left(k) <= N) {
			// 先假设左边节点比较大
			int order = left(k);
			// 如果右边节点存在，比一下大小
			if(right(k) <= N && less(order, right(k))) {
				order = right(k);
			}
			// 节点 k 比俩孩子都大，就不必下沉了
			if(less(order, k)) break;
			// 否则，不符合最大堆的结构，下沉k节点
			exch(k, order);
			k = order;
		}
	}

	/* 以下实现优先队列 */
	public void insert(Key e) {
		N++;
		// 先把新元素 加到最后
		pq[N] = e;
		// 然后让它上浮到 正确的位置
		swim(N);
	}

	public Key delMax() {
		// 最大堆的堆顶 就是最大元素
		Key max = pq[1];
		// 把这个最大元素换到 最后，删除之
		exch(1, N);
		pq[N] = null;
		N--;
		// 让 pq[1] 下沉到正确位置
		sink(1);
		return max;
	}

};



// Part2.4 手把手设计数据结构
// 题目：2.4.10 栈实现队列，队列实现栈
1.用栈实现队列
class MyQueue {
	private Stack<Integer> s1, s2;
	public MyQueue() {
		s1 = new Stack<>();
		s2 = new Stack<>();
	}

	/* 添加元素到队尾 */
	public void push(int x) {
		s1.push(x);
	}

	/* 返回队头元素 */
	public int peek() {
		if(s2.isEmpty()) {
			// 把s1 的元素 压入 s2
			while(!s1.isEmpty()) {
				s2.push(s1.pop());
			}
		}
		return s2.peek();
	}

	/* 删除队头的元素 并返回 */
	public int pop() {
		// 先调用 peek 保证 s2 非空
		peek();
		return s2.pop();
	}

	/* 判断队列是否为空 */
	public boolean empty() {
		return s1.isEmpty() && s2.isEmpty();
	}
};

2.用队列实现栈
class MyStack {
	Queue<Integer> q = new LinkedList<>();
	int top_elem = 0;
	/* 添加元素到栈顶 */
	public void push(int x) {
		// x 是队列的队尾，是栈的栈顶
		q.offer(x);
		top_elem = x;
	}
	/* 返回栈顶元素 */
	public int top() {
		return top_elem;
	}

	/* 删除栈顶的元素 并返回 */
	public int pop() {
		int size = q.size();
		while(size > 1) {
			q.offer(q.poll());
			size--;
		}
		// 之前的队尾元素 已经到了队头
		return q.poll();
	}
	// 但是这里的问题是，原来的队尾元素被提前到队头，并删除了，但是top_elem变量没有更新
	// 需要修改一下
	public int pop() {
		int size = q.size();
		// 留下队尾两个元素
		while(size > 2) {
			q.offer(q.poll());
			size--;
		}
		// 记录新的队尾元素
		top_elem = q.peek();
		q.offer(q.poll());
		//删除之前的队尾元素
		return q.poll();
	}

	/* 判断栈 是否为空 */
	public boolean empty() {
		return q.isEmpty();
	}
};


// Part2.5 手把手刷数组题目
// 题目：2.5.1 如何运用二分查找算法
1.爱吃香蕉的珂珂
珂珂喜欢吃香蕉。这里有 N 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 H 小时后回来。
珂珂可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。
  如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  
珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。

示例1：
输入：piles = [3,6,7,11], H = 8
输出：4

示例2：
输入：piles = [30, 11, 23, 4, 20]

1.1 暴力解法：
    最小速度为1，最大速度为max(piles),遍历一下
int minEatingSpeed(vector<int>& piles, int H) {
	// piles 数组的最大值
	int max_val = 0;
	for (int n : piles) {
		max_val = max(n, max_val);
	}
	for (int speed = 1; speed < max_val; speed++) {
		// 以 speed 是否能在 H 小时内 吃完香蕉
		if(canFinish(piles, speed, H)) {
			return speed;
		}
	}
	return max_val;
}

// 时间复杂度 O(N)
bool canFinish(vector<int>& piles, int speed, int H) {
	int time = 0;
	for (int n : piles) {
		time += timeOf(n, speed);
	}
	return time <= H;
}

int timeOf(int n, int speed) {
	return (n / speed) + ((n % speed) > 0 ? 1 : 0);
}

以上方法，超时加越界

1.2 二分查找法，因为搜索区间是单调线性的

int minEatingSpeed(vector<int>& piles, int H) {
    int max_val = getMax(piles);
    int left = 1; 
    int right = max_val + 1;
    while(left < right) {
        // 防止溢出
        int mid = left + (right - left) / 2;
        if(canFinish(piles, mid, H)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

// 时间复杂度 O(N)
bool canFinish(vector<int>& piles, int speed, int H) {
    int time = 0;
    for (int n : piles) {
        time += timeOf(n, speed);
    }
    return time <= H;
}

int timeOf(int n, int speed) {
    return (n / speed) + ((n % speed > 0) ? 1 : 0);
}

int getMax(vector<int>& piles) {
    int max_val = 0;
    for(int n : piles) {
        max_val = max(n, max_val);
    }
    return max_val;
}


2. 在D天内 送达包裹的能力
传送带上的包裹必须在 D 天内从一个港口运送到另一个港口。

传送带上的第 i 个包裹的重量为 weights[i]。
每一天，我们都会按给出重量的顺序往传送带上装载包裹。
我们装载的重量不会超过船的最大运载重量。

返回能在 D 天内将传送带上的所有包裹送达的船的最低运载能力。


示例1：
输入：weights = [1,2,3,4,5,6,7,8,9,10], D = 5
输出：15
解释：
船舶最低载重 15 就能够在 5 天内送达所有包裹，如下所示：
第 1 天：1, 2, 3, 4, 5
第 2 天：6, 7
第 3 天：8
第 4 天：9
第 5 天：10

请注意，货物必须按照给定的顺序装运，
因此使用载重能力为 14 的船舶并将包装分成 (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) 是不允许的。 

2.1 分析，本题本质上 和 吃香蕉的问题是一样的，首先确定
    cap的最大值和最小值分别是：max(weights), sum(weights);
可以用 搜索左侧边界的二分查找算法 优化线性搜索
// 寻找左侧边界的二分查找
int shipWithinDays(vector<int>& weights, int D) {
	//载重可能的最小值
	int left = getMax(weights);
	// 载重可能的最大值 + 1
	int right = getSum(weights) + 1;
	while(left < right) {
		int mid = left + (right - left) / 2;
		if(canFinish(weights, D, mid)) {
			right = mid;
		} else {
			left = mid + 1;
		}
	}
	return left;
}
// 如果转载为cap, 能否在 D 天内运完货物？
bool canFinish(vector<int>& weights, int D, int cap) {
	int i = 0;
	for(int day = 0; day < D; day++) {
		int maxCap = cap;
		while((maxCap -= weights[i]) >= 0) {
			i++;
			if(i == weights.size()) {
				return true;
			}
		}
	}
	return false;
}
int getMax(vector<int>& weights) {
	int max_val = 0;
	for(int val : weights) {
		max_val = max(val, max_val);
	}
	return max_val;
}
int getSum(vector<int>& weights) {
	int sum = 0;
	for(int val : weights) {
		sum += val;
	}
	return sum;
}

3.思考：使用for循环 暴力解决问题，观察代码是否形式：
for(int i = 0; i < n; i++) {
	if(isOK(i)) {
		return answer;
	}
}

// Part2.5 手把手刷数组题目
// 题目：2.5.2 双指针技巧总结
0.1 双指针可以分为两类；
    一类是：快慢指针，主要解决链表中的问题，比如典型的判断链表中，是否包含环？
    一类是：左右指针，主要解决数组(或字符串)中的问题，比如二分查找。

1.快慢指针的常见算法，一般都初始化指向链表的头结点head, 前进时快指针fast在前，慢指针slow在后，
  巧妙的解决一些链表中的问题。

1.1 判断链表中是否含有环？
bool hasCycle(ListNode* head) {
	ListNode* fast, slow;
	fast = slow = head;

	while(fast != NULL && fast->next != NULL) {
		fast = fast->next->next;
		slow = slow->next;

		if(fast == slow)
			return true;
	}
	return false;
}

1.2 已知链表中含有环，返回这个环的起始位置
这个问题其实不难，有点类似脑筋急转弯
ListNode* detectCycle(ListNode* head) {
	ListNode* fast;
	ListNode* slow;
	fast = slow = head;
	while(fast != NULL && fast->next != NULL) {
		fast = fast->next->next;
		slow = slow->next;
		if(fast == slow) break;
	}
	// 以上代码 类似 hasCycle 函数
	slow = head;
	while(slow != fast) {
		slow = slow->next;
		fast = fast->next;
	}
	return slow;
}

1.3 寻找链表的中点
ListNode* slow;
ListNode* fast;
slow = fast = head;
while(fast != NULL && fast->next != NULL) {
	fast = fast->next->next;
	slow = slow->next;
}
// slow 就在中间位置
return slow;

1.4 寻找链表的倒数第 k 个元素 
    先让快指针走 k 步，然后快慢指针同步前进
ListNode* slow;
ListNode* fast;
slow = fast = head;
while(k-- > 0) {
	fast = fast->next;
}
while(fast != NULL) {
	slow = slow->next;
	fast = fast->next;
}
return slow;


2.0 左右指针的常用算法
    左右指针在数组中 实际是指两个索引值，一般初始化为left = 0, right = nums.size() - 1;

2.1 二分查找
int binarySearch(vector<int>& nums, int target) {
	int left = 0;
	int right = nums.size() - 1;
	while(left <= right) {
		int mid = left + (right - left) / 2;
		if(nums[mid] == target) {
			return mid;
		} else if(nums[mid] < target) {
			left = mid + 1;
		} else if (nums[mid] > target) {
			right = mid - 1;
		}
	}
	return -1;
}

2.2 两数之和
    即：给定一个按照升序排列的有序数组，找到两个数使得它们相加之和等于目标数
示例:
	输入：nums = [2, 7, 11, 15], target = 9
	return [1, 2]

vector<int> twoSum(vector<int>& nums, int target) {
	int left = 0;
	int right = nums.size() - 1;
	while(left < right) {
		int sum = nums[left] + nums[right];
		if(sum == target) {
			// 题目要求从1开始
			return {left + 1, right + 1};
		} else if (sum < target) {
			left++;
		} else if (sum > target) {
			right--;
		}
	}
	return {-1, -1};
} 


2.3 翻转数组
void reverse(vector<int>& nums) {
	int left = 0;
	int right = nums.size() - 1;
	while(left < right) {
		// swap(nums[left], nums[right])
		int temp = nums[left];
		nums[left] = nums[right];
		nums[right] = temp;
		left++;
		right--;
	}
}

2.4 滑动窗口算法
   这也许是双指针技巧的做高境界了，如果掌握了此算法，
   可以解决一大类 子 字符串匹配的问题，
   不过[滑动窗口] 算法比上述的这些算法 稍微复杂些
   具体内容：请参考滑动窗口算法模板



// Part2.5 手把手刷数组题目
// 题目：2.5.3 滑动窗口框架
0.1 该算法的大致逻辑如下：
int left = 0， right = 0;
while(right < s.size()) {
	// 增大窗口
	window.add(s[right]);
	right++;

	while(window needs shrink) {
		// 缩小窗口
		window.remove(s[left]);
		left++;
	}
}
注：这个算法的时间复杂度为O(N), 比一般的字符串暴力算法要高效的多。

0.2 注重细节，滑动窗口框架
/* 滑动窗口算法框架 */
void slidingWindow(string s, string t) {
	unordered_map<char, int> need, window;
	for(char ch : t) {
		need[c]++;
	}
	int left = 0, right = 0;
	int valid = 0;
	while(right < s.size()) {
		// c 是将移入窗口的字符
		char c = s[right];
		// 右移窗口
		right++;
		// 进行窗口内数据的 一系列更新
		...

		/* debug 输出的位置 */
		printf("window: [%d, %d]\n", left, right);
		/*******************/

		// 判断左窗口是否要收缩
		while(window needs shrink) {
			// d 是将移出 窗口的字符
			char d = s[left];
			// 左移窗口
			left++;
			// 进行窗口内数据的一系列更新
			...
		}
	}
}

1.最小覆盖子串
题目：给你一个字符串s、一个字符串t。返回s中涵盖t所有字符的最小子串。
     如果 s 中不存在 涵盖 t 所有字符的子串，则返回空字符串 “”
注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。
示例1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"

示例2：
输入： s = "a", t = "a"
输出： "a"

1.1 解法1 暴力解法
for(int i = 0; i < s.size(); i++) {
	for(int j = i + 1; j < s.size(); j++) {
		if(s[i:j] 包含 t 的所有字母)
			更新答案
	}
}

1.2 滑动窗口算法 
细节点：
a.使用 双指针的 左右指针技巧，初始化 left = right = 0，
       把索引左闭右开区间[left, right)称为一个窗口
b.我们先不断地增加right 指针扩大窗口[left, right)，直到窗口中的字符串符合要求(包含了T中的所有字符)

c.此时，停止增加right，转而不断增加left指针，缩小窗口[left, right),直到窗口中的字符串不再符合要求
  (不包含T中的所有字符了)。同时，每次增加left, 我们都要更新一轮结果。

d.重复第2步到第3步，直到right到达 字符串 s 的尽头。

string minWindow(string s, string t) {
	unordered_map<char, int> need, window;
	for(char ch : t) {
		need[ch]++;
	}

	int left = 0, right = 0;
	int valid = 0;
	// 记录最小覆盖子串的 起始索引及长度
	int start = 0, len = INT_MAX;
	while(right < s.size()) {
		// c 是将移入窗口的字符
		char c = s[right];
		// 右移窗口
		right++;
		// 进行窗口内数据的一系列更新
		if(need.count(c)) {
			window[c]++;
			if(window[c] == need[c]) {
				valid++;
			}
		}

		// 判断左侧窗口 是否要 收缩？
		while(valid == need.size()) {
			// 在这里更新 最小覆盖子串
			if(right - left < len) {
				start = left;
				len = right - left;
			}
			// d 是将移出窗口的字符
			char d = s[left];
			// 左移窗口
			left++;
			// 进行窗口内的一系列更新
			if(need.count(c)) {
				if(window[c] == need[c]) {
					valid--;
				}
				window[c]--;
			}
		}
	}
	// 返回最小覆盖子串
	return len == INT_MAX ? "" : s.substr(start, len);
}


2 字符串排列 // leetcode 567
题目：给定两个字符串s1 和 s2, 写一个函数来判断 s2 是否包含s1 的排列
换句话说，第一个字符串的排列之一 是第二个字符串的子串。

示例1：
输入：s1 = "ab", s2 = "eidbaooo"
输出：True 
解释：s2 包含 s1的全排列之一 ("ba")

分析：这种题目，是明显的滑动窗口算法，相当于给你一个S和T, 请问你S中
     是否存在一个子串，包含T中所有字符且不包含其他字符？

bool checkInclusion(string s, string t) {
	unordered_map<char, int> need, window;
	for(char ch : t) {
		need[ch]++;
	}

	int left = 0, right = 0;
	int valid = 0;
	while(right < s.size()) {
		char c = s[right];
		right++;
		// 进行窗口内数据的一系列更新
		if(need.count(c)) {
			window[c]++;
			if(window[c] == need[c]) {
				valid++;
			}
		}

		// 判断最侧窗口 是否要收缩
		while(right - left >= t.size()) {
			// 在这里判断 是否找到了合法的子串
			if(valid == need.size()) {
				return true;
			}
			char d = s[left];
			left++;
			// 进行窗口内数据的 一系列更新
			if(need.count(d)) {
				if(window[d] == need[d]) {
					valid--;
				}
				window[d]--;
			}
		}
	}
	// 未找到符合条件的子串
	return false;
}

3 找所有字母异位词 // leetcode 438
题目：给定一个字符串 s 和 一个非空字符串 p, 找到 s 中 所有是 p 的字母异位词的子串，
     返回这些子串的起始索引。
注：异位词是指，字母相同，但排列不同的字符串
相当于：输入一个串 S, 一个串 T, 找到 S 中 所有 T 的排列，返回它们的起始索引。

vector<int> findAnagrams(string s, string t) {
	unordered_map<char, int> need, window;
	for(char ch : t) {
		need[ch]++;
	}

	int left = 0, right = 0;
	int valid = 0;
	vector<int> res; // 记录结果
	while(right < s.size()) {
		char c = s[right];
		right++;
		// 进行窗口内的一系列更新
		if(need.count(c)) {
			window[c]++;
			if(window[c] == need[c]) {
				valid++;
			}
		}
		// 判断左侧窗口是否要 收缩？
		while(right - left >= t.size()) {
			// 当窗口符合条件时，把起始索引加入 res
			if(valid == need.size()) {
				res.push_back(left);
			}
			char d = s[left];
			left++;
			// 进行窗口内数据的一系列更新
			if(need.count(d)) {
				if(window[d] == need[d]) {
					valid--;
				}
				window[d]--;
			}
		}
	}
	return res;
} 


4 最长无重复子串 // leetcode 3
题目：给定一个字符串，请你找出其中不含有 重复字符串的 最长子串 的长度。

示例1：
输入：“abcabcbb”
输出：3

示例2：
输入：”bbbbb“
输出：1 

// 这个代码简答，不需要need和valid
int lengthOfLongestSubstring(string s) {
	unordered_map<char, int> window;

	int left = 0, right = 0;
	int res = 0; // 记录结果
	while(right < s.size()) {
		char c = s[right];
		right++;
		// 进行窗口内数据的一系列更新
		window[c]++;

		// 判断左侧窗口是否要收缩
		while(window[c] > 1) {
			char d = s[left];
			left++;
			// 进行窗口内数据的一系列更新
			window[d]--;
		}
		// 在这里更新答案
		res = max(res, right - left);
	}
	return res;
}

// Part2.5 手把手刷数组题目
// 题目：2.5.4 O(1)时间 查找/删除数组中任意元素
0. 技巧在于，如何结合哈希表和数组，使得数组的删除和查找操作的时间复杂度稳定在O(1)?
1. leetcode 380
   题目：常数时间插入、删除和获取随机元素。
   设计一个支持在平均时间复杂度O(1)下，执行以下操作的数据结构。
   1.insert(val): 当元素 val 不存在时，向集合中插入该项。
   2.remove(val): 元素 val 存在时，从集合中移出该项。
   3.getRandom(): 随机返回现有集合中的一项，每个元素应该有相同的概率被返回。

class RandomizedSet {
	/* 如果 val 不存在集合中，则插入并返回true, 否则直接返回false */
	public bool insert(int val) {}

	/* 如果 val 在集合中，则删除并返回true, 否则直接返回false */
	public bool remove(int val) {}

	/* 从集合中 等概率 地 获取一个元素 */
};

1.1 本题的难点在于：
	a.插入、删除，获取随机元素这三个操作的时间复杂度必须是O(1)
	b.getRandom 方法返回的元素 必须 等概率返回随机元素

1.2 分析：对于getRandom方法，如果想等概率在O(1)的时间取出元素，一定要满足：
    底层用数组实现，且数组必须是紧凑的。

1.3 但是，如果用数组存储元素的话，插入、删除的时间复杂度怎么可能是O(1)呢？

所以，如果我们想在O(1)的时间删除数组中的某一个元素val, 可以先把这个元素交换到数组的尾部，然后在pop掉；
交换两个元素 必须通过索引进行交换，那么需要一个哈希表 valToIndex 来记录每个元素值对应的索引。

class RandomizedSet {
public:
	// 存储元素的值
	vector<int> nums;
	// 记录每个元素对应在 nums 中的索引
	unordered_map<int, int> valToIndex;

	bool insert(int val) {
		// 若 val 已存在，不用再插入
		if(valToIndex.count(val)) {
			return false;
		}
		// 若 val 不存在，插入到 nums 尾部
		// 并记录 val 对应的索引值
		valToIndex[val] = nums.size();
		nums.push_back(val);
		return true;
	}

	bool remove(int val) {
		// 若 val 不存在，不用再删除
		if(!valToIndex.count(val)) {
			return false;
		}
		// 先拿到 val 的索引
		int index = valToIndex[val];
		// 将最后一个元素对应的索引 修改为 index
		valToIndex[nums.back()] = index;
		// 交换 val 和 最后一个元素
		swap(nums[index], nums.back());
		// 在数组中删除 元素 val
		nums.pop_back();
		// 删除元素 val 对应的索引
		valToIndex.erase(val);
		return true;
	}

	int getRandom() {
		// 随机获取 nums 中的一个元素
		return nums[rand() % nums.size()];
	}

};

2.避开黑名单的随机数
// leetcode 710 
题目描述：给你输入一个正整数N, 代表着左闭右开区间[0, N), 再给你输入一个数组blacklist,
        其中包含一些[黑名单数字]，且blacklist中的数字都是区间[0, N)中的数字。
先要求你，设计如下数据结构：
class Solution {
public:
	// 构造函数，输入参数
	Solution(int N, vector<int>& blacklist) {}

	// 在区间[0, N)中 等概率 随机选取一个元素 并返回
	// 这个元素不能是 blacklist 中 的元素
	int pick() {}
};

pick函数 会被多次调用，每次调用都要在区间[0, N) 中[等概率随机] 返回一个 不在 blacklist 中的整数
题目要求：在pick函数中 应该尽可能少调用随机数 生成函数 rand()

所以，不能采用以下做法;
int pick() {
	int res = rand() % N;
	while(res exists in blacklist) {
		// 重新随机一个结果
		res = rand() % N;
	}
	return res;
}

2.1 聪明的解法类似上一题，可以将区间[0, N) 看做一个数组，然后将blacklist中的元素，
    移到数组的最末尾，同时用一个哈希表进行映射;

2.2 以下是第一版代码(存在几处错误)
class Solution {
	int sz;
	unordered_map<int, int> mapping;

	Solution(int N, vector<int>& blacklist) {
		// 最终数组中的元素个数
		sz = N - blacklist.size();
		// 最后一个元素的索引
		int last = N - 1;
		// 将黑名单中的 索引换到 最后去
		for(int b : blacklist) {
			mapping[b] = last;
			last--;
		}
	}
};

int pick() {
	// 随机选取 一个索引
	int index = rand() % sz;
	// 这个索引命中了 黑名单
	// 需要 被映射到 其它位置
	if(mapping.count(index)) {
		return mapping[index];
	}
	// 若没命中 黑名单，则直接返回
	return index;
}

2.3 构造函数，还有两个问题，
   第一个问题，如下这段代码;
int last = N - 1;
// 将黑名单中的索引换到 最后去
for(int b : blacklist) {
	mapping[b] = last;
	last--;
}

我们将黑名单中的 b 映射到 last, 但是能确定 last 不在 blacklist中吗？
在怼mapping[b]赋值时，要保证last 一定不在 blacklist中，可以如下操作;
Solution(int N, vector<int>& blacklist) {
	sz = N - blacklist.size();
	// 先将所有黑名单数字 加入 map
	for(int b : blacklist) {
		// 这里赋值多少都可以
		// 目的仅仅是把 键存在 哈希表中
		// 方便快速判断 数字是否在黑名单内
		mapping[b] = 666;
	}
	int last = N - 1;
	for(int b : blacklist) {
		// 跳过所有 黑名单中的数字
		while(mapping.count(last)) {
			last--;
		}
		// 将黑名单中的索引 映射到 合法数字
		mapping[b] = last;
		last--;
	}
}

2.4 第二个问题，如果blacklist中的黑名单数字 本身就存在区间[sz, N)中， 
    那么就没必要在mapping中建立映射；
    所以整体的代码如下：
class Solution {
public:
	int sz;
	unordered_map<int, int> mapping;
	Solution(int N, vector<int>& blacklist) {
		sz = N - blacklist.size();
		for(int b : blacklist) {
			mapping[b] = 666;
		}
		int last = N - 1;
		for(int b : blacklist) {
			// 如果 b 已经在区间 [sz, N)
			// 可以直接忽略
			if(b > sz) {
				continue;
			}
			while(mapping.count(last)) {
				last--;
			}
			mapping[b] = last;
			last--;
		}
	}
};
pick 代码见上文。

// Part2.5 手把手刷数组题目
// 题目：2.5.5 数组去重最高境界，单调栈应用
// leetcode 316, 1081
1.题目：给你一个字符串 s, 请你去除字符串中重复的字母，使得每个字母只出现一次。
     需保证 返回结果的字典序最小(要求不能打乱其他字符的相对位置)
示例1：
输入：s = "bcabc"
输出："abc"

示例2：
输入：s = "cbacdcbc"
输出："acdb"

1.1 分析，题目要求3点
   要求一、要去重
   要求二、去除字符串中的字符顺序不能打乱 s 中字符出现的相对顺序
   要求三、在所有符合上一条要求的去重字符串中，字典序最小的作为最终结果。

先暂时忽略要求三，用栈来实现一下 要求一 和 要求 二 
String removeDumplicateLetters(String s) {
	// 存放去重的结果
	Stack<Character> stk = new Stack<>();
	// 布尔数组初始值为false, 记录栈中是否存在某个字符
	// 输入字符均为 ASCII 字符，所以大小为 256 够用了
	boolean[] inStack = new boolean[256];

	for(char c : s.toCharArray()) {
		// 如果字符 c 存在栈中，直接跳过
		if(inStack[c]) continue;
		// 若不存在，则插入栈顶并标记为存在
		stk.push(c);
		inStack[c] = true;
	}

	StringBuilder sb = new StringBuilder();
	while(!stk.empty()) {
		sb.append(stk.pop());
	}
	// 栈中元素插入顺序是反的，需要reverse 一下
	return sb.reverse().toString();
}

问题：如果当前字符'a'比自己钱的字符字典序小，就有可能需要把
     前面的字符pop出栈，让'a' 排在前面吧
String removeDumplicateLetters(String s) {
	Stack<Character> stk = new Stack<>();
	boolean[] inStack = new boolean[256];

	for(char c : s.toCharArray()) {
		if(inStack[c]) continue;
		// 插入之前，和之前的元素比较一下大小
		// 如果字典序比前面的小，pop前面的元素
		while(!stk.empty() && stk.peek() > c) {
			// 弹出栈顶元素，并把该元素标记为不在栈中
			inStack[stk.pop()] = false;
		}
		stk.push(c);
		inStack[c] = true;
	}
	StringBuilder sb = new StringBuilder();
	while(!stk.empty()) {
		sb.append(stk.pop());
	}
	return sb.reverse().toString();
}


问题：算法在stk.peek() > c 时，才会pop元素，应该分两种情况：
情况一、如果stk.peek()这个字符之后还会出现，那么可以把它pop出去
情况二、如果stk.peek()这个字符之后不会出现了，就不能把它pop出去
终版代码：
String removeDumplicateLetters(String s) {
	Stack<Character> stk = new Stack<>();
    // 维护一个计数器 记录字符中字符的数量
    // 因为输入为ASCII 字符，大小256 就够用了
    int[] count = new int[256];
    for(int i = 0; i < s.length(); i++) {
        count[s.charAt(i)]++;
    }

    boolean[] inStack = new boolean[256];
    for(char c : s.toCharArray()) {
        // 每遍历一个字符，都讲对应的计数减一
        count[c]--;

        if(inStack[c]) continue;

        while(!stk.isEmpty() && stk.peek() > c) {
            // 若之后不存在栈顶元素了，则停止pop
            if(count[stk.peek()] == 0) {
                break;
            }
            // 若之后还有，则可以pop
            inStack[stk.pop()] = false;
        }
        stk.push(c);
        inStack[c] = true;
    }
    StringBuilder sb = new StringBuilder();
    while(!stk.empty()) {
        sb.append(stk.pop());
    }
    return sb.reverse().toString();
    }
}

终极代码，C++实现 
string removeDuplicateLetters(string s) {
    stack<char> stk;
    vector<int> count(26, 0);
    for(auto ch : s) {
        count[ch - 'a']++;
    }
    vector<bool> inStack(26, false);
    for (char c : s) {
        count[c - 'a']--;
        
        if(inStack[c - 'a']) continue;

        while(!stk.empty() && stk.top() > c) {
            if(count[stk.top() - 'a'] == 0) {
                break;
            }

            inStack[stk.top() - 'a'] = false;
            stk.pop();
        }
        stk.push(c);
        inStack[c - 'a'] = true;
    }
    string res;
    while(!stk.empty()) {
        res.push_back(stk.top());
        stk.pop();
    }
    reverse(res.begin(), res.end());
    return res;
}

2. 总结，非常好的一道题
   算法时间、空间复杂度都为O(N),现在看下开头将的三个要求，是如何达成的？
2.1 要求一、通过inStack 这个布尔数组做到栈stk中不存在重复元素
2.2 要求二、顺序遍历字符串s, 通过 栈 这种顺序结构的push 和 pop操作记录 结果字符串
    保证了字符出现的顺序 和 s 中出现 的顺序一致
2.3 要求三、用类似单调栈思路，配合计数器count 不断pop掉不符合最小字典序的字符，
	保证了最终得到的结果字典序最小
	最后需要把栈中元素取出后 再反转一次才是最终结果。


// Part2.5 手把手刷数组题目
// 题目：2.5.6 双指针应用，秒杀四道数组、链表题目
1.删除排序数组中的重复项 leetcode 26
题目：
    给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
    不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成;
int removeDuplicates(vector<int>& nums) {
	if(nums.size() == 0) {
		return 0;
	}
	int slow = 0, fast = 0;
	while(fast < nums.size()) {
		if(nums[fast] != nums[slow]) {
			slow++;
			// 维护nums[0,...,slow] 不重复
			nums[slow] = nums[fast];
		}
		fast++;
	}
	// 数组长度为索引 + 1
	return slow + 1;
}

1.1 扩展一下，给你一个有序链表，如何去重呢？// leetcode 83
ListNode* deleteDuplicates(ListNode* head) {
	if(head == nullptr)
		return nullptr;
	ListNode* slow = head;
	ListNode* fast = head;

	while(fast != nullptr) {
		if(fast->val != slow->val) {
			// nums[slow] = nums[fast];
			slow->next = fast;
			slow = slow->next;
		}
		// fast++;
		fast = fast->next;
	}
	// 断开与后面重复元素的链接
	slow->next = nullptr;
	return head;
}

2.0 移出元素 // leetcode 27
题目：
    给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
	不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
	元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素

示例1：
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
     你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，
     而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案;

int removeElement(vector<int>& nums, int val) {
	int fast = 0, slow = 0;

	while(fast < nums.size()) {
		if(nums[fast] != val) {
			nums[slow] = nums[fast];
			slow++;
		}
		fast++;
	}
	return slow;
}

3.0 移动零 //leetcode 283
题目：
    给定一个数组nums, 编写一个函数将所有 0 移动到 数组的末尾，同时保持非零元素的相对位置
示例1：
输入：[0, 1, 0, 3, 12]
输出：[1, 3, 12, 0, 0]
注：必须在原数组上操作，不能拷贝额外的数组
   尽量减少操作次数

void moveElement(vector<int>& nums) {
	// 去除 nums 中的所有 0 
	// 返回去除 0 之后的数组长度
	int p = removeElement(nums, 0);
	// 将 p 之后的所有元素 赋值为 0
	for(; p < nums.size(); p++ {
		nums[p] = 0;
	}
}
// 见上文代码实现
int removeElement(vector<int>& nums, int val) {}


// Part2.5 手把手刷数组题目
// 题目：2.5.7 twoSum问题的 核心思想
0. 最基本形式：给你一个数组和一个整数 target, 
   可以保证数组中存在两个数的和为 target, 请你返回这两个数的索引

0.1 简单粗暴解法;
vector<int> twoSum(vector<int>& nums, int target) {
	for(int i = 0; i < nums.size(); i++) {
		for(int j = i + 1; j < nums.size(); j++) {
			if(nums[j] == target - nums[i])
				return {i, j};
		}
	}
	return {-1, -1};
}

0.2 采用哈希表 减少时间复杂度
vector<int> twoSum(vector<int>& nums, int target) {
	int n = nums.size();
	unordered_map<int, int> index;
	// 构造一个哈希表，元素映射到相应的索引
	for(int i = 0; i < n; i++) {
		index[nums[i]] = i;
	}
	for(int i = 0; i < n; i++) {
		int other = target - nums[i];
		// 如果 other 存在 且 不是 nums[i]本身
		if(index.count(other) && index[other] != i) {
			return {i, index[other]};
		}
	}
	return {-1, -1};
}

1. TwoSum II 题目要求设计一个类，拥有两个API
class TwoSum {
	// 向数据结构中添加一个数
	public void add(int number);
	// 寻找当前数据结构中是否存在两个数的和为 value
	public boolean find(int val);
};

1.1 采用哈希表辅助find方法：
class TwoSum{
	Map<Integer, Integer> freq = new HashMap<>();

	public void add(int number) {
		// 记录 number 出现的次数
		freq.put(number, freq.getOrDefault(number, 0) + 1); 
	}

	public boolean find(int val) {
		for(Integer key : freq.keySet()) {
			int other = value - key;
			// 情况1
			if(other == key && freq.get(key) > 1) {
				return true;
			} 
			// 情况2
			if(other != key && freq.containsKey(other)) {
				return true;
			}
		}
		return false;
	}
};

1.2 对于API设计，要考虑现实情况的。
    find()方法使用比较频繁，每次都是O(N)时间，太浪费时间了
    借助哈希集合 针对性的优化 find方法;
class TwoSum {
	Set<Integer> sum = new HashSet<>();
	List<Integer> nums = new ArrayList<>();

	public void add(int number) {
		// 记录所有可能组成的和
		for(int n : nums) {
			sum.add(n + number);
		}
		nums.add(number);
	}

	public boolean find(int val) {
		return sum.contains(value);
	}
};

1.3 总结，对于TwoSum问题，一个难点是 给的数组无序，
    对于一个无序的数组，似乎什么技巧都没有，只能暴力穷举所有可能
    一般情况下，先把数组排序，再考虑双指针技巧

 如果twosum中给的数组时有序的，应该如何编写算法呢？简单
 vector<int> twoSum(vector<int>& nums, int target) {
 	int left = 0;
 	int right = nums.size() - 1;
 	while(left < right) {
 		int sum = nums[left] + nums[right];
 		if(sum == target) {
 			return {left, right};
 		} else if (sum < target) {
 			left++;  // 让 sum 大一点
 		} else if (sum > target) {
 			right--;  // 让 sum 小一点
 		}
 	}
 	// 不存在这样两个数
 	return {-1, -1};
 }

