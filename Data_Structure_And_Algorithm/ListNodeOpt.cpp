/***
递归的思想相对迭代思想，稍微有点难以理解，
处理的技巧是：不要跳进递归，而是利用明确的定义来实现算法逻辑。
处理看起来比较困难的问题，可以尝试化整为零，把一些简单的解法进行修改，解决困难的问题。
值得一提的是，递归操作链表并不高效。
和迭代解法相比，虽然时间复杂度都是 O(N)，但是迭代解法的空间复杂度是 O(1)，而递归解法需要堆栈，空间复杂度是 O(N)。
所以递归操作链表可以作为对递归算法的练习或者拿去和小伙伴装逼，但是考虑效率的话还是使用迭代算法更好。
***/
//经典题型：反转链表，包括1.整体反转，2.前n个元素反转，3.[m,n]区间反转
//参考文章系列
//https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/2.2-shou-ba-shou-shua-lian-biao-ti-mu-xun-lian-di-gui-si-wei/di-gui-fan-zhuan-lian-biao-de-yi-bu-fen
//206. 反转链表
//反转一个单链表。
//方法：递归解法
ListNode* reverseList(ListNode* head) {
    if (!head || !head->next ) return head;
    ListNode* last = reverseList(head->next);
    head->next->next = head;
    head->next = nullptr;
    return last;
}
//92. 反转链表 II
//反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。

//反转链表前 N 个节点
ListNode* successer = nullptr; // 后驱节点
// 反转以 head为起点的 n 个节点，返回新的头节点
ListNode* reverseN(ListNode* head, int n) {
    if (n == 1) {
        // 记录第 n + 1 个节点
        successer = head->next;
        return head;
    }
    // 以 head.next 为起点，需要反转前 n - 1 个节点
    ListNode* last = reverseN(head->next, n - 1);
    
    head->next->next = head;
    // 让反转之后的 head 节点和后面的节点连起来
    head->next = successer;
    return last;
}

ListNode* reverseBetween(ListNode* head, int m, int n) {
    if (!head || !head->next) return head;
    // base case
    if (1 == m) {
        // 相当于反转前 n 个元素
        return reverseN(head, n);
    }
    // 前进到反转的起点，触发base case
    head->next = reverseBetween(head->next, m - 1, n - 1);
    return head;
}

//206. 反转链表
//反转一个单链表。
//方法：迭代解法
ListNode* reverseList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* curr = head;
    ListNode* next = head;
    ListNode* pre = nullptr;
    while(curr) {
        next = curr->next;
        curr->next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}
//92. 反转链表 II
//反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
// 以下是循环迭代解法，非常巧妙，非常经典，需要理解其精髓
ListNode* reverseBetween(ListNode* head, int m, int n) {
    if (!head || !head->next) return head;
    ListNode* dump = new ListNode(-1);
    ListNode* pre = dump;
    dump->next = head;

    // 首先找到pre节点
    for(int i = 0; i < m - 1; i++) {
        pre = pre->next;
    }
    // 定义循环迭代中 curr和next节点，非常重要
    ListNode* curr = pre->next;
    ListNode* next = curr;
    for (int i = m; i < n; i++) { //这里非常巧妙的完成循环迭代，你品，你细品
        next = curr->next;
        curr->next = next->next;
        next->next = pre->next;
        pre->next = next;
    }
    return dump->next;
}

//25. K 个一组翻转链表
/***
给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
k 是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

示例：
给你这个链表：1->2->3->4->5
当 k = 2 时，应当返回: 2->1->4->3->5
当 k = 3 时，应当返回: 3->2->1->4->5
***/
//方法1：迭代+递归
// 首先定义反转链表的操作
ListNode* reverse(ListNode* node) {
    if (!node) return node;
    ListNode* pre = nullptr;
    ListNode* curr = node;
    ListNode* next = node;
    while (curr != nullptr) {
        next = curr->next;
        curr->next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}
// 反转区间 [a, b) 的元素，注意是左闭右开 
ListNode* reverse(ListNode* a, ListNode* b) {
    ListNode* pre = nullptr;
    ListNode* curr = a;
    ListNode* next = a;
    while (curr != b) {
        next = curr->next;
        curr->next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}

ListNode* reverseKGroup2(ListNode* head, int k) {
    if (!head || !head->next) return head;
    // 区间 [a, b) 包含 k 个待反转元素
    ListNode* a;
    ListNode* b;
    a = b = head;
    for (int i = 0; i != k; i++) {
        // 不足 k 个，不需要反转，base case
        if (!b) return head;
        b = b->next;
    }
    // 反转前 k 个元素
    ListNode* newNode = reverse(a, b);
    // 递归反转后续链表并连接起来
    a->next = reverseKGroup2(b, k);
    return newNode;
}
方法2: 纯迭代做法
// 翻转 head 与tail 之间的node
pair<ListNode*, ListNode*> myReverse(ListNode* head, ListNode* tail) {
    // 先找到下游节点，为了让翻转的链表可以指向下游
    ListNode* pre = tail->next;
    ListNode* curr = head;
    ListNode* next = head;
    while(pre != tail) {
        next = curr->next;
        curr->next = pre;
        pre = curr;
        curr = next;
    }
    // 返回翻转后的头结点和尾结点
    return {tail, head};
}
// 方法2：以下是纯迭代方法，需要细品
ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* dump = new ListNode(0);
    dump->next = head;
    ListNode* pre = dump;
    while(head != nullptr) {
        ListNode* tail = pre;
        for (int i = 0; i < k; i++) {
            tail = tail->next;
            if(!tail) return dump->next;
        }
        // 定义next节点，主要是为了连接 后驱节点
        ListNode* next = tail->next;
        pair<ListNode*, ListNode*> result = myReverse(head, tail);
        head = result.first;
        tail = result.second;
        // 翻转之后，头结点需要设置
        pre->next = head;
        // 翻转之后，尾结点需要设置
        tail->next = next;
        // 更新pre节点 为tail
        pre = tail;
        // 更新head节点 为tail->next
        head = tail->next;
    }
    return dump->next;
}

// 234. 回文链表
/*请判断一个链表是否为回文链表。
示例 1:
输入: 1->2
输出: false
示例 2:
输入: 1->2->2->1
输出: true
***/
ListNode* left;
bool traverse(ListNode* right) {
    if (right == nullptr) return true;
    bool res = traverse(right->next);
    // 后序遍历代码
    res = res && (right->val == left->val);
    left = left->next;
    return res;
}
// 该方法的核心逻辑是：实际上就是把链表节点放入一个栈，
// 然后再拿出来，这时候元素顺序就是反的，只不过我们利用的是递归函数的堆栈而已。
// 方法的时间和空间复杂度都为O(N)
bool isPalindrome2(ListNode* head) {
    left = head;
    return traverse(head);
}
// 方法2：通过双指针技巧
bool isPalindrome(ListNode* head) {
    if (!head || !head->next) return true;
    ListNode* slow, *fast;
    slow = fast = head;
    while(fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    // 如果链表个数为奇数，则fast指向最后一个元素
    // 如果链表个数为偶数，则fast指向NULL
    if (fast != nullptr) {
        slow = slow->next;
    }
    // 以下从slow 开始反转后面的链表，就可以比较回文串了
    ListNode* left = head;
    ListNode* right = reverse(slow);
    while(right) {
        if (left->val != right->val)
            return false;
        left = left->next;
        right = right->next;
    }
    return true;
}
ListNode* reverse(ListNode* head) {
    ListNode* pre = nullptr;
    ListNode* curr = head;
    ListNode* next = head;
    while (curr != nullptr) {
        next = curr->next;
        curr->next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}


