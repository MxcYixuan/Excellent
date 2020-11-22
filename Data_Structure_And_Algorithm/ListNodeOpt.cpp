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
