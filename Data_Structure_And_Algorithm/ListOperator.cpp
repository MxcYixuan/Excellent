//
// Created by qisheng.cxw on 2020/6/6.
//

//19. 删除链表的倒数第N个节点
//说明 给定的 n 保证是有效的。
//你能尝试使用一趟扫描实现吗？
ListNode* removeNthFromEnd(ListNode* head, int n) {
    if (!head) return nullptr;
    ListNode* fast = head;
    ListNode* slow = head;
    for (int id = 0; id < n; id++)
        fast = fast->next;
    if (fast) {
        while(fast->next) {
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
    } else head = head->next;
    return head;
}


//21. 合并两个有序链表
//将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
//方法1 迭代，时间复杂度O(M+N),空间复杂度O(1)
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if (!l1 && !l2) return nullptr;
    if (!l1) return l2;
    if (!l2) return l1;
    ListNode* head = new ListNode(0);
    ListNode* curr = head;
    while(l1 && l2) {
        if (l1->val < l2->val) {
            curr->next = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            l2 = l2->next;
        }
        curr = curr->next;
    }
    if (l1) curr->next = l1;
    if (l2) curr->next = l2;
    return head->next;
}

//方法2 递归，时间复杂度O(M+N),空间复杂度O(M+N)
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if(!l1) return l2;
    if(!l2) return l1;
    if(l1->val < l2->val) {
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    } else {
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    }
}

