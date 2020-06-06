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
