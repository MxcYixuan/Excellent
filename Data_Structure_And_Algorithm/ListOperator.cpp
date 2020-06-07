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


//23. 合并K个排序链表
//方法1： 思想：按照两两合并链表的思想，进行合并
//时间复杂度O(k平方N),空间复杂度O(1)
ListNode* mergeTwoLists(ListNode *a, ListNode *b) {
    if (!a) return b;
    if (!b) return a;
    ListNode head;
    ListNode *tail = &head, *aPtr = a, *bPtr = b;
    while (aPtr && bPtr) {
        if (aPtr->val < bPtr->val) {
            tail->next = aPtr;
            aPtr = aPtr->next;
        } else {
            tail->next = bPtr;
            bPtr = bPtr->next;
        }
        tail = tail->next;
    }
    tail->next= (aPtr ? aPtr : bPtr);

    return head.next;
}

ListNode* mergeKLists(vector<ListNode*> &lists) {
    ListNode* ans = nullptr;
    for (int i = 0; i != lists.size(); i++) {
        ans = mergeTwoLists(ans, lists[i]);
    }
    return ans;
}


/*方法二：分治合并
思路
将 k 个链表配对并将同一对中的链表合并；

考虑优化方法一，用分治的方法进行合并。
重复这一过程，直到我们得到了最终的有序链表。
*/

ListNode* mergeTwoLists(ListNode *a, ListNode *b) {
    if (!a) return b;
    if (!b) return a;
    ListNode head;
    ListNode *tail = &head, *aPtr = a, *bPtr = b;
    while (aPtr && bPtr) {
        if (aPtr->val < bPtr->val) {
            tail->next = aPtr;
            aPtr = aPtr->next;
        } else {
            tail->next = bPtr;
            bPtr = bPtr->next;
        }
        tail = tail->next;
    }
    tail->next= (aPtr ? aPtr : bPtr);
    return head.next;
}

ListNode* merge(vector<ListNode*>&lists, int left, int right) {
    if (left == right) return lists[left];
    if (left > right) return nullptr;
    int mid = left + ((right - left) >> 1);
    return mergeTwoLists(merge(lists, left, mid), merge(lists, mid+1, right));
}
ListNode* mergeKLists(vector<ListNode*>& lists) {
    return merge(lists, 0, lists.size() - 1);
}



