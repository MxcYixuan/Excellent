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


ListNode* merge(vector<ListNode*> &lists, int left, int right) {
    if (left == right) return lists[left];
    if (left > right) return nullptr;
    int mid = left + ((right - left) >> 1);
    return mergeTwoLists(merge(lists, left, mid), merge(lists, mid+1, right));
}
ListNode* mergeKLists(vector<ListNode*>& lists) {
    return merge(lists, 0, lists.size() - 1);
}

//24. 两两交换链表中的节点
/*给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
示例:
给定 1->2->3->4, 你应该返回 2->1->4->3.
*/
ListNode* swapPairs(ListNode* head) {    
    if(!head) return nullptr;
    if(!head->next) return head;
    ListNode* pre = new ListNode(0);
    ListNode* root = pre;
    pre->next = head;
    ListNode* cur = head;
    ListNode* next = cur->next;
    while (next && cur) {
        cur->next = next->next;
        pre->next = next;
        pre->next->next = cur;     
        pre = cur;
        if (cur) cur = cur->next;
        if (cur) next = cur->next;
    }
    return root->next;
}

ListNode* swapPairs(ListNode* head) {
    // 1. 对特殊情况提前返回
    if (head == nullptr || head->next == nullptr) {
        return head;
    }
    // 2. 创建一个临时节点，使它的next指向整个链表的head
    ListNode headNext = ListNode(-1);
    headNext.next = head;
    // 3. 确定双指针指向的两个待交换元素非空
    ListNode* prev = &headNext;
    while (prev->next != nullptr && prev->next->next != nullptr) {
        ListNode* first = prev->next;
        ListNode* second = prev->next->next;
        // 4. 实现交换空间外部界面的处理
        prev->next = second;
        first->next = second->next;
        // 5. 实现交换空间内部的处理
        second->next = first;
        // 6. 向前移动两格
        prev = prev->next->next;
    }
    // 7. 临时节点默默的一直帮我们保存着head，这时终于用到了
    return headNext.next;
}



