/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
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

};
