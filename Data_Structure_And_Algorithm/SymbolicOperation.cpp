//
// Created by qisheng.cxw on 2020/6/6.
//
//20.有效的括号
/*给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。
*/
bool isValid(string s) {
    unordered_map<char, int> umap{
            {'(', 1},
            {'{', 2},
            {'[', 3},
            {')', 4},
            {'}', 5},
            {']', 6}
    };
    stack<char> sta;
    bool flag = true;
    for (auto ele : s) {
        int id = umap[ele];
        if (id >= 1 && id <= 3) sta.push(ele);
        else if (!sta.empty() && umap[sta.top()] == id - 3)
            sta.pop();
        else {
            flag = false; break;
        }
    }
    if (!sta.empty()) return false;
    return flag;
}

//22. 括号生成
/*
 * 数字 n 代表生成括号的对数，请你设计一个函数，
 * 用于能够生成所有可能的并且 有效的 括号组合。
 */

vector<string> generateParenthesis(int n) {
    vector<string> res;
    dfs(res, "", n, 0, 0);
    return res;
}

void dfs(vector<string> & res, string  str,
         int n, int left,int right) {
    if (left > n || right > n || right > left) return;
    if (left == n && right == n) {
        res.push_back(str);
        return;
    }
    dfs(res, str + "(", n, left + 1, right);
    dfs(res, str + ")", n, left, right + 1);
}
