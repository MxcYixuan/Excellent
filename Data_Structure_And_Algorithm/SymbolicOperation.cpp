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