//133. 克隆图
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

class Node {
    public int val;
    public List<Node> neighbors;
}

//采用stack的迭代法
Node* cloneGraph(Node* node) {
    if(!node) return nullptr;
    stack<Node*> s;
    s.push(node);
    unordered_map<Node*, Node*> u_map;
    u_map[node] = new Node(node->val);

    while(!s.empty()) {
        Node* tmp = s.top();
        s.pop();
        Node* cloneNode = u_map[tmp];
        for (auto &neighbor: tmp->neighbors) {
            if(!u_map.count(neighbor)) {
                Node* t = new Node(neighbor->val);
                u_map[neighbor] = t;
                s.push(neighbor);
            }
            cloneNode->neighbors.push_back(u_map[neighbor]);
        }
    }
    return u_map[node];

}
//递归法
unordered_map<Node*, Node*> u_map;
Node* cloneGraph2(Node* node) {
    if(!node) return nullptr;
    if (u_map.count(node)) {
        return u_map[node];
    }
    Node* cloneNode = new Node(node->val);
    u_map[node] = cloneNode;
    for (auto neighbor : node->neighbors) {
        cloneNode->neighbors.push_back(cloneGraph(neighbor));
    }
    return cloneNode;
}
