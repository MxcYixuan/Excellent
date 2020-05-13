
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

class priority_deque {
public:
	vector<int> num;
	void shift(vector<int>&num, int id) {
		int i = id * 2 + 1;
		int j = i + 1;
        if (i < num.size() && num[id] > num[i])
            swap(num[i], num[id]);
        if (j < num.size() && num[id] > num[j])
            swap(num[j], num[id]);
    }


	void push (int val) {
        num.push_back(val);
		if (num.size() > 1) {
            for (int id = num.size() / 2 - 1; id >= 0; id--) {
                shift(num, id);
            }
        }
		return ;
	}

	int pop() {
		int res;
		int len = num.size();
		if (len == 0) return INT_MIN;
        res = num[0];
		if (num.size() == 1) {
			num.pop_back();
		} else {
			swap(num[0], num[len - 1]);
            num.pop_back();
            if (num.size() > 1) {
                for (int id = 0; id <= num.size() / 2 - 1; id++) {
                    shift(num, id);
                }
			}
		}
		return res;
	}
};


int main()
{
	vector<vector<int>> vec = { {1,4,3}, {1,4,7} , {1,3,5} , {2,9,4} , {2,5,8} , {3,9,6} };

	sort(vec.begin(), vec.end());

	for(auto p : vec)
		cout<<p[0]<<' '<<p[1]<<' '<<p[2]<<endl;

    priority_deque pd;
    pd.push(5);
    pd.push(1);
	pd.push(-100);
	pd.push(2);
	pd.push(9);
	pd.push(0);


	cout << "size: " << pd.num.size() << endl;

	cout << "--------------" << endl;
    //pd.push(-1);
    //pd.push(10);
    //pd.push(2);

    //cout << pd.pop() << endl;
    cout << pd.pop() << endl;
    cout << pd.pop() << endl;
	cout << pd.pop() << endl;
	cout << pd.pop() << endl;
	cout << pd.pop() << endl;
	cout << pd.pop() << endl;


	return 0;
}
