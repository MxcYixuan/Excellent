
/*两数之和
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
*/

//两边hash
vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> umap;
        vector<int> res;
        for(int i = 0; i != nums.size(); i++) {
            umap[nums[i]] = i;
        }
        for (int i = 0; i != nums.size(); i++) {
            if (umap.find(target - nums[i]) != umap.end() 
                && umap[target - nums[i]] != i) {
                return {i, umap[target - nums[i]]};
            }
        }
        return res;
    }

//一遍hash
vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> umap;
        vector<int> res;
        for (int i = 0; i != nums.size(); i++) {
            if (umap.find(target - nums[i]) != umap.end() )
                return {i, umap[target - nums[i]]};
            umap[nums[i]] = i;
        }
        return res;
    }

/*
三数之和(自己首先版本)
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。
*/
vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int index = 0;
        int start = index + 1;
        int len = nums.size();
        int end = nums.size() - 1;
        while(index < len && nums[index] <= 0) {
            int nn = nums[index];
            while (end > start) {
                //[-4,-1,-1,-1,0,1,2,2]
                int sumNum = nums[start] + nums[end] + nums[index];
                if (sumNum > 0) end--;
                else if (sumNum < 0) start++;
                else if (sumNum == 0) {
                    while(start + 1 < end && nums[start] == nums[start + 1]) {
                        start++;
                    }
                    while(end - 1 > start && nums[end]==nums[end - 1]) {
                        end--;
                    }
                    res.push_back({nums[index], nums[start], nums[end]});
                    start++, end--;
                    continue;   
                } 
            } 
            while(index+1<len && nums[index+1] == nums[index])
                index++;
            index++;
            start = index + 1;
            end = len - 1;
        }
        return res;
    }
//三数之和，解题答案部分
vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        if (nums.size() < 3) return res;
        sort(nums.begin(), nums.end());
        // -1 -1 0 1
        int start, end;
        int size = nums.size();
        for (int i = 0; i <= size - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue; 
            if (nums[i] > 0) break;
            start = i + 1;
            end = size - 1;
            while(start < end) {
                int sumNum = nums[i] + nums[start] + nums[end];
                if (sumNum > 0) end--;
                else if (sumNum < 0) start++;
                else {
                    res.push_back({nums[i], nums[start], nums[end]});
                    while(start < end && nums[start + 1] == nums[start])
                        start++;
                    while(start < end && nums[end] == nums[end - 1])
                        end--;
                    start++;
                    end--;
                }
            }
        }
        return res;
    }

/* 习题18 四数之和
给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d 18 ，
使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

注意：
答案中不可以包含重复的四元组。
*/
//双指针解法

vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(),nums.end());
        vector<vector<int>> res;
        if(nums.size()<4) return res;
        int id1, id2, left, right;
        int size = nums.size();
        for(id1 = 0; id1 <= size - 4; id1++){
        	if(id1 > 0 && nums[id1] == nums[id1 - 1]) continue;      //确保nums[a] 改变了
        	for(id2 = id1 + 1; id2 <= size - 3; id2++){
        		if(id2 > id1 + 1 && nums[id2] == nums[id2 - 1]) continue;   //确保nums[b] 改变了
        		left = id2 + 1,right = size - 1;
        		while(left < right){
                        int sumNum = nums[id1]+nums[id2]+nums[left]+nums[right];
        			if(sumNum < target)
        			    left++;
        			else if(sumNum > target)
        			    right--;
        			else{
        				res.push_back({nums[id1],nums[id2],nums[left],nums[right]});
        				while(left < right && nums[left + 1]==nums[left])      //确保nums[c] 改变了
        				    left++;
        				while(left < right && nums[right - 1]==nums[right])      //确保nums[d] 改变了
        				    right--;
        				left++;
        				right--;
					}
				}
			}
		}
		return res;
    }



