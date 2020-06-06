/*
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

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
