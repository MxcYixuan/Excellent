//
// Created by qisheng.cxw on 2020/5/15.
//
#include<iostream>
using namespace std;
class CSingleton
{
private:
    CSingleton() {
    }
    ~CSingleton() {
        if (m_pInstance == NULL) {
            return;
        }
        delete m_pInstance;
        m_pInstance = NULL;
    }
    static CSingleton *m_pInstance;
public:
    static CSingleton * GetInstance() {
        if(m_pInstance == NULL)
            m_pInstance = new CSingleton();
        return m_pInstance;
    }
};
CSingleton* CSingleton::m_pInstance = NULL;//类的静态成员变量需要在类外边初始化

int main() {

    CSingleton* single1 = CSingleton::GetInstance();
    CSingleton* single2 = CSingleton::GetInstance();

    if (single1 == single2) {
        cout<<"Same"<<endl;
    }
    return 0;
}
