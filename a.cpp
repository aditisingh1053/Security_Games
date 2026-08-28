#include <iostream>
#include <typeinfo> 
int main(){
    int * arr = new int[5];
    int * arr2 = new int;
    int arr1[] = {1,2,3};  
    std::cout << typeid(arr).name() << "\n";
    std::cout << typeid(arr1).name() << "\n";
    std::cout << typeid(arr2).name() << "\n";

}