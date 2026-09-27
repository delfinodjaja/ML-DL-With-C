#include "autograd.cpp"
#include <iostream>
using namespace std;

int main(){
    double lr = 0.001;
    auto a = make_shared<Node>(0.1);
    auto b = make_shared<Node>(0.1);

    vector<double> x_train = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    vector<double> y_train = {3.1, 4.9, 7.2, 8.8, 11.3, 12.9, 15.4, 16.8, 19.2};
    vector<double> x_test = {10.0, 11.0, 12.0};
    vector<double> y_test = {21.1, 22.9, 25.3};

    for(int i=0;i<50;i++){
        vector<shared_ptr<Node>> y_pred;  

        for(int j=0;j<x_train.size();j++){
            auto x = make_shared<Node>(x_train[j]);
            auto ax = Node::mul(a, x);
            auto y  = Node::sum(ax, b);
            y_pred.push_back(y);          
        }

        shared_ptr<Node> error = make_shared<Node>(0.0); 

        for(int j = 0;j<y_pred.size();j++){
            auto y = make_shared<Node>(y_train[j]);
            auto y_ = y_pred[j];       
            auto x = make_shared<Node>(2.0);

            auto tmp = Node::power(Node::sub(y, y_), x);

            error = Node::sum(error, tmp); 
        }

        auto n = make_shared<Node>((double)y_pred.size());  

        auto loss = Node::div(error, n); 

        cout << "Loss " << loss->data << endl; 
        
        loss->zeroGrad();

        loss->backward();
        
		a->data -= lr * a->grad;
		b->data -= lr * b->grad;

    }
	cout << "Final a (weight): " << a->data << endl;
	cout << "Final b (bias):   " << b->data << endl;
	
	//test
	vector<shared_ptr<Node>> y_pred;  

    for(int j=0;j<x_test.size();j++){
        auto x = make_shared<Node>(x_test[j]);
        auto ax = Node::mul(a, x);
        auto y  = Node::sum(ax, b);
        y_pred.push_back(y);          
    }

    shared_ptr<Node> error = make_shared<Node>(0.0); 

    for(int j = 0;j<y_pred.size();j++){
        auto y = make_shared<Node>(y_test[j]);
        auto y_ = y_pred[j];       
        auto x = make_shared<Node>(2.0);

        auto tmp = Node::power(Node::sub(y, y_), x);

        error = Node::sum(error, tmp); 
    }

    auto n = make_shared<Node>((double)y_pred.size());  

    auto loss = Node::div(error, n); 

    cout << "Test Loss " << loss->data << endl; 	
}
