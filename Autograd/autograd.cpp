#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <stack>
#include <math.h>

using namespace std;

class Node : public enable_shared_from_this<Node> {
public:
    double data = 0.0;
    double grad = 0.0;

    Node(double d) : data(d) {}

    static shared_ptr<Node> mul(shared_ptr<Node> a, shared_ptr<Node> b) {
        auto out = make_shared<Node>(a->data * b->data);
        out->prev = {a, b};
        out->backward_fn = [a, b, out]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };
        return out;
    }
    
    static shared_ptr<Node> sum(shared_ptr<Node> a, shared_ptr<Node> b){
    	auto out = make_shared<Node>(a->data + b->data);
        out->prev = {a, b};
        out->backward_fn = [a, b, out]() {
            a->grad += 1 * out->grad;
            b->grad += 1 * out->grad;
        };
        return out;
	}

    static shared_ptr<Node> sub(shared_ptr<Node> a, shared_ptr<Node> b){
    	auto out = make_shared<Node>(a->data - b->data);
        out->prev = {a, b};
        out->backward_fn = [a, b, out]() {
            a->grad += 1 * out->grad;
            b->grad += -1 * out->grad;
        };
        return out;
	}
	
	static shared_ptr<Node> div(shared_ptr<Node> a, shared_ptr<Node> b){
    	auto out = make_shared<Node>(a->data / b->data);
        out->prev = {a, b};
        out->backward_fn = [a, b, out]() {
            a->grad += 1/b->data * out->grad;
            b->grad += -a->data/(b->data*b->data) * out->grad;
        };
        return out;
	}
	
	static shared_ptr<Node> power(shared_ptr<Node> a, shared_ptr<Node> b){
    	auto out = make_shared<Node>(pow(a->data, b->data));
        out->prev = {a, b};
        out->backward_fn = [a, b, out]() {
            a->grad += b->data * pow(a->data, (b->data-1)) * out->grad;
            b->grad += pow(a->data, (b->data))*log(a->data) * out->grad;
        };
        return out;
	}
	
	void findTopoSort(shared_ptr<Node> node, unordered_set<shared_ptr<Node>>& vis, stack<shared_ptr<Node>>& st) {
	    vis.insert(node);
	    for (shared_ptr<Node> it : node->prev) {
	        if (!vis.count(it)) {
	            findTopoSort(it, vis, st);
	        }
	    }
	    st.push(node);
	}
	
	vector<shared_ptr<Node>> topoSort(shared_ptr<Node> start) {
	    stack<shared_ptr<Node>> st;
	    unordered_set<shared_ptr<Node>> vis;
	
	    findTopoSort(start, vis, st);
	
	    vector<shared_ptr<Node>> topo;
	    while (!st.empty()) {
	        topo.push_back(st.top());
	        st.pop();
	    }
	    return topo;
	}
	
	void backward(){
		vector<shared_ptr<Node>> topo = topoSort(shared_from_this());
		
		this->grad = 1.0;
		
		for (auto it = topo.begin(); it != topo.end(); ++it) {
	        (*it)->backward_fn();
	    }
	}
	
	
	void zeroGrad() {
		vector<shared_ptr<Node>> topo = topoSort(shared_from_this());
		
		this->grad = 0.0;
		for (auto it = topo.begin(); it != topo.end(); ++it) {
	        (*it)->grad=0.0;
	    }
	}
	
private:
    vector<shared_ptr<Node>> prev;
    function<void()> backward_fn = [] {};
};



int test(){
	auto a = make_shared<Node>(2.0);
    auto x = make_shared<Node>(3.0);
    auto b = make_shared<Node>(4.0);

    // Forward pass: y = (2.0 * 3.0) + 4.0 = 10.0
    auto ax = Node::mul(a, x);
    auto y  = Node::sum(ax, b);

    // Run backpropagation starting at y
    y->backward();

    // Print Results
    cout << "--- Forward Pass ---" << endl;
    cout << "y.data (2 * 3 + 4): " << y->data << endl; // Expect: 10.0

    cout << "\n--- Backward Pass (Gradients) ---" << endl;
    cout << "dy/dy : " << y->grad << endl; // Expect: 1.0 (seed)
    cout << "dy/da : " << a->grad << endl; // Expect: 3.0 (x's value)
    cout << "dy/dx : " << x->grad << endl; // Expect: 2.0 (a's value)
    cout << "dy/db : " << b->grad << endl; // Expect: 1.0 (addition passes gradient unchanged)


}
