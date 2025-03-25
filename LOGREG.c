#include<stdio.h>
#include<stdlib.h>
#include <float.h>
double min_tresh=0.00001;
double log_tresh=0.00001;
const double e=2.718281;


double factorial(double x){
	if(x==0||x==1)return 1;
	double result=1;
	int i;
	for(i=2;i<=x;i++){
		result*=i;
	}
	return result;
}
double NaturalLog(double x){
	double y = (x-1)/(x+1);
	double result=0,temp = y;
	int i,power = 1;
	do{
		for(i=1;i<power;i++){
			temp*=y;
		}
		result+=temp/power;
		power+=2;
	}while (temp > min_tresh || temp < -min_tresh);
	return 2*result;
}
double Exp(double a, double b){
	double power = b*NaturalLog(a);
	double quad,result=1+power,temp=FLT_MAX;
	int i,n=2;
	do{
		quad=1;
		for(i=0;i<n;i++){
			quad*=power;
		}
		temp=(quad/factorial(n));
		result+=temp;
		n++;
	}while (temp > min_tresh || temp < -min_tresh);
	return result;
}
double Log(double x,double base){
	double result = NaturalLog(x)/NaturalLog(base);
	return result;
}
double sigmoid(double x){
	double result;
	result = 1/(1+Exp(e,-x));
	
	return result;
}
void getminmax(double arr[],int size, double *max,double*min){
	int i;
	*max=INT_MIN,*min=INT_MAX;
	for(i=0;i<size;i++){
		if(*max<arr[i]){
			*max=arr[i];
		}
		if(*min>arr[i]){
			*min=arr[i];
		}
	}
}
double* minmaxscaler(double arr[],int size,double n_min,double n_max){
	int i;
	double min,max;
	getminmax(arr,size,&max,&min);
	double* scaled_arr = (double*)malloc(size * sizeof(double));
	for(i=0;i<size;i++){
		scaled_arr[i]=((arr[i]-min)/(max-min))*(n_max-n_min)+n_min;
	}
	return scaled_arr;		
}
double BCE(double y[], double pred[], int len){
	int i;
	double result, sum=0;
	for(i=0;i<len;i++){
		sum+=(y[i]*NaturalLog(pred[i])+(1-y[i])*NaturalLog(1-pred[i]));
	}
	result=-(sum/len);
	return result;
}
double logreg(double x, double m, double b){
	double pred,activated_pred;
	pred = m*x+b;
	activated_pred = sigmoid(pred);
	return activated_pred;
}
double BCE_temp(double y, double pred){
	double result;
	result=-(y*NaturalLog(pred)+(1-y)*NaturalLog(1-pred));
	return result;
}
void SGD(double x[], double y[], int len, double *grad_m, double *grad_b, double m, double b) {
    int i = rand() % len;  
    double dm, db;
    dm = (BCE_temp(y[i],logreg(x[i], m + log_tresh, b)) - BCE_temp(y[i],logreg(x[i], m, b))) / log_tresh;
    db = (BCE_temp(y[i],logreg(x[i], m, b + log_tresh)) - BCE_temp(y[i],logreg(x[i], m, b))) / log_tresh;
    *grad_m = dm;
    *grad_b = db;
}

void step(double lr, double *m, double *b, double grad_m, double grad_b){
	*m-=grad_m*lr;
	*b-=grad_b*lr;
}
double acc(double y[], double pred[], int len){
	int i;
	double simillar=0;
	double result;
	int ypred;
	for(i=0;i<len;i++){
		ypred=(pred[i]>0.5)?1:0;
		if(y[i]==ypred){
			simillar++;
		}
	}
	return (simillar/len)*100;
}
int main(){
	double x_train[] = {
	    0.45, 1.23, 2.85, 3.67, 4.12, 4.88, 5.01, 5.25, 6.12, 6.89,
	    7.43, 8.56, 9.24, 0.98, 2.47, 3.85, 4.36, 5.14, 6.55, 7.98,
	    1.12, 2.67, 3.45, 4.78, 5.32, 6.87, 7.21, 8.99, 9.42, 0.54,
	    1.89, 3.01, 4.23, 5.45, 6.78, 7.36, 8.21, 9.87, 0.76, 2.34,
	    3.92, 5.01, 6.23, 7.45, 8.67, 9.11, 0.34, 1.56, 2.78, 4.12,
	    5.23, 6.45, 7.89, 9.02, 0.98, 2.45, 3.78, 5.09, 6.44, 7.65,
	    8.97, 9.56, 0.67, 2.34, 3.98, 5.22, 6.78, 7.54, 8.11, 9.45,
	    1.45, 3.21, 4.89, 5.66, 7.12, 8.76, 9.24, 0.34
	};

	double y_train[] = {
	    0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
	    1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
	    0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
	    0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
	    0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	    1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
	    1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
	    0, 0, 0, 1, 1, 1, 1, 0
	};
	
	double x_test[] = {
	    2.78, 4.56, 5.99, 7.45, 8.23, 9.87, 0.12, 2.45, 3.78, 5.01, 
	    6.34, 7.89, 9.01, 0.98, 2.67, 4.23, 5.78, 7.12, 8.99, 9.54
	};
	
	double y_test[] = {
	    0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 
	    1, 1, 1, 0, 0, 0, 1, 1, 1, 1
	};
	int len=sizeof(y_train)/sizeof(y_train[0]);
 	double ypred[len];
    double m = (double)rand() / RAND_MAX; 
    double b = (double)rand() / RAND_MAX;
 	double grad_m,grad_b;
 	int epochs=6000;
 	int i,j;
	double lr=0.01;
 	for(i=0;i<epochs;i++){
 		for(j=0;j<len;j++){
 			ypred[j]=logreg(x_train[j],m,b);
		}
		SGD(x_train,y_train,len,&grad_m,&grad_b,m,b);
		step(lr,&m,&b,grad_m,grad_b);
		printf("Epoch %d: lr = %lf, m = %lf, b = %lf, BCE = %lf\n", i, lr, m, b, BCE(y_train, ypred, len));

	}
 	for(j=0;j<len;j++){
 			ypred[j]=logreg(x_test[j],m,b);
		 }
    double accuracy = acc(y_test, ypred, 20);

    printf("Final equation: y = %.4lfx + %.4lf\n", m, b);
    printf("Final ACC: %lf\n", accuracy);	
 }

