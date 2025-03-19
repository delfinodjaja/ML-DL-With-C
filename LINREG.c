 #include<stdio.h>
 #include<stdlib.h>
 #include<float.h>
 
 double mse(double y[], double pred[], int len){
 	int i;
 	double diff=0;
 	for(i=0;i<len;i++){
 		diff+=(y[i]-pred[i])*(y[i]-pred[i]);
	 }
	double mean=diff/len;
	return mean;
}
void backward(double y[], double pred[],double x[], int len, double *grad_m, double *grad_b){
 	int i;
 	double dm=0,db=0;
 	for(i=0;i<len;i++){
 		dm+=-2*x[i]*(y[i]-pred[i]);
 		db+=-2*(y[i]-pred[i]);
	 }

	*grad_m = dm / len;
    *grad_b = db / len;
}
void backward_SGD(double y[], double pred[],double x[], int len, double *grad_m, double *grad_b){
 	int i=rand()%len;
 	double dm=0,db=0;
 	dm+=-2*x[i]*(y[i]-pred[i]);
 	db+=-2*(y[i]-pred[i]);

	*grad_m = dm;
    *grad_b = db;
}
void backward_SGD_batch(double y[], double pred[],double x[], int len, double *grad_m, double *grad_b, int batch_size){
	int i;
	double dm=0,db=0;
	int index;
	for(i=0;i<batch_size;i++){
		index=rand()%len;
 		dm+=-2*x[index]*(y[index]-pred[index]);
 		db+=-2*(y[index]-pred[index]);
	}
	*grad_m = dm/batch_size;
    *grad_b = db/batch_size;
}
void step(double lr, double *m, double *b, double grad_m, double grad_b){
	*m-=grad_m*lr;
	*b-=grad_b*lr;
}
void linreg(double x[],double m,double b,double ypred[], int len){
	int i;
	for(i=0;i<len;i++){
		ypred[i]=x[i]*m+b;
	}
}
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
	double result,temp = y;
	int i,power = 1;
	while(temp>=0.0001){
		for(i=1;i<power;i++){
			temp*=y;
		}
		result+=temp/power;
		power+=2;
	}
	return 2*result;
}

double Exp(double a, double b){
	double power = b*NaturalLog(a);
	double quad,result=1+power,temp=FLT_MAX;
	int i,n=2;
	while(temp>0.001||temp<-0.001){
		quad=1;
		for(i=0;i<n;i++){
			quad*=power;
		}
		temp=(quad/factorial(n));
		printf("%lf\n",temp);
		result+=temp;
		n++;
	}
	return result;
}

void getminmax(double arr[],int size, double *max,double*min){
	int i;
	*max=DBL_MAX,*min=DBL_MIN;
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
 int main(){
	double x_train[] = {
		3.745, 9.507, 7.320, 5.987, 1.560, 1.560, 0.581, 8.662, 6.011, 7.081, 
        	0.206, 9.699, 8.324, 2.123, 1.818, 1.834, 3.042, 5.248, 4.319, 2.912, 
        	6.119, 1.395, 2.921, 3.664, 4.561, 7.852, 1.997, 5.142, 5.924, 0.465, 
        	6.075, 1.705, 0.651, 9.489, 9.656, 8.084, 3.046, 0.977, 6.842, 4.402, 
		1.220, 4.952, 0.344, 9.093, 2.588, 6.625, 3.117, 5.201, 5.467, 1.849
	};
	
	double y_train[] = {
		11.975, 28.693, 21.844, 17.659, 3.202, 3.960, 1.282, 27.042, 18.377, 19.479, 
        	0.942, 28.712, 24.296, 6.982, 6.486, 6.433, 8.288, 15.433, 13.290, 9.712, 
        	17.876, 3.999, 7.658, 9.795, 14.495, 24.912, 5.918, 16.431, 18.134, 0.748, 
        	18.588, 6.654, 1.916, 30.031, 26.349, 25.074, 9.225, 2.631, 20.619, 11.217, 
        	3.441, 15.212, 2.510, 26.761, 6.955, 19.374, 10.267, 15.931, 15.872, 6.059
	};
	
	double x_test[] = {
		2.153, 7.438, 9.149, 1.860, 3.716, 5.230, 0.817, 6.283, 8.746, 2.590, 
        	4.919, 7.241, 1.442, 3.109, 5.761, 9.612, 0.532, 6.792, 7.964, 4.307
	};
	
	double y_test[] = {
		7.281, 22.417, 26.853, 5.482, 11.071, 15.963, 2.981, 18.786, 24.715, 7.814, 
        	13.997, 22.070, 4.134, 9.452, 18.326, 28.822, 2.042, 20.758, 23.254, 12.732
	};
 	int len=sizeof(y_train)/sizeof(y_train[0]);
 	int len_x=sizeof(x_train)/sizeof(x_train[0]);
	if(len_x!=len){
		printf("MISMATCH\nOUT= %d IN= %d\n",len,len_x);
	}  	
 	double *scaled_x=minmaxscaler(x_train,len,0,1);
 	double *scaled_y=minmaxscaler(y_train,len,0,1);
 	double *scaled_x_test=minmaxscaler(x_test,len,0,1);
 	double *scaled_y_test=minmaxscaler(y_test,len,0,1);
 	double ypred[len];
    double m = (double)rand() / RAND_MAX; 
    double b = (double)rand() / RAND_MAX;
 	double grad_m,grad_b;
 	int epochs=100;
 	int i,j;
	double lr=0.01;
 	for(i=0;i<epochs;i++){
 		linreg(scaled_x,m,b,ypred,len);
		backward_SGD_batch(scaled_y,ypred,scaled_x,len,&grad_m,&grad_b,10);
		step(lr,&m,&b,grad_m,grad_b);
		printf("Epoch %d: lr = %lf, m = %lf, b = %lf, MSE = %lf\n", i, lr, m, b, mse(scaled_y, ypred, len));

	 }
 	double pred[20];
    linreg(scaled_x_test, m, b, pred, 20);
    double final_mse = mse(scaled_y_test, pred, 20);

    printf("Final equation: y = %.4lfx + %.4lf\n", m, b);
    printf("Final MSE: %.6lf\n", final_mse);	
 }
