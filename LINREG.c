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
	    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
	    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
	    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
	    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
	    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
	    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
	    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
	    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
	    91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
	    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
	    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
	    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
	    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
	    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
	    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
	    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
	    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
	    181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
	    191, 192, 193, 194, 195, 196, 197, 198, 199, 200
	};
	
	double y_train[] = {
	    3.04, 3.46, 5.29, 7.22, 6.51, 7.56, 10.42, 10.66, 10.48, 12.54,
	    12.58, 13.63, 15.39, 14.28, 15.52, 17.73, 18.33, 20.71, 20.54, 21.08,
	    25.01, 24.37, 25.71, 25.27, 27.20, 28.91, 28.69, 31.27, 31.34, 32.70,
	    33.44, 36.95, 36.13, 36.14, 39.07, 38.07, 40.55, 39.44, 41.12, 43.69,
	    45.28, 45.77, 46.53, 47.39, 47.27, 49.08, 50.38, 52.95, 53.29, 52.23,
	    55.37, 55.71, 56.47, 58.81, 60.28, 61.23, 60.51, 62.09, 63.78, 65.47,
	    65.07, 66.41, 66.54, 67.50, 70.56, 72.15, 71.77, 73.90, 74.31, 74.35,
	    76.41, 78.63, 78.11, 80.76, 77.63, 82.12, 82.43, 83.10, 84.54, 83.51,
	    86.33, 87.95, 90.12, 89.18, 89.94, 91.29, 93.76, 94.22, 94.42, 96.51,
	    97.14, 99.06, 98.44, 99.87, 100.85, 100.83, 103.64, 104.66, 105.45, 106.26,
	    108.1, 109.3, 110.7, 112.2, 113.6, 115.4, 116.8, 118.5, 119.7, 121.3,
	    122.8, 124.1, 125.6, 127.3, 128.7, 130.5, 132.0, 133.4, 135.0, 136.5,
	    138.3, 139.7, 141.2, 142.8, 144.3, 146.0, 147.6, 149.2, 150.8, 152.3,
	    154.0, 155.7, 157.3, 158.9, 160.5, 162.0, 163.8, 165.2, 166.8, 168.3
	};
	
	double x_test[] = {
	    201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
	    211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
	    221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
	    231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
	    241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
	    251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
	    261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
	    271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
	    281, 282, 283, 284, 285, 286, 287, 288, 289, 290,
	    291, 292, 293, 294, 295, 296, 297, 298, 299, 300
	};
	
	double y_test[] = {
	    170.0, 171.8, 173.5, 175.0, 176.8, 178.3, 180.0, 181.6, 183.0, 184.8,
	    186.5, 188.3, 190.0, 191.6, 193.2, 195.0, 196.5, 198.0, 199.8, 201.2,
	    202.8, 204.3, 206.0, 207.5, 209.3, 210.8, 212.5, 214.2, 216.0, 217.5,
	    219.3, 220.8, 222.5, 224.2, 226.0, 227.5, 229.3, 230.8, 232.5, 234.2,
	    236.0, 237.5, 239.3, 240.8, 242.5, 244.2, 246.0, 247.5, 249.3, 250.8,
	    252.5, 254.2, 256.0, 257.5, 259.3, 260.8, 262.5, 264.2, 266.0, 267.5,
	    269.3, 270.8, 272.5, 274.2, 276.0, 277.5, 279.3, 280.8, 282.5, 284.2,
	    286.0, 287.5, 289.3, 290.8, 292.5, 294.2, 296.0, 297.5, 299.3, 300.8,
	    302.5, 304.2, 306.0, 307.5, 309.3, 310.8, 312.5, 314.2, 316.0, 317.5,
	    319.3, 320.8, 322.5, 324.2, 326.0, 327.5, 329.3, 330.8, 332.5, 334.2
	};
 	int len=sizeof(y_train)/sizeof(y_train[0]);
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
 	double pred[100];
    linreg(scaled_x_test, m, b, pred, 100);
    double final_mse = mse(scaled_y_test, pred, 100);

    printf("Final equation: y = %.4lfx + %.4lf\n", m, b);
    printf("Final MSE: %.6lf\n", final_mse);	
 }
