#include<stdio.h>
#include<stdlib.h>
#include<float.h>
double min_tresh=0.00001;
double log_tresh=0.00001;

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
double* flatten(int outer_dim, int inner_dim, double arr[outer_dim][inner_dim]){
	double* flattened = (double*)malloc(outer_dim * inner_dim * sizeof(double));
	int i,j,k=0;
	for(i=0;i<outer_dim;i++){
		for(j=0;j<inner_dim;j++){
			flattened[k]=arr[i][j];
			k++;
		}
	}
	return flattened;
}
double MSE(int in, int out, double y[], double ypred[], int len){
	int i,j,k;
	double diff=0;
	for(i=0;i<len;i++){
		for(j=0;j<out;j++){
			diff+=(y[out*i+j]-ypred[out*i+j])*(y[out*i+j]-ypred[out*i+j]);
		}
	}
	return diff/(len*out);
}
void LINREG(int in, int out, double x[], double m[], double b[],int len, double ypred[]){
	int i,j,k;
	for(i=0;i<len;i++){
		for(j=0;j<out;j++){
			ypred[out*i+j]=b[j];
			for(k=0;k<in;k++){
				ypred[out*i+j]+=m[in*j+k]*x[in*i+k];
			}
		}
	}
}
double LINREG_var_2(int in, int out, double x[], double m[], double b){
	int i,j;
	double pred=b;
	for(j=0;j<in;j++){
		pred+=m[j]*x[j];
		
	}
	return pred;
}
void SGD(int in, int out, double y[], double pred[], double x[], double m[], double b[],int len, double grad_m[], double grad_b[]){
	int i=rand()%len;
 	int j,k;
	for(j=0;j<out;j++){
		for(k=0;k<in;k++){
			grad_m[j*in+k]=-2*x[i*in+k]*(y[i*out+j]-pred[i*out+j]);
		}
		grad_b[j] = -2 * (y[i * out + j] - pred[j]);
	}
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
void step(int out, int in, double lr,double m[], double b[], double grad_m[],double grad_b[]){
	int j,k;
	for(j=0;j<out;j++){
		for(k=0;k<in;k++){
			m[j*in+k]-=grad_m[j*in+k]*lr;

		}
		b[j]-=grad_b[j]*lr;
	}
}
int main(){
	int len=50,in=2,out=3;
	double x[50][2] = {
	    {1.2, 2.1}, {2.3, 3.2}, {3.1, 4.0}, {4.4, 5.3}, {5.0, 6.1}, 
	    {6.2, 7.3}, {7.5, 8.4}, {8.1, 9.2}, {9.3, 10.4}, {10.5, 11.6},
	    {11.2, 12.3}, {12.4, 13.5}, {13.1, 14.2}, {14.6, 15.7}, {15.0, 16.1},
	    {16.3, 17.4}, {17.7, 18.8}, {18.2, 19.3}, {19.4, 20.5}, {20.1, 21.2},
	    {21.5, 22.6}, {22.0, 23.1}, {23.6, 24.7}, {24.2, 25.3}, {25.0, 26.1},
	    {1.8, 2.9}, {2.7, 3.8}, {3.9, 5.0}, {4.8, 5.9}, {5.6, 6.7},
	    {6.9, 8.0}, {7.8, 8.9}, {8.7, 9.8}, {9.6, 10.7}, {10.8, 12.0},
	    {11.7, 12.8}, {12.6, 13.7}, {13.8, 14.9}, {14.7, 15.8}, {15.9, 17.0},
	    {16.8, 17.9}, {17.9, 19.0}, {18.8, 19.9}, {19.7, 20.8}, {20.9, 22.0},
	    {21.8, 22.9}, {22.7, 23.8}, {23.9, 25.0}, {24.8, 25.9}, {25.7, 26.8}
	};
	double y[50][3] = {
	    {0.3, 1.2, 0.9}, {1.4, 3.1, 2.2}, {2.1, 3.3, 4.0}, {3.5, 4.6, 5.1}, {1.6, 5.2, 3.5},
	    {3.8, 4.9, 6.5}, {5.3, 6.1, 7.9}, {7.2, 8.5, 9.0}, {8.3, 9.8, 10.2}, {9.5, 10.9, 11.3},
	    {10.4, 11.8, 12.5}, {11.6, 12.9, 14.0}, {12.3, 13.5, 15.2}, {13.7, 14.8, 16.1}, {14.2, 15.3, 17.0},
	    {15.5, 16.6, 18.3}, {16.9, 17.8, 19.5}, {17.4, 18.5, 20.2}, {18.6, 19.7, 21.5}, {19.3, 20.4, 22.1},
	    {20.7, 21.8, 23.5}, {21.2, 22.3, 24.0}, {22.8, 23.9, 25.6}, {23.4, 24.5, 26.2}, {24.1, 25.2, 27.0},
	    {0.7, 1.8, 1.5}, {1.9, 3.0, 2.8}, {3.2, 4.1, 5.3}, {4.3, 5.4, 6.2}, {2.0, 5.8, 4.1},
	    {4.5, 5.9, 7.2}, {6.0, 7.1, 8.5}, {7.8, 8.7, 10.0}, {8.9, 10.1, 11.2}, {10.2, 11.5, 12.8},
	    {11.3, 12.6, 14.1}, {12.5, 13.8, 15.3}, {14.0, 15.1, 16.8}, {15.2, 16.3, 18.0}, {16.1, 17.2, 19.3},
	    {17.5, 18.6, 20.5}, {18.7, 19.8, 21.6}, {19.9, 21.0, 22.8}, {20.8, 21.9, 23.7}, {22.1, 23.2, 25.0},
	    {23.0, 24.1, 26.2}, {24.3, 25.4, 27.1}, {25.5, 26.6, 28.0}, {26.2, 27.3, 29.1}, {27.0, 28.1, 30.0}
	};
	double x_test[10][2] = {
	    {1.5, 2.5}, {2.5, 3.5}, {3.5, 4.5}, {4.5, 5.5}, {5.5, 6.5},
	    {6.5, 7.5}, {7.5, 8.5}, {8.5, 9.5}, {9.5, 10.5}, {10.5, 11.5}
	};
	
	double y_test[10][3] = {
	    {0.5, 1.6, 1.2}, {1.3, 3.1, 2.4}, {2.2, 3.5, 4.0}, {3.3, 4.8, 5.2}, {1.8, 5.6, 3.5},
	    {3.7, 5.0, 6.5}, {5.2, 6.4, 7.5}, {7.4, 8.6, 9.3}, {8.3, 10.1, 10.5}, {9.9, 11.0, 11.6}
	};
	double m[6]; 
	double b[3];
	int i,j;
	for(i=0;i<in*out;i++){
		m[i]=(double)rand() / RAND_MAX;
	}
	for(i=0;i<out;i++){
		b[i]=(double)rand() / RAND_MAX;
	}
	double* flatten_x=flatten(len,in,x);
	double* flatten_y=flatten(len,out,y);
	double* flatten_x_test=flatten(len,in,x_test);
	double* flatten_y_test=flatten(len,out,y_test);
	double ypred[len*out];
 	double *scaled_x=minmaxscaler(flatten_x,len*out,0,1);
 	double *scaled_y=minmaxscaler(flatten_y,len*out,0,1);
 	double *scaled_x_test=minmaxscaler(flatten_x_test,len*out,0,1);
 	double *scaled_y_test=minmaxscaler(flatten_y_test,len*out,0,1);
	double grad_m[out*in];
	double grad_b[out];
	int epochs = 500;
	double lr=0.01;
	for(i=0;i<epochs;i++){
		LINREG(in,out,scaled_x,m,b,len,ypred);
		SGD(in,out,scaled_y,ypred,scaled_x,m,b,len,grad_m,grad_b);
		step(in,out,lr,m,b,grad_m,grad_b);
		printf("Epoch %d: lr = %lf, MSE = %lf\n", i, lr, MSE(in,out,scaled_y,ypred,len));
	}
	double pred[len*out];
	LINREG(in,out,scaled_x_test,m,b,10,pred);
	printf("MSE = %lf\n", MSE(in,out,scaled_y_test,pred,10));

	/**
	Linear regression with multidimension array as input instead of flattened
	double x2[3][2] = {{1.0, 2.0},   
	                   {3.0, 4.0},   
				       {5.0, 6.0}};  
				       
	double m2[2][2] = {{0.5, -0.2}, 
				  	   {0.8,  0.3}}; 
	double b2[] = {0.1, -0.3};
	double pred_y[3][2];
	for(i=0;i<3;i++){
		for(j=0;j<2;j++){
			pred_y[i][j] = LINREG_var_2(2,2,x2[i],m2[j],b2[j]);
		}
	}
	for(i=0;i<3;i++){
		for(j=0;j<2;j++){
			printf("%lf ",pred_y[i][j]);
		}
		printf("\n");
	}
	double* flat=flatten(2,2,m2);
	for(i=0;i<4;i++){
		printf("%lf ",flat[i]);
	}
	**/
}
