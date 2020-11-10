#pragma once
#include<Proto.H>

void test_reduction_linear(double* ptr, double a_coef, double a_pt, unsigned int a_size)
{
  for(int i = 0 ; i < a_size ; i++)
  {
    ptr[i] = i * a_coef + a_pt;
  }
}

bool test_reduction_min_linear_init_val(double a_val, double a_pt)
{
  unsigned int size = 32;
  double * data = new double[size];
 
  test_reduction_linear(data, a_val, a_pt, size);

  double * device;
  protoMalloc(&device,sizeof(double)*size); 
  protoMemcpy(device, data, sizeof(double)*size, protoMemcpyHostToDevice);

  Reduction<double,Operation::Min> red;
  red.reset();

  red.reduce(device, size);
 
  double result = red.fetch();
 
  bool check = false;

  if(a_val < 0) 
    check = result == ((size-1) * a_val + a_pt);
  else
    check = result == a_pt;

  if(!check)
    std::cout << result << std::endl; 
 
  return check;
}

bool test_reduction_min_linear_init_1()
{
  return test_reduction_min_linear_init_val(1,-1.01);
}

bool test_reduction_min_linear_init_minus_2()
{
  return test_reduction_min_linear_init_val(-2,1.01);
}
