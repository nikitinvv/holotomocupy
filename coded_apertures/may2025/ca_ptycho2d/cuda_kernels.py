import cupy as cp
Efast_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ Efast(float2* res, float2 *psi, int* stx, int* sty, int npos, int npatch, int npsi)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= npatch || ty >= npatch || tz >= npos)
        return;        
    int ind_out = tz*npatch*npatch+ty*npatch+tx;    
    int ind_in = (sty[tz]+ty)*npsi+stx[tz]+tx;                                                           
    res[ind_out] = psi[ind_in];
}
""",
    "Efast",
)

ETfast_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ ETfast(float2* res, float2 *psi, int* stx, int* sty, int npos, int npatch, int npsi)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= npatch || ty >= npatch || tz >= npos)
        return;        
    int ind_out = tz*npatch*npatch+ty*npatch+tx;    
    int ind_in = (sty[tz]+ty)*npsi+stx[tz]+tx;                                                           
    atomicAdd(&psi[ind_in].x,res[ind_out].x);
    atomicAdd(&psi[ind_in].y,res[ind_out].y);    
}
""",
    "ETfast",
)




pad_sym_kernel = cp.RawKernel(r'''                              
extern "C" 
void __global__ pad_sym(float2* g, float2 *f, int pad_width, int n0, int n1, int ntheta, bool direction)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  int txx,tyy;
  if (tx >= n0+2*pad_width || ty >= n1+2*pad_width || tz >= ntheta)
    return;           
  if (ty < pad_width)
      tyy = pad_width-ty-1;
  else 
      if (ty >= n1 + pad_width)
        tyy = 2*n1-ty+pad_width-1;           
      else                
        tyy = ty-pad_width;
  if (tx < pad_width)
      txx = pad_width-tx-1;
  else 
  if (tx >= n0 + pad_width)
    txx = 2*n0-tx+pad_width-1;
  else                
    txx = tx-pad_width;
  int id1 = tz*(n0+2*pad_width)*(n1+2*pad_width)+ty*(n0+2*pad_width)+tx;
  int id2 = tz*n0*n1+tyy*n0+txx;
  if (direction == 0) 
  {
    g[id1].x = f[id2].x;
    g[id1].y = f[id2].y;
  } else {
    atomicAdd(&f[id2].x, g[id1].x);
    atomicAdd(&f[id2].y, g[id1].y);
  }
}
''', 'pad_sym')

pad_kernel = cp.RawKernel(r'''                              
extern "C" void __global__ pad(float2* g, float2 *f, int n, int ntheta, bool direction)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  int txx, tyy;
  if (tx >= 2*n || ty >= 2*n || tz >= ntheta)
    return;    
  if (ty < n/2)
      tyy = n/2-ty-1;
  else 
      if (ty >= n + n/2)
        tyy = 2*n-ty+n/2-1;           
      else                
        tyy = ty-n/2;
  if (tx < n/2)
      txx = n/2-tx-1;
  else 
  if (tx >= n + n/2)
    txx = 2*n-tx+n/2-1;
  else                
    txx = tx-n/2;
  int id1 = tz*2*n*2*n+ty*2*n+tx;
  int id2 = tz*n*n+tyy*n+txx;
  if (direction == 0) 
  {
    g[id1].x = f[id2].x;
    g[id1].y = f[id2].y;
  } else {
    atomicAdd(&f[id2].x, g[id1].x);
    atomicAdd(&f[id2].y, g[id1].y);
  }
}
''', 'pad')

