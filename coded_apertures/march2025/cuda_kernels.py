import cupy as cp
# placing/extracting patches is better to do with CUDA C
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


wrap_kernel = cp.RawKernel(r'''
extern "C" __global__ void __global__ wrap(float2 *f, int n, int nz, int m)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
		return;
	if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
	{
		int tx0 = (tx - m + 2 * n) % (2 * n);
		int ty0 = (ty - m + 2 * n) % (2 * n);
		int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
		int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
		f[id1].x = f[id2].x;
		f[id1].y = f[id2].y;
	}
}
                           
''', 'wrap')

wrapadj_kernel = cp.RawKernel(r'''                
extern "C" __global__ void wrapadj(float2 *f, int n, int nz, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;
  if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
  {
    int tx0 = (tx - m + 2 * n) % (2 * n);
    int ty0 = (ty - m + 2 * n) % (2 * n);
    int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
    int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
    
    atomicAdd(&f[id2].x, f[id1].x);
    atomicAdd(&f[id2].y, f[id1].y);
  }
}

''', 'wrapadj')

gather_kernel = cp.RawKernel(r'''
extern "C" __global__ void gather(float2 *g, float2 *f, float *theta, int m,
                       float *mu, int n, int ntheta, int nz, bool direction)
{

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= ntheta || tz >= nz)
    return;
  float M_PI = 3.141592653589793238f;
  float2 g0, g0t;
  float w, coeff0;
  float w0, w1, x0, y0, coeff1;
  int ell0, ell1, g_ind, f_ind;

  g_ind = tx + tz * n + ty * n * nz;//swapped axes
  if (direction == 0) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x;
    g0.y = g[g_ind].y;
  }

  coeff0 = M_PI / mu[0];
  coeff1 = -M_PI * M_PI / mu[0];
  x0 = (tx - n / 2) / (float)n * __cosf(theta[ty]);
  y0 = -(tx - n / 2) / (float)n * __sinf(theta[ty]);
  if (x0 >= 0.5f)
    x0 = 0.5f - 1e-5;
  if (y0 >= 0.5f)
    y0 = 0.5f - 1e-5;
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;                             
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      w0 = ell0 / (float)(2 * n) - x0;
      w1 = ell1 / (float)(2 * n) - y0;
      w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1)); 
      f_ind = n + m + ell0 + (2 * n + 2 * m) * (n + m + ell1) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
      if (direction == 0) {
        g0.x += w * f[f_ind].x;
        g0.y += w * f[f_ind].y;
      } else {
        float *fx = &(f[f_ind].x);
        float *fy = &(f[f_ind].y);
        atomicAdd(fx, w * g0.x);
        atomicAdd(fy, w * g0.y);
      }
    }
  }
  if (direction == 0){
    g[g_ind].x = g0.x / n;
    g[g_ind].y = g0.y / n;
  }
}

''', 'gather')
