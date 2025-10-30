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
  int ell0, ell1, g_ind, f_ind, f_indx, f_indy;

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
                               
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;                             
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      w0 = ell0 / (float)(2 * n) - x0;
      w1 = ell1 / (float)(2 * n) - y0;
      w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1)); 
      f_indx = (n + ell0 + 2 * n)%(2 * n);
      f_indy = (n + ell1 + 2 * n)%(2 * n); 
      f_ind = f_indx  + (2 * n) * f_indy + tz * (2 * n) * (2 * n);
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



gather_mag_kernel = cp.RawKernel(r'''
extern "C" __global__ void gather_mag(float2 *g, float2 *f, float *magnification, int m,
                       float *mu, int n, int ne, int ntheta, bool direction)
{

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= n || tz >= ntheta)
    return;
  float M_PI = 3.141592653589793238f;
  float2 g0;
  float w, coeff0;
  float w0, w1, x0, y0, coeff1;
  int ell0, ell1, g_ind, f_ind,f_indx,f_indy;

  g_ind = tx + ty * n + tz * n * n;
  if (direction == 0) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x;
    g0.y = g[g_ind].y;
  }

  coeff0 = M_PI / mu[0];
  coeff1 = -M_PI * M_PI / mu[0];
  float s =  - (magnification[0]-1)*0.5-magnification[0]*(n/(float)ne-1)*0.5;
  x0 = -(tx - n / 2 + s) / (float)n / magnification[0];
  y0 = -(ty - n / 2 + s) / (float)n / magnification[0];

  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * ne * y0) - m + i1;
                                 
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * ne * x0) - m + i0;
  
      w0 = ell0 / (float)(2 * ne) - x0;
      w1 = ell1 / (float)(2 * ne) - y0;
      w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1)); 
      f_indx = (ne + ell0 + 2 * ne)%(2 * ne);
      f_indy = (ne + ell1 + 2 * ne)%(2 * ne);                                                           
      
      f_ind = f_indx + (2 * ne) * f_indy + tz * (2 * ne) * (2 * ne);
      
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
    g[g_ind].x = g0.x / ne;
    g[g_ind].y = g0.y / ne;
  }
}

''', 'gather_mag')


ishift_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ ishift(float2* res, float2 *psi, int* x, int* y, int n, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= n || tz >= ntheta)
        return;  
    int indx = (tx-x[tz]+n)%n;         
    int indy = (ty-y[tz]+n)%n;         
    res[tx+ty*n+tz*n*n].x = psi[indx+indy*n+tz*n*n].x;
    res[tx+ty*n+tz*n*n].y = psi[indx+indy*n+tz*n*n].y;
}
""", "ishift",
)

fun_phi = r"""
__device__ float phi(float t, float m)
{
  int sgn=0;
  float w = 0;

  t/=m;

  if (-2<t && t<=-1)
  {
    w =  (t+2)*(t+2)*(t+2);
  }
  else if (-1<t && t<=1)
  {
    sgn = (t > 0) ? 1 : ((t < 0) ? -1 : 0);
    w = 4-6*t*t+3*t*t*t*sgn;
  }
  else if (1<t && t<=2)
  {
    w = (2-t)*(2-t)*(2-t);
  }
  else
    w = 0;
  
  return w;
}
"""

fun_dphi = r"""
__device__ float dphi(float t, float m)
{
  int sgn=0;  
  float w = 0;
  t/=m;
  if (-2<t && t<=-1)
  {
    w =  3*(t+2)*(t+2);
  }
  else if (-1<t && t<=1)
  {
    sgn = (t > 0) ? 1 : ((t < 0) ? -1 : 0);
    w = -12*t+9*t*t*sgn;
  }
  else if (1<t && t<=2)
  {
    w = -3*(2-t)*(2-t);
  }
  else
    w = 0;
  
  w/=m;
  return w;
}
"""

fun_d2phi = r"""
__device__ float d2phi(float t,float m)
{
  int sgn=0;  
  float w = 0;
  t/=m;

  if (-2<t && t<=-1)
  {
    w =  6*(t+2);
  }
  else if (-1<t && t<=1)
  {
    sgn = (t > 0) ? 1 : ((t < 0) ? -1 : 0);
    w = -12+18*t*sgn;
  }
  else if (1<t && t<=2)
  {
    w = 6*(2-t);
  }
  else
    w = 0;
  
  w/=(m*m);
  return w;
}
"""

s_kernel = cp.RawKernel(
    r"""                                 
extern "C" 
{"""+fun_phi+
r"""
void __global__ s(float2* g, float2 *f, float* r, float* mag, int n, int npsi, int ntheta, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= n || tz >= ntheta)
        return;  

    
    int ix,iy;
    int f_indx,f_indy,f_ind,g_ind;
    float x,y;
    float dx,dy;  
    float dxm,dym;   
    float w;
    float2 g0;
        
    x = (mag[0]*(tx-n/2)-r[2*tz+1]+(mag[0]-1)/2)+npsi/2;
    y = (mag[0]*(ty-n/2)-r[2*tz+0]+(mag[0]-1)/2)+npsi/2;
    
    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x-ix;
    dy = y-iy;

    g_ind = tx+ty*n+tz*n*n;

    if (dir==0) 
    {
      g0.x = 0.0f;
      g0.y = 0.0f;
    } 
    else 
    {
      g0.x = g[g_ind].x;
      g0.y = g[g_ind].y;
    }

    
    for (int jy=ceil(dy-2*mag[0]);jy<dy+2*mag[0];jy++)
      for (int jx=ceil(dx-2*mag[0]);jx<dx+2*mag[0];jx++)
      {
        dxm = dx-jx;
        dym = dy-jy;
        w = phi(dxm,mag[0])*phi(dym,mag[0]);
        f_indx = (ix+jx+npsi)%npsi;         
        f_indy = (iy+jy+npsi)%npsi;         
        f_ind = f_indx+f_indy*npsi+tz*npsi*npsi;
        if (dir==0)
        {
          g0.x += w*f[f_ind].x;
          g0.y += w*f[f_ind].y;
        }
        else
        {
          float *fx = &(f[f_ind].x);
          float *fy = &(f[f_ind].y);
          atomicAdd(fx, w * g0.x);
          atomicAdd(fy, w * g0.y);          
        }

      }
    if (dir==0)
    {
      g[g_ind].x=g0.x;
      g[g_ind].y=g0.y;    
    }
}
}
""", "s",
)


dt_kernel = cp.RawKernel(
    r"""                                 
extern "C" 
{"""+fun_phi+fun_dphi+
r"""
void __global__ dt(float2* res, float2 *c, float* r,float* mag, float* Deltar,  int n, int npsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= n || tz >= ntheta)
        return;  

    
    int ix,iy;
    int indx,indy;
    float x,y;
    float dx,dy;    
    float dxm,dym; 
    float Deltarx,Deltary;   
    float w;
    float2 r0 = {};
        
    x = (mag[0]*(tx-n/2)-r[2*tz+1]+(mag[0]-1)/2)+npsi/2;
    y = (mag[0]*(ty-n/2)-r[2*tz+0]+(mag[0]-1)/2)+npsi/2;
    
    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x-ix;
    dy = y-iy;
    
    Deltarx = Deltar[2*tz+1];
    Deltary = Deltar[2*tz+0];
    
    for (int jy=ceil(dy-2*mag[0]);jy<dy+2*mag[0];jy++)
      for (int jx=ceil(dx-2*mag[0]);jx<dx+2*mag[0];jx++)
      {      
        dxm = dx - jx;
        dym = dy - jy;
        w = dphi(dxm,mag[0])*phi(dym,mag[0])*Deltarx+
            dphi(dym,mag[0])*phi(dxm,mag[0])*Deltary;

        indx = (ix+jx+npsi)%npsi;         
        indy = (iy+jy+npsi)%npsi;         
        r0.x -= w*c[indx+indy*npsi+tz*npsi*npsi].x;
        r0.y -= w*c[indx+indy*npsi+tz*npsi*npsi].y;
      }
    res[tx+ty*n+tz*n*n].x=r0.x;
    res[tx+ty*n+tz*n*n].y=r0.y;    
}
}
""", "dt",
)

d2t_kernel = cp.RawKernel(
    r"""                                 
extern "C" 
{"""+fun_phi+fun_dphi+fun_d2phi+
r"""
void __global__ d2t(float2* res, float2 *c, float* r, float* mag, float* Deltar1, float* Deltar2, int n,int npsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= n || tz >= ntheta)
        return;  
    
    int ix,iy;
    int indx,indy;
    float x,y;
    float dx,dy;   
    float dxm,dym;
    float Deltar1x,Deltar1y;  
    float Deltar2x,Deltar2y;
    float w;
    float2 r0 = {};
        
    x = (mag[0]*(tx-n/2)-r[2*tz+1]+(mag[0]-1)/2)+npsi/2;
    y = (mag[0]*(ty-n/2)-r[2*tz+0]+(mag[0]-1)/2)+npsi/2;
    
    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x-ix;
    dy = y-iy;
    
    Deltar1x = Deltar1[2*tz+1];
    Deltar1y = Deltar1[2*tz+0];
    Deltar2x = Deltar2[2*tz+1];
    Deltar2y = Deltar2[2*tz+0];


    for (int jy=ceil(dy-2*mag[0]);jy<dy+2*mag[0];jy++)
      for (int jx=ceil(dx-2*mag[0]);jx<dx+2*mag[0];jx++)
      {       
        dxm = dx - jx;
        dym = dy - jy;
        w = d2phi(dxm,mag[0])*phi(dym,mag[0])*Deltar1x*Deltar2x;
        w += dphi(dxm,mag[0])*dphi(dym,mag[0])*(Deltar1x * Deltar2y + Deltar1y * Deltar2x);
        w += phi(dxm,mag[0])*d2phi(dym,mag[0])*Deltar1y*Deltar2y;

        indx = (ix+jx+npsi)%npsi;         
        indy = (iy+jy+npsi)%npsi;         
        r0.x += w*c[indx+indy*npsi+tz*npsi*npsi].x;
        r0.y += w*c[indx+indy*npsi+tz*npsi*npsi].y;
      }
    res[tx+ty*n+tz*n*n].x=r0.x;
    res[tx+ty*n+tz*n*n].y=r0.y;    
}
}
""", "d2t",
)



dtadj_kernel = cp.RawKernel(
    r"""                                 
extern "C" 
{"""+fun_phi+fun_dphi+
r"""
void __global__ dtadj(float2* dt1, float2* dt2, float2 *c, float* r, float* mag, int n,int npsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= n || tz >= ntheta)
        return;  

    
    int ix,iy;
    int indx,indy;
    float x,y;
    float dx,dy;   
    float dxm,dym;   
    float w1,w2;
        
    x = (mag[0]*(tx-n/2)-r[2*tz+1]+(mag[0]-1)/2)+npsi/2;
    y = (mag[0]*(ty-n/2)-r[2*tz+0]+(mag[0]-1)/2)+npsi/2;

    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x-ix;
    dy = y-iy;
    
    float2 dt10 = {};
    float2 dt20 = {};    

    for (int jy=ceil(dy-2*mag[0]);jy<dy+2*mag[0];jy++)
      for (int jx=ceil(dx-2*mag[0]);jx<dx+2*mag[0];jx++)
      {
        dxm = dx-jx;
        dym = dy-jy;
        w1 = -dphi(dym,mag[0]) * phi(dxm,mag[0]);
        w2 = -dphi(dxm,mag[0]) * phi(dym,mag[0]);
        
        indx = (ix+jx+npsi)%npsi;         
        indy = (iy+jy+npsi)%npsi;        
        
        dt10.x += w1*c[indx+indy*npsi+tz*npsi*npsi].x;
        dt10.y += w1*c[indx+indy*npsi+tz*npsi*npsi].y;

        dt20.x += w2*c[indx+indy*npsi+tz*npsi*npsi].x;
        dt20.y += w2*c[indx+indy*npsi+tz*npsi*npsi].y;
      }

    dt1[tx+ty*n+tz*n*n].x=dt10.x;
    dt1[tx+ty*n+tz*n*n].y=dt10.y;    
    dt2[tx+ty*n+tz*n*n].x=dt20.x;
    dt2[tx+ty*n+tz*n*n].y=dt20.y;    
}
}
""", "dtadj",
)


fwdline_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ fwd(float *data, float *u, float *theta, int n, int nz, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= nz || tz >= ntheta)
            return;
            
        float x = 0;
        float y = 0;
        float z = 0;
        int xr = 0;
        int yr = 0;
        float data0 = 0;
        
        float ctheta = __cosf(theta[tz]);
        float stheta = __sinf(theta[tz]);
        
        for (int t = 0; t<n; t++)
        {           
            x = stheta*(t-n/2)+ctheta*(tx-n/2) + n/2;
            y = ctheta*(t-n/2)-stheta*(tx-n/2) + n/2;      

            xr = (int)x;
            yr = (int)y;
            
            // linear interp            
            if ((xr >= 0) & (xr < n - 1) & (yr >= 0) & (yr < n - 1))
            {
                x = x-xr;
                y = y-yr;
                data0 +=u[xr+0+(yr+0)*n+ty*n*n]*(1-x)*(1-y)+
                        u[xr+1+(yr+0)*n+ty*n*n]*(0+x)*(1-y)+
                        u[xr+0+(yr+1)*n+ty*n*n]*(1-x)*(0+y)+
                        u[xr+1+(yr+1)*n+ty*n*n]*(0+x)*(0+y);                        
            }
        }
        data[tx + ty * n + tz * n * nz] = data0*n;        
    }    
""",
    "fwd",
)


adjline_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ adj(float *u, float *data, float *theta, int n, int nz, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nz)
            return;
        float p = 0;
        int pr = 0;        
        float u0 = 0;
        float ctheta = 0;
        float stheta = 0;
            
        for (int t = 0; t<ntheta; t++)
        {
            ctheta = __cosf(theta[t]);
            stheta = __sinf(theta[t]);
            
            p = ctheta*(tx-n/2)-stheta*(ty-n/2) + n/2;
            
            pr = (int)p;
            // linear interp            
            if ((pr >= 0) & (pr < n - 1))
            {
                p = p-pr;
                u0 +=   data[pr+0+tz*n+t*n*nz]*(1-p)+
                        data[pr+1+tz*n+t*n*nz]*(0+p);
                        
            }
        }
        u[tx + ty * n + tz * n * n] = u0*n;        
    }    
""",
    "adj",
)