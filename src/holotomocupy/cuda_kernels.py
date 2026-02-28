import cupy as cp

gather_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void gather(float2* g, float2* f, float* theta, int m, float* mu,
                                  int n, int ntheta, int nz, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= ntheta || tz >= nz) return;

    const float PI     = 3.141592653589793238f;
    const int   twon   = 2 * n;
    const float ftwon  = (float)twon;
    const float mu0    = mu[0];
    const float coeff0 = PI / mu0;
    const float coeff1 = -PI * PI / mu0;
    const float inv_twon = 1.0f / ftwon;

    const float x0 =  (tx - n / 2) / (float)n * __cosf(theta[ty]);
    const float y0 = -(tx - n / 2) / (float)n * __sinf(theta[ty]);

    const int g_ind = tx + tz * n + ty * n * nz;  // swapped axes
    float2 g0 = (dir == 0) ? make_float2(0.0f, 0.0f) : g[g_ind];

    const int base_x  = (int)floorf(ftwon * x0) - m;
    const int base_y  = (int)floorf(ftwon * y0) - m;
    const int tz_off  = tz * twon * twon;
    const int len     = 2 * m + 1;

    // Precompute x-direction exponential factors once.
    // Reduces expf calls from (2m+1)^2 to 2*(2m+1).
    float ex[32];  // 2*m+1 entries; m is small (typically 4-5)
    for (int i0 = 0; i0 < len; i0++) {
        float w0 = (base_x + i0) * inv_twon - x0;
        ex[i0] = __expf(coeff1 * w0 * w0);
    }

    for (int i1 = 0; i1 < len; i1++)
    {
        int   ell1    = base_y + i1;
        float w1      = ell1 * inv_twon - y0;
        float ey      = coeff0 * __expf(coeff1 * w1 * w1);
        int   f_indy  = (n + ell1 + twon) % twon;
        int   row_off = twon * f_indy + tz_off;

        for (int i0 = 0; i0 < len; i0++)
        {
            float w    = ex[i0] * ey;
            int   ell0 = base_x + i0;
            int   f_ind = (n + ell0 + twon) % twon + row_off;

            if (dir == 0)
            {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            }
            else
            {
                atomicAdd(&(f[f_ind].x), w * g0.x);
                atomicAdd(&(f[f_ind].y), w * g0.y);
            }
        }
    }

    if (dir == 0)
    {
        g[g_ind].x = g0.x / n;
        g[g_ind].y = g0.y / n;
    }
}
""",
    "gather",
)

pad_kernel = cp.RawKernel(
    r"""
extern "C" void __global__ pad(float2* g, float2* f, int n, int nz, int ntheta, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    int txx, tyy;

    if (tx >= 2 * n || ty >= 2 * nz || tz >= ntheta) return;

    if (ty < nz / 2) tyy = nz / 2 - ty - 1;
    else if (ty >= nz + nz / 2) tyy = 2 * nz - ty + nz / 2 - 1;
    else tyy = ty - nz / 2;

    if (tx < n / 2) txx = n / 2 - tx - 1;
    else if (tx >= n + n / 2) txx = 2 * n - tx + n / 2 - 1;
    else txx = tx - n / 2;

    int id1 = tz * 2 * n * 2 * nz + ty * 2 * n + tx;
    int id2 = tz * n * nz + tyy * n + txx;

    if (dir == 0)
    {
        g[id1] = f[id2];
    }
    else
    {
        atomicAdd(&f[id2].x, g[id1].x);
        atomicAdd(&f[id2].y, g[id1].y);
    }
}
""",
    "pad",
)

# B-spline basis functions and derivatives.
# Use fabsf instead of an integer sgn variable to avoid branching.
fun_phi = r"""
__device__ __forceinline__ float phi(float t)
{
    if (-2.0f < t && t <= -1.0f) return (t + 2.0f) * (t + 2.0f) * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return 4.0f - 6.0f*t*t + 3.0f*fabsf(t)*t*t;
    if ( 1.0f < t && t <=  2.0f) return (2.0f - t) * (2.0f - t) * (2.0f - t);
    return 0.0f;
}
"""

fun_dphi = r"""
__device__ __forceinline__ float dphi(float t)
{
    if (-2.0f < t && t <= -1.0f) return 3.0f * (t + 2.0f) * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return -12.0f*t + 9.0f*fabsf(t)*t;
    if ( 1.0f < t && t <=  2.0f) return -3.0f * (2.0f - t) * (2.0f - t);
    return 0.0f;
}
"""

fun_d2phi = r"""
__device__ __forceinline__ float d2phi(float t)
{
    if (-2.0f < t && t <= -1.0f) return 6.0f * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return -12.0f + 18.0f*fabsf(t);
    if ( 1.0f < t && t <=  2.0f) return 6.0f * (2.0f - t);
    return 0.0f;
}
"""

s_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + r"""
void __global__ s(float2* g, float2* f, float* r, float* mag,
                  int n, int npsi, int nz, int nzpsi, int ntheta, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[0];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * n + tz * n * nz;
    const int   tz_off = tz * npsi * nzpsi;

    float2 g0 = (dir == 0) ? make_float2(0.0f, 0.0f) : g[g_ind];

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float pdym   = phi(dy - jy);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float w   = phi(dx - jx) * pdym;
            int   idx = indx + row_off;

            if (dir == 0)
            {
                g0.x += w * f[idx].x;
                g0.y += w * f[idx].y;
            }
            else
            {
                atomicAdd(&(f[idx].x), w * g0.x);
                atomicAdd(&(f[idx].y), w * g0.y);
            }
        }
    }

    if (dir == 0) g[g_ind] = g0;
}
}
""",
    "s",
)


sf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + r"""
void __global__ s(float* g, float* f, float* r, float* mag,
                  int n, int npsi, int nz, int nzpsi, int ntheta, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[0];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * n + tz * n * nz;
    const int   tz_off = tz * npsi * nzpsi;

    float g0 = (dir == 0) ? 0.0f : g[g_ind];

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float pdym    = phi(dy - jy);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float w   = phi(dx - jx) * pdym;
            int   idx = indx + row_off;

            if (dir == 0)
                g0 += w * f[idx];
            else
                atomicAdd(&(f[idx]), w * g0);
        }
    }

    if (dir == 0) g[g_ind] = g0;
}
}
""",
    "s",
)

dt_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dt(float2* res, float2* c, float* r, float* mag, float* Deltar,
                   int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0    = mag[0];
    const float half    = (mag0 - 1.0f) / 2.0f;
    const float x       = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y       = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix      = (int)floorf(x);
    const int   iy      = (int)floorf(y);
    const float dx      = x - ix;
    const float dy      = y - iy;
    const float Deltarx = Deltar[2 * tz + 1];
    const float Deltary = Deltar[2 * tz + 0];
    const int   tz_off  = tz * npsi * nzpsi;

    float2 r0 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float dxm = dx - jx;
            float w   = dphi(dxm) * pdym  * Deltarx
                      + dpdym      * phi(dxm) * Deltary;
            int   idx = indx + row_off;
            r0.x -= w * c[idx].x;
            r0.y -= w * c[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "dt",
)


dtf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dt(float* res, float* c, float* r, float* mag, float* Deltar,
                   int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0    = mag[0];
    const float half    = (mag0 - 1.0f) / 2.0f;
    const float x       = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y       = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix      = (int)floorf(x);
    const int   iy      = (int)floorf(y);
    const float dx      = x - ix;
    const float dy      = y - iy;
    const float Deltarx = Deltar[2 * tz + 1];
    const float Deltary = Deltar[2 * tz + 0];
    const int   tz_off  = tz * npsi * nzpsi;

    float r0 = 0.0f;

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float dxm = dx - jx;
            float w   = dphi(dxm) * pdym  * Deltarx
                      + dpdym      * phi(dxm) * Deltary;
            r0 -= w * c[indx + row_off];
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "dt",
)

d2t_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
void __global__ d2t(float2* res, float2* c, float* r, float* mag,
                    float* Deltar1, float* Deltar2,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0     = mag[0];
    const float half     = (mag0 - 1.0f) / 2.0f;
    const float x        = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y        = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix       = (int)floorf(x);
    const int   iy       = (int)floorf(y);
    const float dx       = x - ix;
    const float dy       = y - iy;
    const float Deltar1x = Deltar1[2 * tz + 1];
    const float Deltar1y = Deltar1[2 * tz + 0];
    const float Deltar2x = Deltar2[2 * tz + 1];
    const float Deltar2y = Deltar2[2 * tz + 0];
    const float cross    = Deltar1x * Deltar2y + Deltar1y * Deltar2x;
    const int   tz_off   = tz * npsi * nzpsi;

    float2 r0 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        float d2pdym  = d2phi(dym);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float dxm  = dx - jx;
            float pdxm  = phi(dxm);
            float dpdxm = dphi(dxm);
            float w  = d2phi(dxm) * pdym   * Deltar1x * Deltar2x
                     + dpdxm      * dpdym   * cross
                     + pdxm       * d2pdym  * Deltar1y * Deltar2y;
            int idx = indx + row_off;
            r0.x += w * c[idx].x;
            r0.y += w * c[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2t",
)


d2tf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
void __global__ d2t(float* res, float* c, float* r, float* mag,
                    float* Deltar1, float* Deltar2,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0     = mag[0];
    const float half     = (mag0 - 1.0f) / 2.0f;
    const float x        = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y        = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix       = (int)floorf(x);
    const int   iy       = (int)floorf(y);
    const float dx       = x - ix;
    const float dy       = y - iy;
    const float Deltar1x = Deltar1[2 * tz + 1];
    const float Deltar1y = Deltar1[2 * tz + 0];
    const float Deltar2x = Deltar2[2 * tz + 1];
    const float Deltar2y = Deltar2[2 * tz + 0];
    const float cross    = Deltar1x * Deltar2y + Deltar1y * Deltar2x;
    const int   tz_off   = tz * npsi * nzpsi;

    float r0 = 0.0f;

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float dym    = dy - jy;
        float pdym   = phi(dym);
        float dpdym  = dphi(dym);
        float d2pdym = d2phi(dym);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float dxm  = dx - jx;
            float pdxm  = phi(dxm);
            float dpdxm = dphi(dxm);
            float w  = d2phi(dxm) * pdym   * Deltar1x * Deltar2x
                     + dpdxm      * dpdym   * cross
                     + pdxm       * d2pdym  * Deltar1y * Deltar2y;
            r0 += w * c[indx + row_off];
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2t",
)

dtadj_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dtadj(float2* dt1, float2* dt2, float2* c, float* r, float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[0];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;

    float2 dt10 = {};
    float2 dt20 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float dxm = dx - jx;
            float w1  = -dpdym    * phi(dxm);
            float w2  = -dphi(dxm) * pdym;
            int   idx = indx + row_off;

            dt10.x += w1 * c[idx].x;
            dt10.y += w1 * c[idx].y;
            dt20.x += w2 * c[idx].x;
            dt20.y += w2 * c[idx].y;
        }
    }

    int out_ind = tx + ty * n + tz * n * nz;
    dt1[out_ind] = dt10;
    dt2[out_ind] = dt20;
}
}
""",
    "dtadj",
)


dtadjf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dtadj(float* dt1, float* dt2, float* c, float* r, float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[0];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;

    float dt10 = 0.0f;
    float dt20 = 0.0f;

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float dxm = dx - jx;
            float w1  = -dpdym     * phi(dxm);
            float w2  = -dphi(dxm) * pdym;
            float cv  = c[indx + row_off];
            dt10 += w1 * cv;
            dt20 += w2 * cv;
        }
    }

    int out_ind = tx + ty * n + tz * n * nz;
    dt1[out_ind] = dt10;
    dt2[out_ind] = dt20;
}
}
""",
    "dtadj",
)


# extra for paganin

fun_phi_back = r"""
__device__ __forceinline__ float phi(float t, float m)
{
    t /= m;
    if (-2.0f < t && t <= -1.0f) return (t + 2.0f) * (t + 2.0f) * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return 4.0f - 6.0f*t*t + 3.0f*fabsf(t)*t*t;
    if ( 1.0f < t && t <=  2.0f) return (2.0f - t) * (2.0f - t) * (2.0f - t);
    return 0.0f;
}
"""

sback_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi_back
    + r"""
void __global__ sback(float2* g, float2* f, float* r, float* mag,
                  int n, int npsi, int nz, int nzpsi, int ntheta, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[0];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * n + tz * n * nz;
    const int   tz_off = tz * npsi * nzpsi;
    const float span   = 2.0f * mag0;

    float2 g0 = (dir == 0) ? make_float2(0.0f, 0.0f) : g[g_ind];

    for (int jy = (int)ceilf(dy - span); jy < (int)(dy + span); jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nzpsi) continue;
        float pdym    = phi(dy - jy, mag0);
        int   row_off = indy * npsi + tz_off;

        for (int jx = (int)ceilf(dx - span); jx < (int)(dx + span); jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= npsi) continue;

            float w   = phi(dx - jx, mag0) * pdym;
            int   idx = indx + row_off;

            if (dir == 0)
            {
                g0.x += w * f[idx].x;
                g0.y += w * f[idx].y;
            }
            else
            {
                atomicAdd(&(f[idx].x), w * g0.x);
                atomicAdd(&(f[idx].y), w * g0.y);
            }
        }
    }

    if (dir == 0) g[g_ind] = g0;
}
}
""",
    "sback",
)
