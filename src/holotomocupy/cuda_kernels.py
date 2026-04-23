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

pad_fwd_kernel = cp.RawKernel(
    r"""
extern "C" void __global__ pad_fwd(float2* __restrict__ g,
                                    const float2* __restrict__ f,
                                    int n, int nz, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= 2*n || ty >= 2*nz || tz >= ntheta) return;

    int txx = (tx < n/2)       ? (n/2  - tx - 1)         :
              (tx >= n + n/2)   ? (2*n  - tx + n/2  - 1)  : (tx - n/2);
    int tyy = (ty < nz/2)      ? (nz/2 - ty - 1)         :
              (ty >= nz + nz/2) ? (2*nz - ty + nz/2 - 1)  : (ty - nz/2);

    g[tz*2*n*2*nz + ty*2*n + tx] = f[tz*n*nz + tyy*n + txx];
}
""",
    "pad_fwd",
)

pad_adj_kernel = cp.RawKernel(
    r"""
/* Adjoint of pad_fwd: launch over f (n x nz).
   Each f[tx,ty] gathers from exactly 4 symmetric locations in g — no atomics. */
extern "C" void __global__ pad_adj(const float2* __restrict__ g,
                                    float2* __restrict__ f,
                                    int n, int nz, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n || ty >= nz || tz >= ntheta) return;

    int gx_c = tx + n/2;
    int gx_m = (tx < n/2) ? (n/2 - 1 - tx) : (2*n + n/2 - 1 - tx);
    int gy_c = ty + nz/2;
    int gy_m = (ty < nz/2) ? (nz/2 - 1 - ty) : (2*nz + nz/2 - 1 - ty);

    const float2* base = g + tz * 2*n * 2*nz;
    float2 v0 = base[gy_c*2*n + gx_c];
    float2 v1 = base[gy_c*2*n + gx_m];
    float2 v2 = base[gy_m*2*n + gx_c];
    float2 v3 = base[gy_m*2*n + gx_m];
    f[tz*n*nz + ty*n + tx] = {v0.x+v1.x+v2.x+v3.x, v0.y+v1.y+v2.y+v3.y};
}
""",
    "pad_adj",
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

    const float mag0   = mag[tz];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * n + tz * n * nz;
    const int   tz_off = tz * npsi * nzpsi;

    // Precompute x-direction phi values once (4 evals instead of 16).
    float px[4];
    for (int jx = -1; jx < 3; jx++) px[jx + 1] = phi(dx - jx);

    float2 g0 = (dir == 0) ? make_float2(0.0f, 0.0f) : g[g_ind];

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

            float w   = px[jx + 1] * pdym;
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

    const float mag0   = mag[tz];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * n + tz * n * nz;
    const int   tz_off = tz * npsi * nzpsi;

    // Precompute x-direction phi values once (4 evals instead of 16).
    float px[4];
    for (int jx = -1; jx < 3; jx++) px[jx + 1] = phi(dx - jx);

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

            float w   = px[jx + 1] * pdym;
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

    const float mag0   = mag[tz];
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








d2s_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
void __global__ d2s(float2* res, float2* c, float2* c1, float2* c2, float* r, float* mag,
                    float* Deltar1, float* Deltar2,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0     = mag[tz];
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

    // Precompute x-direction phi, dphi, d2phi values (12 evals instead of 48).
    float px[4], dpx[4], d2px[4];
    for (int jx = -1; jx < 3; jx++) {
        float d   = dx - jx;
        px[jx + 1]   = phi(d);
        dpx[jx + 1]  = dphi(d);
        d2px[jx + 1] = d2phi(d);
    }

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

            float w  = d2px[jx + 1] * pdym    * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym   * cross
                     + px[jx + 1]   * d2pdym  * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym  * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym  * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;
            int idx = indx + row_off;
            r0.x += w  * c[idx].x;
            r0.y += w  * c[idx].y;
            r0.x -= w1 * c1[idx].x;
            r0.y -= w1 * c1[idx].y;
            r0.x -= w2 * c2[idx].x;
            r0.y -= w2 * c2[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2s",
)


d2sf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
void __global__ d2s(float* res, float* c, float* c1, float* c2, float* r, float* mag,
                    float* Deltar1, float* Deltar2,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0     = mag[tz];
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

    // Precompute x-direction phi, dphi, d2phi values (12 evals instead of 48).
    float px[4], dpx[4], d2px[4];
    for (int jx = -1; jx < 3; jx++) {
        float d   = dx - jx;
        px[jx + 1]   = phi(d);
        dpx[jx + 1]  = dphi(d);
        d2px[jx + 1] = d2phi(d);
    }

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

            float w  = d2px[jx + 1] * pdym   * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym  * cross
                     + px[jx + 1]   * d2pdym * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym       * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym       * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;

            int idx = indx + row_off;
            r0 += w  * c[idx];
            r0 -= w1 * c1[idx];
            r0 -= w2 * c2[idx];
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2s",
)




ds_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ ds(float2* res, float2* c, float2* c1, float* r, float* mag, float* Deltar,
                   int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0    = mag[tz];
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

    // Precompute x-direction phi and dphi values (8 evals instead of 32).
    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

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

            float w   = dpx[jx + 1] * pdym  * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;

            int   idx = indx + row_off;
            r0.x -= w * c[idx].x;
            r0.y -= w * c[idx].y;
            r0.x += w1 * c1[idx].x;
            r0.y += w1 * c1[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "ds",
)


dsf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ ds(float* res, float* c, float* c1, float* r, float* mag, float* Deltar,
                   int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0    = mag[tz];
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

    // Precompute x-direction phi and dphi values (8 evals instead of 32).
    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

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

            float w   = dpx[jx + 1] * pdym       * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;
            int   idx = indx + row_off;
            r0 -= w  * c[idx];
            r0 += w1 * c1[idx];
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "ds",
)


dsadj_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsadj(float2* f, float2* dt1, float2* dt2, float2* c, float2 *g, float* r, float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[tz];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;
    const int   g_ind  = tx + ty * n + tz * n * nz;

    // Precompute x-direction phi and dphi values (8 evals instead of 32).
    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

    float2 g0 = g[g_ind];
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

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx + row_off;

            dt10.x += w1 * c[idx].x;
            dt10.y += w1 * c[idx].y;
            dt20.x += w2 * c[idx].x;
            dt20.y += w2 * c[idx].y;

            float w3 = px[jx + 1] * pdym;
            atomicAdd(&(f[idx].x), w3 * g0.x);
            atomicAdd(&(f[idx].y), w3 * g0.y);
        }
    }

    int out_ind = tx + ty * n + tz * n * nz;
    dt1[out_ind] = dt10;
    dt2[out_ind] = dt20;
}
}
""",
    "dsadj",
)


dsadjf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsadj(float* f, float* dt1, float* dt2, float* c, float* g, float* r,  float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float mag0   = mag[tz];
    const float half   = (mag0 - 1.0f) / 2.0f;
    const float x      = (mag0 * (tx - n / 2) - r[2 * tz + 1] + half) + npsi  / 2;
    const float y      = (mag0 * (ty - nz / 2) - r[2 * tz + 0] + half) + nzpsi / 2;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;

    const int g_ind  = tx + ty * n + tz * n * nz;
    float g0 = g[g_ind];

    // Precompute x-direction phi and dphi values (8 evals instead of 32).
    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

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

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx + row_off;
            float cv  = c[idx];
            dt10 += w1 * cv;
            dt20 += w2 * cv;

            float w3 = px[jx + 1] * pdym;
            atomicAdd(&(f[idx]), w3 * g0);
        }
    }

    int out_ind = tx + ty * n + tz * n * nz;
    dt1[out_ind] = dt10;
    dt2[out_ind] = dt20;
}
}
""",
    "dsadj",
)

