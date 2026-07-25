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

    const float cx  = n * 0.5f;
    const float x0 =  (tx - cx) / (float)n * __cosf(theta[ty]);
    const float y0 = -(tx - cx) / (float)n * __sinf(theta[ty]);

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
__device__ __forceinline__ int sym_idx(int i, int N)
{
    if (i < 0)   i = -i;
    if (i >= N)  i = 2*N - 2 - i;
    return i;
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

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float pdym    = phi(dy - jy);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = px[jx + 1] * pdym;
            int   idx = indx_s + row_off;

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

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float pdym    = phi(dy - jy);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = px[jx + 1] * pdym;
            int   idx = indx_s + row_off;

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
    + fun_phi
    + r"""
void __global__ sback(float2* g, float2* f, float* r, float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;  // in [0, npsi)
    int ty = blockDim.y * blockIdx.y + threadIdx.y;  // in [0, nzpsi)
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= npsi || ty >= nzpsi || tz >= ntheta) return;

    const float x      = (tx - (npsi-1) * 0.5f + r[2 * tz + 1]) / mag[2 * tz + 1] + (n-1)   * 0.5f;
    const float y      = (ty - (nzpsi-1)* 0.5f + r[2 * tz + 0]) / mag[2 * tz + 0] + (nz-1)  * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * npsi + tz * npsi * nzpsi;
    const int   tz_off = tz * n * nz;

    float px[4];
    for (int jx = -1; jx < 3; jx++) px[jx + 1] = phi(dx - jx);

    float2 g0 = make_float2(0.0f, 0.0f);

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nz) continue;
        float pdym    = phi(dy - jy);
        int   row_off = indy * n + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= n) continue;
            float w   = px[jx + 1] * pdym;
            int   idx = indx + row_off;
            g0.x += w * f[idx].x;
            g0.y += w * f[idx].y;
        }
    }

    g[g_ind] = g0;
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

    const float x        = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y        = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        float d2pdym  = d2phi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w  = d2px[jx + 1] * pdym    * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym   * cross
                     + px[jx + 1]   * d2pdym  * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym  * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym  * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;
            int idx = indx_s + row_off;
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


# -----------------------------------------------------------------------------
# d2s_m: single-pass 2nd directional derivative on (c, r, m).
#
# Bilinear form on directions (c1, Deltar1, Deltam1) and (c2, Deltar2, Deltam2).
# Substitutes the identity d/dm_axis = -tau_axis(pixel) * d/dr_axis into the
# existing d2s formula: use per-pixel effective Delta_r_eff = Deltar - tau*Deltam
# for BOTH direction slots. Because the d2s kernel is quadratic in Delta_r,
# swapping Delta_r -> Delta_r_eff produces exactly:
#     r,r + m,m + r,m + c,r + c,m
# 2nd-derivative contributions in a single kernel launch.
# -----------------------------------------------------------------------------
d2sm_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
void __global__ d2sm(float2* res, float2* c, float2* c1, float2* c2, float* r, float* mag,
                     float* Deltar1, float* Deltam1, float* Deltar2, float* Deltam2,
                     int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x        = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y        = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix       = (int)floorf(x);
    const int   iy       = (int)floorf(y);
    const float dx       = x - ix;
    const float dy       = y - iy;

    const float taux     = tx - (n  - 1) * 0.5f;
    const float tauy     = ty - (nz - 1) * 0.5f;
    const float Deltar1x = Deltar1[2 * tz + 1] - taux * Deltam1[2 * tz + 1];
    const float Deltar1y = Deltar1[2 * tz + 0] - tauy * Deltam1[2 * tz + 0];
    const float Deltar2x = Deltar2[2 * tz + 1] - taux * Deltam2[2 * tz + 1];
    const float Deltar2y = Deltar2[2 * tz + 0] - tauy * Deltam2[2 * tz + 0];
    const float cross    = Deltar1x * Deltar2y + Deltar1y * Deltar2x;
    const int   tz_off   = tz * npsi * nzpsi;

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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        float d2pdym  = d2phi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w  = d2px[jx + 1] * pdym    * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym   * cross
                     + px[jx + 1]   * d2pdym  * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym  * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym  * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;
            int idx = indx_s + row_off;
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
    "d2sm",
)


d2smf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
void __global__ d2sm(float* res, float* c, float* c1, float* c2, float* r, float* mag,
                     float* Deltar1, float* Deltam1, float* Deltar2, float* Deltam2,
                     int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x        = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y        = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix       = (int)floorf(x);
    const int   iy       = (int)floorf(y);
    const float dx       = x - ix;
    const float dy       = y - iy;

    const float taux     = tx - (n  - 1) * 0.5f;
    const float tauy     = ty - (nz - 1) * 0.5f;
    const float Deltar1x = Deltar1[2 * tz + 1] - taux * Deltam1[2 * tz + 1];
    const float Deltar1y = Deltar1[2 * tz + 0] - tauy * Deltam1[2 * tz + 0];
    const float Deltar2x = Deltar2[2 * tz + 1] - taux * Deltam2[2 * tz + 1];
    const float Deltar2y = Deltar2[2 * tz + 0] - tauy * Deltam2[2 * tz + 0];
    const float cross    = Deltar1x * Deltar2y + Deltar1y * Deltar2x;
    const int   tz_off   = tz * npsi * nzpsi;

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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        float d2pdym  = d2phi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w  = d2px[jx + 1] * pdym    * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym   * cross
                     + px[jx + 1]   * d2pdym  * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym  * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym  * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;
            int idx = indx_s + row_off;
            r0 += w  * c[idx];
            r0 -= w1 * c1[idx];
            r0 -= w2 * c2[idx];
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2sm",
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

    const float x        = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y        = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float dym    = dy - jy;
        float pdym   = phi(dym);
        float dpdym  = dphi(dym);
        float d2pdym = d2phi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w  = d2px[jx + 1] * pdym   * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym  * cross
                     + px[jx + 1]   * d2pdym * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym       * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym       * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;

            int idx = indx_s + row_off;
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

    const float x       = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y       = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = dpx[jx + 1] * pdym  * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;

            int   idx = indx_s + row_off;
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


# -----------------------------------------------------------------------------
# ds_m: single-pass (c, r, m) directional derivative.
#
# Same as ds_kernel but with an extra `Deltam` argument. Uses the identity
#     d/dm_axis  =  - tau_axis(pixel) * d/dr_axis
# to fold the m-direction into a per-pixel effective r-direction:
#     Delta_r_eff_axis(pixel) = Delta_r_axis - tau_axis(pixel) * Delta_m_axis
# and then applies the existing r-derivative formula. This costs the same as
# ds_kernel: one kernel launch, no extra global memory traffic. Returns
#     curlySc(c1, r, m) + d/dr curlySc(c, r, m) * Delta_r
#                      + d/dm curlySc(c, r, m) * Delta_m
# -----------------------------------------------------------------------------
dsm_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsm(float2* res, float2* c, float2* c1,
                    float* r, float* mag, float* Deltar, float* Deltam,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x       = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y       = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix      = (int)floorf(x);
    const int   iy      = (int)floorf(y);
    const float dx      = x - ix;
    const float dy      = y - iy;

    // Effective per-pixel r-direction: Delta_r - tau * Delta_m.
    const float taux    = tx - (n  - 1) * 0.5f;
    const float tauy    = ty - (nz - 1) * 0.5f;
    const float Deltarx = Deltar[2 * tz + 1] - taux * Deltam[2 * tz + 1];
    const float Deltary = Deltar[2 * tz + 0] - tauy * Deltam[2 * tz + 0];
    const int   tz_off  = tz * npsi * nzpsi;

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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = dpx[jx + 1] * pdym  * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;

            int   idx = indx_s + row_off;
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
    "dsm",
)


dsmf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsm(float* res, float* c, float* c1,
                    float* r, float* mag, float* Deltar, float* Deltam,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x       = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y       = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix      = (int)floorf(x);
    const int   iy      = (int)floorf(y);
    const float dx      = x - ix;
    const float dy      = y - iy;

    const float taux    = tx - (n  - 1) * 0.5f;
    const float tauy    = ty - (nz - 1) * 0.5f;
    const float Deltarx = Deltar[2 * tz + 1] - taux * Deltam[2 * tz + 1];
    const float Deltary = Deltar[2 * tz + 0] - tauy * Deltam[2 * tz + 0];
    const int   tz_off  = tz * npsi * nzpsi;

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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = dpx[jx + 1] * pdym  * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;

            int   idx = indx_s + row_off;
            r0 -= w * c[idx];
            r0 += w1 * c1[idx];
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "dsm",
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

    const float x       = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y       = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = dpx[jx + 1] * pdym       * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;
            int   idx = indx_s + row_off;
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

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx_s + row_off;

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

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx_s + row_off;
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


# -----------------------------------------------------------------------------
# dsm_adj: adjoint of dcurlySmc (the (c, r, m) directional derivative).
#
# Same as dsadj_kernel but also writes the per-pixel m-adjoint fields
#     dtm1 = -tau_y(ty) * dt1     (∂curlySc / ∂m_y   at each pixel)
#     dtm2 = -tau_x(tx) * dt2     (∂curlySc / ∂m_x   at each pixel)
# so the m-direction adjoint reduces to a plain redot in Python — no
# broadcast multiply of a big (ntheta, nz, n) array by tau.
# -----------------------------------------------------------------------------
dsmadj_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsmadj(float2* f,
                       float2* dt1,  float2* dt2,
                       float2* dtm1, float2* dtm2,
                       float2* c, float2 *g, float* r, float* mag,
                       int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;
    const int   g_ind  = tx + ty * n + tz * n * nz;

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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx_s + row_off;

            dt10.x += w1 * c[idx].x;
            dt10.y += w1 * c[idx].y;
            dt20.x += w2 * c[idx].x;
            dt20.y += w2 * c[idx].y;

            float w3 = px[jx + 1] * pdym;
            atomicAdd(&(f[idx].x), w3 * g0.x);
            atomicAdd(&(f[idx].y), w3 * g0.y);
        }
    }

    const float tauy = ty - (nz - 1) * 0.5f;
    const float taux = tx - (n  - 1) * 0.5f;
    int out_ind = tx + ty * n + tz * n * nz;
    dt1 [out_ind] = dt10;
    dt2 [out_ind] = dt20;
    dtm1[out_ind].x = -tauy * dt10.x;
    dtm1[out_ind].y = -tauy * dt10.y;
    dtm2[out_ind].x = -taux * dt20.x;
    dtm2[out_ind].y = -taux * dt20.y;
}
}
""",
    "dsmadj",
)


dsmadjf_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsmadj(float* f,
                       float* dt1,  float* dt2,
                       float* dtm1, float* dtm2,
                       float* c, float* g, float* r, float* mag,
                       int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;
    const int   g_ind  = tx + ty * n + tz * n * nz;

    float g0 = g[g_ind];

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
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx_s + row_off;
            float cv  = c[idx];
            dt10 += w1 * cv;
            dt20 += w2 * cv;

            float w3 = px[jx + 1] * pdym;
            atomicAdd(&(f[idx]), w3 * g0);
        }
    }

    const float tauy = ty - (nz - 1) * 0.5f;
    const float taux = tx - (n  - 1) * 0.5f;
    int out_ind = tx + ty * n + tz * n * nz;
    dt1 [out_ind] = dt10;
    dt2 [out_ind] = dt20;
    dtm1[out_ind] = -tauy * dt10;
    dtm2[out_ind] = -taux * dt20;
}
}
""",
    "dsmadj",
)


# ── Patch extract / scatter-add for RecFFP ─────────────────────────────────
# Used by rec_ffp_mpi.py's F3 pipeline: gather (nz+2M, n+2M) patches around
# each ipos = round(pos) from the full object grid, then scatter-add per-theta
# patch adjoints back into the full obj-space gradient buffer.
# Semantics:
#     patches[t, py, px] = obj[cy - ipos_y[t] + py, cx - ipos_x[t] + px]
# with (cy, cx) = (nzobj//2 - npad_y//2, nobj//2 - npad_x//2), so ipos=0
# puts the patch center on the object center. Out-of-bounds indices give 0
# on extract and are dropped on scatter-add.

patch_extract_c64_kernel = cp.RawKernel(r'''
extern "C" __global__
void patch_extract_c64(
    const float2* __restrict__ obj,
    const int*    __restrict__ ipos_y,
    const int*    __restrict__ ipos_x,
    float2*       __restrict__ patches,
    int nzobj, int nobj, int npad_y, int npad_x, int ntheta,
    int cy, int cx
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int t  = blockIdx.z;
    if (px >= npad_x || py >= npad_y || t >= ntheta) return;
    int oy = cy - ipos_y[t] + py;
    int ox = cx - ipos_x[t] + px;
    float2 v;
    if (oy >= 0 && oy < nzobj && ox >= 0 && ox < nobj) {
        v = obj[oy * nobj + ox];
    } else {
        v.x = 0.0f; v.y = 0.0f;
    }
    patches[(t * npad_y + py) * npad_x + px] = v;
}
''', 'patch_extract_c64')

patch_extract_f32_kernel = cp.RawKernel(r'''
extern "C" __global__
void patch_extract_f32(
    const float* __restrict__ obj,
    const int*   __restrict__ ipos_y,
    const int*   __restrict__ ipos_x,
    float*       __restrict__ patches,
    int nzobj, int nobj, int npad_y, int npad_x, int ntheta,
    int cy, int cx
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int t  = blockIdx.z;
    if (px >= npad_x || py >= npad_y || t >= ntheta) return;
    int oy = cy - ipos_y[t] + py;
    int ox = cx - ipos_x[t] + px;
    float v = (oy >= 0 && oy < nzobj && ox >= 0 && ox < nobj)
                ? obj[oy * nobj + ox] : 0.0f;
    patches[(t * npad_y + py) * npad_x + px] = v;
}
''', 'patch_extract_f32')

patch_scatter_add_c64_kernel = cp.RawKernel(r'''
extern "C" __global__
void patch_scatter_add_c64(
    float2*       __restrict__ obj,
    const int*    __restrict__ ipos_y,
    const int*    __restrict__ ipos_x,
    const float2* __restrict__ patches,
    int nzobj, int nobj, int npad_y, int npad_x, int ntheta,
    int cy, int cx
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int t  = blockIdx.z;
    if (px >= npad_x || py >= npad_y || t >= ntheta) return;
    int oy = cy - ipos_y[t] + py;
    int ox = cx - ipos_x[t] + px;
    if (oy < 0 || oy >= nzobj || ox < 0 || ox >= nobj) return;
    float2 v = patches[(t * npad_y + py) * npad_x + px];
    float* dst = reinterpret_cast<float*>(&obj[oy * nobj + ox]);
    atomicAdd(&dst[0], v.x);
    atomicAdd(&dst[1], v.y);
}
''', 'patch_scatter_add_c64')

patch_scatter_add_f32_kernel = cp.RawKernel(r'''
extern "C" __global__
void patch_scatter_add_f32(
    float*       __restrict__ obj,
    const int*   __restrict__ ipos_y,
    const int*   __restrict__ ipos_x,
    const float* __restrict__ patches,
    int nzobj, int nobj, int npad_y, int npad_x, int ntheta,
    int cy, int cx
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int t  = blockIdx.z;
    if (px >= npad_x || py >= npad_y || t >= ntheta) return;
    int oy = cy - ipos_y[t] + py;
    int ox = cx - ipos_x[t] + px;
    if (oy < 0 || oy >= nzobj || ox < 0 || ox >= nobj) return;
    atomicAdd(&obj[oy * nobj + ox],
              patches[(t * npad_y + py) * npad_x + px]);
}
''', 'patch_scatter_add_f32')
