import cupy as cp


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
