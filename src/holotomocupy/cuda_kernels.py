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

    float M_PI = 3.141592653589793238f;
    float2 g0, g0t;
    float w, coeff0;
    float w0, w1, x0, y0, coeff1;
    int ell0, ell1, g_ind, f_ind, f_indx, f_indy;

    g_ind = tx + tz * n + ty * n * nz;  // swapped axes

    if (dir == 0) g0 = {};
    else g0 = g[g_ind];

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

            f_indx = (n + ell0 + 2 * n) % (2 * n);
            f_indy = (n + ell1 + 2 * n) % (2 * n);
            f_ind = f_indx + (2 * n) * f_indy + tz * (2 * n) * (2 * n);

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

fun_phi = r"""
__device__ float phi(float t, float m)
{
    int sgn = 0;
    float w = 0;

    t /= m;

    if (-2 < t && t <= -1)
    {
        w = (t + 2) * (t + 2) * (t + 2);
    }
    else if (-1 < t && t <= 1)
    {
        sgn = (t > 0) ? 1 : ((t < 0) ? -1 : 0);
        w = 4 - 6 * t * t + 3 * t * t * t * sgn;
    }
    else if (1 < t && t <= 2)
    {
        w = (2 - t) * (2 - t) * (2 - t);
    }
    else
    {
        w = 0;
    }

    return w;
}
"""

fun_dphi = r"""
__device__ float dphi(float t, float m)
{
    int sgn = 0;
    float w = 0;

    t /= m;

    if (-2 < t && t <= -1)
    {
        w = 3 * (t + 2) * (t + 2);
    }
    else if (-1 < t && t <= 1)
    {
        sgn = (t > 0) ? 1 : ((t < 0) ? -1 : 0);
        w = -12 * t + 9 * t * t * sgn;
    }
    else if (1 < t && t <= 2)
    {
        w = -3 * (2 - t) * (2 - t);
    }
    else
    {
        w = 0;
    }

    w /= m;
    return w;
}
"""

fun_d2phi = r"""
__device__ float d2phi(float t, float m)
{
    int sgn = 0;
    float w = 0;

    t /= m;

    if (-2 < t && t <= -1)
    {
        w = 6 * (t + 2);
    }
    else if (-1 < t && t <= 1)
    {
        sgn = (t > 0) ? 1 : ((t < 0) ? -1 : 0);
        w = -12 + 18 * t * sgn;
    }
    else if (1 < t && t <= 2)
    {
        w = 6 * (2 - t);
    }
    else
    {
        w = 0;
    }

    w /= (m * m);
    return w;
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

    int ix, iy;
    int f_indx, f_indy, f_ind, g_ind;
    float x, y;
    float dx, dy;
    float dxm, dym;
    float w;
    float2 g0;

    x = (mag[0] * (tx - n / 2) - r[2 * tz + 1] + (mag[0] - 1) / 2) + npsi / 2;
    y = (mag[0] * (ty - nz / 2) - r[2 * tz + 0] + (mag[0] - 1) / 2) + nzpsi / 2;

    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x - ix;
    dy = y - iy;

    g_ind = tx + ty * n + tz * n * nz;

    if (dir == 0) g0 = {};
    else g0 = g[g_ind];

    for (int jy = ceil(dy - 2 * mag[0]); jy < dy + 2 * mag[0]; jy++)
        for (int jx = ceil(dx - 2 * mag[0]); jx < dx + 2 * mag[0]; jx++)
        {
            dxm = dx - jx;
            dym = dy - jy;

            w = phi(dxm, mag[0]) * phi(dym, mag[0]);

            f_indx = (ix + jx + npsi) % npsi;
            f_indy = (iy + jy + nzpsi) % nzpsi;
            f_ind = f_indx + f_indy * npsi + tz * npsi * nzpsi;

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

    int ix, iy;
    int indx, indy;
    float x, y;
    float dx, dy;
    float dxm, dym;
    float Deltarx, Deltary;
    float w;
    float2 r0 = {};

    x = (mag[0] * (tx - n / 2) - r[2 * tz + 1] + (mag[0] - 1) / 2) + npsi / 2;
    y = (mag[0] * (ty - nz / 2) - r[2 * tz + 0] + (mag[0] - 1) / 2) + nzpsi / 2;

    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x - ix;
    dy = y - iy;

    Deltarx = Deltar[2 * tz + 1];
    Deltary = Deltar[2 * tz + 0];

    for (int jy = ceil(dy - 2 * mag[0]); jy < dy + 2 * mag[0]; jy++)
        for (int jx = ceil(dx - 2 * mag[0]); jx < dx + 2 * mag[0]; jx++)
        {
            dxm = dx - jx;
            dym = dy - jy;

            w = dphi(dxm, mag[0]) * phi(dym, mag[0]) * Deltarx
              + dphi(dym, mag[0]) * phi(dxm, mag[0]) * Deltary;

            indx = (ix + jx + npsi) % npsi;
            indy = (iy + jy + nzpsi) % nzpsi;

            int idx = indx + indy * npsi + tz * npsi * nzpsi;
            r0.x -= w * c[idx].x;
            r0.y -= w * c[idx].y;
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

    int ix, iy;
    int indx, indy;
    float x, y;
    float dx, dy;
    float dxm, dym;
    float Deltar1x, Deltar1y;
    float Deltar2x, Deltar2y;
    float w;
    float2 r0 = {};

    x = (mag[0] * (tx - n / 2) - r[2 * tz + 1] + (mag[0] - 1) / 2) + npsi / 2;
    y = (mag[0] * (ty - nz / 2) - r[2 * tz + 0] + (mag[0] - 1) / 2) + nzpsi / 2;

    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x - ix;
    dy = y - iy;

    Deltar1x = Deltar1[2 * tz + 1];
    Deltar1y = Deltar1[2 * tz + 0];
    Deltar2x = Deltar2[2 * tz + 1];
    Deltar2y = Deltar2[2 * tz + 0];

    for (int jy = ceil(dy - 2 * mag[0]); jy < dy + 2 * mag[0]; jy++)
        for (int jx = ceil(dx - 2 * mag[0]); jx < dx + 2 * mag[0]; jx++)
        {
            dxm = dx - jx;
            dym = dy - jy;

            w  = d2phi(dxm, mag[0]) * phi(dym, mag[0]) * Deltar1x * Deltar2x;
            w += dphi(dxm, mag[0]) * dphi(dym, mag[0])
                 * (Deltar1x * Deltar2y + Deltar1y * Deltar2x);
            w += phi(dxm, mag[0]) * d2phi(dym, mag[0]) * Deltar1y * Deltar2y;

            indx = (ix + jx + npsi) % npsi;
            indy = (iy + jy + nzpsi) % nzpsi;

            int idx = indx + indy * npsi + tz * npsi * nzpsi;
            r0.x += w * c[idx].x;
            r0.y += w * c[idx].y;
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

    int ix, iy;
    int indx, indy;
    float x, y;
    float dx, dy;
    float dxm, dym;
    float w1, w2;

    x = (mag[0] * (tx - n / 2) - r[2 * tz + 1] + (mag[0] - 1) / 2) + npsi / 2;
    y = (mag[0] * (ty - nz / 2) - r[2 * tz + 0] + (mag[0] - 1) / 2) + nzpsi / 2;

    ix = (int)floorf(x);
    iy = (int)floorf(y);

    dx = x - ix;
    dy = y - iy;

    float2 dt10 = {};
    float2 dt20 = {};

    for (int jy = ceil(dy - 2 * mag[0]); jy < dy + 2 * mag[0]; jy++)
        for (int jx = ceil(dx - 2 * mag[0]); jx < dx + 2 * mag[0]; jx++)
        {
            dxm = dx - jx;
            dym = dy - jy;

            w1 = -dphi(dym, mag[0]) * phi(dxm, mag[0]);
            w2 = -dphi(dxm, mag[0]) * phi(dym, mag[0]);

            indx = (ix + jx + npsi) % npsi;
            indy = (iy + jy + nzpsi) % nzpsi;

            int idx = indx + indy * npsi + tz * npsi * nzpsi;

            dt10.x += w1 * c[idx].x;
            dt10.y += w1 * c[idx].y;

            dt20.x += w2 * c[idx].x;
            dt20.y += w2 * c[idx].y;
        }

    dt1[tx + ty * n + tz * n * nz] = dt10;
    dt2[tx + ty * n + tz * n * nz] = dt20;
}
}
""",
    "dtadj",
)