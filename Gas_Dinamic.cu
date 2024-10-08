#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include "Header.h"
#include <omp.h>
#include <mutex>

using namespace std;

//__device__ int sign_(const double& x);
//__device__ double minmod_(double x, double y);
//__device__ double linear_(double x1, double t1, double x2, double t2, double x3, double t3, double y);
//__device__ void linear2_(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
//    double& A, double& B);


__device__ double minmod_(double x, double y)
{
    if (sign_(x) + sign_(y) == 0)
    {
        return 0.0;
    }
    else
    {
        return   ((sign_(x) + sign_(y)) / 2.0) * min(fabs(x), fabs(y));  ///minmod
        //return (2*x*y)/(x + y);   /// vanleer
    }
}

__device__ double linear_(double x1, double t1, double x2, double t2, double x3, double t3, double y)
{
    double d = minmod_((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    return  (d * (y - x2) + t2);
}

__device__ void linear2_(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B)
{
    // ������� �������� - ����������� - �� ������� �� ����
    double d = minmod_((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    A = (d * (y1 - x2) + t2);
    B = (d * (y2 - x2) + t2);
    //printf("%lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf \n", x1, t1, x2, t2, x3, t3, y1, y2, A, B);
    return;
}

__device__ int sign_(const double& x)
{
    if (x > 0)
    {
        return 1;
    }
    else if (x < 0)
    {
        return  -1;
    }
    else
    {
        return 0;
    }
}


__device__ void TVD(const double2& s_1, const double2& s_2, const double2& s_3, const double2& s_4, const double2& s_5,//
    const double2& s_6, const double2& s_7, const double2& s_8, const double2& s_9, double2& s12,//
    double2& s13, double2& s14, double2& s15, double2& s21, double2& s31, double2& s41, double2& s51, double ddx, double ddy, bool zero)
{
    // ��� ��������� � �������� zero ������ ���� ����� true
    linear2_(-ddx, s_4.x, 0.0, s_1.x, ddx, s_2.x, -ddx / 2.0, ddx / 2.0, s14.x, s12.x);
    if (zero == true)
    {
        if (s14.x <= 0.0)
        {
            s14.x = s_1.x;
        }
        if (s12.x <= 0.0)
        {
            s12.x = s_1.x;
        }
    }
   
    linear2_(-ddx, s_4.y, 0.0, s_1.y, ddx, s_2.y, -ddx / 2.0, ddx / 2.0, s14.y, s12.y);
    if (zero == true)
    {
        if (s14.y <= 0.0)
        {
            s14.y = s_1.y;
        }
        if (s12.y <= 0.0)
        {
            s12.y = s_1.y;
        }
    }

    linear2_(-ddy, s_3.x, 0.0, s_1.x, ddy, s_5.x, -ddy / 2.0, ddy / 2.0, s13.x, s15.x);
    if (zero == true)
    {
        if (s13.x <= 0.0)
        {
            s13.x = s_1.x;
        }
        if (s15.x <= 0.0)
        {
            s15.x = s_1.x;
        }
    }

    linear2_(-ddy, s_3.y, 0.0, s_1.y, ddx, s_5.y, -ddy / 2.0, ddy / 2.0, s13.y, s15.y);
    if (zero == true)
    {
        if (s13.y <= 0.0)
        {
            s13.y = s_1.y;
        }
        if (s15.y <= 0.0)
        {
            s15.y = s_1.y;
        }
    }

    s21.x = linear_(0.0, s_1.x, ddx, s_2.x, 2.0 * ddx, s_6.x, ddx / 2.0);
    if (s21.x <= 0) s21.x = s_2.x;
    s21.y = linear_(0.0, s_1.y, ddx, s_2.y, 2.0 * ddx, s_6.y, ddx / 2.0);
    if (s21.y <= 0) s21.y = s_2.y;

    s41.x = linear_(0.0, s_1.x, - ddx, s_4.x, - 2.0 * ddx, s_8.x, - ddx / 2.0);
    if (s41.x <= 0 && zero == true) s41.x = s_4.x;
    s41.y = linear_(0.0, s_1.y, - ddx, s_4.y, - 2.0 * ddx, s_8.y, - ddx / 2.0);
    if (s41.y <= 0 && zero == true) s41.y = s_4.y;

    s31.x = linear_(0.0, s_1.x, - ddy, s_3.x, - 2.0 * ddy, s_7.x, - ddy / 2.0);
    if (s31.x <= 0 && zero == true) s31.x = s_3.x;
    s31.y = linear_(0.0, s_1.y, - ddy, s_3.y, - 2.0 * ddy, s_7.y, - ddy / 2.0);
    if (s31.y <= 0 && zero == true) s31.y = s_3.y;

    s51.x = linear_(0.0, s_1.x, + ddy, s_5.x, + 2.0 * ddy, s_9.x, + ddy / 2.0);
    if (s51.x <= 0) s51.x = s_5.x;
    s51.y = linear_(0.0, s_1.y, + ddy, s_5.y, + 2.0 * ddy, s_9.y, + ddy / 2.0);
    if (s51.y <= 0)  s51.y = s_5.y;

    return;
}


__device__ double HLLC_Korolkov_2D(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double n1, const double n2, double2& Ps, double2& Pu, const double rad)
{
    double u_L, v_L;
    double u_R, v_R;

    double ro1 = Ls.x;
    double u1 = Lu.x;
    double v1 = Lu.y;
    double p1 = Ls.y;

    double ro2 = Rs.x;
    double u2 = Ru.x;
    double v2 = Ru.y;
    double p2 = Rs.y;

    double t1 = -n2;    // ����������� ������
    double t2 = n1;

    u_L = u1 * n1 + v1 * n2;
    v_L = u1 * t1 + v1 * t2;

    u_R = u2 * n1 + v2 * n2;
    v_R = u2 * t1 + v2 * t2;

    double cL = sqrt(ga * p1 / ro1);
    double cR = sqrt(ga * p2 / ro2);

    /*double SL = min((u_L - cL), (u_R - cR));
    double SR = max((u_L + cL), (u_R + cR));*/

    double SL = min(u_L, u_R) - max(cL, cR);
    double SR = max(u_L, u_R) + max(cL, cR);

   /* double SL = min(u_L, u_R) - max(cL, cR);
    double SR = max(u_L, u_R) + max(cL, cR);*/

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    if (SL >= 0.0)
    {
        Ps.x = ro1 * u_L;
        Ps.y = ( ga * p1/(g1) + 0.5 * ro1 * (kv(u1) + kv(v1)) ) * u_L;
        Pu.x = (ro1 * u_L * u_L + p1) * n1 + (ro1 * u_L * v_L) * t1;
        Pu.y = (ro1 * u_L * u_L + p1) * n2 + (ro1 * u_L * v_L) * t2;
        return time;
    }
    else if (SR <= 0.0)
    {
        Ps.x = ro2 * u_R;
        Ps.y = (ga * p2 / (g1) + 0.5 * ro2 * (kv(u2) + kv(v2))) * u_R;
        Pu.x = (ro2 * u_R * u_R + p2) * n1 + (ro2 * u_R * v_R) * t1;
        Pu.y = (ro2 * u_R * u_R + p2) * n2 + (ro2 * u_R * v_R) * t2;
        return time;
    }
    else
    {
        double SM = ( (SR - u_R)*ro2 * u_R - (SL - u_L)*ro1*u_L - p2 + p1 )/( (SR - u_R)*ro2 - (SL - u_L)*ro1 );
        double pp = p1 + ro1 * (SL - u_L) * (SM - u_L);

        if (SM <= 0.0)
        {
            double rr = ro2 * (SR - u_R) / (SR - SM);
            double e = p2 / g1 + 0.5 * ro2 * (kv(u2) + kv(v2));
            double ee = ((SR - u_R) * e - p2 * u_R + pp * SM) / (SR - SM);
            Ps.x = SR * (rr - ro2) + ro2 * u_R;
            Ps.y = SR * (ee - e) + (ga * p2 / (g1) + 0.5 * ro2 * (kv(u2) + kv(v2))) * u_R;

            double F1 = (ro2 * u_R * u_R + p2) + SR * (rr * SM - ro2 * u_R);
            double F2 = (ro2 * u_R * v_R) + SR * (rr * v_R - ro2 * v_R);
            Pu.x = F1 * n1 + F2 * t1;
            Pu.y = F1 * n2 + F2 * t2;
            return time;
        }
        else if (SM >= 0.0)
        {
            double rr = ro1 * (SL - u_L) / (SL - SM);
            double e = p1 / g1 + 0.5 * ro1 * (kv(u1) + kv(v1));
            double ee = ( (SL - u_L)*e - p1*u_L + pp * SM )/(SL - SM);
            Ps.x = SL * (rr - ro1) + ro1 * u_L;
            Ps.y = SL * (ee - e) + (ga * p1 / (g1) + 0.5 * ro1 * (kv(u1) + kv(v1))) * u_L;
            double F1 = (ro1 * u_L * u_L + p1) + SL * (rr * SM - ro1 * u_L);
            double F2 = (ro1 * u_L * v_L ) + SL * (rr * v_L - ro1 * v_L);
            Pu.x = F1 * n1 + F2 * t1;
            Pu.y = F1 * n2 + F2 * t2;
            return time;
        }
        else
        {
            printf("ERROR HLLC_KOROLKOV_2d   kod oshibki: 1jdt27453h\n");
            return time;
        }
    }
    return time;
}

__device__ double HLLCQ_Korolkov_2D(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double& LQ, const double& RQ, double n1, double n2, double2& Ps, double2& Pu, double& PQ, double rad)
{
    double u_L, v_L;
    double u_R, v_R;

    double ro1 = Ls.x;
    double u1 = Lu.x;
    double v1 = Lu.y;
    double p1 = Ls.y;
    double Q_L = LQ;

    double ro2 = Rs.x;
    double u2 = Ru.x;
    double v2 = Ru.y;
    double p2 = Rs.y;
    double Q_R = RQ;

    double t1 = -n2;    // ����������� ������
    double t2 = n1;

    u_L = u1 * n1 + v1 * n2;
    v_L = u1 * t1 + v1 * t2;

    u_R = u2 * n1 + v2 * n2;
    v_R = u2 * t1 + v2 * t2;

    double cL = sqrt(ga * p1 / ro1);
    double cR = sqrt(ga * p2 / ro2);

    double SL = min((u_L - cL), (u_R - cR));
    double SR = max((u_L + cL), (u_R + cR));

   /* double SL = min(u_L, u_R) - max(cL, cR);
    double SR = max(u_L, u_R) + max(cL, cR);*/

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double FL1 = ro1 * u_L * u_L + p1;
    double FL2 = ro1 * u_L * v_L;

    double FR1 = ro2 * u_R * u_R + p2;
    double FR2 = ro2 * u_R * v_R;

    if (SL >= 0.0)
    {
        PQ = Q_L * u_L;
        Ps.x = ro1 * u_L;
        Ps.y = (ga * p1 / (g1) + 0.5 * ro1 * (kv(u1) + kv(v1))) * u_L;
        Pu.x = (FL1) * n1 + (FL2) * t1;
        Pu.y = (FL1) * n2 + (FL2) * t2;
        return time;
    }
    else if (SR <= 0.0)
    {
        PQ = Q_R * u_R;
        Ps.x = ro2 * u_R;
        Ps.y = (ga * p2 / (g1) + 0.5 * ro2 * (kv(u2) + kv(v2))) * u_R;
        Pu.x = (FR1) * n1 + (FR2) * t1;
        Pu.y = (FR1) * n2 + (FR2) * t2;
        return time;
    }
    else
    {
        double SM = ((SR - u_R) * ro2 * u_R - (SL - u_L) * ro1 * u_L - p2 + p1) / ((SR - u_R) * ro2 - (SL - u_L) * ro1);
        double pp = p1 + ro1 * (SL - u_L) * (SM - u_L);

        if (SM <= 0.0)
        {
            double rr = ro2 * (SR - u_R) / (SR - SM);
            double e = p2 / g1 + 0.5 * ro2 * (kv(u2) + kv(v2));
            double ee = ((SR - u_R) * e - p2 * u_R + pp * SM) / (SR - SM);
            PQ = SR * (rr * Q_R/ro2 - Q_R) + Q_R * u_R;
            Ps.x = SR * (rr - ro2) + ro2 * u_R;
            Ps.y = SR * (ee - e) + (ga * p2 / (g1) + 0.5 * ro2 * (kv(u2) + kv(v2))) * u_R;

            double F1 = (ro2 * u_R * u_R + p2) + SR * (rr * SM - ro2 * u_R);
            double F2 = (ro2 * u_R * v_R) + SR * (rr * v_R - ro2 * v_R);
            Pu.x = F1 * n1 + F2 * t1;
            Pu.y = F1 * n2 + F2 * t2;
            return time;
        }
        else if (SM >= 0.0)
        {
            double rr = ro1 * (SL - u_L) / (SL - SM);
            double e = p1 / g1 + 0.5 * ro1 * (kv(u1) + kv(v1));
            double ee = ((SL - u_L) * e - p1 * u_L + pp * SM) / (SL - SM);
            PQ = SL * (rr * Q_L / ro1 - Q_L) + Q_L * u_L;
            Ps.x = SL * (rr - ro1) + ro1 * u_L;
            Ps.y = SL * (ee - e) + (ga * p1 / (g1) + 0.5 * ro1 * (kv(u1) + kv(v1))) * u_L;
            double F1 = FL1 + SL * (rr * SM - ro1 * u_L);
            double F2 = FL2 + SL * (rr * v_L - ro1 * v_L);
            Pu.x = F1 * n1 + F2 * t1;
            Pu.y = F1 * n2 + F2 * t2;
            return time;
        }
        else
        {
            printf("ERROR HLLC_KOROLKOV_2d   kod oshibki: 1jdt27453h\n");
            return time;
        }
    }
    return time;
}

__device__ double HLLCQ_Aleksashov(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double& LQ, const double& RQ, double n1, double n2, double2& Ps, double2& Pu, double& PQ, double rad)
{
    double n[3];
    n[0] = n1;
    n[1] = n2;
    n[2] = 0.0;
    //int id_bn = 1;
    //int n_state = 1;
    double FR[8], FL[8];
    double UL[8], UZ[8], UR[8];
    double UZL[8], UZR[8];

    double vL[3], vR[3], bL[3], bR[3];
    double vzL[3], vzR[3], bzL[3], bzR[3];
    double qv[3];
    double aco[3][3];

    double wv = 0.0;
    double r1 = Ls.x;
    double u1 = Lu.x;
    double v1 = Lu.y;
    double w1 = 0.0;
    double p1 = Ls.y;
    double bx1 = 0.0;
    double by1 = 0.0;
    double bz1 = 0.0;
    double Q_L = LQ;


    double r2 = Rs.x;
    double u2 = Ru.x;
    double v2 = Ru.y;
    double w2 = 0.0;
    double p2 = Rs.y;
    double bx2 = 0.0;
    double by2 = 0.0;
    double bz2 = 0.0;
    double Q_R = RQ;

    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;
    double abx = (bx2 + bx1) / 2.0;
    double aby = (by2 + by1) / 2.0;
    double abz = (bz2 + bz1) / 2.0;


    double bk = abx * n[0] + aby * n[1] + abz * n[2];
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double d = b2 - kv(bk);
    aco[0][0] = n[0];
    aco[1][0] = n[1];
    aco[2][0] = n[2];
    if (d > 0.000000001)
    {
        d = sqrt(d);
        aco[0][1] = (abx - bk * n[0]) / d;
        aco[1][1] = (aby - bk * n[1]) / d;
        aco[2][1] = (abz - bk * n[2]) / d;
        aco[0][2] = (aby * n[2] - abz * n[1]) / d;
        aco[1][2] = (abz * n[0] - abx * n[2]) / d;
        aco[2][2] = (abx * n[1] - aby * n[0]) / d;
    }
    else
    {
        double aix, aiy, aiz;
        if ((fabs(n[0]) < fabs(n[1])) && (fabs(n[0]) < fabs(n[2])))
        {
            aix = 1.0;
            aiy = 0.0;
            aiz = 0.0;
        }
        else if (fabs(n[1]) < fabs(n[2]))
        {
            aix = 0.0;
            aiy = 1.0;
            aiz = 0.0;
        }
        else
        {
            aix = 0.0;
            aiy = 0.0;
            aiz = 1.0;
        }

        double aik = aix * n[0] + aiy * n[1] + aiz * n[2];
        d = sqrt(1.0 - kv(aik));
        aco[0][1] = (aix - aik * n[0]) / d;
        aco[1][1] = (aiy - aik * n[1]) / d;
        aco[2][1] = (aiz - aik * n[2]) / d;
        aco[0][2] = (aiy * n[2] - aiz * n[1]) / d;
        aco[1][2] = (aiz * n[0] - aix * n[2]) / d;
        aco[2][2] = (aix * n[1] - aiy * n[0]) / d;
    }

    for (int i = 0; i < 3; i++)
    {
        vL[i] = aco[0][i] * u1 + aco[1][i] * v1 + aco[2][i] * w1;
        vR[i] = aco[0][i] * u2 + aco[1][i] * v2 + aco[2][i] * w2;
        bL[i] = aco[0][i] * bx1 + aco[1][i] * by1 + aco[2][i] * bz1;
        bR[i] = aco[0][i] * bx2 + aco[1][i] * by2 + aco[2][i] * bz2;
    }

    double aaL = bL[0] / sqrt(r1);
    double b2L = kv(bL[0]) + kv(bL[1]) + kv(bL[2]);
    double b21 = b2L / r1;
    double cL = sqrt(ga * p1 / r1);
    double qp = sqrt(b21 + cL * (cL + 2.0 * aaL));
    double qm = sqrt(b21 + cL * (cL - 2.0 * aaL));
    double cfL = (qp + qm) / 2.0;
    double ptL = p1 + b2L / 2.0;

    double aaR = bR[0] / sqrt(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = sqrt(ga * p2 / r2);
    qp = sqrt(b22 + cR * (cR + 2.0 * aaR));
    qm = sqrt(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / 2.0;
    double ptR = p2 + b2R / 2.0;

    double aC = (aaL + aaR) / 2.0;
    double b2o = (b22 + b21) / 2.0;
    double cC = sqrt(ga * ap / ro);
    qp = sqrt(b2o + cC * (cC + 2.0 * aC));
    qm = sqrt(b2o + cC * (cC - 2.0 * aC));
    double cfC = (qp + qm) / 2.0;
    double vC1 = (vL[0] + vR[0]) / 2.0;

    double SL = min((vL[0] - cfL), (vR[0] - cfR));
    double SR = max((vL[0] + cfL), (vR[0] + cfR));

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    if (SR <= SL)
    {
        printf("231\n");
    }

    double SM00 = SM;
    double SR00 = SR;
    double SL00 = SL;
    double SM01, SR01, SL01;
    if ((SM00 >= SR00) || (SM00 <= SL00))
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
        suR = SR - vR[0];
        suL = SL - vL[0];
        SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);
        SM01 = SM;
        SR01 = SR;
        SL01 = SL;
        if ((SM01 >= SR01) || (SM01 <= SL01))
        {
            printf("251\n");
        }
    }


    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / 2.0;
    double e2 = p2 / g1 + r2 * upt2 + b2R / 2.0;

    double FL0 = Q_L * vL[0];
    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + ptL - kv(bL[0]);
    FL[2] = r1 * vL[0] * vL[1] - bL[0] * bL[1];
    FL[3] = r1 * vL[0] * vL[2] - bL[0] * bL[2];
    FL[4] = (e1 + ptL) * vL[0] - bL[0] * sbv1;
    FL[5] = 0.0;
    FL[6] = vL[0] * bL[1] - vL[1] * bL[0];
    FL[7] = vL[0] * bL[2] - vL[2] * bL[0];

    double FR0 = Q_R * vR[0];
    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + ptR - kv(bR[0]);
    FR[2] = r2 * vR[0] * vR[1] - bR[0] * bR[1];
    FR[3] = r2 * vR[0] * vR[2] - bR[0] * bR[2];
    FR[4] = (e2 + ptR) * vR[0] - bR[0] * sbv2;
    FR[5] = 0.0;
    FR[6] = vR[0] * bR[1] - vR[1] * bR[0];
    FR[7] = vR[0] * bR[2] - vR[2] * bR[0];

    UL[0] = r1;
    UL[4] = e1;
    UR[0] = r2;
    UR[4] = e2;


    for (int ik = 0; ik < 3; ik++)
    {
        UL[ik + 1] = r1 * vL[ik];
        UL[ik + 5] = bL[ik];
        UR[ik + 1] = r2 * vR[ik];
        UR[ik + 5] = bR[ik];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }

    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;
    vzR[0] = SM;
    vzL[0] = SM;
    double ptzR = ptR + r2 * suR * (SM - vR[0]);
    double ptzL = ptL + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;
    bzR[0] = UZ[5];
    bzL[0] = UZ[5];

    vzR[1] = UZ[2] / UZ[0];
    vzR[2] = UZ[3] / UZ[0];
    vzL[1] = vzR[1];
    vzL[2] = vzR[2];

    vzR[1] = vR[1] + UZ[5] * (bR[1] - UZ[6]) / suR / r2;
    vzR[2] = vR[2] + UZ[5] * (bR[2] - UZ[7]) / suR / r2;
    vzL[1] = vL[1] + UZ[5] * (bL[1] - UZ[6]) / suL / r1;
    vzL[2] = vL[2] + UZ[5] * (bL[2] - UZ[7]) / suL / r1;

    bzR[1] = UZ[6];
    bzR[2] = UZ[7];
    bzL[1] = bzR[1];
    bzL[2] = bzR[2];

    double sbvz = (UZ[5] * UZ[1] + UZ[6] * UZ[2] + UZ[7] * UZ[3]) / UZ[0];

    double ezR = e2 * suRm + (ptz * SM - ptR * vR[0] + UZ[5] * (sbv2 - sbvz)) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - ptL * vL[0] + UZ[5] * (sbv1 - sbvz)) / (SL - SM);

    if (fabs(UZ[5]) < 0.000000001)
    {
        vzR[1] = vR[1];
        vzR[2] = vR[2];
        vzL[1] = vL[1];
        vzL[2] = vL[2];
        bzR[1] = bR[1] * suRm;
        bzR[2] = bR[2] * suRm;
        bzL[1] = bL[1] * suLm;
        bzL[2] = bL[2] * suLm;
    }
    UZL[0] = rzL;
    UZL[4] = ezL;
    UZR[0] = rzR;
    UZR[4] = ezR;

    for (int ik = 0; ik < 3; ik++)
    {
        UZL[ik + 1] = vzL[ik] * rzL;
        UZL[ik + 5] = bzL[ik];
        UZR[ik + 1] = vzR[ik] * rzR;
        UZR[ik + 5] = bzR[ik];
    }

    if (SL > wv)
    {
        PQ = FL0;
        Ps.x = FL[0] - wv * UL[0];
        Ps.y = FL[4] - wv * UL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] - wv * UL[ik];
        }
    }
    else if ((SL <= wv) && (SM >= wv))
    {
        PQ = FL0 + SL * (rzL * Q_L/r1 - Q_L);
        Ps.x = FL[0] + SL * (rzL - r1) - wv * UZL[0];
        Ps.y = FL[4] + SL * (ezL - e1) - wv * UZL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
        }
    }
    else if ((SM <= wv) && (SR >= wv))
    {
        PQ = FR0 + SR * (rzR * Q_R/r2 - Q_R);
        Ps.x = FR[0] + SR * (rzR - r2) - wv * UZR[0];
        Ps.y = FR[4] + SR * (ezR - e2) - wv * UZR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
        }
    }
    else if (SR < wv)
    {
        PQ = FR0;
        Ps.x = FR[0] - wv * UR[0];
        Ps.y = FR[4] - wv * UR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + -wv * UR[ik];
        }
    }
    else
    {
        printf("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n");
    }


    Pu.x = aco[0][0] * qv[0] + aco[0][1] * qv[1] + aco[0][2] * qv[2];
    Pu.y = aco[1][0] * qv[0] + aco[1][1] * qv[1] + aco[1][2] * qv[2];

    return time;
}

__device__ double HLLC_Aleksashov_2D(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double n[2];
    n[0] = n1;
    n[1] = n2;
    //int id_bn = 1;
    //int n_state = 1;
    double FR[5], FL[5];
    double UL[5], UZ[5], UR[5];
    double UZL[5], UZR[5];

    double vL[2], vR[2];
    double vzL[2], vzR[2];
    double qv[2];
    double aco[2][2];

    double r1 = Ls.x;
    double u1 = Lu.x;
    double v1 = Lu.y;
    double p1 = Ls.y;


    double r2 = Rs.x;
    double u2 = Ru.x;
    double v2 = Ru.y;
    double p2 = Rs.y;

    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;


    aco[0][0] = n[0];
    aco[1][0] = n[1];

    aco[0][1] = -n[1];
    aco[1][1] = n[0];
    

    for (int i = 0; i < 2; i++)
    {
        vL[i] = aco[0][i] * u1 + aco[1][i] * v1;
        vR[i] = aco[0][i] * u2 + aco[1][i] * v2;
    }


    double cL = sqrt(ga * p1 / r1);
    double cR = sqrt(ga * p2 / r2);


    double cC = sqrt(ga * ap / ro);

    double vC1 = (vL[0] + vR[0]) / 2.0;

    //double SL = min((vL[0] - cL), (vR[0] - cR));
    //double SR = max((vL[0] + cL), (vR[0] + cR));

    double SL = min(vL[0], vR[0]) - max(cL, cR);
    double SR = max(vL[0], vR[0]) + max(cL, cR);


    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - p2 + p1 - suL * r1 * vL[0]) / (suR * r2 - suL * r1);


    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double upt1 = (kv(u1) + kv(v1)) / 2.0;

    double upt2 = (kv(u2) + kv(v2)) / 2.0;

    double e1 = p1 / g1 + r1 * upt1;
    double e2 = p2 / g1 + r2 * upt2;

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + p1;
    FL[2] = r1 * vL[0] * vL[1];
    FL[4] = (e1 + p1) * vL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + p2;
    FR[2] = r2 * vR[0] * vR[1];
    FR[4] = (e2 + p2) * vR[0];

    UL[0] = r1;
    UL[4] = e1;
    UR[0] = r2;
    UR[4] = e2;


    for (int ik = 0; ik < 2; ik++)
    {
        UL[ik + 1] = r1 * vL[ik];
        UR[ik + 1] = r2 * vR[ik];
    }

    for (int ik = 0; ik < 5; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }

    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;
    vzR[0] = SM;
    vzL[0] = SM;
    double ptzR = p2 + r2 * suR * (SM - vR[0]);
    double ptzL = p1 + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;


    vzR[1] = vR[1];
    vzL[1] = vL[1];



    double ezR = e2 * suRm + (ptz * SM - p2 * vR[0]) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - p1 * vL[0]) / (SL - SM);

    UZL[0] = rzL;
    UZL[4] = ezL;
    UZR[0] = rzR;
    UZR[4] = ezR;

    for (int ik = 0; ik < 2; ik++)
    {
        UZL[ik + 1] = vzL[ik] * rzL;
        UZR[ik + 1] = vzR[ik] * rzR;
    }

    if (SL > 0.0)
    {
        Ps.x = FL[0];
        Ps.y = FL[4];
        for (int ik = 1; ik < 3; ik++)
        {
            qv[ik - 1] = FL[ik];
        }
    }
    else if ((SL <= 0.0) && (SM >= 0.0))
    {
        Ps.x = FL[0] + SL * (rzL - r1);
        Ps.y = FL[4] + SL * (ezL - e1);
        for (int ik = 1; ik < 3; ik++)
        {
            qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]);
        }
    }
    else if ((SM <= 0.0) && (SR >= 0.0))
    {
        Ps.x = FR[0] + SR * (rzR - r2);
        Ps.y = FR[4] + SR * (ezR - e2);
        for (int ik = 1; ik < 3; ik++)
        {
            qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]);
        }
    }
    else if (SR < 0.0)
    {
        Ps.x = FR[0];
        Ps.y = FR[4];
        for (int ik = 1; ik < 3; ik++)
        {
            qv[ik - 1] = FR[ik];
        }
    }
    else
    {
        printf("hllc 2d ERROR\n");
    }


    Pu.x = aco[0][0] * qv[0] + aco[0][1] * qv[1];
    Pu.y = aco[1][0] * qv[0] + aco[1][1] * qv[1];

    return time;
}

__device__ int sign(double& x)
{
    if (x > 0)
    {
        return 1;
    }
    else if (x < 0)
    {
        return  -1;
    }
    else
    {
        return 0;
    }
}

__device__ double HLLDQ_Korolkov(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, double& PQ, const double& n1, const double& n2, const double& n3, double& rad, int metod)
{// �� ��������, ���� �������� ����� �� �������
 // ������� ����� ��������� �� ���� ���������

    double bx_L = Bx_L / spi4;
    double by_L = By_L / spi4;
    double bz_L = Bz_L / spi4;

    double bx_R = Bx_R / spi4;
    double by_R = By_R / spi4;
    double bz_R = Bz_R / spi4;

    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    double m1 = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;

    if (n1 > 0.1)
    {
        t2 = 1.0;
        m3 = 1.0;
    }
    else if (n2 > 0.1)
    {
        t3 = 1.0;
        m1 = 1.0;
    }
    else if (n3 > 0.1)
    {
        t1 = 1.0;
        m2 = 1.0;
    }
    else if (n1 < -0.1)
    {
        t3 = -1.0;
        m2 = -1.0;
    }
    else if (n2 < -0.1)
    {
        t1 = -1.0;
        m3 = -1.0;
    }
    else if (n3 < -0.1)
    {
        t1 = -1.0;
        m2 = -1.0;
    }
    else
    {
        printf("EROROR 1421  normal_error\n");
    }


    double u1, v1, w1, u2, v2, w2;
    u1 = v1_L * n1 + v2_L * n2 + v3_L * n3;
    v1 = v1_L * t1 + v2_L * t2 + v3_L * t3;
    w1 = v1_L * m1 + v2_L * m2 + v3_L * m3;
    u2 = v1_R * n1 + v2_R * n2 + v3_R * n3;
    v2 = v1_R * t1 + v2_R * t2 + v3_R * t3;
    w2 = v1_R * m1 + v2_R * m2 + v3_R * m3;

    double bn1, bt1, bm1, bn2, bt2, bm2;
    bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
    bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
    bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
    bn2 = bx_R * n1 + by_R * n2 + bz_R * n3;
    bt2 = bx_R * t1 + by_R * t2 + bz_R * t3;
    bm2 = bx_R * m1 + by_R * m2 + bz_R * m3;

    //cout << " = " << bt2 * bt2 + bm2 * bm2 << endl;

    double sqrtroL = sqrt(ro_L);
    double sqrtroR = sqrt(ro_R);
    double ca_L = bn1 / sqrtroL;
    double ca_R = bn2 / sqrtroR;
    double cL = sqrt(ggg * p_L / ro_L);
    double cR = sqrt(ggg * p_R / ro_R);

    double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
    double bb_R = kv(bx_R) + kv(by_R) + kv(bz_R);

    double aL = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;
    double aR = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;

    double uu_L = (kv(v1_L) + kv(v2_L) + kv(v3_L)) / 2.0;
    double uu_R = (kv(v1_R) + kv(v2_R) + kv(v3_R)) / 2.0;

    double cfL = sqrt((ggg * p_L + bb_L + //
        sqrt(kv(ggg * p_L + bb_L) - 4.0 * ggg * p_L * kv(bn1))) / (2.0 * ro_L));
    double cfR = sqrt((ggg * p_R + bb_R + //
        sqrt(kv(ggg * p_R + bb_R) - 4.0 * ggg * p_R * kv(bn2))) / (2.0 * ro_R));


    double SL = min(u1, u2) - max(cfL, cfR);
    double SR = max(u1, u2) + max(cfL, cfR);

    double pTL = p_L + bb_L / 2.0;
    double pTR = p_R + bb_R / 2.0;

    double suR = (SR - u2);
    double suL = (SL - u1);

    double SM = (suR * ro_R * u2 - suL * ro_L * u1 - pTR + pTL) //
        / (suR * ro_R - suL * ro_L);

    double PTT = (suR * ro_R * pTL - suL * ro_L * pTR + ro_L * ro_R * suR * suL * (u2 - u1))//
        / (suR * ro_R - suL * ro_L);

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double FL[9], FR[9], UL[9], UR[9];

    double e1 = p_L / g1 + ro_L * uu_L + bb_L / 2.0;
    double e2 = p_R / g1 + ro_R * uu_R + bb_R / 2.0;


    FL[0] = ro_L * u1;
    FL[1] = ro_L * u1 * u1 + pTL - kv(bn1);
    FL[2] = ro_L * u1 * v1 - bn1 * bt1;
    FL[3] = ro_L * u1 * w1 - bn1 * bm1;
    FL[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
    //cout << uu_L << endl;
    FL[5] = 0.0;
    FL[6] = u1 * bt1 - v1 * bn1;
    FL[7] = u1 * bm1 - w1 * bn1;
    FL[8] = Q_L * u1;

    FR[0] = ro_R * u2;
    FR[1] = ro_R * u2 * u2 + pTR - kv(bn2);
    FR[2] = ro_R * u2 * v2 - bn2 * bt2;
    FR[3] = ro_R * u2 * w2 - bn2 * bm2;
    FR[4] = (e2 + pTR) * u2 - bn2 * (u2 * bn2 + v2 * bt2 + w2 * bm2);
    FR[5] = 0.0;
    FR[6] = u2 * bt2 - v2 * bn2;
    FR[7] = u2 * bm2 - w2 * bn2;
    FR[8] = Q_R * u2;

    UL[0] = ro_L;
    UL[1] = ro_L * u1;
    UL[2] = ro_L * v1;
    UL[3] = ro_L * w1;
    UL[4] = e1;
    UL[5] = bn1;
    UL[6] = bt1;
    UL[7] = bm1;
    UL[8] = Q_L;

    UR[0] = ro_R;
    UR[1] = ro_R * u2;
    UR[2] = ro_R * v2;
    UR[3] = ro_R * w2;
    UR[4] = e2;
    UR[5] = bn2;
    UR[6] = bt2;
    UR[7] = bm2;
    UR[8] = Q_R;

    double bn = (SR * UR[5] - SL * UL[5] + FL[5] - FR[5]) / (SR - SL);
    double bt = (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) / (SR - SL);
    double bm = (SR * UR[7] - SL * UL[7] + FL[7] - FR[7]) / (SR - SL);
    double bbn = bn * bn;

    double ro_LL = ro_L * (SL - u1) / (SL - SM);
    double ro_RR = ro_R * (SR - u2) / (SR - SM);
    double Q_LL = Q_L * (SL - u1) / (SL - SM);
    double Q_RR = Q_R * (SR - u2) / (SR - SM);

    if (metod == 2)   // HLLC  + mgd
    {
        double sbv1 = u1 * bn1 + v1 * bt1 + w1 * bm1;
        double sbv2 = u2 * bn2 + v2 * bt2 + w2 * bm2;

        double UZ0 = (SR * UR[0] - SL * UL[0] + FL[0] - FR[0]) / (SR - SL);
        double UZ1 = (SR * UR[1] - SL * UL[1] + FL[1] - FR[1]) / (SR - SL);
        double UZ2 = (SR * UR[2] - SL * UL[2] + FL[2] - FR[2]) / (SR - SL);
        double UZ3 = (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) / (SR - SL);
        double UZ4 = (SR * UR[4] - SL * UL[4] + FL[4] - FR[4]) / (SR - SL);
        double vzL, vzR, vLL, wLL, vRR, wRR, ppLR, btt1, bmm1, btt2, bmm2, ee1, ee2;


        double suRm = suR / (SR - SM);
        double suLm = suL / (SL - SM);
        double rzR = ro_R * suRm;
        double rzL = ro_L * suLm;

        double ptzR = pTR + ro_R * suR * (SM - u2);
        double ptzL = pTL + ro_L * suL * (SM - u1);
        double ptz = (ptzR + ptzL) / 2.0;


        vRR = UZ2 / UZ0;
        wRR = UZ3 / UZ0;
        vLL = vRR;
        wLL = wRR;

        /*vRR = v2 + bn * (bt2 - bt) / suR / ro_R;
        wRR = w2 + bn * (bm2 - bm) / suR / ro_R;
        vLL = v1 + bn * (bt1 - bt) / suL / ro_L;
        wLL = w1 + bn * (bm1 - bm) / suL / ro_L;*/

        btt2 = bt;
        bmm2 = bm;
        btt1 = btt2;
        bmm1 = bmm2;

        double sbvz = (bn * UZ1 + bt * UZ2 + bm * UZ3) / UZ0;

        ee2 = e2 * suRm + (ptz * SM - pTR * u2 + bn * (sbv2 - sbvz)) / (SR - SM);
        ee1 = e1 * suLm + (ptz * SM - pTL * u1 + bn * (sbv1 - sbvz)) / (SL - SM);

        /*if (fabs(bn) < 0.000001 )
        {
            vRR = v2;
            wRR = w2;
            vLL = v1;
            wLL = w1;
            btt2 = bt2 * suRm;
            bmm2 = bm2 * suRm;
            btt1 = bt1 * suLm;
            bmm1 = bm1 * suLm;
        }*/

        /*ppLR = (pTL + ro_L * (SL - u1) * (SM - u1) + pTR + ro_R * (SR - u2) * (SM - u2)) / 2.0;

        if (fabs(bn) < 0.000001)
        {
            vLL = v1;
            wLL = w1;
            vRR = v2;
            wRR = w2;

            btt1 = bt1 * (SL - u1) / (SL - SM);
            btt2 = bt2 * (SR - u2) / (SR - SM);

            bmm1 = bm1 * (SL - u1) / (SL - SM);
            bmm2 = bm2 * (SR - u2) / (SR - SM);

            ee1 = ((SL - u1) * e1 - pTL * u1 + ppLR * SM) / (SL - SM);
            ee2 = ((SR - u2) * e2 - pTL * u2 + ppLR * SM) / (SR - SM);
        }
        else
        {
            btt2 = btt1 = (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) / (SR - SL);
            bmm2 = bmm1 = (SR * UR[7] - SL * UL[7] + FL[7] - FR[7]) / (SR - SL);
            vLL = v1 + bn * (bt1 - btt1) / (ro_L * (SL - u1));
            vRR = v2 + bn * (bt2 - btt2) / (ro_R * (SR - u2));

            wLL = w1 + bn * (bm1 - bmm1) / (ro_L * (SL - u1));
            wRR = w2 + bn * (bm2 - bmm2) / (ro_R * (SR - u2));

            double sks1 = u1 * bn1 + v1 * bt1 + w1 * bm1 - SM * bn - vLL * btt1 - wLL * bmm1;
            double sks2 = u2 * bn2 + v2 * bt2 + w2 * bm2 - SM * bn - vRR * btt2 - wRR * bmm2;

            ee1 = ((SL - u1) * e1 - pTL * u1 + ppLR * SM + bn * sks1) / (SL - SM);
            ee2 = ((SR - u2) * e2 - pTR * u2 + ppLR * SM + bn * sks2) / (SR - SM);
        }*/


        double  ULL[9], URR[9], PO[9];
        ULL[0] = ro_LL;
        ULL[1] = ro_LL * SM;
        ULL[2] = ro_LL * vLL;
        ULL[3] = ro_LL * wLL;
        ULL[4] = ee1;
        ULL[5] = bn;
        ULL[6] = btt1;
        ULL[7] = bmm1;
        ULL[8] = Q_LL;

        URR[0] = ro_RR;
        URR[1] = ro_RR * SM;
        URR[2] = ro_RR * vRR;
        URR[3] = ro_RR * wRR;
        URR[4] = ee2;
        URR[5] = bn;
        URR[6] = btt2;
        URR[7] = bmm2;
        URR[8] = Q_RR;

        if (SL >= 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i];
            }
        }
        else if (SL < 0.0 && SM >= 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
            }
        }
        else if (SR > 0.0 && SM < 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
            }
        }
        else if (SR <= 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i];
            }
        }



        double SN = max(fabs(SL), fabs(SR));

        PO[5] = -SN * (bn2 - bn1);

        P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
        P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
        P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
        P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
        P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
        P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
        P[0] = PO[0];
        P[4] = PO[4];
        PQ = PO[8];

        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;
        return time;

    }
    else if (metod == 3)  // HLLD
    {

        double ttL = ro_L * suL * (SL - SM) - bbn;
        double ttR = ro_R * suR * (SR - SM) - bbn;

        double vLL, wLL, vRR, wRR, btt1, bmm1, btt2, bmm2;

        if (fabs(ttL) >= 0.000001)
        {
            vLL = v1 - bn * bt1 * (SM - u1) / ttL;
            wLL = w1 - bn * bm1 * (SM - u1) / ttL;
            btt1 = bt1 * (ro_L * suL * suL - bbn) / ttL;
            bmm1 = bm1 * (ro_L * suL * suL - bbn) / ttL;
        }
        else
        {
            vLL = v1;
            wLL = w1;
            btt1 = 0.0;
            bmm1 = 0.0;
        }

        if (fabs(ttR) >= 0.000001)
        {
            vRR = v2 - bn * bt2 * (SM - u2) / ttR;
            wRR = w2 - bn * bm2 * (SM - u2) / ttR;
            btt2 = bt2 * (ro_R * suR * suR - bbn) / ttR;
            bmm2 = bm2 * (ro_R * suR * suR - bbn) / ttR;
            //cout << "tbr = " << (ro_R * suR * suR - bbn) / ttR << endl;
            //cout << "bt2 = " << bt2 << endl;
        }
        else
        {
            vRR = v2;
            wRR = w2;
            btt2 = 0.0;
            bmm2 = 0.0;
        }

        double eLL = (e1 * suL + PTT * SM - pTL * u1 + bn * //
            ((u1 * bn1 + v1 * bt1 + w1 * bm1) - (SM * bn + vLL * btt1 + wLL * bmm1))) //
            / (SL - SM);
        double eRR = (e2 * suR + PTT * SM - pTR * u2 + bn * //
            ((u2 * bn2 + v2 * bt2 + w2 * bm2) - (SM * bn + vRR * btt2 + wRR * bmm2))) //
            / (SR - SM);

        double sqrtroLL = sqrt(ro_LL);
        double sqrtroRR = sqrt(ro_RR);
        double SLL = SM - fabs(bn) / sqrtroLL;
        double SRR = SM + fabs(bn) / sqrtroRR;

        double idbn = 1.0;
        if (fabs(bn) > 0.001)
        {
            idbn = 1.0 * sign(bn);
        }
        else
        {
            idbn = 0.0;
            SLL = SM;
            SRR = SM;
        }

        double vLLL = (sqrtroLL * vLL + sqrtroRR * vRR + //
            idbn * (btt2 - btt1)) / (sqrtroLL + sqrtroRR);

        double wLLL = (sqrtroLL * wLL + sqrtroRR * wRR + //
            idbn * (bmm2 - bmm1)) / (sqrtroLL + sqrtroRR);

        double bttt = (sqrtroLL * btt2 + sqrtroRR * btt1 + //
            idbn * sqrtroLL * sqrtroRR * (vRR - vLL)) / (sqrtroLL + sqrtroRR);

        double bmmm = (sqrtroLL * bmm2 + sqrtroRR * bmm1 + //
            idbn * sqrtroLL * sqrtroRR * (wRR - wLL)) / (sqrtroLL + sqrtroRR);

        double eLLL = eLL - idbn * sqrtroLL * ((SM * bn + vLL * btt1 + wLL * bmm1) //
            - (SM * bn + vLLL * bttt + wLLL * bmmm));
        double eRRR = eRR + idbn * sqrtroRR * ((SM * bn + vRR * btt2 + wRR * bmm2) //
            - (SM * bn + vLLL * bttt + wLLL * bmmm));
        //cout << " = " << bn << " " << btt2 << " " << bmm2 << endl;
        //cout << "sbvr = " << (SM * bn + vRR * btt2 + wRR * bmm2) << endl;
        double  ULL[9], URR[9], ULLL[9], URRR[9];

        ULL[0] = ro_LL;
        ULL[1] = ro_LL * SM;
        ULL[2] = ro_LL * vLL;
        ULL[3] = ro_LL * wLL;
        ULL[4] = eLL;
        ULL[5] = bn;
        ULL[6] = btt1;
        ULL[7] = bmm1;
        ULL[8] = Q_LL;

        URR[0] = ro_RR;
        //cout << ro_RR << endl;
        URR[1] = ro_RR * SM;
        URR[2] = ro_RR * vRR;
        URR[3] = ro_RR * wRR;
        URR[4] = eRR;
        URR[5] = bn;
        URR[6] = btt2;
        URR[7] = bmm2;
        URR[8] = Q_RR;

        ULLL[0] = ro_LL;
        ULLL[1] = ro_LL * SM;
        ULLL[2] = ro_LL * vLLL;
        ULLL[3] = ro_LL * wLLL;
        ULLL[4] = eLLL;
        ULLL[5] = bn;
        ULLL[6] = bttt;
        ULLL[7] = bmmm;
        ULLL[8] = Q_LL;

        URRR[0] = ro_RR;
        URRR[1] = ro_RR * SM;
        URRR[2] = ro_RR * vLLL;
        URRR[3] = ro_RR * wLLL;
        URRR[4] = eRRR;
        URRR[5] = bn;
        URRR[6] = bttt;
        URRR[7] = bmmm;
        URRR[8] = Q_RR;

        double PO[9];

        if (SL >= 0.0)
        {
            //cout << "SL >= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i];
            }
        }
        else if (SL < 0.0 && SLL >= 0.0)
        {
            //cout << "SL < 0.0 && SLL >= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
            }
            //cout << ULL[0] << endl;
        }
        else if (SLL <= 0.0 && SM >= 0.0)
        {
            //cout << "SLL <= 0.0 && SM >= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i] + SLL * ULLL[i] - (SLL - SL) * ULL[i] - SL * UL[i];
            }
        }
        else if (SM < 0.0 && SRR > 0.0)
        {
            //cout << "SM < 0.0 && SRR > 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i] + SRR * URRR[i] - (SRR - SR) * URR[i] - SR * UR[i];
            }
            //cout << "P4 = " << URRR[4] << endl;
        }
        else if (SR > 0.0 && SRR <= 0.0)
        {
            //cout << "SR > 0.0 && SRR <= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
            }
            //cout << URR[0] << endl;
        }
        else if (SR <= 0.0)
        {
            //cout << "SR <= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i];
            }
        }



        double SN = max(fabs(SL), fabs(SR));

        PO[5] = -SN * (bn2 - bn1);

        P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
        P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
        P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
        P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
        P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
        P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
        P[0] = PO[0];
        P[4] = PO[4];
        PQ = PO[8];

        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;
        return time;
    }

}



void spherical_skorost(double x, double y, double z, double Vx, double Vy, double Vz, double& Vr, double& Vphi, double& Vtheta)
{
    double r_1 = sqrt(x * x + y * y + z * z);
    double the_1 = acos(z / r_1);
    double phi_1 = polar_angle(x, y);

    Vr = Vx * sin(the_1) * cos(phi_1) + Vy * sin(the_1) * sin(phi_1) + Vz * cos(the_1);
    Vtheta = Vx * cos(the_1) * cos(phi_1) + Vy * cos(the_1) * sin(phi_1) - Vz * sin(the_1);
    Vphi = -Vx * sin(phi_1) + Vy * cos(phi_1);
}

void dekard_skorost(double x, double y, double z, double Vr, double Vphi, double Vtheta, double& Vx, double& Vy, double& Vz)
{
    double r_2 = sqrt(x * x + y * y + z * z);
    double the_2 = acos(z / r_2);
    double phi_2 = polar_angle(x, y);

    Vx = Vr * sin(the_2) * cos(phi_2) + Vtheta * cos(the_2) * cos(phi_2) - Vphi * sin(phi_2);
    Vy = Vr * sin(the_2) * sin(phi_2) + Vtheta * cos(the_2) * sin(phi_2) + Vphi * cos(phi_2);
    Vz = Vr * cos(the_2) - Vtheta * sin(the_2);
    
}

double polar_angle(double x, double y)
{
    if (x < 0)
    {
        return atan(y / x) + 1.0 * PI;
    }
    else if (x > 0 && y >= 0)
    {
        return atan(y / x);
    }
    else if (x > 0 && y < 0)
    {
        return atan(y / x) + 2.0 * PI;
    }
    else if (y > 0 && x >= 0 && x <= 0)
    {
        return PI / 2.0;
    }
    else if (y < 0 && x >= 0 && x <= 0)
    {
        return  3.0 * PI / 2.0;
    }
    return 0.0;
}


double minplus(double x, double y)
{
    if (x < 0.00000000001 && y < 0.00000000001)
    {
        return -1.0;
    }
    else if(x < 0.00000000001)
    {
        return y;
    }
    else if (y < 0.00000000001)
    {
        return x;
    }
    else if (x < y)
    {
        return x;
    }
    else
    {
        return y;
    }
}

bool Peresechenie(const double& x0, const double& y0, const double& x, const double& y, const double& z, const double& Vx,//
    const double& Vy, const double& Vz, int& mode, double& t)
{
    mode = 0;
    double t1 = -1.0, t2 = -1.0, t3 = -1.0, t4 = -1.0, t5 = -1.0, t6 = -1.0;
    if (fabs(Vx) > 0.0000001) 
    {
        t1 = (x0 + dx / 2.0 - x) / Vx;
        t2 = (x0 - dx / 2.0 - x) / Vx;
    }
    if (kv(Vy) + kv(Vz) > 0.0000001)
    {
        peresich(y, z, Vy, Vz, y0 + dy / 2.0, t3, t4);
        peresich(y, z, Vy, Vz, y0 - dy / 2.0, t5, t6);
        t3 = minplus(t3, t4);
        t4 = minplus(t5, t6);
    }
    t = 100000000.0;
    if (t > t1 && t1 > 0.000000001)
    {
        t = t1;
        mode = 1;
    }
    if (t > t2 && t2 > 0.000000001)
    {
        t = t2;
        mode = 2;
    }
    if (t > t3 && t3 > 0.000000001)
    {
        t = t3;
        mode = 3;
    }
    if (t > t4 && t4 > 0.000000001)
    {
        t = t4;
        mode = 4;
    }

    if (fabs(Vx) > 0.0000001 && t1 < 0.000000001 && t2 < 0.000000001)
    {
        return false;
    }

    if(mode == 0)
    {
        return false;
    }
    return true;
}

void peresich(const double& y, const double& z, const double& Vy, const double& Vz, const double& R, double& t1, double& t2)
{
    double b = (2.0 * y * Vy + 2.0 * z * Vz);
    double a = (kv(Vy) + kv(Vz));
    double D = b * b - 4.0 * (kv(y) + kv(z) - kv(R)) * a;
    if (D < 0)
    {
        t1 = -1.0;
        t2 = -1.0;
        return;
    }
    D = sqrt(D);
    t1 = (-b + D) / (2.0 * a);
    t2 = (-b - D) / (2.0 * a);
}

void Belong_point(const double& x, const double& y, int& n, int& m)
{
    m = (int)(y / dy);
    n = (int)((x - x_min + dx/2.0) / dx);
    return;
}

double Velosity_1(const double& u, const double& cp)
{
    if (u < 0.001)
    {
        return 2.0 * cp / sqrtpi + 2.0 * u * u / (3.0 * cp * sqrtpi) - u * u * u * u / (15.0 * cp * cp * cp * sqrtpi);
    }
    else
    {
        return  exp(-u * u / kv(cp)) * cp / sqrtpi + (u + kv(cp) / (2.0 * u)) * erf(u / cp);
    }
}

double Velosity_2(const double& u, const double& cp)  // ������� �� ������ ��������, � ������ � ��������� (��. ������)
{
    if (u < 0.001)
    {
        return (8.0/3.0) * kv(cp) * kv(cp) * pi * u + (8.0/15.0)*kv(cp) * pi * u * u * u - (4.0/105.0) * pi * kv(u) * kv(u) * u;
    }
    else
    {
        return  cp * cp * cp * pi * ( exp(-u * u / kv(cp)) * cp * u * 2.0 * (kv(cp) + 2.0 * kv(u)) +//
            sqrtpi * (4.0 * kv(u) * kv(u) + 4.0 * cp * cp * kv(u) - kv(cp) * kv(cp)) * erf(u / cp) ) / (4.0 * u * u);
    }
}

double Velosity_3(const double& u, const double& cp)
{
    if (u < 0.001)
    {
        return 8.0 * cp / (3.0 * sqrtpi) + 8.0 * u * u / (9.0 * cp * sqrtpi) - 44.0 * u * u * u * u / (135.0 * cp * cp * cp * sqrtpi);
    }
    else
    {
        return  exp(-u * u / kv(cp)) * cp * (5.0 * kv(cp) + 2.0 * kv(u)) / (sqrtpi * (3.0 * kv(cp) + 2.0 * kv(u))) +//
            (4.0 * kv(u) * kv(u) + 12.0 * cp * cp * kv(u) + 3.0 * kv(cp) * kv(cp)) * erf(u / cp) / (2.0 * u * (3.0 * kv(cp) + 2.0 * kv(u)));
    }
}

__device__ double HLLC_2d_Korolkov_b_s(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L,//
    const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& W, //
    double* P, double& PQ, const double& n1, const double& n2, const double& rad, int metod, double& Vl, double& Vc, double& Vp, bool nul_potok)
    // BestSeries
    // ������ ���������� 2� ���������
    //
    //  �����:
    // P[1]       // ��������
    // P[2]
    // P[0]       // �����
    // P[3]       // �������
{
    double t1 = n2;
    double t2 = -n1;

    double u1, v1, u2, v2;
    u1 = v1_L * n1 + v2_L * n2;
    v1 = v1_L * t1 + v2_L * t2;
    u2 = v1_R * n1 + v2_R * n2;
    v2 = v1_R * t1 + v2_R * t2;

    double sqrtroL = sqrt(ro_L);
    double sqrtroR = sqrt(ro_R);
    double cL = sqrt(ggg * p_L / ro_L);
    double cR = sqrt(ggg * p_R / ro_R);


    double uu_L = (kv(v1_L) + kv(v2_L)) / 2.0;
    double uu_R = (kv(v1_R) + kv(v2_R)) / 2.0;



    double SL = min(u1, u2) - max(cL, cR);
    double SR = max(u1, u2) + max(cL, cR);

    Vl = SL;
    Vp = SR;

    double suR = (SR - u2);
    double suL = (SL - u1);

    double SM = (suR * ro_R * u2 - suL * ro_L * u1 - p_R + p_L) //
        / (suR * ro_R - suL * ro_L);

    Vc = SM;

    double PTT = (suR * ro_R * p_L - suL * ro_L * p_R + ro_L * ro_R * suR * suL * (u2 - u1)) / (suR * ro_R - suL * ro_L);

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double FL[5], FR[5], UL[5], UR[5];

    double e1 = p_L / g1 + ro_L * uu_L;
    double e2 = p_R / g1 + ro_R * uu_R;


    FL[0] = ro_L * u1;
    FL[1] = ro_L * u1 * u1 + p_L;
    FL[2] = ro_L * u1 * v1;
    FL[3] = (e1 + p_L) * u1;
    FL[4] = Q_L * u1;

    FR[0] = ro_R * u2;
    FR[1] = ro_R * u2 * u2 + p_R;
    FR[2] = ro_R * u2 * v2;
    FR[3] = (e2 + p_R) * u2;
    FR[4] = Q_R * u2;


    UL[0] = ro_L;
    UL[1] = ro_L * u1;
    UL[2] = ro_L * v1;
    UL[3] = e1;
    UL[4] = Q_L;

    UR[0] = ro_R;
    UR[1] = ro_R * u2;
    UR[2] = ro_R * v2;
    UR[3] = e2;
    UR[4] = Q_R;

    if (SL >= W)
    {
        P[1] = n1 * (FL[1] - W * UL[1]) + t1 * (FL[2] - W * UL[2]);     // ��������
        P[2] = n2 * (FL[1] - W * UL[1]) + t2 * (FL[2] - W * UL[2]);
        P[0] = FL[0] - W * UL[0];                       // �����
        P[3] = FL[3] - W * UL[3];                       // �������
        PQ = FL[4] - W * UL[4];
        return time;
    }

    if (SR <= W)
    {
        P[1] = n1 * (FR[1] - W * UR[1]) + t1 * (FR[2] - W * UR[2]);     // ��������
        P[2] = n2 * (FR[1] - W * UR[1]) + t2 * (FR[2] - W * UR[2]);
        P[0] = FR[0] - W * UR[0];                       // �����
        P[3] = FR[3] - W * UR[3];                       // �������
        PQ = FR[4] - W * UR[4];
        return time;
    }


    double ro_LL = ro_L * (SL - u1) / (SL - SM);
    double ro_RR = ro_R * (SR - u2) / (SR - SM);
    double Q_LL = Q_L * (SL - u1) / (SL - SM);
    double Q_RR = Q_R * (SR - u2) / (SR - SM);


    double UZ0 = (SR * UR[0] - SL * UL[0] + FL[0] - FR[0]) / (SR - SL);
    double UZ1 = (SR * UR[1] - SL * UL[1] + FL[1] - FR[1]) / (SR - SL);
    double UZ2 = (SR * UR[2] - SL * UL[2] + FL[2] - FR[2]) / (SR - SL);
    double UZ3 = (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) / (SR - SL);
    double UZ4 = (SR * UR[4] - SL * UL[4] + FL[4] - FR[4]) / (SR - SL);
    double vzL, vzR, vLL, vRR, ppLR, ee1, ee2;

    if (metod == 0)
    {
        double  PO[5];
        for (int i = 0; i < 5; i++)
        {
            PO[i] = (SR * FL[i] - SL * FR[i] + SR * SL * (UR[i] - UL[i])) / (SR - SL);
        }

        P[1] = n1 * (PO[1] - W * UZ1) + t1 * (PO[2] - W * UZ2);     // ��������
        P[2] = n2 * (PO[1] - W * UZ1) + t2 * (PO[2] - W * UZ2);
        P[0] = PO[0] - W * UZ0;                       // �����
        P[3] = PO[3] - W * UZ3;                       // �������
        PQ = PO[4] - W * UZ4;
        return time;
    }


    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = ro_R * suRm;
    double rzL = ro_L * suLm;

    double ptzR = p_R + ro_R * suR * (SM - u2);
    double ptzL = p_L + ro_L * suL * (SM - u1);
    double ptz = (ptzR + ptzL) / 2.0;


    /*if( fabs(v1 - v2) > 0.1)
    {
        vLL = v1;
        vRR = v2;
    }
    else
    {
        vRR = UZ2 / UZ0;
        vLL = vRR;
    }*/


    if (nul_potok == true)
    {
        vRR = UZ2 / UZ0;
        vLL = vRR;
    }
    else
    {
        vLL = v1;
        vRR = v2;
    }



    ee2 = e2 * suRm + (ptz * SM - p_R * u2) / (SR - SM);
    ee1 = e1 * suLm + (ptz * SM - p_L * u1) / (SL - SM);


    double  ULL[5], URR[5], PO[5];
    ULL[0] = ro_LL;
    ULL[1] = ro_LL * SM;
    ULL[2] = ro_LL * vLL;
    ULL[3] = ee1;
    ULL[4] = Q_LL;

    URR[0] = ro_RR;
    URR[1] = ro_RR * SM;
    URR[2] = ro_RR * vRR;
    URR[3] = ee2;
    URR[4] = Q_RR;

    if (SL < W && SM >= W)
    {
        for (int i = 0; i < 5; i++)
        {
            PO[i] = FL[i] + SL * ULL[i] - SL * UL[i] - W * ULL[i];
        }
    }
    else if (SR > W && SM < W)
    {
        for (int i = 0; i < 5; i++)
        {
            PO[i] = FR[i] + SR * URR[i] - SR * UR[i] - W * URR[i];
        }
    }

    P[1] = n1 * PO[1] + t1 * PO[2];     // ��������
    P[2] = n2 * PO[1] + t2 * PO[2];
    P[0] = PO[0];                       // �����
    P[3] = PO[3];                       // �������
    PQ = PO[4];

    return time;
}


__device__ double HLLC_2d_Korolkov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad, bool nul_potok)
{
    double P[4];
    double PQ;
    double Vl, Vc, Vp;
    double t = HLLC_2d_Korolkov_b_s(Ls.x, 1.0, Ls.y, Lu.x, Lu.y, Rs.x, 1.0, Rs.y, Ru.x, Ru.y, 0.0, P, PQ, n1, n2, rad, 2, Vl, Vc, Vp, nul_potok);
    Ps.x = P[0];
    Ps.y = P[3];
    Pu.x = P[1];
    Pu.y = P[2];
    return t;
}