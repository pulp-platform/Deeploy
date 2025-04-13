/*
 * Standalone implementation of expf(x) function
 * Based on FreeBSD's implementation from /usr/src/lib/msun/src/e_expf.c
 * Modified to remove all external dependencies
 */
#include "DeeployBasicMath.h"
 // Helper macros to replace libm.h functionality
 #define GET_FLOAT_WORD(i, f) do {                 \
     union {float f; uint32_t i;} __u;             \
     __u.f = (f);                                  \
     (i) = __u.i;                                  \
 } while (0)
 
 #define SET_FLOAT_WORD(f, i) do {                 \
     union {float f; uint32_t i;} __u;             \
     __u.i = (i);                                  \
     (f) = __u.f;                                  \
 } while (0)
 
 #define STRICT_ASSIGN(type, lval, rval) do {      \
     volatile type __v = (rval);                   \
     (lval) = __v;                                 \
 } while (0)
 
 #define FORCE_EVAL(x) do {                        \
     volatile float __dummy = (x);                 \
     (void)__dummy;                                \
 } while (0)
 
 // Replacement for scalbnf
 static float scalbnf(float x, int n) {
     union {float f; uint32_t i;} u;
     float y = x;
     
     if (n > 127) {
         y *= 0x1p127f;
         n -= 127;
         if (n > 127) {
             y *= 0x1p127f;
             n -= 127;
             if (n > 127)
                 n = 127;
         }
     } else if (n < -126) {
         y *= 0x1p-126f;
         n += 126;
         if (n < -126) {
             y *= 0x1p-126f;
             n += 126;
             if (n < -126)
                 n = -126;
         }
     }
     
     u.f = 1.0f;
     u.i = (uint32_t)(0x7f + n) << 23;
     return y * u.f;
 }
 
 // Main expf implementation
 float fexpf_openlibm(float x) {
     static const float
     half[2] = {0.5f, -0.5f},
     ln2hi   = 6.9314575195e-1f,  /* 0x3f317200 */
     ln2lo   = 1.4286067653e-6f,  /* 0x35bfbe8e */
     invln2  = 1.4426950216e+0f,  /* 0x3fb8aa3b */
     /*
      * Domain [-0.34568, 0.34568], range ~[-4.278e-9, 4.447e-9]:
      * |x*(exp(x)+1)/(exp(x)-1) - p(x)| < 2**-27.74
      */
     P1 =  1.6666625440e-1f, /*  0xaaaa8f.0p-26 */
     P2 = -2.7667332906e-3f; /* -0xb55215.0p-32 */
     
     float hi, lo, c, xx;
     int k, sign;
     uint32_t hx;
     
     GET_FLOAT_WORD(hx, x);
     sign = hx >> 31;   /* sign bit of x */
     hx &= 0x7fffffff;  /* high word of |x| */
     
     /* special cases */
     if (hx >= 0x42b17218) {  /* if |x| >= 88.722839f or NaN */
         if (hx > 0x7f800000)  /* NaN */
             return x;
         if (!sign) {
             /* overflow if x!=inf */
             STRICT_ASSIGN(float, x, x * 0x1p127f);
             return x;
         }
         if (hx == 0x7f800000)  /* -inf */
             return 0;
         if (hx >= 0x42cff1b5) { /* x <= -103.972084f */
             /* underflow */
             STRICT_ASSIGN(float, x, 0x1p-100f*0x1p-100f);
             return x;
         }
     }
     
     /* argument reduction */
     if (hx > 0x3eb17218) {  /* if |x| > 0.5 ln2 */
         if (hx > 0x3f851592)  /* if |x| > 1.5 ln2 */
             k = invln2*x + half[sign];
         else
             k = 1 - sign - sign;
         hi = x - k*ln2hi;  /* k*ln2hi is exact here */
         lo = k*ln2lo;
         STRICT_ASSIGN(float, x, hi - lo);
     } else if (hx > 0x39000000) {  /* |x| > 2**-14 */
         k = 0;
         hi = x;
         lo = 0;
     } else {
         /* raise inexact */
         FORCE_EVAL(0x1p127f + x);
         return 1 + x;
     }
     
     /* x is now in primary range */
     xx = x*x;
     c = x - xx*(P1+xx*P2);
     x = 1 + (x*c/(2-c) - lo + hi);
     if (k == 0)
         return x;
     return scalbnf(x, k);
 }
 
 // Example usage
 #ifdef TEST_EXPF
 #include <stdio.h>
 
 int main() {
     float values[] = {-5.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 5.0f};
     int count = sizeof(values) / sizeof(values[0]);
     
     for (int i = 0; i < count; i++) {
         printf("exp(%.1f) = %f\n", values[i], expf(values[i]));
     }
     
     return 0;
 }
 #endif