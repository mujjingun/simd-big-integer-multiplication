// This program multiplies two 300,000-size decimal integers
// Utilizing the Fast Number Theoretic Transform

#ifdef __GNUC__
#pragma GCC target("avx2")
#endif

#if __cpp_if_constexpr >= 201606
#define CONSTIF if constexpr
#else
#define CONSTIF if
#endif

#define AVX_TARGET __attribute__((target("avx2")))

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int LOG_MAX_SZ = 17;
constexpr int MAX_SZ = 1 << LOG_MAX_SZ;

// Helpers
constexpr int32_t modpow(int32_t x, int y, int32_t P) {
  int32_t r = x;
  for (--y; y; y >>= 1) {
    if (y & 1)
      r = int64_t(r) * x % P;
    x = int64_t(x) * x % P;
  }
  return r;
}

constexpr int32_t modinv(int32_t x, int32_t P) {
  return modpow(x, P - 2, P);
}

// Modular Arithmetic
// P: modulus, R: primitive root of P
// MR: Montgomery number, MRR : Inverse of MR
template <int32_t P_, int32_t R_>
struct Ring {
  // Convert to Montgomery form and back
  static constexpr int32_t montify(int32_t x) { return (int64_t(x) << 32) % P; }

  static constexpr int32_t unmontify(int32_t x) { return int64_t(x) * MRR % P; }

  static constexpr int32_t add(int32_t a, int32_t b) {
    int32_t c = a + b;
    return c < P ? c : c - P;
  }

  static constexpr int32_t sub(int32_t a, int32_t b) {
    int32_t c = P + a - b;
    return c < P ? c : c - P;
  }

  static constexpr int32_t mul(int32_t a, int32_t b) {
    int64_t x = int64_t(a) * b;
    int64_t s = ((x & (MR - 1)) * K) & (MR - 1);
    int64_t u = (x + s * P) >> 32;
    int32_t up = static_cast<int32_t>(u - P);
    return static_cast<int32_t>(up < 0 ? u : up);
  }

  static constexpr int32_t pow(int32_t a, int n) {
    int32_t r = ONE;
    for (; n; n >>= 1) {
      if (n & 1)
        r = mul(r, a);
      a = mul(a, a);
    }
    return r;
  }

  static constexpr int32_t inv(int32_t a) { return pow(a, P - 2); }

  AVX_TARGET static __m256i add(__m256i a, __m256i b) {
    auto add = _mm256_add_epi32(a, b);
    auto mmP = _mm256_set1_epi32(P);
    auto cmp = _mm256_cmpgt_epi32(mmP, add);
    return _mm256_sub_epi32(add, _mm256_andnot_si256(cmp, mmP));
  }

  AVX_TARGET static __m256i sub(__m256i a, __m256i b) {
    auto sub = _mm256_sub_epi32(a, b);
    auto cmp = _mm256_cmpgt_epi32(_mm256_setzero_si256(), sub);
    auto mmP = _mm256_set1_epi32(P);
    return _mm256_add_epi32(sub, _mm256_and_si256(cmp, mmP));
  }

  AVX_TARGET static __m256i mul(__m256i a, __m256i b) {
    auto mmK64 = _mm256_set1_epi64x(K);
    auto mmP64 = _mm256_set1_epi64x(P);
    auto shft_a = _mm256_bsrli_epi128(a, 4);
    auto shft_b = _mm256_bsrli_epi128(b, 4);
    auto ab_hi = _mm256_mul_epu32(shft_a, shft_b);
    auto s_hi = _mm256_mul_epu32(ab_hi, mmK64);
    auto u_hi = _mm256_add_epi64(_mm256_mul_epu32(s_hi, mmP64), ab_hi);

    auto ab_lo = _mm256_mul_epu32(a, b);
    auto s_lo = _mm256_mul_epu32(ab_lo, mmK64);
    auto u_lo = _mm256_add_epi64(_mm256_mul_epu32(s_lo, mmP64), ab_lo);

    auto mask = _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1);
    auto u = _mm256_blendv_epi8(_mm256_bsrli_epi128(u_lo, 4), u_hi, mask);
    auto mmP32 = _mm256_set1_epi32(P);
    auto cmp = _mm256_cmpgt_epi32(mmP32, u);
    return _mm256_sub_epi32(u, _mm256_andnot_si256(cmp, mmP32));
  }

  static __m256i pow(__m256i a, int n) {
    __m256i r = _mm256_set1_epi32(ONE);
    for (; n; n >>= 1) {
      if (n & 1)
        r = mul(r, a);
      a = mul(a, a);
    }
    return r;
  }

  static __m256i inv(__m256i a) { return pow(a, P - 2); }

  // N-th Primitive root of unity
  template <int f>
  static constexpr int32_t proot(int32_t N) {
    CONSTIF(f > 0) return pow(RRI, P / N);
    else return pow(RR, P / N);
  }

  template <int f>
  AVX_TARGET static __m256i roots(int N) {
    alignas(32) int32_t t[8] = {ONE};
    int32_t root = proot<f>(N);
    for (int i = 1; i < 8; ++i)
      t[i] = mul(t[i - 1], root);
    return _mm256_load_si256(reinterpret_cast<__m256i*>(t));
  }

  template <int f>
  AVX_TARGET static void ntt8(__m256i* a, int N) {
    constexpr auto w4 = proot<f>(4), w8 = proot<f>(8), w4w8 = mul(w4, w8);
    const auto f1 = _mm256_setr_epi32(ONE, ONE, ONE, w4, ONE, ONE, ONE, w4);
    const auto f2 = _mm256_setr_epi32(ONE, ONE, ONE, ONE, ONE, w8, w4, w4w8);
    const auto mmP = _mm256_set1_epi32(P);
    const auto mP = _mm_set1_epi32(P);
    for (int i = 0; i < N; ++i) {
      a[i] = _mm256_permutevar8x32_epi32(
          a[i], _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7));
      auto mm1 = _mm256_hadd_epi32(a[i], _mm256_setzero_si256());
      auto cmp = _mm256_cmpgt_epi32(mmP, mm1);
      mm1 = _mm256_sub_epi32(mm1, _mm256_andnot_si256(cmp, mmP));
      auto mm2 = _mm256_hsub_epi32(a[i], _mm256_setzero_si256());
      cmp = _mm256_cmpgt_epi32(_mm256_setzero_si256(), mm2);
      mm2 = _mm256_add_epi32(mm2, _mm256_and_si256(cmp, mmP));
      a[i] = _mm256_unpacklo_epi32(mm1, mm2);
      a[i] = mul(a[i], f1);
      auto s1 = _mm256_bsrli_epi128(a[i], 8);
      auto s2 = add(a[i], s1);
      auto s3 = sub(a[i], s1);
      auto s4 = _mm256_bslli_epi128(s3, 8);
      a[i] = _mm256_blend_epi32(s2, s4, 204);
      a[i] = mul(a[i], f2);
      auto m1 = _mm256_extracti128_si256(a[i], 0);
      auto m2 = _mm256_extracti128_si256(a[i], 1);
      auto m3 = _mm_add_epi32(m1, m2);
      auto c = _mm_cmpgt_epi32(mP, m3);
      m3 = _mm_sub_epi32(m3, _mm_andnot_si128(c, mP));
      auto m4 = _mm_sub_epi32(m1, m2);
      c = _mm_cmpgt_epi32(_mm_setzero_si128(), m4);
      m4 = _mm_add_epi32(m4, _mm_and_si128(c, mP));
      a[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(m3), (m4), 1);
    }
  }

  template <int f>
  AVX_TARGET static void conj_fft_rec(__m256i* __restrict__ out,
                                      __m256i const* __restrict__ in,
                                      size_t offs,
                                      size_t mask,
                                      size_t stride,
                                      size_t N) {
    if (N == 1) {
      out[0] = in[offs & mask];
    } else if (N == 2) {
      auto a = in[offs & mask];
      auto b = in[(offs + stride) & mask];
      out[0] = add(a, b);
      out[1] = sub(a, b);
    } else if (N == 4) {
      auto x0 = in[offs & mask];
      auto x1 = in[(offs + stride) & mask];
      auto x2 = in[(offs + stride * 2) & mask];
      auto x3 = in[(offs + stride * 3) & mask];

      auto A = add(x0, x2);
      auto B = sub(x0, x2);
      auto C = add(x1, x3);
      auto D = mul(_mm256_set1_epi32(proot<-f>(4)), sub(x1, x3));

      out[0] = add(A, C);
      out[1] = add(B, D);
      out[2] = sub(A, C);
      out[3] = sub(B, D);
    } else {
      conj_fft_rec<f>(out, in, offs, mask, stride * 2, N / 2);
      conj_fft_rec<f>(out + N / 2, in, offs + stride, mask, stride * 4, N / 4);
      conj_fft_rec<f>(out + 3 * N / 4, in, offs - stride, mask, stride * 4,
                      N / 4);

      auto const p4 = proot<f>(4);
      for (size_t k = 0; k < N / 4; k++) {
        auto Uk = out[k];
        auto Zk = out[k + N / 2];
        auto Uk2 = out[k + N / 4];
        auto Zdk = out[k + 3 * N / 4];
        auto const w = T.s_twiddles[N / 4 + k][f < 0];
        auto const wi = T.s_twiddles[N / 4 + k][f > 0];

        // Twiddle
        Zk = mul(Zk, _mm256_set1_epi32(w));
        Zdk = mul(Zdk, _mm256_set1_epi32(wi));

        // Z butterflies
        auto Zsum = add(Zk, Zdk);
        auto Zdif = mul(_mm256_set1_epi32(p4), sub(Zk, Zdk));

        out[k] = add(Uk, Zsum);
        out[k + N / 2] = sub(Uk, Zsum);
        out[k + N / 4] = sub(Uk2, Zdif);
        out[k + 3 * N / 4] = add(Uk2, Zdif);
      }
    }
  }

  template <int f>
  AVX_TARGET static void conj_fft(__m256i* out, int N) {
    alignas(32) static __m256i in[MAX_SZ / 8];
    std::copy(out, out + N, in);
    conj_fft_rec<f>(out, in, 0, N - 1, 1, N);
    std::reverse(out + 1, out + N);
    if (f == -1) {
      auto z = _mm256_set1_epi32(inv(montify(N)));
      for (int i = 0; i < N; ++i) {
        out[i] = mul(z, out[i]);
      }
    }
  }

  // Performs the 8-step Number Theoretic Transform
  // transposed as a 8 x N/8 matrix in parallel.
  AVX_TARGET static void NTT(int32_t* a, int N) {
    auto* va = reinterpret_cast<__m256i*>(a);
    conj_fft<1>(va, N / 8);
    // Apply twiddle factors and perform 8-point fft
    const __m256i wN = roots<1>(N);
    auto w = _mm256_set1_epi32(ONE);
    for (int i = 0; i < N / 8; ++i) {
      va[i] = mul(va[i], w);
      w = mul(w, wN);
    }
    ntt8<1>(va, N / 8);
  }

  AVX_TARGET static void iNTT(int32_t* a, int N) {
    // Perform a 8-point FFT and apply twiddle factors
    auto* va = reinterpret_cast<__m256i*>(a);
    ntt8<-1>(va, N / 8);
    const __m256i wN = roots<-1>(N);
    auto w = _mm256_set1_epi32(ONE);
    for (int i = 0; i < N / 8; ++i) {
      va[i] = mul(va[i], w);
      w = mul(w, wN);
    }
    conj_fft<-1>(va, N / 8);
    // Normalize
    auto z = _mm256_set1_epi32(montify(modinv(8, P)));
    for (int i = 0; i < N / 8; i++)
      va[i] = mul(va[i], z);
  }

  AVX_TARGET static void polymul_ring(int32_t* f, int fn, int32_t* g, int gn) {
    int N = 8;
    while (N < fn + gn + 1)
      N *= 2;
    for (int i = 0; i < fn; ++i)
      f[i] = montify(f[i]);
    for (int i = 0; i < gn; ++i)
      g[i] = montify(g[i]);
    NTT(f, N);
    NTT(g, N);
    auto* va = reinterpret_cast<__m256i*>(f);
    auto* vb = reinterpret_cast<__m256i*>(g);
    for (int i = 0; i < N / 8; ++i)
      va[i] = mul(va[i], vb[i]);

    iNTT(f, N);
  }

  static constexpr int64_t MR = 1LL << 32;
  static constexpr int32_t P = P_, R = R_;
  static constexpr int32_t MRR = modinv(MR % P, P);
  static constexpr int32_t K = (int64_t(MR) * MRR - 1) / P;
  static constexpr int32_t ONE = MR % P;
  static constexpr int32_t RR = int64_t(R) * MR % P;
  static constexpr int32_t RRI = inv(RR);

  struct twiddle_lookup_table {
    constexpr twiddle_lookup_table() : s_twiddles{} {
      for (int N = 1; N <= MAX_SZ / 4; N *= 2) {
        auto w = proot<-1>(N * 4);
        auto wi = proot<1>(N * 4);
        int32_t wk = ONE, wik = ONE;
        for (int k = 0; k < N; k++) {
          s_twiddles[N + k][0] = wk;
          s_twiddles[N + k][1] = wik;
          wk = mul(wk, w);
          wik = mul(wik, wi);
        }
      }
    }

    int32_t s_twiddles[(MAX_SZ / 4) * 2][2];
  };

  static constexpr twiddle_lookup_table T{};
};
template <int32_t P_, int32_t R_>
constexpr typename Ring<P_, R_>::twiddle_lookup_table Ring<P_, R_>::T;

using R1 = Ring<469762049, 3>;
using R2 = Ring<754974721, 11>;

int from_chars(char* s, char* l) {
  int n = 0;
  while (s < l) {
    n = n * 10 + *s - '0';
    s++;
  }
  return n;
}

int parse(char* s, int l, int* a) {
  int i, j = 0;
  for (i = l; i >= 6; i -= 6) {
    a[j++] = from_chars(s + i - 6, s + i);
  }
  a[j++] = from_chars(s, s + i);
  return j;
}

int string_mul(char* s1, int l1, char* s2, int l2) {
  alignas(32) int32_t F[MAX_SZ]{0}, G[MAX_SZ]{0};
  alignas(32) int32_t a[MAX_SZ]{0}, b[MAX_SZ]{0};
  int64_t R[100010]{0};
  int i, t;
  l1 = parse(s1, l1, a);
  l2 = parse(s2, l2, b);
  std::copy(a, a + l1, F);
  std::copy(b, b + l2, G);
  R1::polymul_ring(a, l1, b, l2);
  R2::polymul_ring(F, l1, G, l2);
  // Apply the Chinese Remainder Theorem
  constexpr int32_t X1 = R1::inv(R1::montify(R2::P));
  constexpr int32_t X2 = R2::inv(R2::montify(R1::P));
  constexpr int64_t M = int64_t(R1::P) * R2::P;
  for (int i = 0; i < l1 + l2 + 1; ++i) {
    int64_t t1 = R1::unmontify(R1::mul(a[i], X1)) * int64_t(R2::P);
    int64_t t2 = R2::unmontify(R2::mul(F[i], X2)) * int64_t(R1::P);
    R[i] = t1 + t2;
    R[i] = R[i] < M ? R[i] : R[i] - M;
  }
  for (i = 0; i < l1 + l2 + 1; ++i) {
    R[i + 1] += R[i] / 1000000;
    R[i] %= 1000000;
  }
  while (!R[--i] && i > 0)
    ;
  sprintf(s1, "%ld%n", R[i--], &t);
  for (int j = 5; j >= 0; --j) {
    for (int k = i, l = t; k >= 0; --k, l += 6)
      s1[l + j] = (R[k] % 10) | 48, R[k] /= 10;
  }
  s1[t + i * 6 + 6] = 0;
  return t + i * 6 + 6;
}

constexpr int SIZE = 300016;
int main() {
  char s1[SIZE * 2] = {0}, *s2 = s1 + SIZE;
  scanf("%s%s", s1, s2);
  int l1 = strlen(s1), l2 = strlen(s2);
  int len = string_mul(s1, l1, s2, l2);
  fwrite(s1, 1, len, stdout);
  return 0;
}
