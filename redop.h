//
// Created by kihiro on 2/5/20.
//

#ifndef DG_REDOP_H
#define DG_REDOP_H

#include <algorithm>
#include <iterator>
#include "legion.h"

using namespace Legion;

template <int n>
class ReductionSum {
  public:
    typedef struct LHS {
      public:
        LHS() {
            for (int i=0; i<n; i++) value[i] = 0.;
        }

        LHS(std::vector<rtype> vec) {
            for (int i=0; i<n; i++) value[i] = vec[i];
        }

        LHS &operator=(const LHS &lhs) {
            for (int i=0; i<n; i++) value[i] = lhs.value[i];
            return *this;
        }

        rtype value[n];
    } LHS;

    typedef LHS RHS;

    static const LHS identity;

    template<bool EXCLUSIVE>
    void static apply(LHS &lhs, RHS rhs) {
        for (auto i = 0; i < n; ++i) {
            SumReduction<rtype>::apply<EXCLUSIVE>(lhs.value[i], rhs.value[i]);
        }
    }

    template<bool EXCLUSIVE>
    void static fold(RHS &rhs1, RHS rhs2) {
        for (auto i = 0; i < n; ++i) {
            SumReduction<rtype>::fold<EXCLUSIVE>(rhs1.value[i], rhs2.value[i]);
        }
    }
};

template<int n>
const typename ReductionSum<n>::LHS ReductionSum<n>::identity = ReductionSum<n>::LHS();

// N_REDOP = ns*nb
// #define N_REDOP 4  // quad, p=0
// #define N_REDOP 16 // quad, p=1
// #define N_REDOP 36 // quad, p=2
// #define N_REDOP 64 // quad, p=3
// #define N_REDOP 5 // hex, p = 0
// #define N_REDOP 40 // hex, p = 1
#define N_REDOP 135// hex, p = 2
// #define N_REDOP 320 // hex, p = 3

#endif //DG_REDOP_H
