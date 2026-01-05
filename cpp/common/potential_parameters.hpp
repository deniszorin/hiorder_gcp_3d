#pragma once

#include "geometry_connectivity.hpp"

namespace ipc {

struct PotentialParameters {
    double alpha;
    double p;
    double epsilon;
    bool localized;
    bool one_sided;
};

// ****************************************************************************
// Potential blending/localization functions

template <typename F>
inline F H(const F& z)
{
    if (z < -1.0) {
        return make_constant(0.0, z);
    }
    if (z > 1.0) {
        return make_constant(1.0, z);
    }
    return ((2.0 - z) * (z + 1.0) * (z + 1.0)) / 4.0;
}

template <typename F>
inline F H_alpha(const F& t, const double alpha)
{
    return H(t / alpha);
}

template <typename F>
inline F h_local(const F& z)
{
    if (z > 1.0) {
        return make_constant(0.0, z);
    }
    return (2.0 * z + 1.0) * (z - 1.0) * (z - 1.0);
}

template <typename F>
inline F h_epsilon(const F& z, const double epsilon)
{
    return h_local(z / epsilon);
}

} // namespace ipc
