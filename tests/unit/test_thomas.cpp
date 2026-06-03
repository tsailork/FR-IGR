#include "../doctest.h"
#include "../../src/igr/adi_solver.hpp"
#include <vector>

TEST_CASE("Thomas algorithm - 3x3 known solution") {
    std::vector<double> a = {0.0, -1.0, -1.0};
    std::vector<double> b = {2.0, 2.0, 2.0};
    std::vector<double> c = {-1.0, -1.0, 0.0};
    std::vector<double> d = {1.0, 2.0, 3.0};
    std::vector<double> x(3, 0.0);

    solve_tridiagonal(a, b, c, d, x);

    CHECK(x[0] == doctest::Approx(2.5));
    CHECK(x[1] == doctest::Approx(4.0));
    CHECK(x[2] == doctest::Approx(3.5));
}

TEST_CASE("Thomas algorithm - Identity system") {
    std::vector<double> a = {0.0, 0.0, 0.0};
    std::vector<double> b = {1.0, 1.0, 1.0};
    std::vector<double> c = {0.0, 0.0, 0.0};
    std::vector<double> d = {5.0, 10.0, 15.0};
    std::vector<double> x(3, 0.0);

    solve_tridiagonal(a, b, c, d, x);

    CHECK(x[0] == doctest::Approx(5.0));
    CHECK(x[1] == doctest::Approx(10.0));
    CHECK(x[2] == doctest::Approx(15.0));
}

TEST_CASE("Thomas algorithm - Diagonal dominance") {
    std::vector<double> a = {0.0, 1.0, 1.0};
    std::vector<double> b = {4.0, 4.0, 4.0};
    std::vector<double> c = {1.0, 1.0, 0.0};
    std::vector<double> d = {5.0, 6.0, 5.0};
    std::vector<double> x(3, 0.0);

    solve_tridiagonal(a, b, c, d, x);

    CHECK(x[0] == doctest::Approx(1.0));
    CHECK(x[1] == doctest::Approx(1.0));
    CHECK(x[2] == doctest::Approx(1.0));
}

TEST_CASE("Thomas algorithm - Large system") {
    int N = 100;
    std::vector<double> a(N, -1.0);
    std::vector<double> b(N, 2.0);
    std::vector<double> c(N, -1.0);
    std::vector<double> d(N, 0.0);
    std::vector<double> x(N, 0.0);

    a[0] = 0.0;
    c[N-1] = 0.0;
    d[0] = 1.0; 
    d[N-1] = 1.0;

    solve_tridiagonal(a, b, c, d, x);

    CHECK(x[50] == doctest::Approx(1.0));
}
