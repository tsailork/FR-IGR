#pragma once

#include <stdexcept>
#include <string>

namespace fr {

/**
 * @brief Base exception class for all FR-IGR solver errors.
 */
class SolverException : public std::runtime_error {
public:
    explicit SolverException(const std::string& message)
        : std::runtime_error(message) {}
    explicit SolverException(const char* message)
        : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when non-physical states (NaN/Inf, negative density/pressure) occur.
 */
class DivergenceException : public SolverException {
public:
    explicit DivergenceException(const std::string& message)
        : SolverException("Divergence Exception: " + message) {}
};

/**
 * @brief Exception thrown when input parameters or domain configuration files are invalid.
 */
class ConfigurationException : public SolverException {
public:
    explicit ConfigurationException(const std::string& message)
        : SolverException("Configuration Exception: " + message) {}
};

/**
 * @brief Exception thrown when a numerical algorithm (e.g. Gauss nodes, linear solver) fails to converge.
 */
class NumericalException : public SolverException {
public:
    explicit NumericalException(const std::string& message)
        : SolverException("Numerical Exception: " + message) {}
};

} // namespace fr
