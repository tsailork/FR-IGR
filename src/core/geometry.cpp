/**
 * @file geometry.cpp
 * @brief Implementation of shape containment and intersection query functions.
 */

#include "geometry.hpp"
#include <cmath>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <iostream>
#include <filesystem>

namespace Geometry {

bool point_in_polygon(double x, double y, const std::vector<double>& px, const std::vector<double>& py) {
    if (px.empty()) return false;
    int n = px.size();
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (((py[i] > y) != (py[j] > y)) &&
            (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i])) {
            inside = !inside;
        }
    }
    return inside;
}

bool segment_intersects_aabb(double x1, double y1, double x2, double y2,
                             double xmin, double xmax, double ymin, double ymax) {
    double tmin = 0.0;
    double tmax = 1.0;
    double dx = x2 - x1;
    double dy = y2 - y1;

    // Check X boundaries
    for (int i = 0; i < 2; ++i) {
        double p = (i == 0) ? -dx : dx;
        double q = (i == 0) ? (x1 - xmin) : (xmax - x1);
        if (std::abs(p) < 1e-12) {
            if (q < 0.0) return false; // Parallel and outside
        } else {
            double t = q / p;
            if (p < 0.0) {
                if (t > tmax) return false;
                if (t > tmin) tmin = t;
            } else {
                if (t < tmin) return false;
                if (t < tmax) tmax = t;
            }
        }
    }

    // Check Y boundaries
    for (int i = 0; i < 2; ++i) {
        double p = (i == 0) ? -dy : dy;
        double q = (i == 0) ? (y1 - ymin) : (ymax - y1);
        if (std::abs(p) < 1e-12) {
            if (q < 0.0) return false; // Parallel and outside
        } else {
            double t = q / p;
            if (p < 0.0) {
                if (t > tmax) return false;
                if (t > tmin) tmin = t;
            } else {
                if (t < tmin) return false;
                if (t < tmax) tmax = t;
            }
        }
    }

    return tmin <= tmax;
}

bool polygon_intersects_aabb(const std::vector<double>& px, const std::vector<double>& py,
                             double xmin, double xmax, double ymin, double ymax) {
    if (px.empty()) return false;
    int n = px.size();

    // 1. Is any polygon vertex inside the AABB?
    for (int i = 0; i < n; ++i) {
        if (px[i] >= xmin && px[i] <= xmax && py[i] >= ymin && py[i] <= ymax) {
            return true;
        }
    }

    // 2. Is any corner of the AABB inside the polygon?
    // (Handles the case where AABB is fully inside the polygon)
    if (point_in_polygon(xmin, ymin, px, py)) return true;

    // 3. Do any polygon edges intersect the AABB?
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (segment_intersects_aabb(px[j], py[j], px[i], py[i], xmin, xmax, ymin, ymax)) {
            return true;
        }
    }

    return false;
}

bool Naca::contains(double px, double py) const {
    double m = 0.0;
    double p = 0.0;
    double t = 0.12;

    if (naca_code.length() == 4 && std::all_of(naca_code.begin(), naca_code.end(), ::isdigit)) {
        m = (naca_code[0] - '0') * 0.01;
        p = (naca_code[1] - '0') * 0.1;
        t = ((naca_code[2] - '0') * 10 + (naca_code[3] - '0')) * 0.01;
    }

    static const double PI = 3.14159265358979323846;
    double alpha = aoa_deg * PI / 180.0;

    double dx_pt = px - x_le;
    double dy_pt = py - y_le;

    // Rotate query point by -alpha around leading edge
    double x_rot = dx_pt * std::cos(alpha) + dy_pt * std::sin(alpha);
    double y_rot = -dx_pt * std::sin(alpha) + dy_pt * std::cos(alpha);

    double xc = x_rot;
    double yc = y_rot;

    double phi = 0.0;

    if (xc < 0.0) {
        phi = std::sqrt(xc * xc + yc * yc);
    } else if (xc > chord) {
        phi = std::sqrt((xc - chord) * (xc - chord) + yc * yc);
    } else {
        double xc_frac = xc / chord;

        double yt = 5.0 * t * chord * (
            0.2969 * std::sqrt(xc_frac)
            - 0.1260 * xc_frac
            - 0.3516 * xc_frac * xc_frac
            + 0.2843 * xc_frac * xc_frac * xc_frac
            - 0.1015 * xc_frac * xc_frac * xc_frac * xc_frac
        );

        double y_camber = 0.0;
        if (m > 0.0 && p > 0.0) {
            if (xc_frac <= p) {
                y_camber = (m * chord / (p * p)) * (2.0 * p * xc_frac - xc_frac * xc_frac);
            } else {
                y_camber = (m * chord / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xc_frac - xc_frac * xc_frac);
            }
        }

        phi = std::abs(yc - y_camber) - yt;
    }
    return phi <= 0.0;
}

Polygon Naca::to_polygon(int num_points) const {
    double m = 0.0;
    double p = 0.0;
    double t = 0.12;

    if (naca_code.length() == 4 && std::all_of(naca_code.begin(), naca_code.end(), ::isdigit)) {
        m = (naca_code[0] - '0') * 0.01;
        p = (naca_code[1] - '0') * 0.1;
        t = ((naca_code[2] - '0') * 10 + (naca_code[3] - '0')) * 0.01;
    }

    static const double PI = 3.14159265358979323846;
    double alpha = aoa_deg * PI / 180.0;

    std::vector<double> x_upper, y_upper;
    std::vector<double> x_lower, y_lower;

    for (int i = 0; i <= num_points; ++i) {
        double theta = i * PI / num_points;
        double xc_frac = 0.5 * (1.0 - std::cos(theta));

        double yt = 5.0 * t * chord * (
            0.2969 * std::sqrt(xc_frac)
            - 0.1260 * xc_frac
            - 0.3516 * xc_frac * xc_frac
            + 0.2843 * xc_frac * xc_frac * xc_frac
            - 0.1015 * xc_frac * xc_frac * xc_frac * xc_frac
        );

        double y_camber = 0.0;
        double dyc_dxc = 0.0;
        if (m > 0.0 && p > 0.0) {
            if (xc_frac <= p) {
                y_camber = (m * chord / (p * p)) * (2.0 * p * xc_frac - xc_frac * xc_frac);
                dyc_dxc = (2.0 * m / (p * p)) * (p - xc_frac);
            } else {
                y_camber = (m * chord / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xc_frac - xc_frac * xc_frac);
                dyc_dxc = (2.0 * m / ((1.0 - p) * (1.0 - p))) * (p - xc_frac);
            }
        }

        double theta_c = std::atan(dyc_dxc);

        double xu = xc_frac * chord - yt * std::sin(theta_c);
        double yu = y_camber + yt * std::cos(theta_c);

        double xl = xc_frac * chord + yt * std::sin(theta_c);
        double yl = y_camber - yt * std::cos(theta_c);

        // Rotate and translate
        double xu_rot = xu * std::cos(alpha) - yu * std::sin(alpha) + x_le;
        double yu_rot = xu * std::sin(alpha) + yu * std::cos(alpha) + y_le;

        double xl_rot = xl * std::cos(alpha) - yl * std::sin(alpha) + x_le;
        double yl_rot = xl * std::sin(alpha) + yl * std::cos(alpha) + y_le;

        x_upper.push_back(xu_rot);
        y_upper.push_back(yu_rot);
        x_lower.push_back(xl_rot);
        y_lower.push_back(yl_rot);
    }

    std::vector<double> px, py;
    for (size_t i = 0; i < x_upper.size(); ++i) {
        px.push_back(x_upper[i]);
        py.push_back(y_upper[i]);
    }
    for (int i = (int)x_lower.size() - 2; i >= 0; --i) {
        px.push_back(x_lower[i]);
        py.push_back(y_lower[i]);
    }

    Polygon poly;
    poly.x = px;
    poly.y = py;
    return poly;
}

bool Naca::intersects_aabb(double xmin, double xmax, double ymin, double ymax) const {
    Polygon poly = to_polygon(50);
    return poly.intersects_aabb(xmin, xmax, ymin, ymax);
}

} // namespace Geometry

namespace fr::geometry {

void Triangle::update_bounds_and_normal() {
    double e0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    double e1[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
    normal[0] = e0[1] * e1[2] - e0[2] * e1[1];
    normal[1] = e0[2] * e1[0] - e0[0] * e1[2];
    normal[2] = e0[0] * e1[1] - e0[1] * e1[0];
    double len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (len > 1e-14) {
        normal[0] /= len;
        normal[1] /= len;
        normal[2] /= len;
    } else {
        normal[0] = 0.0; normal[1] = 0.0; normal[2] = 1.0;
    }

    bbox_min[0] = std::min({v0[0], v1[0], v2[0]});
    bbox_min[1] = std::min({v0[1], v1[1], v2[1]});
    bbox_min[2] = std::min({v0[2], v1[2], v2[2]});

    bbox_max[0] = std::max({v0[0], v1[0], v2[0]});
    bbox_max[1] = std::max({v0[1], v1[1], v2[1]});
    bbox_max[2] = std::max({v0[2], v1[2], v2[2]});
}

double Triangle::sq_distance_to_point(const double p[3], double closest_pt[3]) const {
    double B[3] = { v0[0], v0[1], v0[2] };
    double E0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    double E1[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
    double D[3] = { B[0] - p[0], B[1] - p[1], B[2] - p[2] };

    double a = E0[0]*E0[0] + E0[1]*E0[1] + E0[2]*E0[2];
    double b = E0[0]*E1[0] + E0[1]*E1[1] + E0[2]*E1[2];
    double c = E1[0]*E1[0] + E1[1]*E1[1] + E1[2]*E1[2];
    double d = E0[0]*D[0] + E0[1]*D[1] + E0[2]*D[2];
    double e = E1[0]*D[0] + E1[1]*D[1] + E1[2]*D[2];

    double det = a * c - b * b;
    double s = b * e - c * d;
    double t = b * d - a * e;

    if (s + t <= det) {
        if (s < 0.0) {
            if (t < 0.0) {
                if (d < 0.0) {
                    t = 0.0; s = (-d >= a ? 1.0 : -d / a);
                } else {
                    s = 0.0; t = (e >= 0.0 ? 0.0 : (-e >= c ? 1.0 : -e / c));
                }
            } else {
                s = 0.0;
                t = (e >= 0.0 ? 0.0 : (-e >= c ? 1.0 : -e / c));
            }
        } else if (t < 0.0) {
            t = 0.0;
            s = (d >= 0.0 ? 0.0 : (-d >= a ? 1.0 : -d / a));
        } else {
            double invDet = (std::abs(det) > 1e-14 ? 1.0 / det : 0.0);
            s *= invDet;
            t *= invDet;
        }
    } else {
        if (s < 0.0) {
            double tmp0 = b + d;
            double tmp1 = c + e;
            if (tmp1 > tmp0) {
                double numer = tmp1 - tmp0;
                double denom = a - 2.0 * b + c;
                s = (numer >= denom ? 1.0 : numer / denom);
                t = 1.0 - s;
            } else {
                s = 0.0;
                t = (tmp1 <= 0.0 ? 1.0 : (e >= 0.0 ? 0.0 : -e / c));
            }
        } else if (t < 0.0) {
            double tmp0 = b + e;
            double tmp1 = a + d;
            if (tmp1 > tmp0) {
                double numer = tmp1 - tmp0;
                double denom = a - 2.0 * b + c;
                t = (numer >= denom ? 1.0 : numer / denom);
                s = 1.0 - t;
            } else {
                t = 0.0;
                s = (tmp1 <= 0.0 ? 1.0 : (d >= 0.0 ? 0.0 : -d / a));
            }
        } else {
            double numer = (c + e) - (b + d);
            if (numer <= 0.0) {
                s = 0.0; t = 1.0;
            } else {
                double denom = a - 2.0 * b + c;
                s = (numer >= denom ? 1.0 : (denom > 1e-14 ? numer / denom : 0.0));
                t = 1.0 - s;
            }
        }
    }

    closest_pt[0] = B[0] + s * E0[0] + t * E1[0];
    closest_pt[1] = B[1] + s * E0[1] + t * E1[1];
    closest_pt[2] = B[2] + s * E0[2] + t * E1[2];

    double dx = p[0] - closest_pt[0];
    double dy = p[1] - closest_pt[1];
    double dz = p[2] - closest_pt[2];
    return dx * dx + dy * dy + dz * dz;
}

void CadSdfEngine::clear() {
    triangles_.clear();
    bvh_nodes_.clear();
    root_idx_ = 0;
}

bool CadSdfEngine::load_mesh(const std::string& filepath) {
    if (filepath.empty()) return false;
    std::string ext = "";
    size_t dot = filepath.rfind('.');
    if (dot != std::string::npos) {
        ext = filepath.substr(dot);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == ".stl") {
        return load_stl(filepath);
    } else if (ext == ".obj") {
        return load_obj(filepath);
    } else if (ext == ".csv" || ext == ".dat" || ext == ".txt") {
        return load_csv(filepath);
    }
    return false;
}

bool CadSdfEngine::load_stl(const std::string& filepath) {
    clear();
    std::string actual_path = filepath;
    if (!std::filesystem::exists(actual_path)) {
        if (std::filesystem::exists("./" + filepath)) {
            actual_path = "./" + filepath;
        }
    }
    std::ifstream is(actual_path, std::ios::binary);
    if (!is.is_open()) {
        std::cerr << "[CAD ERROR] Unable to open STL file at path: '" << filepath 
                  << "'. Current directory: " << std::filesystem::current_path() << "\n";
        return false;
    }

    // Check if ASCII or Binary STL by sampling first 1KB
    char sample_buf[1024];
    is.read(sample_buf, sizeof(sample_buf) - 1);
    std::streamsize bytes_read = is.gcount();
    sample_buf[bytes_read] = '\0';
    std::string sample(sample_buf, bytes_read);

    bool is_ascii = (sample.find("solid") != std::string::npos &&
                     (sample.find("facet") != std::string::npos || sample.find("endsolid") != std::string::npos));
    is.close();

    if (is_ascii) {
        std::ifstream ascii_is(actual_path);
        std::string line;
        Triangle tri;
        int v_idx = 0;
        int raw_lines = 0;
        while (std::getline(ascii_is, line)) {
            raw_lines++;
            std::stringstream ss(line);
            std::string token;
            ss >> token;
            if (token == "facet") {
                ss >> token; // "normal"
                ss >> tri.normal[0] >> tri.normal[1] >> tri.normal[2];
                v_idx = 0;
            } else if (token == "vertex") {
                if (v_idx == 0) ss >> tri.v0[0] >> tri.v0[1] >> tri.v0[2];
                else if (v_idx == 1) ss >> tri.v1[0] >> tri.v1[1] >> tri.v1[2];
                else if (v_idx == 2) ss >> tri.v2[0] >> tri.v2[1] >> tri.v2[2];
                v_idx++;
            } else if (token == "endfacet") {
                tri.update_bounds_and_normal();
                triangles_.push_back(tri);
            }
        }
    } else {
        // Binary STL
        std::ifstream bin_is(actual_path, std::ios::binary);
        uint32_t num_triangles = 0;
        bin_is.seekg(80, std::ios::beg);
        bin_is.read(reinterpret_cast<char*>(&num_triangles), sizeof(num_triangles));

        triangles_.reserve(num_triangles);
        for (uint32_t i = 0; i < num_triangles; ++i) {
            float norm[3], v0[3], v1[3], v2[3];
            uint16_t attr;
            bin_is.read(reinterpret_cast<char*>(norm), 12);
            bin_is.read(reinterpret_cast<char*>(v0), 12);
            bin_is.read(reinterpret_cast<char*>(v1), 12);
            bin_is.read(reinterpret_cast<char*>(v2), 12);
            bin_is.read(reinterpret_cast<char*>(&attr), 2);

            Triangle tri;
            tri.v0[0] = v0[0]; tri.v0[1] = v0[1]; tri.v0[2] = v0[2];
            tri.v1[0] = v1[0]; tri.v1[1] = v1[1]; tri.v1[2] = v1[2];
            tri.v2[0] = v2[0]; tri.v2[1] = v2[1]; tri.v2[2] = v2[2];
            tri.update_bounds_and_normal();
            triangles_.push_back(tri);
        }
    }

    return !triangles_.empty();
}

bool CadSdfEngine::load_obj(const std::string& filepath) {
    clear();
    std::ifstream is(filepath);
    if (!is.is_open()) return false;

    std::vector<std::vector<double>> verts;
    std::string line;

    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        if (token == "v") {
            double x, y, z;
            ss >> x >> y >> z;
            verts.push_back({x, y, z});
        } else if (token == "f") {
            std::vector<int> face_indices;
            std::string v_str;
            while (ss >> v_str) {
                size_t slash = v_str.find('/');
                if (slash != std::string::npos) v_str = v_str.substr(0, slash);
                int idx = std::stoi(v_str) - 1;
                face_indices.push_back(idx);
            }
            // Triangulate polygon fan
            for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                int i0 = face_indices[0];
                int i1 = face_indices[i];
                int i2 = face_indices[i + 1];
                if (i0 >= 0 && i0 < (int)verts.size() &&
                    i1 >= 0 && i1 < (int)verts.size() &&
                    i2 >= 0 && i2 < (int)verts.size()) {
                    Triangle tri;
                    tri.v0[0] = verts[i0][0]; tri.v0[1] = verts[i0][1]; tri.v0[2] = verts[i0][2];
                    tri.v1[0] = verts[i1][0]; tri.v1[1] = verts[i1][1]; tri.v1[2] = verts[i1][2];
                    tri.v2[0] = verts[i2][0]; tri.v2[1] = verts[i2][1]; tri.v2[2] = verts[i2][2];
                    tri.update_bounds_and_normal();
                    triangles_.push_back(tri);
                }
            }
        }
    }

    return !triangles_.empty();
}

bool CadSdfEngine::load_csv(const std::string& filepath) {
    clear();
    std::string actual_path = filepath;
    if (!std::filesystem::exists(actual_path)) {
        if (std::filesystem::exists("./" + filepath)) {
            actual_path = "./" + filepath;
        }
    }
    std::ifstream is(actual_path);
    if (!is.is_open()) {
        std::cerr << "[CAD ERROR] Unable to open CSV file at path: '" << filepath << "'\n";
        return false;
    }

    std::vector<std::vector<double>> verts;
    std::string line;

    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '/') continue;
        std::string clean_line = line;
        for (char& c : clean_line) {
            if (c == ',' || c == ';') c = ' ';
        }

        std::stringstream ss(clean_line);
        double x, y, z = 0.0;
        if (ss >> x >> y) {
            if (!(ss >> z)) z = 0.0;
            verts.push_back({x, y, z});
        }
    }

    if (verts.size() < 3) return false;

    // Extrude 2D polygon into 3D lateral triangles along Z in [-1, 1]
    for (size_t i = 0; i < verts.size(); ++i) {
        size_t next_i = (i + 1) % verts.size();
        double x0 = verts[i][0], y0 = verts[i][1];
        double x1 = verts[next_i][0], y1 = verts[next_i][1];

        Triangle tri1;
        tri1.v0[0] = x0; tri1.v0[1] = y0; tri1.v0[2] = -1.0;
        tri1.v1[0] = x0; tri1.v1[1] = y0; tri1.v1[2] =  1.0;
        tri1.v2[0] = x1; tri1.v2[1] = y1; tri1.v2[2] =  1.0;
        tri1.update_bounds_and_normal();
        triangles_.push_back(tri1);

        Triangle tri2;
        tri2.v0[0] = x0; tri2.v0[1] = y0; tri2.v0[2] = -1.0;
        tri2.v1[0] = x1; tri2.v1[1] = y1; tri2.v1[2] =  1.0;
        tri2.v2[0] = x1; tri2.v2[1] = y1; tri2.v2[2] = -1.0;
        tri2.update_bounds_and_normal();
        triangles_.push_back(tri2);
    }

    return !triangles_.empty();
}

void CadSdfEngine::transform(double scale, double translate_x, double translate_y, double translate_z,
                           double pitch_deg, double yaw_deg, double roll_deg) {
    if (triangles_.empty()) return;

    static const double PI = 3.14159265358979323846;
    double rad_p = pitch_deg * PI / 180.0;
    double rad_y = yaw_deg   * PI / 180.0;
    double rad_r = roll_deg  * PI / 180.0;

    double cp = std::cos(rad_p), sp = std::sin(rad_p);
    double cy = std::cos(rad_y), sy = std::sin(rad_y);
    double cr = std::cos(rad_r), sr = std::sin(rad_r);

    // Rotation Matrix R = Rz(roll) * Ry(pitch) * Rx(yaw)
    auto rotate = [&](double v[3]) {
        double x = v[0] * scale;
        double y = v[1] * scale;
        double z = v[2] * scale;

        // Yaw (X-axis)
        double y1 = y * cy - z * sy;
        double z1 = y * sy + z * cy;

        // Pitch (Y-axis)
        double x2 = x * cp + z1 * sp;
        double z2 = -x * sp + z1 * cp;

        // Roll (Z-axis)
        double x3 = x2 * cr - y1 * sr;
        double y3 = x2 * sr + y1 * cr;

        v[0] = x3 + translate_x;
        v[1] = y3 + translate_y;
        v[2] = z2 + translate_z;
    };

    for (auto& tri : triangles_) {
        rotate(tri.v0);
        rotate(tri.v1);
        rotate(tri.v2);
        tri.update_bounds_and_normal();
    }
}

void CadSdfEngine::build_bvh(int max_leaf_triangles) {
    bvh_nodes_.clear();
    if (triangles_.empty()) return;

    std::vector<Triangle> reordered_triangles;
    reordered_triangles.reserve(triangles_.size());

    std::vector<int> tri_indices(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i) tri_indices[i] = i;

    BvhNode root;
    bvh_nodes_.push_back(root);
    root_idx_ = 0;

    build_bvh_recursive(root_idx_, tri_indices, max_leaf_triangles, 0, reordered_triangles);
    triangles_ = std::move(reordered_triangles);
}

void CadSdfEngine::build_bvh_recursive(int node_idx, std::vector<int>& tri_indices, int max_leaf_triangles, int depth, std::vector<Triangle>& reordered_triangles) {
    if (tri_indices.empty()) return;

    // Compute bounding box for current node
    double bmin[3] = { 1e30,  1e30,  1e30 };
    double bmax[3] = {-1e30, -1e30, -1e30 };

    for (int idx : tri_indices) {
        const auto& tri = triangles_[idx];
        for (int k = 0; k < 3; ++k) {
            bmin[k] = std::min(bmin[k], tri.bbox_min[k]);
            bmax[k] = std::max(bmax[k], tri.bbox_max[k]);
        }
    }

    bvh_nodes_[node_idx].bbox_min[0] = bmin[0]; bvh_nodes_[node_idx].bbox_min[1] = bmin[1]; bvh_nodes_[node_idx].bbox_min[2] = bmin[2];
    bvh_nodes_[node_idx].bbox_max[0] = bmax[0]; bvh_nodes_[node_idx].bbox_max[1] = bmax[1]; bvh_nodes_[node_idx].bbox_max[2] = bmax[2];

    if ((int)tri_indices.size() <= max_leaf_triangles || depth > 32) {
        bvh_nodes_[node_idx].triangle_start = reordered_triangles.size();
        bvh_nodes_[node_idx].triangle_count = tri_indices.size();
        // Re-order triangle buffer contiguous for leaf
        for (int idx : tri_indices) {
            reordered_triangles.push_back(triangles_[idx]);
        }
        return;
    }

    // Split along longest axis
    double extent[3] = { bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2] };
    int axis = 0;
    if (extent[1] > extent[0]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;

    double mid = 0.5 * (bmin[axis] + bmax[axis]);

    std::vector<int> left_indices, right_indices;
    for (int idx : tri_indices) {
        double center = 0.5 * (triangles_[idx].bbox_min[axis] + triangles_[idx].bbox_max[axis]);
        if (center < mid) left_indices.push_back(idx);
        else right_indices.push_back(idx);
    }

    if (left_indices.empty() || right_indices.empty()) {
        left_indices.assign(tri_indices.begin(), tri_indices.begin() + tri_indices.size() / 2);
        right_indices.assign(tri_indices.begin() + tri_indices.size() / 2, tri_indices.end());
    }

    int left_idx = bvh_nodes_.size();
    bvh_nodes_.push_back(BvhNode());
    int right_idx = bvh_nodes_.size();
    bvh_nodes_.push_back(BvhNode());

    bvh_nodes_[node_idx].left_child = left_idx;
    bvh_nodes_[node_idx].right_child = right_idx;

    build_bvh_recursive(left_idx, left_indices, max_leaf_triangles, depth + 1, reordered_triangles);
    build_bvh_recursive(right_idx, right_indices, max_leaf_triangles, depth + 1, reordered_triangles);
}

void CadSdfEngine::query_bvh_recursive(int node_idx, const double p[3],
                                       double& min_sq_dist, double closest_pt[3],
                                       int& closest_tri_idx) const {
    if (node_idx < 0 || node_idx >= (int)bvh_nodes_.size()) return;
    const auto& node = bvh_nodes_[node_idx];

    // Compute lower bound distance to node AABB box
    double box_sq_dist = 0.0;
    for (int k = 0; k < 3; ++k) {
        if (p[k] < node.bbox_min[k]) {
            double d = node.bbox_min[k] - p[k];
            box_sq_dist += d * d;
        } else if (p[k] > node.bbox_max[k]) {
            double d = p[k] - node.bbox_max[k];
            box_sq_dist += d * d;
        }
    }

    if (box_sq_dist >= min_sq_dist) return;

    if (node.is_leaf()) {
        for (int i = 0; i < node.triangle_count; ++i) {
            int tri_idx = node.triangle_start + i;
            double cur_closest[3];
            double sq_d = triangles_[tri_idx].sq_distance_to_point(p, cur_closest);
            if (sq_d < min_sq_dist) {
                min_sq_dist = sq_d;
                closest_pt[0] = cur_closest[0];
                closest_pt[1] = cur_closest[1];
                closest_pt[2] = cur_closest[2];
                closest_tri_idx = tri_idx;
            }
        }
    } else {
        query_bvh_recursive(node.left_child, p, min_sq_dist, closest_pt, closest_tri_idx);
        query_bvh_recursive(node.right_child, p, min_sq_dist, closest_pt, closest_tri_idx);
    }
}

bool CadSdfEngine::ray_triangle_intersection(const double orig[3], const double dir[3],
                                            const Triangle& tri, double& t) const {
    static const double EPS = 1e-10;
    double e1[3] = { tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2] };
    double e2[3] = { tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2] };
    double pvec[3] = {
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0]
    };

    double det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
    if (std::abs(det) < EPS) return false;
    double inv_det = 1.0 / det;

    double tvec[3] = { orig[0] - tri.v0[0], orig[1] - tri.v0[1], orig[2] - tri.v0[2] };
    double u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
    if (u < 0.0 || u > 1.0) return false;

    double qvec[3] = {
        tvec[1] * e1[2] - tvec[2] * e1[1],
        tvec[2] * e1[0] - tvec[0] * e1[2],
        tvec[0] * e1[1] - tvec[1] * e1[0]
    };
    double v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * inv_det;
    if (v < 0.0 || u + v > 1.0) return false;

    t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
    return t > EPS;
}

int CadSdfEngine::compute_ray_intersections(const double p[3], const double dir[3]) const {
    int count = 0;
    for (const auto& tri : triangles_) {
        double t = 0.0;
        if (ray_triangle_intersection(p, dir, tri, t)) {
            count++;
        }
    }
    return count;
}

double CadSdfEngine::query_sdf(const double p[3], double normal[3]) const {
    if (triangles_.empty()) {
        if (normal) { normal[0] = 0.0; normal[1] = 0.0; normal[2] = 1.0; }
        return 1.0;
    }

    double min_sq_dist = 1e30;
    double closest_pt[3] = {0.0, 0.0, 0.0};
    int closest_tri_idx = -1;

    if (!bvh_nodes_.empty()) {
        query_bvh_recursive(root_idx_, p, min_sq_dist, closest_pt, closest_tri_idx);
    } else {
        for (size_t i = 0; i < triangles_.size(); ++i) {
            double cur_closest[3];
            double sq_d = triangles_[i].sq_distance_to_point(p, cur_closest);
            if (sq_d < min_sq_dist) {
                min_sq_dist = sq_d;
                closest_pt[0] = cur_closest[0];
                closest_pt[1] = cur_closest[1];
                closest_pt[2] = cur_closest[2];
                closest_tri_idx = i;
            }
        }
    }

    double dist = std::sqrt(min_sq_dist);
    double vec[3] = { p[0] - closest_pt[0], p[1] - closest_pt[1], p[2] - closest_pt[2] };

    // Surface normal output
    if (normal) {
        if (closest_tri_idx >= 0 && closest_tri_idx < (int)triangles_.size()) {
            normal[0] = triangles_[closest_tri_idx].normal[0];
            normal[1] = triangles_[closest_tri_idx].normal[1];
            normal[2] = triangles_[closest_tri_idx].normal[2];
        } else {
            normal[0] = (dist > 1e-12 ? vec[0] / dist : 0.0);
            normal[1] = (dist > 1e-12 ? vec[1] / dist : 0.0);
            normal[2] = (dist > 1e-12 ? vec[2] / dist : 1.0);
        }
    }

    // Determine sign: 2D cross-section polygon extraction and point-in-polygon testing for 2D solver classification.
    struct Pt2D { double x, y; };
    std::vector<Pt2D> poly_pts;

    for (const auto& tri : triangles_) {
        if (std::abs(tri.normal[2]) > 0.9) continue; // Skip Z-endcaps
        Pt2D pts[3] = { {tri.v0[0], tri.v0[1]}, {tri.v1[0], tri.v1[1]}, {tri.v2[0], tri.v2[1]} };
        for (int k = 0; k < 3; ++k) {
            bool exists = false;
            for (const auto& existing : poly_pts) {
                if (std::abs(existing.x - pts[k].x) < 1e-5 && std::abs(existing.y - pts[k].y) < 1e-5) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                poly_pts.push_back(pts[k]);
            }
        }
    }

    bool inside = false;
    if (poly_pts.size() >= 3) {
        double cx = 0.0, cy = 0.0;
        for (const auto& pt : poly_pts) { cx += pt.x; cy += pt.y; }
        cx /= poly_pts.size(); cy /= poly_pts.size();

        std::sort(poly_pts.begin(), poly_pts.end(), [cx, cy](const Pt2D& a, const Pt2D& b) {
            return std::atan2(a.y - cy, a.x - cx) < std::atan2(b.y - cy, b.x - cx);
        });

        std::vector<double> px(poly_pts.size()), py(poly_pts.size());
        for (size_t i = 0; i < poly_pts.size(); ++i) {
            px[i] = poly_pts[i].x;
            py[i] = poly_pts[i].y;
        }

        inside = Geometry::point_in_polygon(p[0], p[1], px, py);
    }

    if (inside) {
        return -dist; // Inside solid
    }
    return dist; // Fluid domain
}

} // namespace fr::geometry

