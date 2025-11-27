#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX_JULIA_DIM 1024

struct Complex {
	float r;
	float i;
public:
    Complex(float a=0, float b=0) : r(a), i(b) {}
    Complex(const Complex& a) : r(a.r), i(a.i) {}
    Complex operator* (const Complex& a)
	{
		return Complex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	Complex operator+ (const Complex& a)
	{
		return Complex(r + a.r, i + a.i);
	}
    std::string str() {
        std::stringstream oss;
        oss << "(" << r << "," << i << ")";
        return oss.str();
    }
    float m2() {
        return r * r + i * i;
    }	//计算复数的模值
};


class JuliaSet {
    float *_graymap = nullptr;
    int _max_iter = 50;
    Complex _c;
    float _threshold = 2.0f;
    int _height;
    int _width;
    // float _scale = 0.001f;
    float _xmin = -1.5f;
    float _xmax = 1.5f;
    float _ymin = -1.5f;
    float _ymax = 1.5f;
public:
    JuliaSet(int width=MAX_JULIA_DIM,
        int height=MAX_JULIA_DIM, 
        int max_iter=50, 
        Complex c=Complex(), 
        float threshold=2.0f) {
            _graymap = new float[width * height];
            _max_iter = max_iter;
            _c = c;
            _threshold = threshold;
            _height = height;
            _width = width;
    }

    virtual ~JuliaSet() {
        if (_graymap != nullptr) {
            delete[] _graymap;
            _graymap = nullptr;
        }
    }

    float* graymap() {
        return _graymap;
    }

    std::string str() {
        if (_graymap == nullptr) {
            return std::string("Empty Graymap");
        }
        std::stringstream oss;
        for (int y = 0; y < _height; y++) {
            for (int x = 0; x < _width; x++) {
                oss << (_graymap[y * _width + x] > 0.5f ? "*" : " ");
            }
            oss << "\n";
        }
        return oss.str();
    }
    void compute_by_cpu() {
        for (int y = 0; y < _height; y++) {
            for (int x = 0; x < _width; x++) {
                float ix = _xmin + (float)x / (float)_width * (_xmax - _xmin);
                float iy = _ymin + (float)y / (float)_height * (_ymax - _ymin);
                Complex z = Complex(ix, iy) + _c;
                bool overflow = false;
                int iter = 0;
                printf("Computing pixel (%d, %d), initial z: %s\n", x, y, z.str().c_str());
                for (; iter < _max_iter && z.m2() <= _threshold * _threshold; iter++) {
                    z = z * z + _c;
                    // std::cout << "z: " << z.str() << ", m2:" << z.m2() << ", less than "
                    //     << _threshold * _threshold <<"\n";
                }

                std::cout << "Final z: " << z.str() << ", m2:" << z.m2() << ", vs "
                    << _threshold * _threshold <<"\n";
                
                _graymap[y * _width + x] = iter * 1.0 / float(_max_iter);
            }
        }
    }
};


void gen_julia_set() {
    JuliaSet julia_set(100, 100, 100, Complex(-0.1, 0.65), 10.0f);
    julia_set.compute_by_cpu();
    std::cout << julia_set.str();
}