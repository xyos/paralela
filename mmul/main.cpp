#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <ctime>

using namespace std;
using namespace chrono;
typedef high_resolution_clock Clock;

class Mnn {
private:
    vector<double> _data;
    int _n;
public:
    Mnn(const int &n) {
        _n = n;
        _data.resize(_n*_n);
        fill(_data.begin(),_data.end(),0.0);
    }

    // returns size n of matrix
    int get_n()const {
        return _n;
    }
    int get_data_size()const {
        return _data.size();
    }

    // returns index from given row and col
    double operator()(const int& row, const int& col)const {
        return _data[row * _n + col];
    }
    double &operator()(const int& row, const int& col) {
        return _data[row * _n + col];
    }
    // returns index i of data vector
    double &operator[](const int& i){
        return _data[i];
    }
    double operator[](const int& i)const{
        return _data[i];
    }
    void operator=(const Mnn &m){
        _n = m._n;
        _data.resize(_n*_n);
        std::copy(m._data.begin(), m._data.end(),_data.begin());
    }
    // computes a == b where a is this, returns if a and b are equal
    bool operator==(const Mnn &b) {
        if(_n != b.get_n()) return false;
        for (int i = 0; i < _data.size(); i++) {
            if(_data[i] != b[i]){
                return false;
            }
        }
        return true;
    }


    // computes a * b where a is this, returns result
    Mnn operator*(const Mnn& b)const{
        Clock::time_point t1 = Clock::now();
        Mnn result(b.get_n());
        const Mnn *a = this;
        for (int i = 0; i < _n; i++) {
            for (int j = 0; j < _n; j++) {
               for (int k = 0; k < _n; k++) {
                   result(i,j) += (*this)(i,k) * b(k,j);
               }
            }
        }
        Clock::time_point t2 = Clock::now();
        duration<double> time = duration_cast<duration<double>>(t2-t1);
        printf("---- single thread time: %f seconds \n", time.count());
        return result;
    }

    void identity() {
        for (int i = 0; i < _n; i++) {
            for (int j = 0; j < _n; j++) {
                if(i==j){
                    (*this)(i,j) = 1.0;
                }
            }
        }
    }

    void randomize(){
        // Mersene Twister
        mt19937 generator;
        generator.seed(random_device{}());
        uniform_real_distribution<double> distribution(-1.0,1.0);
        for (int i = 0; i < _data.size(); i++) {
            _data[i] = distribution(generator);
        }
    }

    void set_row(const int& i, const int& j, double val){
        _data[i * _n + j] = val;
    }


    Mnn threadMult(const Mnn& b, const unsigned int& n_threads){
        Clock::time_point t1 = Clock::now();
        Mnn c(_n);
        const Mnn *a = this;
        const int size = _data.size();

        vector<thread> threads;
        for (unsigned int id = 0; id < n_threads ; id++) {
            threads.push_back(thread([id,a,b,&c,size,n_threads](){
                for (unsigned int i=id; i<size; i+=n_threads) {
                    int row = i/c.get_n();
                    int col = i-(row*c.get_n());
                    for(int rm = 0; rm < c.get_n(); rm++){
                        c(row,col) += (*a)(row,rm) * b(rm,col);
                    }
                }
            }));
        }
        for (auto& t : threads){
            t.join();
        }
        Clock::time_point t2 = Clock::now();
        duration<double> time = duration_cast<duration<double>>(t2-t1);
        printf("**** %d threads time: %f seconds \n", n_threads, time.count());
        return c;
    }

    void print(){
        for (int i = 0; i < _n; i++) {
            for (int j = 0; j < _n; j++) {
                printf("%f ", (*this)(i,j));
            }
            printf("\n");
        }
    }
};

int main(int argc, const char *argv[])
{
    unsigned int m_size  = atoi(argv[1]),
                 n_threads = atoi(argv[2]);
    printf("running for n:%d and threads:%d\n\n", m_size, n_threads);
    Mnn a(m_size), b(m_size);
    a.randomize();
    b.randomize();
    Mnn c = a * b;
    Mnn d = a.threadMult(b,n_threads);
    printf("TEST: %s", (c == d) ? "PASS": "FAIL");
    return 0;
}
