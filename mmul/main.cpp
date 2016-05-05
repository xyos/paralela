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

class mNN {
private:
    vector<double> _data;
    int _n;
public:
    mNN(const int &n) {
        _n = n;
        _data.resize(_n*_n);
        fill(_data.begin(),_data.end(),0.0);
    }

    // returns size n of matrix
    int get_n()const {
        return _n;
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
    // computes a == b where a is this, returns if a and b are equal
    bool operator==(const mNN &b) {
        if(_n != b.get_n()) return false;
        for (int i = 0; i < _data.size(); i++) {
            if(_data[i] != b[i]){
                return false;
            }
        }
        return true;
    }


    // computes a * b where a is this, returns result
    mNN operator*(const mNN& b)const{
        mNN result(b.get_n());
        const mNN *a = this;
        Clock::time_point t1 = Clock::now();
        for (int i = 0; i < _n; i++) {
            for (int j = 0; j < _n; j++) {
               for (int k = 0; k < _n; k++) {
                   result(i,j) += (*this)(i,k) * b(k,j);
               }
            }
        }
        Clock::time_point t2 = Clock::now();
        duration<double> time = duration_cast<duration<double>>(t2-t1);
        printf("single thread time: %f seconds \n", time.count());
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
    printf("running for n:%d and threads:%d", m_size, n_threads);
    mNN a(m_size), b(m_size);
    a.randomize();
    b.randomize();
    mNN c = a * b;
    printf("a*b:\n");
    return 0;
}


void thread_run(int i){
}

void runThreads(const unsigned int &n_threads, const unsigned int &m_size) {
    vector<thread> threads;
    for (unsigned int id = 0; id < n_threads ; id++) {
        for (unsigned int i = 0; i < m_size; i+=n_threads) {
            thread_run(i);
        }
    }
}
