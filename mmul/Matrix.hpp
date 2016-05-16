#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <random>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace chrono;
typedef high_resolution_clock Clock;
mutex arr_mutex;
int shm_id;

class Mnn {
private:
    // size n of matrix
    int _n;
    // n*n
    int _nn;

    void _loopFork(const int& t_forks, const int& n_forks, const Mnn& b){
        float* arr = (float*)shmat(shm_id, NULL, 0);
        const Mnn *a = this;
        for (unsigned int i=t_forks; i<_nn; i+=n_forks) {
            int row = i/b.get_n();
            int col = i-(row*b.get_n());
            for(int rm = 0; rm < b.get_n(); rm++){
                arr_mutex.lock();
                arr[row * _n + col] += (*a)(row,rm) * b(rm,col);
                arr_mutex.unlock();
            }
        }
    }

    void _fork(const int& n_forks,  const Mnn& b){
        int pid;

        for(int i=0; i<n_forks; i++){
            pid = fork();
            if(pid == -1){
                //error handling
            }
            else if(pid == 0){
                //child, do something
                _loopFork(i, n_forks, b);
                exit(0);
            }
            else{ //parent
            }
        }
        // Need to wait for all
        for(int i=0; i<n_forks; i++){
            wait(NULL);
        }
        wait(NULL);

    }
public:
    float* _data;

    Mnn(const int &n) {
        _n = n;
        _nn = n*n;
        _data = (float*)malloc(_nn * sizeof(float));
        for (int i = 0; i < _nn; i++) {
            _data[i] = 0.0;
        }
    }

    // returns size n of matrix
    int get_n()const {
        return _n;
    }
    int get_data_size()const {
        return _nn;
    }
    void set_data(float* arr){
        for (int i = 0; i < _nn; i++) {
            _data[i] = arr[i];
        }
    }

    // returns index from given row and col
    float operator()(const int& row, const int& col)const {
        return _data[row * _n + col];
    }
    float &operator()(const int& row, const int& col) {
        return _data[row * _n + col];
    }
    // returns index i of data vector
    float &operator[](const int& i){
        return _data[i];
    }
    float operator[](const int& i)const{
        return _data[i];
    }
    // computes a == b where a is this, returns if a and b are equal
    bool operator==(const Mnn &b) {
        if(_n != b.get_n()) return false;
        for (int i = 0; i < _nn; i++) {
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
        for (int i = 0; i < _n; i++) {
            for (int j = 0; j < _n; j++) {
               for (int k = 0; k < _n; k++) {
                   result(i,j) += (*this)(i,k) * b(k,j);
               }
            }
        }
        Clock::time_point t2 = Clock::now();
        duration<float> time = duration_cast<duration<float>>(t2-t1);
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
        for (int i = 0; i < _nn; i++) {
            _data[i] = (float)distribution(generator);
        }
    }

    Mnn forkMult(const Mnn& b, const unsigned int& n_threads){
        Clock::time_point t1 = Clock::now();
        int n_forks = n_threads;

        key_t key;
        key=ftok("matrix_c",1);
        shm_id = shmget(key, _nn*sizeof(float), 0666 | IPC_CREAT);
        float* arr = (float*)shmat(shm_id, NULL, 0);
        for (int i = 0; i < _nn; i++) {
            arr[i] = 0.0;
        }
        _fork(n_forks, b);
        //Attach to shared memory
        Clock::time_point t2 = Clock::now();
        Mnn c(_n);
        c.set_data(arr);
        /* Deallocate the shared memory segment.  */ 
        shmdt (arr); 
        shmctl (shm_id, IPC_RMID, 0); 
        duration<float> time = duration_cast<duration<float>>(t2-t1);
        printf("**** %d processes time: %f seconds \n", n_threads, time.count());
        return c;
    }

    Mnn threadMult(const Mnn& b, const unsigned int& n_threads){
        Clock::time_point t1 = Clock::now();
        Mnn c(_n);
        const Mnn *a = this;
        const int size = _nn;

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
        duration<float> time = duration_cast<duration<float>>(t2-t1);
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

