#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h> // for timing
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <pthread.h>

//const
#define N 1024*1024+7
//#define N 1
#define NX 20


float random_float() 
{
    return ((float)rand() / (float)RAND_MAX) * (1.0f - FLT_TRUE_MIN) + FLT_TRUE_MIN;
}

/* Time function */
double now(){
   struct timeval t; double f_t;
   gettimeofday(&t, NULL); 
   f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
   return f_t; 
}

double dist(const float *U, const float *V, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        float u = *(U+i);
        float v = *(V+i);
        double numerator = u * u + v * v;
        double denominator = 1 + (u * v) * (u * v);
        sum += sqrt(numerator / denominator);
    }
    return sum;
}

double vec_dist(const float *U, const float *V, size_t n)
{
    double sum = 0.0;
    __m256d sum_vec = _mm256_setzero_pd();  // Accumulation vector for results

    for (size_t i = 0; i < n; i += 8)
    {
        // Load and square U and V vectors
        __m256d u_vec_i = _mm256_set_pd((double)U[i], (double)U[i + 1], (double)U[i + 2], (double)U[i + 3]);
        __m256d u_vec_s = _mm256_set_pd((double)U[i+4], (double)U[i + 5], (double)U[i + 6], (double)U[i + 7]);
        __m256d v_vec_i = _mm256_set_pd((double)V[i], (double)V[i + 1], (double)V[i + 2], (double)V[i + 3]);
        __m256d v_vec_s = _mm256_set_pd((double)V[i+4], (double)V[i + 5], (double)V[i + 6], (double)V[i + 7]);

        u_vec_i = _mm256_mul_pd(u_vec_i, u_vec_i);
        u_vec_s = _mm256_mul_pd(u_vec_s, u_vec_s);
        v_vec_i = _mm256_mul_pd(v_vec_i, v_vec_i);
        v_vec_s = _mm256_mul_pd(v_vec_s, v_vec_s);

        // Calculate numerator and denominator
        __m256d numerator_vec_i = _mm256_add_pd(u_vec_i, v_vec_i);
        __m256d numerator_vec_s = _mm256_add_pd(u_vec_s, v_vec_s);
        __m256d denominator_vec_i = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(u_vec_i, v_vec_i));
        __m256d denominator_vec_s = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(u_vec_s, v_vec_s));

        // Perform division and square root
        __m256d result_vec_i = _mm256_sqrt_pd(_mm256_div_pd(numerator_vec_i, denominator_vec_i));
        __m256d result_vec_s = _mm256_sqrt_pd(_mm256_div_pd(numerator_vec_s, denominator_vec_s));

        // Accumulate results into sum_vec
        sum_vec = _mm256_add_pd(sum_vec, result_vec_i);
        sum_vec = _mm256_add_pd(sum_vec, result_vec_s);
    }
    // Horizontal addition to get the final sum
    double *temp= (double*)&sum_vec[0];
    sum += temp[0] + temp[1] + temp[2] + temp[3];
    return sum;
}

double vec_dist_gen(const float *U, const float *V, size_t n)
{
    double sum = 0.0;
    __m256d sum_vec = _mm256_setzero_pd();  // Accumulation vector for results
    size_t i=0;
    for(i = 0; (i+8) <= n; i += 8)
    {
        // Load and square U and V vectors
        __m256d u_vec_i = _mm256_set_pd((double)U[i], (double)U[i + 1], (double)U[i + 2], (double)U[i + 3]);
        __m256d u_vec_s = _mm256_set_pd((double)U[i+4], (double)U[i + 5], (double)U[i + 6], (double)U[i + 7]);
        __m256d v_vec_i = _mm256_set_pd((double)V[i], (double)V[i + 1], (double)V[i + 2], (double)V[i + 3]);
        __m256d v_vec_s = _mm256_set_pd((double)V[i+4], (double)V[i + 5], (double)V[i + 6], (double)V[i + 7]);

        u_vec_i = _mm256_mul_pd(u_vec_i, u_vec_i);
        u_vec_s = _mm256_mul_pd(u_vec_s, u_vec_s);
        v_vec_i = _mm256_mul_pd(v_vec_i, v_vec_i);
        v_vec_s = _mm256_mul_pd(v_vec_s, v_vec_s);

        // Calculate numerator and denominator
        __m256d numerator_vec_i = _mm256_add_pd(u_vec_i, v_vec_i);
        __m256d numerator_vec_s = _mm256_add_pd(u_vec_s, v_vec_s);
        __m256d denominator_vec_i = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(u_vec_i, v_vec_i));
        __m256d denominator_vec_s = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(u_vec_s, v_vec_s));

        // Perform division and square root
        __m256d result_vec_i = _mm256_sqrt_pd(_mm256_div_pd(numerator_vec_i, denominator_vec_i));
        __m256d result_vec_s = _mm256_sqrt_pd(_mm256_div_pd(numerator_vec_s, denominator_vec_s));

        // Accumulate results into sum_vec
        sum_vec = _mm256_add_pd(sum_vec, result_vec_i);
        sum_vec = _mm256_add_pd(sum_vec, result_vec_s);
    }
    double *temp = (double*)&sum_vec;
    sum += temp[0] + temp[1] + temp[2] + temp[3];
    if(i<n)
    {
        sum+=dist(&U[i],&V[i],n-i);
    }
    return sum;
}

typedef struct 
{
    double sum;                
    pthread_mutex_t mutex;
} Sum;

Sum shared_sum;

typedef struct
{
    const float *U;
    const float *V;
    Sum *shared_sum;
    size_t start;
    size_t end;
    const size_t *n;
} ThreadData;

//thread function scalar
void* computeDistanceScalar(void* arg) 
{
    ThreadData *data = (ThreadData*)arg;
    const float *U = data->U;
    const float *V = data->V;
    Sum *shared_sum = data->shared_sum;

    // Local sum for each thread
    double local_sum = 0.0;

    for (size_t i = data->start; i < data->end; ++i) 
    {
        float u = U[i];
        float v = V[i];
        double numerator = u * u + v * v;
        double denominator = 1 + (u * v) * (u * v);
        local_sum += sqrt(numerator / denominator);
    }
    // Lock mutex and update the shared sum
    pthread_mutex_lock(&shared_sum->mutex);
    shared_sum->sum += local_sum;
    pthread_mutex_unlock(&shared_sum->mutex);
    return NULL;
}

void* computeDistanceVec(void* arg) 
{
    ThreadData *data = (ThreadData*)arg;
    const float *U = data->U;
    const float *V = data->V;
    const size_t n = *data->n;
    Sum *shared_sum = data->shared_sum;

    // Local sum for each thread
    double local_sum = 0.0;
    __m256d local_sum_vec = _mm256_setzero_pd();
    size_t i;

    // Vectorized loop processing 8 elements at a time
    for (i = data->start; i + 8 <= data->end; i += 8)
    {
        // Load U and V vectors (8 elements)
        __m256d u_vec_i = _mm256_set_pd((double)U[i], (double)U[i + 1], (double)U[i + 2], (double)U[i + 3]);
        __m256d u_vec_s = _mm256_set_pd((double)U[i+4], (double)U[i + 5], (double)U[i + 6], (double)U[i + 7]);
        __m256d v_vec_i = _mm256_set_pd((double)V[i], (double)V[i + 1], (double)V[i + 2], (double)V[i + 3]);
        __m256d v_vec_s = _mm256_set_pd((double)V[i+4], (double)V[i + 5], (double)V[i + 6], (double)V[i + 7]);

        // Calculate the square of each component
        u_vec_i = _mm256_mul_pd(u_vec_i, u_vec_i);
        u_vec_s = _mm256_mul_pd(u_vec_s, u_vec_s);
        v_vec_i = _mm256_mul_pd(v_vec_i, v_vec_i);
        v_vec_s = _mm256_mul_pd(v_vec_s, v_vec_s);

        // Calculate the numerator and denominator
        __m256d numerator_vec_i = _mm256_add_pd(u_vec_i, v_vec_i);
        __m256d numerator_vec_s = _mm256_add_pd(u_vec_s, v_vec_s);
        __m256d denominator_vec_i = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(u_vec_i, v_vec_i));
        __m256d denominator_vec_s = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(u_vec_s, v_vec_s));

        // Perform the division and square root operations
        __m256d result_vec_i = _mm256_sqrt_pd(_mm256_div_pd(numerator_vec_i, denominator_vec_i));
        __m256d result_vec_s = _mm256_sqrt_pd(_mm256_div_pd(numerator_vec_s, denominator_vec_s));

        // Accumulate results
        local_sum_vec = _mm256_add_pd(local_sum_vec, result_vec_i);
        local_sum_vec = _mm256_add_pd(local_sum_vec, result_vec_s);
    }

    // Handle remaining elements (less than 8) that do not fit into the vectorized loop
    for (; i < data->end; ++i)
    {
        float u = U[i];
        float v = V[i];
        double numerator = u * u + v * v;
        double denominator = 1 + (u * v) * (u * v);
        local_sum += sqrt(numerator / denominator);
    }

    // Reduce the vector sum into a scalar
    double *temp = (double*)&local_sum_vec;
    local_sum += temp[0] + temp[1] + temp[2] + temp[3];

    // Lock mutex and update the shared sum
    pthread_mutex_lock(&shared_sum->mutex);
    shared_sum->sum += local_sum;
    pthread_mutex_unlock(&shared_sum->mutex);

    return NULL;
}

void distPar(const float *U, const float *V, const size_t n, const int nb_threads, int mode)
{
    pthread_mutex_init(&shared_sum.mutex, NULL);
    pthread_t *threads = malloc(nb_threads * sizeof(pthread_t));
    ThreadData *thread_data = malloc(nb_threads * sizeof(ThreadData));
    size_t chunk_size = n / nb_threads;
    size_t rest_thread = n % 8;
    shared_sum.sum = 0.0;
    for (int i = 0; i < nb_threads; ++i) 
    {
        thread_data[i].U = U;
        thread_data[i].V = V;
        thread_data[i].n = &n;
        thread_data[i].shared_sum = &shared_sum;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == nb_threads - 1) ? n : (i + 1) * chunk_size;
        if (!mode) 
        {   
            pthread_create(&threads[i], NULL, computeDistanceScalar, &thread_data[i]);
        } 
        else 
        {
            pthread_create(&threads[i], NULL, computeDistanceVec, &thread_data[i]);
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < nb_threads; ++i) 
    {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&shared_sum.mutex);
    free(threads);
    free(thread_data);
}



void display(const float* U, size_t n)
{
    for(size_t i=0;i<n;++i)
    {
        printf("print u%zu=%8.4f\n",i,*(U+i));
    }
}

int main()
{
    //init
    srand((unsigned int)time(NULL));
    const size_t n = N;
    int nb_threads=NX;
    
    float *U;
    if (posix_memalign((void**)&U, 32, n * sizeof(float)) != 0) 
    {
        printf("Memory allocation of U failed\n");
        return 1;
    }

    float *V;
    if (posix_memalign((void**)&V, 32, n * sizeof(float)) != 0) 
    {
        printf("Memory allocation of U failed\n");
        return 1;
    }

    //filling the arrays
    for(size_t i=0;i<n;++i)
    {
        *(U+i)=random_float();
        *(V+i)=random_float();
    }

    printf("print n=%zu\n",n);
    printf("print n=%d\n",nb_threads);
    double sum=0.0, t0=0.0, t1=0, T1=0.0, T2=0.0, T3=0.0, T4=0, T5=0;
    
    t0=now(); sum=dist(U,V,n); t1=now(); T1=t1-t0;
    printf("sum=%8.4f, dist T1=%8.4f\n",sum, T1);

    t0=now(); sum=vec_dist(U,V,n); t1=now(); T2=t1-t0;
    printf("sum=%8.4f, vec_dist T2=%8.4f, acceleration=%8.4f\n",sum,T2,T1/T2);

    t0=now(); sum=vec_dist_gen(U,V,n); t1=now(); T3=t1-t0;
    printf("sum=%8.4f, vec_dist_gen T3=%8.4f, acceleration=%8.4f\n",sum,T3,T1/T3);

    t0=now(); distPar(U,V,n,nb_threads,0); t1=now(); T4=t1-t0;
    printf("sum=%8.4f, distPar mode=0 T4=%8.4f, acceleration=%8.4f\n",shared_sum.sum,T4,T1/T4);

    t0=now(); distPar(U,V,n,nb_threads,1); t1=now(); T5=t1-t0;
    printf("sum=%8.4f, distPar mode=1 T5=%8.4f, acceleration=%8.4f\n",shared_sum.sum,T5,T1/T5);

    free(U);
    free(V);
    return 0;
}