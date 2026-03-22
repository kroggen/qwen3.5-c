/* Inference for Qwen-3.5 model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define QWEN35_MAGIC 0x51773335

typedef struct {
    int dim;
    int n_heads;
    int n_kv_heads;
    int n_layer;
    int n_mlp;
    int vocab_size;
    int seq_len;
    float rope_theta;
    float rms_norm_eps;
    int tie_word_embeddings;
    int d_head;
    int n_linear_k_heads;
    int n_linear_v_heads;
    int d_linear_k;
    int d_linear_v;
    int linear_conv_kernel;
    int n_full_attn_layers;
    int n_linear_attn_layers;
} Config;

typedef struct {
    float* data;
    float* token_embedding_table;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* q_norm;
    float* k_norm;
    float* in_proj_qkv;
    float* in_proj_z;
    float* in_proj_b;
    float* in_proj_a;
    float* conv1d_weight;
    float* dt_bias;
    float* A_log;
    float* linear_norm;
    float* out_proj;
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} Weights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *gate;
    float *key_cache;
    float *value_cache;
    float *qkv;
    float *z;
    float *beta;
    float *g;
    float *linear_out;
    float *S;
    float *conv_state;
    float *delta_S;
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
    int* layer_types;
    int* attn_layer_indices;
    int* deltanet_layer_indices;
} Qwen35;

void malloc_run_state(RunState* s, Config* p) {
    int dim = p->dim;
    int kv_dim = p->n_kv_heads * p->d_head;
    int hidden_dim = p->n_mlp;
    int d_head = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    int attn_dim = p->n_heads * d_head;  // attention output dimension
    size_t n_kv_layers     = (size_t) p->n_full_attn_layers;
    size_t n_linear_layers = (size_t) p->n_linear_attn_layers;

    s->x      = calloc(dim,                             sizeof(float));
    s->xb     = calloc(attn_dim > dim ? attn_dim : dim, sizeof(float));
    s->xb2    = calloc(dim,                             sizeof(float));
    s->hb     = calloc(hidden_dim,                      sizeof(float));
    s->hb2    = calloc(hidden_dim,                      sizeof(float));
    s->q      = calloc(p->n_heads * d_head * 2,         sizeof(float));
    s->k      = calloc(kv_dim,                          sizeof(float));
    s->v      = calloc(kv_dim,                          sizeof(float));
    s->att    = calloc(p->n_heads * p->seq_len,         sizeof(float));
    s->logits = calloc(p->vocab_size,                   sizeof(float));
    s->gate   = calloc(p->n_heads * d_head,             sizeof(float));

    /* One KV cache block per full-attention layer (same indexing as wq/wk/wv/wo). */
    if (n_kv_layers > 0) {
        s->key_cache   = calloc(n_kv_layers * p->seq_len * kv_dim, sizeof(float));
        s->value_cache = calloc(n_kv_layers * p->seq_len * kv_dim, sizeof(float));
    }

    if (n_linear_layers > 0) {
        int key_dim   = p->n_linear_k_heads * p->d_linear_k;
        int value_dim = p->n_linear_v_heads * p->d_linear_v;

        s->qkv        = calloc(key_dim * 2 + value_dim, sizeof(float));
        s->z          = calloc(value_dim, sizeof(float));
        s->beta       = calloc(p->n_linear_v_heads, sizeof(float));
        s->g          = calloc(p->n_linear_v_heads, sizeof(float));
        s->linear_out = calloc(value_dim, sizeof(float));
        s->delta_S    = calloc(p->n_linear_v_heads * p->d_linear_k * p->d_linear_v, sizeof(float));
        s->S          = calloc(n_linear_layers * p->n_linear_v_heads * p->d_linear_k * p->d_linear_v, sizeof(float));
        s->conv_state = calloc(n_linear_layers * (key_dim * 2 + value_dim) * p->linear_conv_kernel, sizeof(float));
    }

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || !s->gate) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->gate);
    free(s->key_cache);
    free(s->value_cache);
    free(s->qkv);
    free(s->z);
    free(s->beta);
    free(s->g);
    free(s->linear_out);
    free(s->S);
    free(s->conv_state);
    free(s->delta_S);
}

void memory_map_weights(Weights *w, Config* p, float* ptr, int n_full_attn, int n_linear_attn) {
    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int key_dim = p->n_linear_k_heads * p->d_linear_k;
    int value_dim = p->n_linear_v_heads * p->d_linear_v;
    int conv_dim = key_dim * 2 + value_dim;

    w->data = ptr;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    w->rms_att_weight = ptr;
    ptr += p->n_layer * p->dim;

    w->wq = ptr;
    ptr += (long long)n_full_attn * p->dim * (p->n_heads * head_size * 2);

    w->wk = ptr;
    ptr += (long long)n_full_attn * p->dim * kv_dim;

    w->wv = ptr;
    ptr += (long long)n_full_attn * p->dim * kv_dim;

    w->wo = ptr;
    ptr += (long long)n_full_attn * (p->n_heads * head_size) * p->dim;

    w->q_norm = ptr;
    ptr += (long long)n_full_attn * head_size;

    w->k_norm = ptr;
    ptr += (long long)n_full_attn * head_size;

    w->in_proj_qkv = ptr;
    ptr += (long long)n_linear_attn * conv_dim * p->dim;

    w->in_proj_z = ptr;
    ptr += (long long)n_linear_attn * value_dim * p->dim;

    w->in_proj_b = ptr;
    ptr += (long long)n_linear_attn * p->n_linear_v_heads * p->dim;

    w->in_proj_a = ptr;
    ptr += (long long)n_linear_attn * p->n_linear_v_heads * p->dim;

    w->conv1d_weight = ptr;
    ptr += (long long)n_linear_attn * conv_dim * p->linear_conv_kernel;

    w->dt_bias = ptr;
    ptr += (long long)n_linear_attn * p->n_linear_v_heads;

    w->A_log = ptr;
    ptr += (long long)n_linear_attn * p->n_linear_v_heads;

    w->linear_norm = ptr;
    ptr += (long long)n_linear_attn * p->d_linear_v;

    w->out_proj = ptr;
    ptr += (long long)n_linear_attn * p->dim * value_dim;

    w->rms_ffn_weight = ptr;
    ptr += p->n_layer * p->dim;

    w->w1 = ptr;
    ptr += (long long)p->n_layer * p->dim * p->n_mlp;

    w->w2 = ptr;
    ptr += (long long)p->n_layer * p->n_mlp * p->dim;

    w->w3 = ptr;
    ptr += (long long)p->n_layer * p->dim * p->n_mlp;

    w->rms_final_weight = ptr;
    ptr += p->dim;

    w->wcls = p->tie_word_embeddings ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, Weights* weights,
                     int* fd, float** data, ssize_t* file_size, int** layer_types,
                     int** attn_layer_indices, int** deltanet_layer_indices) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    unsigned int magic;
    if (fread(&magic, sizeof(unsigned int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic != QWEN35_MAGIC) {
        fprintf(stderr, "Invalid magic number: 0x%x (expected 0x%x)\n", magic, QWEN35_MAGIC);
        exit(EXIT_FAILURE);
    }

    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 1) {
        fprintf(stderr, "Unsupported version: %d (expected 1)\n", version);
        exit(EXIT_FAILURE);
    }

    if (fread(&config->dim,                  sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_heads,              sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_kv_heads,           sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_layer,              sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_mlp,                sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->vocab_size,           sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->seq_len,              sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->rope_theta,           sizeof(float), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->rms_norm_eps,         sizeof(float), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->tie_word_embeddings,  sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->d_head,               sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_linear_k_heads,     sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_linear_v_heads,     sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->d_linear_k,           sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->d_linear_v,           sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->linear_conv_kernel,   sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_full_attn_layers,   sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&config->n_linear_attn_layers, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }

    *layer_types = calloc(config->n_layer, sizeof(int));
    for (int i = 0; i < config->n_layer; i++) {
        if (fread(&(*layer_types)[i], sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    }

    *attn_layer_indices = calloc(config->n_layer, sizeof(int));
    *deltanet_layer_indices = calloc(config->n_layer, sizeof(int));
    int la = 0, ld = 0;
    for (int i = 0; i < config->n_layer; i++) {
        if ((*layer_types)[i] == 1) {
            (*deltanet_layer_indices)[i] = ld++;
        } else {
            (*attn_layer_indices)[i] = la++;
        }
    }

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + 256 / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, config->n_full_attn_layers, config->n_linear_attn_layers);
}

void build_qwen35(Qwen35 *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size, &t->layer_types, &t->attn_layer_indices, &t->deltanet_layer_indices);
    malloc_run_state(&t->state, &t->config);
}

void free_qwen35(Qwen35* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
    free(t->layer_types);
    free(t->attn_layer_indices);
    free(t->deltanet_layer_indices);
}

float silu(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}

void gemma_rmsnorm(float* o, float* x, float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = (1.0f + weight[j]) * (ss * x[j]);
    }
}

void rmsnorm_gated(float* o, float* x, float* gate, float* weight, int n_heads, int d_v, float eps) {
    // Apply per-head: out = weight * (x / sqrt(mean(x^2) + eps)) * silu(gate)
    for (int h = 0; h < n_heads; h++) {
        float* x_h = x + h * d_v;
        float* gate_h = gate + h * d_v;
        float* o_h = o + h * d_v;

        // Compute variance
        float ss = 0.0f;
        for (int j = 0; j < d_v; j++) {
            ss += x_h[j] * x_h[j];
        }
        ss /= d_v;
        ss += eps;
        ss = 1.0f / sqrtf(ss);

        // Apply norm, weight, and gate
        for (int j = 0; j < d_v; j++) {
            float x_norm = ss * x_h[j];
            o_h[j] = weight[j] * x_norm * silu(gate_h[j]);
        }
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float matmul_scalar(float* x, float* w, int n) {
    float val = 0.0f;
    for (int i = 0; i < n; i++) {
        val += w[i] * x[i];
    }
    return val;
}

void l2norm(float* x, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss + 1e-6f);
    for (int i = 0; i < size; i++) {
        x[i] *= ss;
    }
}

void forward_attention_layer(Qwen35* model, int l, int la, int pos) {
    Config* p = &model->config;
    Weights* w = &model->weights;
    RunState* s = &model->state;
    float *x = s->x;
    int dim          = p->dim;
    int head_size    = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    int kv_dim       = p->n_kv_heads * head_size;
    int q_dim        = p->n_heads * head_size * 2;  // packs [q | gate] per head
    int attn_out_dim = p->n_heads * head_size;
    int kv_mul       = p->n_heads / p->n_kv_heads;
    int loff         = la * p->seq_len * kv_dim;
    float eps        = p->rms_norm_eps;
    float* key_cache_row   = s->key_cache   + loff + pos * kv_dim;
    float* value_cache_row = s->value_cache + loff + pos * kv_dim;

#ifdef DEBUG_ATTN
    if (l == 3) {
        fprintf(stderr, "DEBUG ATTN layer %d (la=%d pos=%d):\n", l, la, pos);
        fprintf(stderr, "  x[:5]: %.6f %.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], x[3], x[4]);
    }
#endif

    // weights for this layer
    float* rms_att_weight = w->rms_att_weight + (long long)l  * dim;
    float* wq             = w->wq             + (long long)la * dim * q_dim;
    float* wk             = w->wk             + (long long)la * dim * kv_dim;
    float* wv             = w->wv             + (long long)la * dim * kv_dim;
    float* wo             = w->wo             + (long long)la * attn_out_dim * dim;
    float* q_norm         = w->q_norm         + (long long)la * head_size;
    float* k_norm         = w->k_norm         + (long long)la * head_size;

    // pre-attention norm
    gemma_rmsnorm(s->xb, x, rms_att_weight, dim, eps);

#ifdef DEBUG_ATTN
    if (l == 3) {
        fprintf(stderr, "  xb[:5] (after norm): %.6f %.6f %.6f %.6f %.6f\n",
                s->xb[0], s->xb[1], s->xb[2], s->xb[3], s->xb[4]);
    }
#endif

    // QKV projections; wq output is [q | gate] interleaved per head
    matmul(s->q, s->xb, wq, dim, q_dim);
    matmul(s->k, s->xb, wk, dim, kv_dim);
    matmul(s->v, s->xb, wv, dim, kv_dim);

#ifdef DEBUG_ATTN
    if (l == 3) {
        fprintf(stderr, "  qg[:5]: %.6f %.6f %.6f %.6f %.6f\n",
                s->q[0], s->q[1], s->q[2], s->q[3], s->q[4]);
        fprintf(stderr, "  k[:5]: %.6f %.6f %.6f %.6f %.6f\n",
                s->k[0], s->k[1], s->k[2], s->k[3], s->k[4]);
        fprintf(stderr, "  v[:5]: %.6f %.6f %.6f %.6f %.6f\n",
                s->v[0], s->v[1], s->v[2], s->v[3], s->v[4]);
    }
#endif

    // split q into [q, gate] per head, apply per-head RMSNorm to q
    for (int h = 0; h < p->n_heads; h++) {
        float* q_ptr = s->q + h * head_size;
        float* gate_ptr = s->gate + h * head_size;
        for (int i = 0; i < head_size; i++) {
            q_ptr[i] = s->q[h * head_size * 2 + i];
            gate_ptr[i] = s->q[h * head_size * 2 + head_size + i];
        }
        gemma_rmsnorm(q_ptr, q_ptr, q_norm, head_size, eps);
    }

    // apply per-head RMSNorm to k
    for (int h = 0; h < p->n_kv_heads; h++) {
        float* k_ptr = s->k + h * head_size;
        gemma_rmsnorm(k_ptr, k_ptr, k_norm, head_size, eps);
    }

    // RoPE rotary positional embeddings on q and k
    float theta = p->rope_theta;
    for (int i = 0; i < head_size; i+=2) {
        float freq = 1.0f / powf(theta, (float)i / head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float q0 = q[i];
            float q1 = q[i+1];
            q[i]   = q0 * fcr - q1 * fci;
            q[i+1] = q0 * fci + q1 * fcr;
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            float* k = s->k + h * head_size;
            float k0 = k[i];
            float k1 = k[i+1];
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }

#ifdef DEBUG_ATTN
    if (l == 3) {
        fprintf(stderr, "  q[0,:5] after norm+rope: %.6f %.6f %.6f %.6f %.6f\n",
                s->q[0], s->q[1], s->q[2], s->q[3], s->q[4]);
        fprintf(stderr, "  k[0,:5] after rope: %.6f %.6f %.6f %.6f %.6f\n",
                s->k[0], s->k[1], s->k[2], s->k[3], s->k[4]);
    }
#endif

    // store k/v into the KV cache for this position
    memcpy(key_cache_row,   s->k, kv_dim * sizeof(float));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

    // multi-head attention
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
        float* q = s->q + h * head_size;
        float* att = s->att + h * p->seq_len;

        // dot-product scores, scaled
        for (int t = 0; t <= pos; t++) {
            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            att[t] = score;
        }

        // softmax over scores
        softmax(att, pos + 1);

        // weighted sum of values
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[t];
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }

        // gated output
        float* gate_ptr = s->gate + h * head_size;
        for (int i = 0; i < head_size; i++) {
            xb[i] *= sigmoid(gate_ptr[i]);
        }
    }

#ifdef DEBUG_ATTN
    if (l == 3) {
        fprintf(stderr, "  xb[:5] before o_proj: %.6f %.6f %.6f %.6f %.6f\n",
                s->xb[0], s->xb[1], s->xb[2], s->xb[3], s->xb[4]);
    }
#endif

    // output projection
    matmul(s->xb2, s->xb, wo, attn_out_dim, dim);

#ifdef DEBUG_ATTN
    if (l == 3) {
        fprintf(stderr, "  attn_out[:5]: %.6f %.6f %.6f %.6f %.6f\n",
                s->xb2[0], s->xb2[1], s->xb2[2], s->xb2[3], s->xb2[4]);
    }
#endif

    // add to residual
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb2[i];
    }
}

void forward_linear_attention_layer(Qwen35* model, int l, int ld, int pos) {
    Config* p = &model->config;
    Weights* w = &model->weights;
    RunState* s = &model->state;
    float *x = s->x;
    int dim = p->dim;
    int head_size = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    float eps = p->rms_norm_eps;

    int n_k_heads = p->n_linear_k_heads;
    int n_v_heads = p->n_linear_v_heads;
    int d_k = p->d_linear_k;
    int d_v = p->d_linear_v;
    int key_dim = n_k_heads * d_k;
    int value_dim = n_v_heads * d_v;
    int conv_dim = key_dim * 2 + value_dim;
    int conv_kernel = p->linear_conv_kernel;

    // weights for this layer
    float* rms_att_weight = w->rms_att_weight + (long long)l  * dim;
    float* in_proj_qkv    = w->in_proj_qkv    + (long long)ld * conv_dim * dim;
    float* in_proj_z      = w->in_proj_z      + (long long)ld * value_dim * dim;
    float* in_proj_b      = w->in_proj_b      + (long long)ld * n_v_heads * dim;
    float* in_proj_a      = w->in_proj_a      + (long long)ld * n_v_heads * dim;
    float* conv1d_weight  = w->conv1d_weight  + (long long)ld * conv_dim * conv_kernel;
    float* dt_bias        = w->dt_bias        + (long long)ld * n_v_heads;
    float* A_log          = w->A_log          + (long long)ld * n_v_heads;
    float* linear_norm    = w->linear_norm    + (long long)ld * d_v;
    float* out_proj       = w->out_proj       + (long long)ld * dim * value_dim;

    // state for this linear-attention block (same index as weight tensors)
    float* conv_state = s->conv_state + (long long)ld * conv_dim * conv_kernel;
    float* S = s->S + (long long)ld * n_v_heads * d_k * d_v;

    // pre-attention norm
    gemma_rmsnorm(s->xb, x, rms_att_weight, dim, eps);

    // project to qkv (pre-conv) and z (value gate)
    matmul(s->qkv, s->xb, in_proj_qkv, dim, conv_dim);
    matmul(s->z, s->xb, in_proj_z, dim, value_dim);

    // beta (write strength) per head
    for (int i = 0; i < n_v_heads; i++) {
        s->beta[i] = sigmoid(matmul_scalar(s->xb, in_proj_b + i * dim, dim));
    }

    // g (decay rate) per head: g = A * softplus(a + dt_bias), A < 0
    for (int i = 0; i < n_v_heads; i++) {
        float a_val = matmul_scalar(s->xb, in_proj_a + i * dim, dim);
        float A = -expf(A_log[i]);
        s->g[i] = A * softplus(a_val + dt_bias[i]);
    }

    // shift conv state buffer and apply depthwise conv1d + SiLU to qkv
    for (int i = 0; i < conv_dim; i++) {
        for (int j = 0; j < conv_kernel - 1; j++) {
            conv_state[i * conv_kernel + j] = conv_state[i * conv_kernel + j + 1];
        }
        conv_state[i * conv_kernel + conv_kernel - 1] = s->qkv[i];
    }

    // apply conv weights and SiLU
    float* qkv_conv = s->qkv;
    for (int i = 0; i < conv_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < conv_kernel; j++) {
            val += conv_state[i * conv_kernel + j] * conv1d_weight[i * conv_kernel + j];
        }
        qkv_conv[i] = silu(val);
    }

    // split conv output into q, k, v
    float* q = qkv_conv;
    float* k = qkv_conv + key_dim;
    float* v = qkv_conv + key_dim * 2;

    // L2-normalize q and k per group-head, scale q
    float scale = 1.0f / sqrtf((float)d_k);
    for (int h = 0; h < n_k_heads; h++) {
        float* k_h = k + h * d_k;
        float* q_h = q + h * d_k;
        l2norm(k_h, d_k);
        l2norm(q_h, d_k);
        for (int i = 0; i < d_k; i++) {
            q_h[i] *= scale;
        }
    }

    // GQA ratio: each k/q head is shared by r value heads; use h/r to index into q and k
    int r = (n_v_heads > n_k_heads) ? n_v_heads / n_k_heads : 1;

    // linear attention state update per head: decay S, delta rule write, then read out
    for (int h = 0; h < n_v_heads; h++) {
        float g_t = expf(s->g[h]);
        float beta_t = s->beta[h];

        float* S_h = S + h * d_k * d_v;
        float* q_h = q + (h / r) * d_k;
        float* k_h = k + (h / r) * d_k;
        float* v_h = v + h * d_v;

        // decay state
        for (int i = 0; i < d_k * d_v; i++) {
            S_h[i] *= g_t;
        }

        // delta = (v - S*k) * beta
        float* delta = s->delta_S + h * d_v;
        for (int j = 0; j < d_v; j++) {
            float dot = 0.0f;
            for (int i = 0; i < d_k; i++) {
                dot += S_h[i * d_v + j] * k_h[i];
            }
            delta[j] = (v_h[j] - dot) * beta_t;
        }

        // S += k outer delta
        for (int i = 0; i < d_k; i++) {
            for (int j = 0; j < d_v; j++) {
                S_h[i * d_v + j] += k_h[i] * delta[j];
            }
        }

        // out = S * q
        float* out_h = s->linear_out + h * d_v;
        for (int j = 0; j < d_v; j++) {
            float val = 0.0f;
            for (int i = 0; i < d_k; i++) {
                val += S_h[i * d_v + j] * q_h[i];
            }
            out_h[j] = val;
        }
    }

    // gated RMSNorm
    rmsnorm_gated(s->linear_out, s->linear_out, s->z, linear_norm, n_v_heads, d_v, eps);

    // output projection
    matmul(s->xb, s->linear_out, out_proj, value_dim, dim);

    // add to residual
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb[i];
    }
}

void forward_mlp_layer(Qwen35* model, int l) {
    Config* p = &model->config;
    Weights* w = &model->weights;
    RunState* s = &model->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->n_mlp;
    float eps = p->rms_norm_eps;

    // weights for this layer
    float* rms_ffn_weight = w->rms_ffn_weight + (long long)l * dim;
    float* w1             = w->w1             + (long long)l * dim * hidden_dim;
    float* w3             = w->w3             + (long long)l * dim * hidden_dim;
    float* w2             = w->w2             + (long long)l * hidden_dim * dim;

    // pre-FFN norm
    gemma_rmsnorm(s->xb, x, rms_ffn_weight, dim, eps);

    // gate (w1) and up (w3) projections
    matmul(s->hb,  s->xb, w1, dim, hidden_dim);
    matmul(s->hb2, s->xb, w3, dim, hidden_dim);

    // SwiGLU: silu(gate) * up
    for (int i = 0; i < hidden_dim; i++) {
        float val = s->hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= s->hb2[i];
        s->hb[i] = val;
    }

    // down projection
    matmul(s->xb, s->hb, w2, hidden_dim, dim);

    // add to residual
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb[i];
    }
}

float* forward(Qwen35* model, int token, int pos) {
    Config* p = &model->config;
    Weights* w = &model->weights;
    RunState* s = &model->state;
    float *x = s->x;
    int dim = p->dim;

    // token embedding lookup
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

#ifdef DEBUG_FORWARD
    fprintf(stderr, "DEBUG: Embedding[:5]: %.6f %.6f %.6f %.6f %.6f\n",
            x[0], x[1], x[2], x[3], x[4]);
#endif

    // transformer layers: attention (or linear attention) + MLP
    int la = 0, ld = 0;
    for (unsigned long long l = 0; l < p->n_layer; l++) {
        int layer_type = model->layer_types[l];

        if (layer_type == 1) {
            forward_linear_attention_layer(model, l, ld, pos);
            ld++;
        } else {
            forward_attention_layer(model, l, la, pos);
            la++;
        }

        forward_mlp_layer(model, l);

#ifdef DEBUG_FORWARD
        if (l < 5 || l == p->n_layer - 1) {
            fprintf(stderr, "DEBUG: After layer %llu x[:5]: %.6f %.6f %.6f %.6f %.6f\n",
                    l, x[0], x[1], x[2], x[3], x[4]);
        }
#endif
    }

#ifdef DEBUG_FORWARD
    fprintf(stderr, "DEBUG: Before final norm x[:5]: %.6f %.6f %.6f %.6f %.6f\n",
            x[0], x[1], x[2], x[3], x[4]);
#endif

    // final norm
    gemma_rmsnorm(x, x, w->rms_final_weight, dim, p->rms_norm_eps);

#ifdef DEBUG_FORWARD
    fprintf(stderr, "DEBUG: After final norm x[:5]: %.6f %.6f %.6f %.6f %.6f\n",
            x[0], x[1], x[2], x[3], x[4]);
#endif

    // project to logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (len > 0) {
            if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

typedef struct {
    const char* token;
    int id;
} SpecialToken;

static const SpecialToken SPECIAL_TOKENS[] = {
    {"<|endoftext|>", 248044},
    {"<|im_start|>", 248045},
    {"<|im_end|>", 248046},
    {"<|object_ref_start|>", 248047},
    {"<|object_ref_end|>", 248048},
    {"<|box_start|>", 248049},
    {"<|box_end|>", 248050},
    {"<|quad_start|>", 248051},
    {"<|quad_end|>", 248052},
    {"<|vision_start|>", 248053},
    {"<|vision_end|>", 248054},
    {"<|vision_pad|>", 248055},
    {"<|image_pad|>", 248056},
    {"<|video_pad|>", 248057},
    {NULL, 0}
};

void encode_segment(Tokenizer* t, char *text, int *tokens, int *n_tokens) {
    if (text[0] == '\0') return;

    char* str_buffer = malloc((t->max_token_length + 1) * sizeof(char));
    char* pos = text;

    while (*pos != '\0') {
        int best_len = 0;
        int best_id = -1;

        for (int len = 1; len <= t->max_token_length && pos[len-1] != '\0'; len++) {
            if ((pos[len-1] & 0xC0) == 0x80 && len < t->max_token_length && pos[len] != '\0') {
                continue;
            }

            strncpy(str_buffer, pos, len);
            str_buffer[len] = '\0';

            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1) {
                best_len = len;
                best_id = id;
            }
        }

        if (best_id != -1) {
            tokens[(*n_tokens)++] = best_id;
            pos += best_len;
        } else {
            tokens[(*n_tokens)++] = (unsigned char)*pos + 3;
            pos++;
        }
    }

    free(str_buffer);
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;

    char* segment = malloc(strlen(text) + 1);
    char* pos = text;

    while (*pos != '\0') {
        int found_special = 0;
        for (int i = 0; SPECIAL_TOKENS[i].token != NULL; i++) {
            size_t len = strlen(SPECIAL_TOKENS[i].token);
            if (strncmp(pos, SPECIAL_TOKENS[i].token, len) == 0) {
                tokens[(*n_tokens)++] = SPECIAL_TOKENS[i].id;
                pos += len;
                found_special = 1;
                break;
            }
        }

        if (!found_special) {
            size_t seg_len = 0;
            char* seg_start = pos;

            while (*pos != '\0') {
                int is_special_start = 0;
                for (int i = 0; SPECIAL_TOKENS[i].token != NULL; i++) {
                    if (strncmp(pos, SPECIAL_TOKENS[i].token, strlen(SPECIAL_TOKENS[i].token)) == 0) {
                        is_special_start = 1;
                        break;
                    }
                }
                if (is_special_start) break;
                pos++;
                seg_len++;
            }

            if (seg_len > 0) {
                strncpy(segment, seg_start, seg_len);
                segment[seg_len] = '\0';
                encode_segment(t, segment, tokens, n_tokens);
            }
        }
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(segment);
}

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare_prob(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Qwen35 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(model, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    fflush(stdout);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    } else {
        buffer[0] = '\0';
    }
}

void chat(Qwen35 *model, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    const int im_end_id = 248046;
    char system_prompt[512];
    char user_prompt[2048];
    int rendered_size = 8192;
    char* rendered_prompt = malloc(rendered_size * sizeof(char));
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(rendered_size * sizeof(int));
    int user_idx;

    int8_t user_turn = 1;
    int8_t first_turn = 1;
    int next = 0;
    int token;
    int pos = 0;
    long start = 0;
    int generated_tokens = 0;

    while (pos < steps) {

        if (user_turn) {
            if (first_turn) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }

            if (first_turn && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
                if (user_prompt[0] == '\0') { break; }
            }

            if (user_prompt[0] == '\0') { continue; }

            if (first_turn) {
                if (system_prompt[0] != '\0') {
                    snprintf(rendered_prompt, rendered_size,
                        "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                        system_prompt, user_prompt);
                } else {
                    snprintf(rendered_prompt, rendered_size,
                        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                        user_prompt);
                }
            } else {
                snprintf(rendered_prompt, rendered_size,
                    "<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                    user_prompt);
            }

            encode(tokenizer, rendered_prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            first_turn = 0;
            generated_tokens = 0;
            start = time_in_ms();
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        float* logits = forward(model, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens) {
            if (next == im_end_id || next == 2) {
                printf("\n");
                long end = time_in_ms();
                if (generated_tokens > 0 && (end - start) > 0) {
                    fprintf(stderr, "tok/s: %.2f\n", generated_tokens / (double)(end - start) * 1000);
                }
                user_turn = 1;
            } else {
                char* piece = decode(tokenizer, token, next);
                safe_printf(piece);
                fflush(stdout);
                generated_tokens++;
            }
        }
    }
    printf("\n");
    free(rendered_prompt);
    free(prompt_tokens);
}

#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   qwen35 <checkpoint> [options]\n");
    fprintf(stderr, "Example: qwen35 model.bin -y \"You are a helpful assistant.\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0 (greedy argmax)\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default max_seq_len\n");
    fprintf(stderr, "  -i <string> first user message (optional)\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -y <string> (optional) system prompt\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "chat";
    char *system_prompt = NULL;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Qwen35 model;
    build_qwen35(&model, checkpoint_path);
    if (steps == 0 || steps > model.config.seq_len) steps = model.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, model.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0) {
        generate(&model, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&model, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_qwen35(&model);
    return 0;
}
#endif
