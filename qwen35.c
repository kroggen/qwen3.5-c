/* Inference for Qwen-3.5 model in pure C */
/* Loads weights directly from safetensors format */

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
    #include <dirent.h>
#endif

#include "csafetensors.h"
#include "json.h"

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
    int* layer_types;
    int* attn_layer_indices;
    int* deltanet_layer_indices;
    csafetensors_t safetensors;
    int use_safetensors;
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

static int load_tensor_auto(const csafetensors_t *st, const char *name, float *dest, size_t expected_size) {
    const csafetensors_tensor_t *tensor = csafetensors_get_tensor(st, name);
    if (!tensor) return -1;

    const uint8_t *data = csafetensors_get_tensor_data(st, tensor);
    if (!data) return -1;

    size_t num_elements = csafetensors_shape_size(tensor);

    if (expected_size > 0 && num_elements != expected_size) {
        fprintf(stderr, "Tensor %s size mismatch: got %zu, expected %zu\n", name, num_elements, expected_size);
    }

    if (tensor->dtype == CSAFETENSORS_DTYPE_BFLOAT16) {
        const uint16_t *bf16_data = (const uint16_t *)data;
        for (size_t i = 0; i < num_elements; i++) {
            dest[i] = csafetensors_bf16_to_f32(bf16_data[i]);
        }
    } else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT16) {
        const uint16_t *f16_data = (const uint16_t *)data;
        for (size_t i = 0; i < num_elements; i++) {
            dest[i] = csafetensors_f16_to_f32(f16_data[i]);
        }
    } else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT32) {
        memcpy(dest, data, num_elements * sizeof(float));
    } else {
        fprintf(stderr, "Unsupported dtype for tensor %s\n", name);
        return -1;
    }

    return 0;
}

static int get_layer_type(int layer_idx, const JsonValue *layer_types) {
    if (!layer_types || layer_types->type != JSON_ARRAY) return 0;
    if (layer_idx >= (int)layer_types->data.array.count) return 0;
    JsonValue *lt = json_array_get(layer_types, layer_idx);
    if (!lt || lt->type != JSON_STRING) return 0;
    const char *type_str = lt->data.string;
    if (strcmp(type_str, "linear_attention") == 0) return 1;
    return 0;
}

static char* find_file_in_dir(const char *dir, const char *name) {
    static char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return path;
    }
    return NULL;
}

static int ends_with_safetensors(const char *name) {
    size_t len = strlen(name);
    return len > 12 && strcmp(name + len - 12, ".safetensors") == 0;
}

static int starts_with_model(const char *name) {
    return strncmp(name, "model", 5) == 0;
}

static int compare_strings(const void *a, const void *b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

static char** find_safetensors_files(const char *model_dir, int *count) {
    DIR *dir = opendir(model_dir);
    if (!dir) return NULL;

    char **files = NULL;
    *count = 0;
    int capacity = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG && entry->d_type != DT_LNK) continue;
        if (!starts_with_model(entry->d_name)) continue;
        if (!ends_with_safetensors(entry->d_name)) continue;

        if (*count >= capacity) {
            capacity = capacity ? capacity * 2 : 16;
            files = realloc(files, capacity * sizeof(char*));
        }
        files[*count] = strdup(entry->d_name);
        (*count)++;
    }
    closedir(dir);

    if (*count == 0) {
        free(files);
        return NULL;
    }

    qsort(files, *count, sizeof(char*), compare_strings);
    return files;
}

static char* get_safetensors_path(const char *model_dir) {
    static char path[4096];

    int count;
    char **files = find_safetensors_files(model_dir, &count);
    if (!files) return NULL;

    snprintf(path, sizeof(path), "%s/%s", model_dir, files[0]);

    for (int i = 0; i < count; i++) free(files[i]);
    free(files);

    return path;
}

int load_config(const char *model_dir, Config *config) {
    char config_path[4096];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    FILE *f = fopen(config_path, "rb");
    if (!f) {
        fprintf(stderr, "Could not open config.json at %s\n", config_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)malloc(size + 1);
    if (!json_str) {
        fclose(f);
        return -1;
    }

    if (fread(json_str, 1, size, f) != (size_t)size) {
        free(json_str);
        fclose(f);
        return -1;
    }
    json_str[size] = '\0';
    fclose(f);

    char error[256] = {0};
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);

    if (!root) {
        fprintf(stderr, "Failed to parse config.json: %s\n", error);
        return -1;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (!cfg) cfg = root;

    memset(config, 0, sizeof(Config));

    config->dim        = json_get_int(json_object_get(cfg, "hidden_size"), 896);
    config->n_heads    = json_get_int(json_object_get(cfg, "num_attention_heads"), 14);
    config->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), config->n_heads);
    config->n_layer    = json_get_int(json_object_get(cfg, "num_hidden_layers"), 24);
    config->n_mlp      = json_get_int(json_object_get(cfg, "intermediate_size"), 4864);
    if (config->n_mlp == 0) {
        config->n_mlp = json_get_int(json_object_get(cfg, "shared_expert_intermediate_size"), 4864);
    }
    config->vocab_size   = json_get_int   (json_object_get(cfg, "vocab_size"), 151936);
    config->rope_theta   = json_get_double(json_object_get(cfg, "rope_theta"), 10000.0);
    config->rms_norm_eps = json_get_double(json_object_get(cfg, "rms_norm_eps"), 1e-6);
    config->d_head       = json_get_int   (json_object_get(cfg, "head_dim"), config->dim / config->n_heads);
    config->tie_word_embeddings = json_get_bool(json_object_get(root, "tie_word_embeddings"), 0);

    config->n_linear_k_heads   = json_get_int(json_object_get(cfg, "linear_num_key_heads"), 0);
    config->n_linear_v_heads   = json_get_int(json_object_get(cfg, "linear_num_value_heads"), 0);
    config->d_linear_k         = json_get_int(json_object_get(cfg, "linear_key_head_dim"), 0);
    config->d_linear_v         = json_get_int(json_object_get(cfg, "linear_value_head_dim"), 0);
    config->linear_conv_kernel = json_get_int(json_object_get(cfg, "linear_conv_kernel_dim"), 4);

    JsonValue *layer_types = json_object_get(cfg, "layer_types");

    config->n_full_attn_layers = 0;
    config->n_linear_attn_layers = 0;

    for (int i = 0; i < config->n_layer; i++) {
        if (get_layer_type(i, layer_types) == 1) {
            config->n_linear_attn_layers++;
        } else {
            config->n_full_attn_layers++;
        }
    }

    config->seq_len = 2048;

    json_free(root);

    fprintf(stderr, "Model config:\n");
    fprintf(stderr, "  dim: %d\n", config->dim);
    fprintf(stderr, "  n_heads: %d\n", config->n_heads);
    fprintf(stderr, "  n_kv_heads: %d\n", config->n_kv_heads);
    fprintf(stderr, "  n_layer: %d\n", config->n_layer);
    fprintf(stderr, "  n_mlp: %d\n", config->n_mlp);
    fprintf(stderr, "  vocab_size: %d\n", config->vocab_size);
    fprintf(stderr, "  d_head: %d\n", config->d_head);
    fprintf(stderr, "  rope_theta: %f\n", config->rope_theta);
    fprintf(stderr, "  rms_norm_eps: %f\n", config->rms_norm_eps);
    fprintf(stderr, "  tie_word_embeddings: %d\n", config->tie_word_embeddings);
    fprintf(stderr, "  n_full_attn_layers: %d\n", config->n_full_attn_layers);
    fprintf(stderr, "  n_linear_attn_layers: %d\n", config->n_linear_attn_layers);
    fprintf(stderr, "  n_linear_k_heads: %d\n", config->n_linear_k_heads);
    fprintf(stderr, "  n_linear_v_heads: %d\n", config->n_linear_v_heads);
    fprintf(stderr, "  d_linear_k: %d\n", config->d_linear_k);
    fprintf(stderr, "  d_linear_v: %d\n", config->d_linear_v);
    fprintf(stderr, "  linear_conv_kernel: %d\n", config->linear_conv_kernel);

    return 0;
}

static float* load_tensor_alloc(const csafetensors_t *st, const char *name, size_t expected_size) {
    const csafetensors_tensor_t *tensor = csafetensors_get_tensor(st, name);
    if (!tensor) return NULL;

    size_t num_elements = csafetensors_shape_size(tensor);
    float *output = (float *)malloc(num_elements * sizeof(float));
    if (!output) return NULL;

    if (load_tensor_auto(st, name, output, expected_size) != 0) {
        free(output);
        return NULL;
    }
    return output;
}

int load_weights_from_safetensors(Qwen35 *model, const char *model_dir) {
    Config *p = &model->config;
    Weights *w = &model->weights;

    char *safetensors_path = get_safetensors_path(model_dir);
    if (!safetensors_path) {
        fprintf(stderr, "Could not find safetensors file in %s\n", model_dir);
        return -1;
    }

    fprintf(stderr, "Loading weights from %s\n", safetensors_path);

    csafetensors_error_t err = csafetensors_load_from_file(safetensors_path, &model->safetensors);
    if (err != CSAFETENSORS_SUCCESS) {
        fprintf(stderr, "Failed to load safetensors: %s\n", csafetensors_get_error(&model->safetensors));
        return -1;
    }

    model->use_safetensors = 1;

    fprintf(stderr, "Loaded %zu tensors\n", model->safetensors.n_tensors);

    int head_size     = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim        = p->n_kv_heads * head_size;
    int key_dim       = p->n_linear_k_heads * p->d_linear_k;
    int value_dim     = p->n_linear_v_heads * p->d_linear_v;
    int conv_dim      = key_dim * 2 + value_dim;
    int q_dim         = p->n_heads * head_size * 2;
    int attn_out_dim  = p->n_heads * head_size;
    int n_full_attn   = p->n_full_attn_layers;
    int n_linear_attn = p->n_linear_attn_layers;

    fprintf(stderr, "Loading embedding weights...\n");
    w->token_embedding_table = load_tensor_alloc(&model->safetensors, "model.embed_tokens.weight", 0);
    if (!w->token_embedding_table) {
        w->token_embedding_table = load_tensor_alloc(&model->safetensors, "model.language_model.embed_tokens.weight", 0);
    }

    fprintf(stderr, "Loading attention layers...\n");
    w->rms_att_weight = (float *)malloc((size_t)p->n_layer * p->dim * sizeof(float));
    w->wq = (float *)malloc((size_t)n_full_attn * p->dim * q_dim * sizeof(float));
    w->wk = (float *)malloc((size_t)n_full_attn * p->dim * kv_dim * sizeof(float));
    w->wv = (float *)malloc((size_t)n_full_attn * p->dim * kv_dim * sizeof(float));
    w->wo = (float *)malloc((size_t)n_full_attn * attn_out_dim * p->dim * sizeof(float));
    w->q_norm = (float *)malloc((size_t)n_full_attn * head_size * sizeof(float));
    w->k_norm = (float *)malloc((size_t)n_full_attn * head_size * sizeof(float));

    int la = 0, ld = 0;
    for (int l = 0; l < p->n_layer; l++) {
        char name[256];

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        if (load_tensor_auto(&model->safetensors, name, w->rms_att_weight + l * p->dim, p->dim) != 0) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", l);
            load_tensor_auto(&model->safetensors, name, w->rms_att_weight + l * p->dim, p->dim);
        }

        int layer_type = model->layer_types[l];

        if (layer_type == 0) {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->wq + la * p->dim * q_dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
                load_tensor_auto(&model->safetensors, name, w->wq + la * p->dim * q_dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->wk + la * p->dim * kv_dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
                load_tensor_auto(&model->safetensors, name, w->wk + la * p->dim * kv_dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->wv + la * p->dim * kv_dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
                load_tensor_auto(&model->safetensors, name, w->wv + la * p->dim * kv_dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->wo + la * attn_out_dim * p->dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
                load_tensor_auto(&model->safetensors, name, w->wo + la * attn_out_dim * p->dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->q_norm + la * head_size, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", l);
                load_tensor_auto(&model->safetensors, name, w->q_norm + la * head_size, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->k_norm + la * head_size, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", l);
                load_tensor_auto(&model->safetensors, name, w->k_norm + la * head_size, 0);
            }

            la++;
        }
    }

    if (n_linear_attn > 0) {
        fprintf(stderr, "Loading linear attention layers...\n");
        w->in_proj_qkv = (float *)malloc((size_t)n_linear_attn * conv_dim * p->dim * sizeof(float));
        w->in_proj_z   = (float *)malloc((size_t)n_linear_attn * value_dim * p->dim * sizeof(float));
        w->in_proj_b   = (float *)malloc((size_t)n_linear_attn * p->n_linear_v_heads * p->dim * sizeof(float));
        w->in_proj_a   = (float *)malloc((size_t)n_linear_attn * p->n_linear_v_heads * p->dim * sizeof(float));
        w->conv1d_weight = (float *)malloc((size_t)n_linear_attn * conv_dim * p->linear_conv_kernel * sizeof(float));
        w->dt_bias     = (float *)malloc((size_t)n_linear_attn * p->n_linear_v_heads * sizeof(float));
        w->A_log       = (float *)malloc((size_t)n_linear_attn * p->n_linear_v_heads * sizeof(float));
        w->linear_norm = (float *)malloc((size_t)n_linear_attn * p->d_linear_v * sizeof(float));
        w->out_proj    = (float *)malloc((size_t)n_linear_attn * p->dim * value_dim * sizeof(float));

        ld = 0;
        for (int l = 0; l < p->n_layer; l++) {
            if (model->layer_types[l] != 1) continue;

            char name[256];

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->in_proj_qkv + ld * conv_dim * p->dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", l);
                load_tensor_auto(&model->safetensors, name, w->in_proj_qkv + ld * conv_dim * p->dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->in_proj_z + ld * value_dim * p->dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", l);
                load_tensor_auto(&model->safetensors, name, w->in_proj_z + ld * value_dim * p->dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->in_proj_b + ld * p->n_linear_v_heads * p->dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", l);
                load_tensor_auto(&model->safetensors, name, w->in_proj_b + ld * p->n_linear_v_heads * p->dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->in_proj_a + ld * p->n_linear_v_heads * p->dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", l);
                load_tensor_auto(&model->safetensors, name, w->in_proj_a + ld * p->n_linear_v_heads * p->dim, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->conv1d_weight + ld * conv_dim * p->linear_conv_kernel, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", l);
                load_tensor_auto(&model->safetensors, name, w->conv1d_weight + ld * conv_dim * p->linear_conv_kernel, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", l);
            if (load_tensor_auto(&model->safetensors, name, w->dt_bias + ld * p->n_linear_v_heads, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", l);
                load_tensor_auto(&model->safetensors, name, w->dt_bias + ld * p->n_linear_v_heads, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", l);
            if (load_tensor_auto(&model->safetensors, name, w->A_log + ld * p->n_linear_v_heads, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", l);
                load_tensor_auto(&model->safetensors, name, w->A_log + ld * p->n_linear_v_heads, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->linear_norm + ld * p->d_linear_v, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", l);
                load_tensor_auto(&model->safetensors, name, w->linear_norm + ld * p->d_linear_v, 0);
            }

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", l);
            if (load_tensor_auto(&model->safetensors, name, w->out_proj + ld * p->dim * value_dim, 0) != 0) {
                snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", l);
                load_tensor_auto(&model->safetensors, name, w->out_proj + ld * p->dim * value_dim, 0);
            }

            ld++;
        }
    }

    fprintf(stderr, "Loading FFN weights...\n");
    w->rms_ffn_weight = (float *)malloc((size_t)p->n_layer * p->dim * sizeof(float));
    w->w1 = (float *)malloc((size_t)p->n_layer * p->dim * p->n_mlp * sizeof(float));
    w->w2 = (float *)malloc((size_t)p->n_layer * p->n_mlp * p->dim * sizeof(float));
    w->w3 = (float *)malloc((size_t)p->n_layer * p->dim * p->n_mlp * sizeof(float));

    for (int l = 0; l < p->n_layer; l++) {
        char name[256];

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        if (load_tensor_auto(&model->safetensors, name, w->rms_ffn_weight + l * p->dim, p->dim) != 0) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", l);
            load_tensor_auto(&model->safetensors, name, w->rms_ffn_weight + l * p->dim, p->dim);
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
        if (load_tensor_auto(&model->safetensors, name, w->w1 + l * p->dim * p->n_mlp, 0) != 0) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_proj.weight", l);
            load_tensor_auto(&model->safetensors, name, w->w1 + l * p->dim * p->n_mlp, 0);
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
        if (load_tensor_auto(&model->safetensors, name, w->w2 + l * p->n_mlp * p->dim, 0) != 0) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.down_proj.weight", l);
            load_tensor_auto(&model->safetensors, name, w->w2 + l * p->n_mlp * p->dim, 0);
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
        if (load_tensor_auto(&model->safetensors, name, w->w3 + l * p->dim * p->n_mlp, 0) != 0) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.up_proj.weight", l);
            load_tensor_auto(&model->safetensors, name, w->w3 + l * p->dim * p->n_mlp, 0);
        }
    }

    fprintf(stderr, "Loading final norm...\n");
    w->rms_final_weight = load_tensor_alloc(&model->safetensors, "model.norm.weight", p->dim);
    if (!w->rms_final_weight) {
        w->rms_final_weight = load_tensor_alloc(&model->safetensors, "model.language_model.norm.weight", p->dim);
    }

    if (!p->tie_word_embeddings) {
        fprintf(stderr, "Loading lm_head...\n");
        w->wcls = load_tensor_alloc(&model->safetensors, "lm_head.weight", 0);
    } else {
        w->wcls = w->token_embedding_table;
    }

    fprintf(stderr, "Weights loaded successfully\n");
    csafetensors_free(&model->safetensors);
    model->use_safetensors = 0;
    return 0;
}

void build_qwen35(Qwen35 *t, char* model_path) {
    memset(t, 0, sizeof(Qwen35));

    if (load_config(model_path, &t->config) != 0) {
        fprintf(stderr, "Failed to load config\n");
        exit(EXIT_FAILURE);
    }

    t->layer_types = calloc(t->config.n_layer, sizeof(int));
    t->attn_layer_indices = calloc(t->config.n_layer, sizeof(int));
    t->deltanet_layer_indices = calloc(t->config.n_layer, sizeof(int));

    char config_path[4096];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_path);
    FILE *f = fopen(config_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *json_str = (char *)malloc(size + 1);
        if (json_str && fread(json_str, 1, size, f) == (size_t)size) {
            json_str[size] = '\0';
            fclose(f);

            char error[256] = {0};
            JsonValue *root = json_parse(json_str, size, error, sizeof(error));
            free(json_str);

            if (root) {
                JsonValue *cfg = json_object_get(root, "text_config");
                if (!cfg) cfg = root;
                JsonValue *layer_types = json_object_get(cfg, "layer_types");

                int la = 0, ld = 0;
                for (int i = 0; i < t->config.n_layer; i++) {
                    t->layer_types[i] = get_layer_type(i, layer_types);
                    if (t->layer_types[i] == 1) {
                        t->deltanet_layer_indices[i] = ld++;
                    } else {
                        t->attn_layer_indices[i] = la++;
                    }
                }
                json_free(root);
            }
        } else {
            if (json_str) free(json_str);
            if (f) fclose(f);
        }
    }

    if (load_weights_from_safetensors(t, model_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        exit(EXIT_FAILURE);
    }

    malloc_run_state(&t->state, &t->config);
}

void free_qwen35(Qwen35* t) {
    Weights *w = &t->weights;

    free(w->data);
    free(w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->q_norm);
    free(w->k_norm);
    free(w->in_proj_qkv);
    free(w->in_proj_z);
    free(w->in_proj_b);
    free(w->in_proj_a);
    free(w->conv1d_weight);
    free(w->dt_bias);
    free(w->A_log);
    free(w->linear_norm);
    free(w->out_proj);
    free(w->rms_ffn_weight);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->rms_final_weight);
    if (!t->config.tie_word_embeddings) {
        free(w->wcls);
    }

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
    /* OpenMP only for large output rows; small d is faster serial + SIMD. */
    #pragma omp parallel for if (d > 256)
    for (int i = 0; i < d; i++) {
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
    for (int i = 0; i < head_size; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float q0 = q[i];
            float q1 = q[i + 1];
            q[i]     = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            float* k = s->k + h * head_size;
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i]     = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
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
    #pragma omp parallel for
    for (int h = 0; h < p->n_heads; h++) {
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
    #pragma omp parallel for
    for (int i = 0; i < n_v_heads; i++) {
        s->beta[i] = sigmoid(matmul_scalar(s->xb, in_proj_b + i * dim, dim));
    }

    // g (decay rate) per head: g = A * softplus(a + dt_bias), A < 0
    #pragma omp parallel for
    for (int i = 0; i < n_v_heads; i++) {
        float a_val = matmul_scalar(s->xb, in_proj_a + i * dim, dim);
        float A = -expf(A_log[i]);
        s->g[i] = A * softplus(a_val + dt_bias[i]);
    }

    // shift conv state buffer
    for (int i = 0; i < conv_dim; i++) {
        for (int j = 0; j < conv_kernel - 1; j++) {
            conv_state[i * conv_kernel + j] = conv_state[i * conv_kernel + j + 1];
        }
        conv_state[i * conv_kernel + conv_kernel - 1] = s->qkv[i];
    }

    // apply depthwise conv1d weights and SiLU to qkv
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
    #pragma omp parallel for
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
        #pragma omp parallel for if (sampler->vocab_size > 4096)
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

static char* resolve_model_path(const char *model_name) {
    static char resolved_path[4096];

    if (model_name[0] == '/' || model_name[0] == '.') {
        return (char*)model_name;
    }

    FILE *f = fopen("models.json", "rb");
    if (!f) {
        return (char*)model_name;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)malloc(size + 1);
    if (!json_str) {
        fclose(f);
        return (char*)model_name;
    }

    if (fread(json_str, 1, size, f) != (size_t)size) {
        free(json_str);
        fclose(f);
        return (char*)model_name;
    }
    json_str[size] = '\0';
    fclose(f);

    char error[256] = {0};
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);

    if (!root) {
        return (char*)model_name;
    }

    JsonValue *model_entry = json_object_get(root, model_name);
    if (!model_entry || model_entry->type != JSON_OBJECT) {
        json_free(root);
        return (char*)model_name;
    }

    JsonValue *path_val = json_object_get(model_entry, "path");
    if (!path_val || path_val->type != JSON_STRING) {
        json_free(root);
        return (char*)model_name;
    }

    strncpy(resolved_path, path_val->data.string, sizeof(resolved_path) - 1);
    resolved_path[sizeof(resolved_path) - 1] = '\0';

    json_free(root);
    return resolved_path;
}

void error_usage() {
    fprintf(stderr, "Usage:   qwen35 <model> [options]\n");
    fprintf(stderr, "         qwen35 <model_dir> [options]\n");
    fprintf(stderr, "Example: qwen35 Qwen/Qwen3.5-0.8B -y \"You are a helpful assistant.\"\n");
    fprintf(stderr, "         qwen35 ./Qwen3.5-0.8B -y \"You are a helpful assistant.\"\n");
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

    char *model_arg = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "chat";
    char *system_prompt = NULL;

    if (argc >= 2) { model_arg = argv[1]; } else { error_usage(); }

    char *model_path = resolve_model_path(model_arg);
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
    build_qwen35(&model, model_path);
    if (steps == 0 || steps > model.config.seq_len) steps = model.config.seq_len;

    Tokenizer tokenizer;
    char tokenizer_full_path[8192];
    if (tokenizer_path[0] != '/') {
        snprintf(tokenizer_full_path, sizeof(tokenizer_full_path), "%s/%s", model_path, tokenizer_path);
        FILE *f = fopen(tokenizer_full_path, "rb");
        if (f) {
            fclose(f);
            tokenizer_path = tokenizer_full_path;
        }
    }
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
