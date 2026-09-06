#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <omp.h>

// mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <ctime>

struct Tokenizer
{
    int max_token_length; 
    int vocab_size; // passed by Config

    // token entry : 

    int* len_token;
    float* scores;
    char** vocab;

    std::unordered_map<std::string, int> hash_map;
};

struct Config
{
    int dim; // dimension of tokens vector
    int hidden_dim; // num of neurons in the hidden layer
    int n_blocks; // number of repetitions of transformer blocks
    int n_heads; // number of attention heads
    int n_kv_heads; // number of key value heads
    int vocab_size; // total number of tokens
    int max_context_window; // in tokens
};

struct Weights
{
    void* mmap_ptr;
    size_t mmap_size;
    
    float* token_embedding_table; // total representation of tokens by vectors

    // attention weights
    float* rms_att_weight; 
    float* wq;
    float* wk;
    float* wv;
    float* wo;

    // neural net weights
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;

    float* rms_final_weight;
    float* wcls;
};

struct State {
    std::vector<int> input_tokens;
    int n_tokens;
    int pos;
    std::vector<float> embeded_input;

    std::vector<float> x;
    std::vector<float> x_buffer;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;

    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<float> logits;
    std::vector<float> gate_buffer;
    std::vector<float> up_buffer;
    std::vector<float> down_buffer;
    std::vector<float> wo_output;

    State(int dim, int hidden_dim, int n_layers, int seq_len, int n_kv_heads, int head_size, int vocab_size) : 
        x(dim),
        x_buffer(dim),
        q(dim),
        k(dim),
        v(dim),
        logits(vocab_size),
        k_cache(n_layers * seq_len * n_kv_heads * head_size),
        v_cache(n_layers * seq_len * n_kv_heads * head_size),
        gate_buffer(hidden_dim),
        up_buffer(hidden_dim),
        down_buffer(dim),
        wo_output(dim) {}
};

void loadWeights(Weights* weights, Config* config, const std::string& model_name) {
    int fd = open(model_name.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open stories110M.bin." << std::endl;
        exit(1);
    }

    //file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Failed to get file size." << std::endl;
        exit(1);
    }
    weights->mmap_size = sb.st_size;


    weights->mmap_ptr = mmap(NULL, weights->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (weights->mmap_ptr == MAP_FAILED) {
        std::cerr << "mmap failed." << std::endl;
        exit(1);
    }
    close(fd);

    std::memcpy(config, weights->mmap_ptr, sizeof(Config));

    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = std::abs(config->vocab_size);


    float* ptr = (float*)((char*)weights->mmap_ptr + sizeof(Config)); // tracking pointer

    int head_size = config->dim / config->n_heads;

    weights->token_embedding_table = ptr; ptr += config->vocab_size * config->dim;

    weights->rms_att_weight = ptr; ptr += config->n_blocks * config->dim; 

    weights->wq = ptr; ptr += (size_t)config->n_blocks * config->dim * (config->n_heads * head_size); // size_t for safety

    weights->wk = ptr; ptr += (size_t)config->n_blocks * config->dim * (config->n_kv_heads * head_size);

    weights->wv = ptr; ptr += (size_t)config->n_blocks * config->dim * (config->n_kv_heads * head_size);

    weights->wo = ptr; ptr += (size_t)config->n_blocks * config->dim * (config->n_heads * head_size);

    weights->rms_ffn_weight = ptr; ptr += config->n_blocks * config->dim;

    weights->w1 = ptr; ptr += (size_t)config->n_blocks * config->dim * config->hidden_dim;

    weights->w2 = ptr; ptr += (size_t)config->n_blocks * config->hidden_dim * config->dim;

    weights->w3 = ptr; ptr += (size_t)config->n_blocks * config->dim * config->hidden_dim;

    weights->rms_final_weight = ptr; ptr += config->dim;

    if (shared_weights) {
        weights->wcls = weights->token_embedding_table;
    } else {
        weights->wcls = ptr; 
    }
}

void loadTokenizer( Tokenizer* tokenizer, int vocab_size , std::string tokenizer_name){
    std::ifstream file(tokenizer_name, std::ios::binary);

    if (!file.is_open()){
        std::cerr << "Failed to open tokenizer.bin." << std::endl;
        return;
    }

    tokenizer->vocab_size = vocab_size;
    file.read(reinterpret_cast<char*>(&tokenizer->max_token_length), sizeof(int));

    tokenizer->scores = new float[vocab_size];
    tokenizer->len_token = new int[vocab_size];
    tokenizer->vocab = new char*[vocab_size];
    tokenizer->hash_map.reserve(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        file.read(reinterpret_cast<char*>(&tokenizer->scores[i]), sizeof(float));
        file.read(reinterpret_cast<char*>(&tokenizer->len_token[i]), sizeof(int));

        tokenizer->vocab[i] = new char[tokenizer->len_token[i] + 1];
        file.read(tokenizer->vocab[i], tokenizer->len_token[i]);
        tokenizer->vocab[i][tokenizer->len_token[i]] = '\0'; // null-terminate string

        tokenizer->hash_map.emplace(tokenizer->vocab[i], i);
    }

}

void encode(std::string input, Tokenizer* tokenizer, State* state, Config* config) {
    
    std::string processed = "\xe2\x96\x81";
    for (char c : input) {
        if (c == ' ') {
            processed += "\xe2\x96\x81";
        } else {
            processed += c;
        }
    }

    std::vector<std::string> chars;

    for (size_t i = 0; i < processed.size(); ) {
        if ((unsigned char)processed[i] == 0xe2 && i + 2 < processed.size()) {
            chars.push_back(processed.substr(i, 3));
            i += 3;
        } else {
            chars.push_back(std::string(1, processed[i]));
            i += 1;
        }
    }

    while (chars.size() > 1 ) {
        float max = -std::numeric_limits<float>::infinity();
        int idx = -1;
        std::string combination;

        for( int i = 0; i < chars.size()-1 ; i++){
            std::string concated = std::string( chars[i] + chars[i + 1]);

            auto it = tokenizer->hash_map.find(concated);
            if (it != tokenizer->hash_map.end()){
                if( tokenizer->scores[it->second] > max) {
                    max = tokenizer->scores[it->second];
                    idx = i;
                    combination = concated;
                }  
            }
        }
        if (idx== -1) break; 

        chars[idx] += chars[idx + 1];
        chars.erase(chars.begin() + idx +1);
    }

    for (int i = 0; i < chars.size(); i++){
        std::cout<<chars[i];
    }
    std::cout<<std::endl;

    state->n_tokens = 0;
    state->input_tokens.resize(config->max_context_window);

    state->input_tokens[state->n_tokens] = 1;
    state->n_tokens++;

    for (size_t i = 0; i < chars.size(); i++) {
        auto it = tokenizer->hash_map.find(chars[i]);
        if (it != tokenizer->hash_map.end()) {
            state->input_tokens[state->n_tokens] = it->second;
            state->n_tokens++;
        }
  
    }
}

void rmsNorm(float* weights, Config *config, float* x, float* xb){
    
    float rms_scalar = 0.f;

    for(int i = 0; i<config->dim; i++){
        rms_scalar += x[i] * x[i];
    }
    rms_scalar = rms_scalar / config->dim;
    rms_scalar += 0.00001;
    rms_scalar = std::sqrt(rms_scalar);
    rms_scalar = 1 / rms_scalar;

    for(int i = 0; i<config->dim; i++){
        xb[i] = x[i] * rms_scalar * weights[i];
    }


}

void matmul(float* out, float* x, float* w, int n_out, int n_in) {
    #pragma omp parallel for // parallelization
    for (int i = 0; i < n_out; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_in; j++) {
            sum += w[i * n_in + j] * x[j];
        }
        out[i] = sum;
    }
}

void computeQKV(float* wq, float* wk, float* wv, Config* config , State *state){
    int dim = config->dim;
    int head_size = dim / config->n_heads;
    int kv_dim = config->n_kv_heads * head_size;

    matmul(state->q.data(), state->x_buffer.data(), wq, dim, dim);
    matmul(state->k.data(), state->x_buffer.data(), wk, kv_dim, dim);
    matmul(state->v.data(), state->x_buffer.data(), wv, kv_dim, dim);
}

void applyRoPE( float* q, float* k, int pos, Config* config){ 
    int head_size = config->dim / config->n_heads;

    for(int i = 0; i<config->n_heads; i++){
        for( int j = 0; j< head_size ; j+=2){
            float freq = 1.0f / pow(10000.0f, (float)j / (float)head_size); // enforce float div
            float angle = pos * freq;

            int idx = i * head_size + j;
            float q0_temp = q[idx];
            float q1_temp = q[idx + 1];
            q[idx] = q0_temp * cos(angle) - q1_temp * sin(angle);
            q[idx+1] = q0_temp * sin(angle) + q1_temp * cos(angle);
        }
    }

    for(int i = 0; i<config->n_kv_heads; i++){
        for( int j = 0; j< head_size ; j+=2){
            float freq = 1.0f / pow(10000.0f, (float)j / (float)head_size);
            float angle = pos * freq;

            int idx = i * head_size + j;
            float k0_temp = k[idx];
            float k1_temp = k[idx + 1];
            k[idx] = k0_temp * cos(angle) - k1_temp * sin(angle);
            k[idx+1] = k0_temp * sin(angle) + k1_temp * cos(angle);
        }
    }
}

float SiLU(float x){
    return x * (1 / (1 + exp(-x)));
}

void findTopK(std::vector<int>& indices, float* logits, int vocab_size, int k){
    indices.resize(vocab_size);
    std::iota(indices.begin(), indices.end(), 0); // populate tokens

    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [logits](int a, int b) {
            return logits[a] > logits[b];
        }
    );
}

void runTransformer(Weights* weights, Config* config, State* state, float temperature, int top_k){
    int head_size = config->dim / config->n_heads;
    int kv_dim = config->n_kv_heads * head_size;
    
    std::vector<float> scores = std::vector<float>(config->max_context_window);

    for(int i = state->pos; i < state->n_tokens; i++){

        int token_id = state->input_tokens[i];
    
        float* token_embed = weights->token_embedding_table + (token_id * config->dim);
        std::memcpy(state->x.data(), token_embed, config->dim * sizeof(float));

        for(int layer = 0; layer<config->n_blocks; layer++){
            float* rms_weight_att = weights->rms_att_weight + (layer * config->dim);
            float* rms_weight_ffn = weights->rms_ffn_weight + (layer * config->dim);

            float* q_layer = weights->wq + (size_t)layer * (config->dim*config->dim);
            float* k_layer = weights->wk + (size_t)layer * (config->dim*kv_dim);
            float* v_layer = weights->wv + (size_t)layer * (config->dim*kv_dim);

            rmsNorm( rms_weight_att, config, state->x.data(), state->x_buffer.data());

            computeQKV( q_layer, k_layer, v_layer, config, state );

            applyRoPE(state->q.data(), state->k.data(), i, config);


            int block_offset = layer * config->max_context_window * kv_dim;
            int pos_offset = i * kv_dim;

            float* k_dest = state->k_cache.data() + block_offset + pos_offset;
            float* v_dest = state->v_cache.data() + block_offset + pos_offset;

            std::memcpy(k_dest, state->k.data(), kv_dim * sizeof(float));
            std::memcpy(v_dest, state->v.data(), kv_dim * sizeof(float));

            for(int h = 0; h < config->n_heads; h++){

                float* slice_q = state->q.data() + h * head_size;
                int kv_head = h / (config->n_heads / config->n_kv_heads);

                // dot prod
                for(int t = 0; t <= i; t++){
                    float dot_prod = 0;

                    float* k_head = state->k_cache.data() + block_offset + (t * kv_dim) + (kv_head * head_size);

                    for(int j = 0; j<head_size; j++){
                        
                        dot_prod += slice_q[j] * k_head[j];
                    }
                    dot_prod = dot_prod / sqrt(head_size);
                    scores[t] = dot_prod;

                }

                // softmax
                float max = -std::numeric_limits<float>::infinity();
                for(int j = 0; j<=i; j++){
                    if( scores[j] > max ) { max = scores[j]; }
                }
                float sum_exp = 0.0;
                for( int j = 0; j<=i; j++ ){
                    scores[j] = exp( scores[j] - max );
                    sum_exp += scores[j];
                }
                for(int j = 0; j<=i; j++){ scores[j] /= sum_exp; }

                // clean buffer
                float* head_offset = state->x_buffer.data() + (head_size * h);
                for(int j = 0; j < head_size; j++){
                    head_offset[j] = 0.0f;
                }

                // attention
                for(int t = 0; t <= i; t++){
                    float* v_head = state->v_cache.data() + block_offset + (t * kv_dim) + (kv_head * head_size);

                    for(int j=0; j<head_size; j++){
                        state->x_buffer[h * head_size + j] += scores[t] * v_head[j];
                    }
                }
            }

            // wo and add
            float* wo_offset = weights->wo + (size_t)layer * (config->dim * config->dim);
            matmul(state->wo_output.data(), state->x_buffer.data(),wo_offset, config->dim, config->dim);

            for(int j = 0; j<config->dim; j++){
                state->x[j] += state->wo_output[j]; 
            }

            // pre ffn norm
            rmsNorm(rms_weight_ffn, config, state->x.data(), state->x_buffer.data());

            //ffn:
            int token = state->input_tokens[i];
            float* w1_offset = weights->w1 + (size_t)layer * (config->dim * config->hidden_dim);
            float* w3_offset = weights->w3 + (size_t)layer * (config->dim * config->hidden_dim);
            float* w2_offset = weights->w2 + (size_t)layer * (config->hidden_dim * config->dim);

            matmul(state->gate_buffer.data(), state->x_buffer.data(), w1_offset, config->hidden_dim, config->dim);
            for(int j= 0; j<config->hidden_dim; j++){
                state->gate_buffer[j] = SiLU(state->gate_buffer[j]);
            }
            matmul(state->up_buffer.data(), state->x_buffer.data(), w3_offset, config->hidden_dim, config->dim);
            for(int j = 0; j<config->hidden_dim; j++){
                state->gate_buffer[j] = state->gate_buffer[j] * state->up_buffer[j];
            }
            matmul(state->down_buffer.data(), state->gate_buffer.data(),w2_offset,config->dim, config->hidden_dim);
            for(int j=0; j<config->dim; j++){
                state->x[j] += state->down_buffer[j];
            }            
        }
    }

    // final norm
    float* rms_weight_final = weights->rms_final_weight;
    rmsNorm(rms_weight_final,config,state->x.data(), state->x_buffer.data());

    matmul(state->logits.data(), state->x_buffer.data(), weights->wcls, config->vocab_size, config->dim);

    // temperature sampling
    for(int j=0; j<config->vocab_size;j++){
        state->logits[j] /= temperature;
    }

    std::vector<int> top_indices;
    findTopK(top_indices, state->logits.data(), config->vocab_size, top_k);

    // softmax
    float max_val = state->logits[top_indices[0]];
    
    std::vector<float> top_probs(top_k);
    float sum_exp = 0.0f;
    for (int j = 0; j < top_k; j++) {
        top_probs[j] = exp(state->logits[top_indices[j]] - max_val);
        sum_exp += top_probs[j];
    }
    for (int j = 0; j < top_k; j++) {
        top_probs[j] /= sum_exp;
    }

    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int idx = top_indices[top_k - 1];

    for (int j = 0; j < top_k; j++) {
        cdf += top_probs[j];
        if (r <= cdf) {
            idx = top_indices[j];
            break;
        }
    }

    state->input_tokens[state->n_tokens] = idx;
    state->n_tokens++;
}

int main() {
    std::string model_name = "stories110M.bin";
    std::string tokenizer_name = "tokenizer.bin";

    std::string input = "Once upon a time ";
    float temperature = 0.3;
    int top_k = 20;


    srand(time(NULL));

    Weights weights;
    Config config;
    loadWeights(&weights, &config, model_name);

    Tokenizer tokenizer;
    loadTokenizer(&tokenizer, config.vocab_size, tokenizer_name);

    int head_size = config.dim / config.n_heads;

    State state(config.dim, config.hidden_dim, config.n_blocks, 
                config.max_context_window, config.n_kv_heads, head_size, config.vocab_size);
    state.pos = 0;



    
    encode(input, &tokenizer, &state, &config);
    int max_tokens = config.max_context_window;
    int n_tokens_start = state.n_tokens;
    auto newline = tokenizer.hash_map.find("<0x0A>");

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i<max_tokens; i++){
        runTransformer(&weights, &config, &state, temperature, top_k);
        state.pos = state.n_tokens - 1;
        int predicted_token = state.input_tokens[state.n_tokens - 1];

        if (predicted_token == 0 || predicted_token == 1 || predicted_token == 2) {
            break; 
        }
        if (predicted_token == newline->second){
            std::cout<<std::endl;
            continue;
        }
        
        std::cout << tokenizer.vocab[predicted_token];
        std::cout.flush(); 
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    float time_taken = duration.count() / 1000000.0f;
    std::cout<<std::endl<<std::endl;
    std::cout << "Time taken: "
         << time_taken
         << "seconds, "<< int((state.n_tokens - n_tokens_start)  /  time_taken)<<" tokens / s" << std::endl << std::endl;

    munmap(weights.mmap_ptr, weights.mmap_size);
    return 0;
}