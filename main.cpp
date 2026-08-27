#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <unordered_map>
#include <cmath>
#include <cstring>

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
    int* input_tokens;
    int n_tokens;
    float* embeded_input;

    float* x;
    float* x_buffer;
    float* q;
    float* k;
    float* v;
};

void loadWeights( Weights *weights, Config *config) {
    
    std::ifstream file("stories15M.bin", std::ios::binary);

    if (!file.is_open()){
        std::cerr << "Failed to open stories15M.bin." << std::endl;
        return;
    }

    file.read(reinterpret_cast<char*>(config), sizeof(Config)); // reinterpret_cast<char*> marks &config as *char for the compiler

    // handle sign of vocab size
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = std::abs(config->vocab_size);
    
    file.seekg(0, std::ios::end); // we jump to the end of the file to measure total size
    size_t file_size = file.tellg(); // calculate file size in byte
    size_t weights_size = file_size - sizeof(Config);
    size_t num_floats = weights_size / sizeof(float);

    file.seekg(sizeof(Config), std::ios::beg);

    float* data = new float[num_floats]; // allocate memory for all weights
    file.read(reinterpret_cast<char*>(data), weights_size);

    float* ptr = data; // tracking pointer

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
        ptr += (size_t)config->vocab_size * config->dim;
    }
}

void loadTokenizer( Tokenizer* tokenizer, int vocab_size ){
    std::ifstream file("tokenizer.bin", std::ios::binary);

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

void encode(std::string input, Tokenizer* tokenizer, State* state) {
    
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
    state->input_tokens = new int[chars.size() + 1];

    state->input_tokens[state->n_tokens] = 1;
    state->n_tokens++;

    for (size_t i = 0; i < chars.size(); i++) {
        auto it = tokenizer->hash_map.find(chars[i]);
        if (it != tokenizer->hash_map.end()) {
            state->input_tokens[state->n_tokens] = it->second;
            state->n_tokens++;
        }
  
    }

    // output
    for (int i = 0; i < state->n_tokens; i++){
        std::cout << state->input_tokens[i] << std::endl;
    }
    std::cout<<std::endl;
}

void embedTokens( Weights* weights, Config *config, State* state ) {
    state->embeded_input = new float[state->n_tokens * config->dim];

    float* ptr = state->embeded_input;

    for(int i = 0; i< state->n_tokens; i++){
        float* src = weights->token_embedding_table + ((size_t)state->input_tokens[i] * config->dim);

        std::memcpy(ptr, src, config->dim * sizeof(float));

        ptr += config->dim;
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

    matmul(state->q, state->x_buffer, wq, dim, dim);

    matmul(state->k, state->x_buffer, wk, kv_dim, dim);
    matmul(state->v, state->x_buffer, wv, kv_dim, dim);
}

void applyRoPE( float* q, float* k, int pos, Config* config, State *state ){ 
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

void runTransformer(Weights* weights, Config* config, State* state){
    int head_size = config->dim / config->n_heads;
    int kv_dim = config->n_kv_heads * head_size;
    
    // one transformer bloc for now
    state->x = new float[config->dim];
    state->x_buffer = new float[config->dim];
    state->q = new float[config->dim];
    state->k = new float[config->dim];
    state->v = new float[config->dim];

    int layer = 0;

    float* rms_weight = weights->rms_att_weight + (layer * config->dim);

    float* q_layer = weights->wq + (size_t)layer * (config->dim*config->dim);
    float* k_layer = weights->wk + (size_t)layer * (config->dim*kv_dim);
    float* v_layer = weights->wv + (size_t)layer * (config->dim*kv_dim);

    for(int i = 0; i < state->n_tokens; i++){
        float* token_embed = state->embeded_input + (i * config->dim);
        std::memcpy(state->x, token_embed, config->dim * sizeof(float));

        rmsNorm( rms_weight, config, state->x, state->x_buffer);

        computeQKV( q_layer, k_layer, v_layer, config, state );

        applyRoPE(q_layer, k_layer, i, config, state);
    }
}

int main() {

    Weights weights;
    Config config;
    loadWeights(&weights, &config);

    Tokenizer tokenizer;
    loadTokenizer(&tokenizer, config.vocab_size);

    State state;

    std::string input = "Once upon a time";
    
    encode(input, &tokenizer, &state);

    embedTokens(&weights, &config, &state);

    runTransformer(&weights, &config, &state);

    return 0;
}