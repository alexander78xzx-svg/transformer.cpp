#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <unordered_map>

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

std::vector<int> encode(std::string input, Tokenizer* tokenizer) { // to optimize
    std::vector<int> tokens;

    std::string processed = "\xe2\x96\x81";
    for (char c : input) {
    if (c == ' ') {
        processed += "\xe2\x96\x81";
    } else {
        processed += c;
    }
}

    tokens.push_back(1); // Beginning of sequence token

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
        float max = -std::numeric_limits<float>::infinity();;
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

    for (size_t i = 0; i < chars.size(); i++) {
        auto it = tokenizer->hash_map.find(chars[i]);
        if (it != tokenizer->hash_map.end()) {
            tokens.push_back(it->second);
        }
    }

    for (int i = 0; i < tokens.size(); i++){
        std::cout<<tokens[i] << " ";
    }
    std::cout<<std::endl;
    return tokens;
}

std::vector<std::vector<float>> embedTokens( std::vector<int>* tokens, Weights* weights, Config *config ) {
    std::vector<std::vector<float>> embeded;
    for(int i = 0; i<tokens->size(); i++){

        std::vector<float> temp;
        int start = (*tokens)[i] * config->dim;
        int end = start + config->dim;
        for(int j = start; j < end; j++){
            temp.push_back( weights->token_embedding_table[j] );
        }
        embeded.push_back(temp);
    }
    return embeded;
}

int main() {
    // load binaries in the memory

    Weights weights;
    Config config;
    loadWeights(&weights, &config);

    Tokenizer tokenizer;
    loadTokenizer(&tokenizer, config.vocab_size);

    std::string input = "Once upon a time";
    std::vector<int> encodedInput = encode(input, &tokenizer);

    std::vector<std::vector<float>> tokenizedInput = embedTokens(&encodedInput, &weights, &config );

    /*
    for(int i = 0; i < tokenizedInput.size(); i++){
        for(int j = 0; j < config.dim; j++){
            std::cout<<tokenizedInput[i][j] << std::endl;
        }
    } */

    return 0;
}