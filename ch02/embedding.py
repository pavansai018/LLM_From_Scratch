import torch

def main():
    vocab_size = 50257
    dimension = 12288

    token_embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=dimension)


if __name__ == '__main__':
    main()