from llava.train.train import train
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")


