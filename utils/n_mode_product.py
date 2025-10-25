import torch

def n_mode_product_einsum(X, U, mode):
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    N = X.ndim
    assert N + 1 <= len(letters), "Increase alphabet if you have many dims."

    idx_in = letters[:N]
    s = idx_in[mode]
    r = letters[N]
    idx_out = idx_in.copy()
    idx_out[mode] = r

    eq = f"{''.join(idx_in)},{s}{r}->{''.join(idx_out)}"
    return torch.einsum(eq, X, U)


if __name__ == "__main__":
    X = torch.randn(2, 3, 4, 5, 6, 7)
    print("Original shape:", tuple(X.shape))

    U1 = torch.randn(X.shape[3], 8)
    Y1 = n_mode_product_einsum(X, U1, mode=3)
    print("After mode-3 product:", tuple(Y1.shape), f": {X.shape[3]}--> {Y1.shape[3]}")

    U2 = torch.randn(Y1.shape[1], 10)
    Y2 = n_mode_product_einsum(Y1, U2, mode=1)
    print("After mode-1 product:", tuple(Y2.shape), f": {Y1.shape[1]}--> {Y2.shape[1]}")
