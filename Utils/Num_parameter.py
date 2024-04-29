# Counts the number of parameters in a nn.Module object
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
