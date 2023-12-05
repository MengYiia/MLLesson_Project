if __name__ == '__main__':
    import torch
    x = torch.Tensor([[1, 2, 3], [1, 2, 3]])
    # 第一列就是1，第三列就是3
    # 这里的temp需要clone一份数据，因为x[:, 0]实际返回的是引用，而不是数据本身，temp还是指向原x[:, 0]，后面修改x[:, 0]的时候也会修改temp
    # 所以需要clone一份
    temp = x[:, 0].clone()
    x[:, 0] = x[:, -1]
    x[:, -1] = temp
    print(x)