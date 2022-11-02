from needle import backend_ndarray as nd


x = nd.NDArray([1, 2, 3], device=nd.cuda())
print(x * 2)