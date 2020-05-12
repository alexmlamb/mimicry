
import torch
from torch import nn
import torch.nn.functional as F

class RIMSEncoder(nn.Module):
    def __init__(self, num_rims, num_channels, num_heads, depth, top_k,
                 spatial_shape=[1, 1], return_all=False):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.num_heads = num_heads
        self.num_channels_per_rim = num_channels_per_rim = num_channels
        self.depth = depth
        self.num_rims = num_rims
        self.top_k = top_k
        self.return_all = return_all
        self.rims = nn.Parameter(
            torch.randn([num_rims, num_channels_per_rim, *spatial_shape]) * 0.01)
        self.dropout = nn.Dropout(0.5)

        # forward weights
        self.forward_key_proj = nn.Sequential(
            nn.Conv2d(num_channels_per_rim,num_rims * num_heads * depth,
                      spatial_shape, padding=spatial_shape[0] // 2),
            nn.ReLU(),
        )
        self.forward_value_proj = nn.Sequential(
            nn.Conv2d(num_channels_per_rim, num_rims * num_heads * depth,
            spatial_shape, padding=spatial_shape[0] // 2),
            nn.ReLU(),
        )
        self.forward_query_proj = nn.Sequential(
            nn.Conv2d(num_rims * num_channels_per_rim,
                      num_rims * num_heads * depth, spatial_shape,
                      groups=num_rims),
            nn.ReLU(),
        )
        self.forward_merge_heads = nn.Conv2d(
            num_rims * num_heads * depth,
            num_rims * num_channels_per_rim, [1, 1], groups=num_rims)
        self.forward_out_proj = nn.Sequential(
            nn.Conv2d(num_rims * num_channels_per_rim if return_all \
                      else num_channels_per_rim, num_channels_per_rim, [1, 1]),
            nn.ReLU(),
        )

    @property
    def rims_query(self):
        rims = self.rims.view(1, -1, *self.spatial_shape)
        return self.forward_query_proj(rims)

    def input_key(self, x):
        x = torch.cat([x, torch.zeros_like(x)], dim=0)
        return self.forward_key_proj(x)

    def input_value(self, x):
        return self.forward_value_proj(x)

    def forward(self, x):
        # print("Input:", x.shape)
        top_k = self.top_k
        key = self.input_key(x)
        # print("Key:", key.shape)
        value = self.input_value(x)
        # print("Value:", value.shape)
        query = self.rims_query
        # print("Query:", query.shape)
        query_ = query.view(
            self.num_rims * self.num_heads, self.depth, 1, 1)
        attn_inner_product = F.conv2d(key, query_, groups=self.num_rims * self.num_heads)
        # print("Attn Inner Product:", attn_inner_product.shape)
        attn_inner_product = attn_inner_product.view(2, -1, *attn_inner_product.shape[1:])
        attn = F.softmax(attn_inner_product, dim=0) # binary selection
        attn_sum = attn.view(2, -1, self.num_rims, self.num_heads, *attn.shape[3:]).sum(3)
        # print("Attn Sum:", attn_sum.shape)
        _, top_k_indice = torch.topk(attn_sum[0], k=top_k, dim=1, sorted=True)
        value = self.forward_merge_heads(value)
        # print("Value 2:", value.shape)
        value_ = value.view(-1, self.num_rims, self.num_channels_per_rim, *value.shape[-2:])
        # print("Value 2 reshaped:", value_.shape)
        top_k_indice = top_k_indice[:, :, None].repeat(1, 1, self.num_channels_per_rim, 1, 1)
        if self.return_all:
            # option: R = R + output
            # in this option we return all the rims, some updated, some not
            rims_ = self.rims.view(
                1, self.num_rims * self.num_channels_per_rim, *self.spatial_shape)
            output = F.conv_transpose2d(
                torch.ones(value_.shape[0], 1, *value_.shape[-2:]).cuda(), rims_,
                padding=self.spatial_shape[0] // 2).view(
                    -1, self.num_rims, self.num_channels_per_rim, *value.shape[-2:])
            update = torch.gather(value_, 1, top_k_indice)
            output = output.scatter_add(1, top_k_indice, update).view(
                -1, self.num_rims * self.num_channels_per_rim, *output.shape[-2:])
            output = self.dropout(output)
        else:
            # option: just output
            # in this option we return just the activated outputs
            # they are summed instead of concat to be invariant to order
            output = value_
            output = torch.gather(value_, 1, top_k_indice)
            # print("Output:", output.shape)
            output = output.mean(1)
            #print("Output 2:", tf.shape(output))
            output = self.dropout(output)
        output = F.relu(output)
        output = self.forward_out_proj(output)
        # print("Output 3:", output.shape)
        return output


class RIMSDecoder(nn.Module):
    def __init__(self, num_rims, num_channels, num_heads, depth, top_k,
                 spatial_shape=[1, 1], return_all=False, rims=None):
        """pass rims if the rims are shared from encoder
        """
        super().__init__()
        self.spatial_shape = spatial_shape
        self.num_heads = num_heads
        self.num_channels_per_rim = num_channels_per_rim = num_channels
        self.depth = depth
        self.num_rims = num_rims
        self.top_k = top_k
        self.return_all = return_all
        if rims is None:
            self.rims = nn.Parameter(
                torch.randn(
                    [num_rims, num_channels_per_rim, *spatial_shape]) * 0.01)
        else:
            self.rims = rims

        # backward weights
        self.backward_key_proj = nn.Conv2d(
            num_rims * num_channels_per_rim,
            num_rims * num_heads * depth,
            spatial_shape,
            groups=num_rims
        )
        self.backward_value_proj = nn.Conv2d(
            num_rims * num_channels_per_rim,
            num_rims * num_heads * depth,
            [1, 1],
            groups=num_rims,
        )
        self.backward_query_proj = nn.Conv2d(
            num_channels_per_rim,
            num_rims * num_heads * depth,
            [1, 1]
        )
        self.backward_merge_heads = nn.Conv2d(
            num_rims * num_heads * depth,
            num_channels_per_rim, [1, 1])

    @property
    def rims_key(self):
        rims = self.rims.view(1, -1, *self.spatial_shape)
        return self.backward_key_proj(rims)

    @property
    def rims_value(self):
        rims = self.rims.view(1, -1, *self.spatial_shape)
        return self.backward_value_proj(rims)

    def z_query(self, z):
        return self.backward_query_proj(z)

    def forward(self, z):
        # print("z:", z.shape)
        query = self.z_query(z)
        # print("Query:", query.shape)
        key = self.rims_key
        # print("Key:", key.shape)
        value = self.rims_value
        # print("Value:", value.shape)
        key_ = key.view(self.num_rims * self.num_heads * self.depth, 1, 1, 1)
        attn_inner_product = F.conv_transpose2d(
            query, key_, groups=self.num_rims * self.num_heads)
        # print("Attn Inner Product:", attn_inner_product.shape)
        attn_inner_product = attn_inner_product.view(
            -1, self.num_rims, self.num_heads, *attn_inner_product.shape[-2:])
        # print("Attn Inner Product Reshaped:", attn_inner_product.shape)
        attn = F.softmax(attn_inner_product, dim=1) # binary selection
        # print("Attn:", attn.shape)
        attn = attn.view(-1, self.num_rims * self.num_heads, *attn.shape[-2:])
        value_ = value.view(
            self.num_rims * self.num_heads, self.depth, *self.spatial_shape)
        # print("Value 2:", value_.shape)
        recons = F.conv_transpose2d(
            attn, value_, padding=self.spatial_shape[0]//2,
            groups=self.num_rims * self.num_heads)
        # print("Recons:", recons.shape)
        recons = self.backward_merge_heads(recons)
        # print("Recons 2:", recons.shape)
        return recons



if __name__ == "__main__":
    num_rims = 10
    num_channels = 12
    num_heads = 4
    spatial_shape = [3, 3]
    depth = 7
    batch_size = 24
    height = width = 32
    in_channels = out_channels = 12
    stride = 1
    return_all = False

    input = torch.randn([batch_size, num_channels, height, width])
    encoder = RIMSEncoder(
        num_rims, num_channels, num_heads, depth, top_k=2,
        spatial_shape=spatial_shape, return_all=False)
    decoder = RIMSDecoder(
        num_rims, num_channels, num_heads, depth, top_k=2,
        spatial_shape=spatial_shape, return_all=False, rims=encoder.rims)
    z = encoder(input)
    x = decoder(z)
    print(z)
    print(x)
